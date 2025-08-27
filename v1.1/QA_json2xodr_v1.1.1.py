#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker — v1.1.1
改进要点（在 v1.1 基础上）：
- [CHANGE v1.1.1] “边界”行的 XODR 口径从 roadMark 实体改为 **拓扑边界数**（与 JSON 唯一边界ID口径一致）
- [NEW v1.1.1] 新增 “标线实体（roadMark）” 行，专门展示已绘制标线的实体数量（不计入综合分）
- [KEEP] sign→signal / 共有边界识别 / debug 输出等保持不变

用法：
    python QA_json2xodr_v1.1.1.py --json "Appendix_1 sample.json" --xodr "Appendix_2 sample.xodr" --out out_v1_1_1.html
"""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set


# -------------------------------
# 数据结构
# -------------------------------
@dataclass
class IntegrityRow:
    name: str
    json_count: int
    xodr_count: int
    subscore: float
    note: str = ""
    include_in_score: bool = True  # v1.1.1: 可选择不计入综合分


@dataclass
class QualityReport:
    completeness_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    rows: List[IntegrityRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# -------------------------------
# 主类
# -------------------------------
class QualityChecker:
    def __init__(self, json_file: str, xodr_file: str):
        self.json_path = Path(json_file)
        self.xodr_path = Path(xodr_file)
        self.json_data: Dict[str, Any] = {}
        self.xodr_root: ET.Element = None

    # 读取
    def load_json(self) -> None:
        with self.json_path.open("r", encoding="utf-8") as f:
            self.json_data = json.load(f)

    def load_xodr(self) -> None:
        tree = ET.parse(str(self.xodr_path))
        self.xodr_root = tree.getroot()

    # -------------------------------
    # JSON 侧解析工具
    # -------------------------------
    def _json_lanes(self) -> List[Dict[str, Any]]:
        return list(self.json_data.get("lanes", []))

    def _json_bounds(self) -> List[Dict[str, Any]]:
        return list(self.json_data.get("bounds", []))

    def _json_objects(self) -> List[Dict[str, Any]]:
        return list(self.json_data.get("objects", []))

    # v1.1: JSON 文件中的“sign”作为数组存储
    def _json_signs(self) -> List[Dict[str, Any]]:
        return list(self.json_data.get("sign", []))

    # 共有边界识别：
    # 若同一个 bound_id 被多个车道同时引用（作为 left/right 边界），则认为其为“共有边界”。
    # 返回：unique_ids, shared_ids
    def _detect_shared_bounds(self) -> Tuple[Set[int], Set[int]]:
        lanes = self._json_lanes()
        usage_count: Dict[int, int] = {}
        for ln in lanes:
            for key in ("left_bound_id", "right_bound_id"):
                bid = ln.get(key)
                if bid is None:
                    continue
                usage_count[bid] = usage_count.get(bid, 0) + 1
        unique_ids = {bid for bid in usage_count.keys()}
        shared_ids = {bid for bid, c in usage_count.items() if c >= 2}
        return unique_ids, shared_ids

    # -------------------------------
    # XODR 侧解析工具
    # -------------------------------
    def _xodr_count_driving_lanes(self) -> int:
        # 统计 type=driving 的 lane 数
        count = 0
        for lane in self.xodr_root.iter("lane"):
            t_attr = lane.get("type")
            if (t_attr and t_attr.lower() == "driving"):
                count += 1
                continue
            t_node = lane.find("type")
            if t_node is not None and (t_node.text or "").strip().lower() == "driving":
                count += 1
        return count

    def _xodr_collect_lane_marks(self) -> List[Dict[str, Any]]:
        """收集 XODR 中的标线：
        1) 遍历 road->lanes->laneSection->(left/center/right)->lane 下的 roadMark / roadMarks 容器；
        2) 若 1) 未检出，则用 root.iter('roadMark') 兜底一遍，避免非典型层级写法漏检。
        返回每条标线的上下文属性字典。
        """
        results: List[Dict[str, Any]] = []
        # 1) 结构化遍历
        for road in self.xodr_root.findall('road'):
            road_id = road.get('id')
            lanes_node = road.find('lanes')
            if lanes_node is None:
                continue
            for ls_idx, ls in enumerate(lanes_node.findall('laneSection')):
                for side in ('left', 'center', 'right'):
                    side_node = ls.find(side)
                    if side_node is None:
                        continue
                    for lane in side_node.findall('lane'):
                        lane_id = lane.get('id')
                        # 直接子节点 roadMark / roadmark
                        for tag in ('roadMark', 'roadmark'):
                            for rm in lane.findall(tag):
                                results.append({
                                    'road_id': road_id,
                                    'laneSection_index': ls_idx,
                                    'lane_side': side,
                                    'lane_id': lane_id,
                                    'tag': rm.tag,
                                    'attrib': dict(rm.attrib),
                                })
                        # 兼容 roadMarks 容器
                        rm_container = lane.find('roadMarks')
                        if rm_container is not None:
                            for rm in rm_container.findall('roadMark'):
                                results.append({
                                    'road_id': road_id,
                                    'laneSection_index': ls_idx,
                                    'lane_side': side,
                                    'lane_id': lane_id,
                                    'tag': rm.tag,
                                    'attrib': dict(rm.attrib),
                                })
        # 2) 兜底
        if not results:
            for node in self.xodr_root.iter('roadMark'):
                results.append({'tag': node.tag, 'attrib': dict(node.attrib)})
        return results

    def _xodr_count_objects(self) -> int:
        return sum(1 for _ in self.xodr_root.iter("object"))

    def _xodr_count_signals(self) -> int:
        return sum(1 for _ in self.xodr_root.iter("signal"))

    def _xodr_count_topo_boundaries(self) -> int:
        """按拓扑口径统计 laneSection 的边界数：
        左侧贡献 = left_lane_count（外缘1 + 内部相邻 left_lane_count-1）≈ left_lane_count
        右侧贡献 = right_lane_count
        中央分隔 = 1（若左右两侧都存在）
        总数 = 每个 laneSection 的上述三项之和，再对所有 road 累加。
        该口径与 JSON 的唯一边界ID统计更对齐；不会受 roadMark 是否绘制影响。
        """
        total = 0
        for road in self.xodr_root.findall('road'):
            lanes_node = road.find('lanes')
            if lanes_node is None:
                continue
            for ls in lanes_node.findall('laneSection'):
                left_node = ls.find('left')
                right_node = ls.find('right')
                left_cnt = len(left_node.findall('lane')) if left_node is not None else 0
                right_cnt = len(right_node.findall('lane')) if right_node is not None else 0
                section_total = 0
                if left_cnt > 0:
                    section_total += left_cnt
                if right_cnt > 0:
                    section_total += right_cnt
                if left_cnt > 0 and right_cnt > 0:
                    section_total += 1  # 中央分隔
                total += section_total
        return total

    # -------------------------------
    # 完整性检查
    # -------------------------------
    def run_integrity(self) -> QualityReport:
        rep = QualityReport()
        # 保护性检查
        if self.xodr_root is None:
            raise RuntimeError("XODR 尚未加载，请先调用 load_xodr() 再运行完整性检查。")
        if not self.json_data:
            raise RuntimeError("JSON 尚未加载，请先调用 load_json() 再运行完整性检查。")

        # JSON 侧
        json_lanes = [ln for ln in self._json_lanes() if (ln.get("type") or "").lower() == "driving"]
        json_lane_count = len(json_lanes)
        json_bounds = self._json_bounds()
        unique_bound_ids, shared_bound_ids = self._detect_shared_bounds()
        json_bounds_unique_cnt = len(unique_bound_ids)
        json_objects_cnt = len(self._json_objects())
        json_signs_cnt = len(self._json_signs())

        # XODR 侧
        xodr_lane_cnt = self._xodr_count_driving_lanes()
        xodr_lane_mark_list = self._xodr_collect_lane_marks()
        xodr_lane_mark_cnt = len(xodr_lane_mark_list)
        # 控制台调试输出 roadMark 采样
        print(f"[DEBUG] XODR roadMark total = {xodr_lane_mark_cnt}")
        for i, it in enumerate(xodr_lane_mark_list[:50], 1):
            rid = it.get('road_id'); ls = it.get('laneSection_index'); sd = it.get('lane_side'); lid = it.get('lane_id'); tg = it.get('tag'); at = it.get('attrib')
            print(f"[DEBUG] #{i} road={rid} ls={ls} side={sd} lane={lid} tag={tg} attrib={at}")
        xodr_objects_cnt = self._xodr_count_objects()
        xodr_signals_cnt = self._xodr_count_signals()
        xodr_topo_bound_cnt = self._xodr_count_topo_boundaries()

        # 子分计算
        def subscore(a: int, b: int) -> float:
            if a == 0 and b == 0:
                return 1.0
            if a == 0 or b == 0:
                return 0.0
            return round(min(a, b) / max(a, b), 2)

        # 车道
        rep.rows.append(IntegrityRow(
            name="车道（driving）",
            json_count=json_lane_count,
            xodr_count=xodr_lane_cnt,
            subscore=subscore(json_lane_count, xodr_lane_cnt),
            note=""
        ))

        # 边界（拓扑口径对齐）
        shared_note = (
            f"唯一边界ID数：{json_bounds_unique_cnt}；"
            f"共有边界：{len(shared_bound_ids)} 个（ID: {sorted(shared_bound_ids)}）"
        )
        rep.rows.append(IntegrityRow(
            name="边界（JSON唯一计数 vs XODR拓扑）",
            json_count=json_bounds_unique_cnt,
            xodr_count=xodr_topo_bound_cnt,
            subscore=subscore(json_bounds_unique_cnt, xodr_topo_bound_cnt),
            note=shared_note
        ))

        # 物体
        rep.rows.append(IntegrityRow(
            name="物体（objects）",
            json_count=json_objects_cnt,
            xodr_count=xodr_objects_cnt,
            subscore=subscore(json_objects_cnt, xodr_objects_cnt),
            note=""
        ))

        # 标识 sign→signal
        rep.rows.append(IntegrityRow(
            name="标识（sign → signal）",
            json_count=json_signs_cnt,
            xodr_count=xodr_signals_cnt,
            subscore=subscore(json_signs_cnt, xodr_signals_cnt),
            note="JSON sign 数量与 XODR signal 数量的匹配度"
        ))

        # 标线实体（roadMark）——用于评估覆盖度（不计入综合分）
        rep.rows.append(IntegrityRow(
            name="标线实体（roadMark）",
            json_count=xodr_topo_bound_cnt,
            xodr_count=xodr_lane_mark_cnt,
            subscore=subscore(xodr_topo_bound_cnt, xodr_lane_mark_cnt),
            note="以XODR拓扑边界为参照衡量已绘制标线覆盖度；中心分隔通常计入两侧车道，外缘若未绘制则不计入。",
            include_in_score=False
        ))

        # 综合分：仅统计 include_in_score=True 的行
        if rep.rows:
            sc_rows = [r.subscore for r in rep.rows if getattr(r, 'include_in_score', True)]
            rep.completeness_score = round(sum(sc_rows) / len(sc_rows), 2) if sc_rows else 0.0

        # 细节
        rep.details.update({
            "json_lane_ids": [ln.get("id") for ln in json_lanes],
            "json_unique_bound_ids": sorted(unique_bound_ids),
            "json_shared_bound_ids": sorted(shared_bound_ids),
            # 调试：预览前 50 条 roadMark 采样
            "xodr_roadmarks_preview": xodr_lane_mark_list[:50],
            "xodr_roadmarks_total": xodr_lane_mark_cnt,
            "xodr_topo_boundaries_total": xodr_topo_bound_cnt,
        })
        return rep

    # -------------------------------
    # HTML 报告
    # -------------------------------
    def save_html(self, report: QualityReport, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)

        def _row(r: IntegrityRow) -> str:
            return (
                f"<tr>"
                f"<td>{r.name}</td>"
                f"<td style='text-align:right'>{r.json_count}</td>"
                f"<td style='text-align:right'>{r.xodr_count}</td>"
                f"<td style='text-align:right'>{r.subscore:.2f}</td>"
                f"<td>{r.note}</td>"
                f"</tr>"
            )

        rows_html = "\n".join(_row(r) for r in report.rows)

        html = f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<title>JSON↔XODR 质检完整性报告 v1.1.1</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(0,0,0,.04); margin-bottom: 20px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
small {{ color: #6b7280; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #eee; padding: 10px; text-align: left; }}
th {{ background: #fafafa; font-weight: 600; }}
.score {{ font-size: 28px; font-weight: 700; }}
.kv {{ display: grid; grid-template-columns: 160px 1fr; gap: 8px; font-size: 13px; color: #374151; }}
code {{ background: #f6f7f9; padding: 2px 6px; border-radius: 6px; }}
</style>
</head>
<body>
  <div class=\"card\">
    <h1>JSON↔XODR 完整性概览 <small>v1.1.1</small></h1>
    <div>综合完整性评分：<span class=\"score\">{report.completeness_score:.2f}</span></div>
    <div class=\"kv\" style=\"margin-top:12px\">
      <div>JSON 文件</div><div>{self.json_path.name}</div>
      <div>XODR 文件</div><div>{self.xodr_path.name}</div>
    </div>
  </div>

  <div class=\"card\">
    <h2>完整性检查</h2>
    <table>
      <thead>
        <tr>
          <th>要素</th>
          <th style=\"text-align:right\">JSON 数量</th>
          <th style=\"text-align:right\">XODR 数量</th>
          <th style=\"text-align:right\">子分数</th>
          <th>备注 / 细分</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    <p style=\"color:#6b7280;font-size:13px;margin-top:10px\">说明：<b>边界</b>（JSON 唯一计数 vs XODR 拓扑）——JSON 统计为“唯一边界ID计数”（去重）；XODR 统计为“拓扑边界数”（按 laneSection 左/右/中央分隔推导）。<b>标线实体（roadMark）</b>单独展示实际绘制的标线数量（不计入综合分），用于评估覆盖度。</p>
  </div>

  <div class=\"card\">
    <h2>细节</h2>
    <div class=\"kv\">
      <div>JSON 驾驶车道ID</div><div>{report.details.get('json_lane_ids')}</div>
      <div>唯一边界ID</div><div>{report.details.get('json_unique_bound_ids')}</div>
      <div>共有边界ID</div><div>{report.details.get('json_shared_bound_ids')}</div>
      <div>XODR 拓扑边界总数</div><div>{report.details.get('xodr_topo_boundaries_total')}</div>
    </div>
  </div>

  <div class=\"card\">
    <h2>调试 · XODR roadMark 采样</h2>
    <div class=\"kv\">
      <div>检出总量</div><div>{report.details.get('xodr_roadmarks_total')}</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>road_id</th>
          <th>laneSection</th>
          <th>side</th>
          <th>lane_id</th>
          <th>tag</th>
          <th>attributes (json)</th>
        </tr>
      </thead>
      <tbody>
      {''.join([f"<tr><td>{it.get('road_id')}</td><td style='text-align:right'>{it.get('laneSection_index')}</td><td>{it.get('lane_side')}</td><td style='text-align:right'>{it.get('lane_id')}</td><td>{it.get('tag')}</td><td><code>{it.get('attrib')}</code></td></tr>" for it in report.details.get('xodr_roadmarks_preview', [])])}
      </tbody>
    </table>
    <p style=\"color:#6b7280;font-size:13px;margin-top:10px\">注：仅预览前 50 条。完整清单可在脚本中扩展导出为 CSV/JSON。</p>
  </div>
</body>
</html>
"""
        out.write_text(html, encoding="utf-8")


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="输入 JSON 文件路径")
    ap.add_argument("--xodr", required=True, help="输入 XODR 文件路径")
    ap.add_argument("--out", default="quality_report_v1_1_1.html", help="输出 HTML 文件路径")
    args = ap.parse_args()

    qc = QualityChecker(args.json, args.xodr)
    qc.load_json()
    qc.load_xodr()

    rep = qc.run_integrity()
    qc.save_html(rep, Path(args.out))
    print(f"✅ 报告已生成: {args.out}")


if __name__ == "__main__":
    main()
