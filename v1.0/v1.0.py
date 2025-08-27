#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker
质检合作方的JSON转XODR代码工具
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


@dataclass
class QualityReport:
    """质检报告数据结构"""
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    warnings: List[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}


class QualityChecker:
    """JSON转XODR质检工具主类"""

    def __init__(self, json_file: str, xodr_file: str, threshold: float = 0.1):
        """
        初始化质检器

        Args:
            json_file: 输入JSON文件路径
            xodr_file: 输出XODR文件路径
            threshold: 偏移阈值(米)，默认0.1m(10cm)
        """
        self.json_file = Path(json_file)
        self.xodr_file = Path(xodr_file)
        self.threshold = threshold

        # 加载数据
        self.json_data = self._load_json()
        self.xodr_data = self._load_xodr()

        # 初始化报告
        self.report = QualityReport()

    def _load_json(self) -> Dict:
        """加载JSON数据"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载JSON文件: {self.json_file}")
            return data
        except Exception as e:
            raise ValueError(f"❌ 无法加载JSON文件 {self.json_file}: {e}")

    def _load_xodr(self) -> ET.Element:
        """加载XODR数据"""
        try:
            tree = ET.parse(self.xodr_file)
            root = tree.getroot()
            print(f"✅ 成功加载XODR文件: {self.xodr_file}")
            return root
        except Exception as e:
            raise ValueError(f"❌ 无法加载XODR文件 {self.xodr_file}: {e}")

    def check_completeness(self) -> float:
        """
        检查元素完整性
        对比JSON输入和XODR输出的关键元素数量
        """
        print("\n🔍 开始元素完整性检查...")

        completeness_details = {}
        total_score = 0
        check_count = 0

        # 1. 检查车道数量
        json_lanes = len(self.json_data.get('lanes', []))
        xodr_lanes = self._count_xodr_lanes()

        lane_score = min(xodr_lanes / json_lanes, 1.0) if json_lanes > 0 else 1.0
        completeness_details['lanes'] = {
            'json_count': json_lanes,
            'xodr_count': xodr_lanes,
            'score': lane_score
        }
        total_score += lane_score
        check_count += 1

        # 2. 检查边界数量
        json_bounds = len(self.json_data.get('bounds', []))
        xodr_road_marks = self._count_xodr_road_marks()

        bound_score = min(xodr_road_marks / (json_bounds * 2),
                          1.0) if json_bounds > 0 else 1.0  # 乘以2因为每个bound可能有多个lane_mark
        completeness_details['bounds'] = {
            'json_count': json_bounds,
            'xodr_count': xodr_road_marks,
            'score': bound_score
        }
        total_score += bound_score
        check_count += 1

        # 3. 检查物体数量
        json_objects = len(self.json_data.get('objects', []))
        xodr_objects = self._count_xodr_objects()

        object_score = min(xodr_objects / json_objects, 1.0) if json_objects > 0 else 1.0
        completeness_details['objects'] = {
            'json_count': json_objects,
            'xodr_count': xodr_objects,
            'score': object_score
        }
        total_score += object_score
        check_count += 1

        # 计算总体完整性得分
        self.report.completeness_score = total_score / check_count if check_count > 0 else 0
        self.report.details['completeness'] = completeness_details

        print(f"📊 完整性检查完成，总体得分: {self.report.completeness_score:.2%}")
        return self.report.completeness_score

    def check_curve_consistency(self) -> float:
        """
        检查曲线一致性：仅对 JSON 的边界点（bounds）计算到 XODR（参考线+车道边界）的最近距离。
        忽略 JSON objects。
        """
        print("\n🔍 开始曲线一致性检查...")

        warnings = []

        # JSON 点：仅保留 bounds
        all_json_points = self._get_all_json_points()
        bound_points = [p for p in all_json_points if p.get('source') == 'bound']
        ignored_objects = len(all_json_points) - len(bound_points)

        # XODR 点（参考线 + 车道边界）
        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_points = []
        xodr_points.extend(xodr_data.get("reference_lines", []))
        xodr_points.extend(xodr_data.get("lane_boundaries", []))

        print(f"   📍 JSON边界点数: {len(bound_points)}  (objects 已忽略: {ignored_objects})")
        print(f"   📍 XODR采样点数: {len(xodr_points)} "
              f"(ref: {len(xodr_data.get('reference_lines', []))}, "
              f"lanes: {len(xodr_data.get('lane_boundaries', []))})")

        if not bound_points or not xodr_points:
            print("   ❌ 数据不足，无法计算一致性")
            self.report.consistency_score = 0.0
            self.report.details['consistency'] = {
                'average_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0,
                'point_count': len(bound_points), 'threshold': self.threshold,
                'warnings_count': 0, 'points_over_threshold': 0
            }
            return 0.0

        # 计算偏移（仅 bounds）
        deviations = []
        for i, json_point in enumerate(bound_points):
            d = self._find_nearest_distance_to_xodr(json_point, xodr_points)
            deviations.append(d)
            if d > self.threshold:
                warnings.append(
                    f"⚠️ 坐标点 ({json_point['x']:.1f}, {json_point['y']:.1f}) "
                    f"偏移超过阈值: {d:.3f}m > {self.threshold}m"
                )
            if (i + 1) % 100 == 0 or i == len(bound_points) - 1:
                print(f"   📊 已处理 {i + 1}/{len(bound_points)} 个点")

        avg_dev = float(np.mean(deviations)) if deviations else 0.0
        max_dev = float(np.max(deviations)) if deviations else 0.0
        min_dev = float(np.min(deviations)) if deviations else 0.0
        over = sum(1 for d in deviations if d > self.threshold)

        # 分数：1 - 平均偏移/阈值（下限 0）
        consistency_score = max(0.0, 1.0 - (avg_dev / self.threshold)) if self.threshold > 0 else 0.0

        self.report.consistency_score = consistency_score
        self.report.warnings.extend(warnings)
        self.report.details['consistency'] = {
            'average_deviation': avg_dev,
            'max_deviation': max_dev,
            'min_deviation': min_dev,
            'point_count': len(bound_points),
            'threshold': self.threshold,
            'warnings_count': len(warnings),
            'points_over_threshold': over,
        }

        print(f"📊 一致性检查完成，得分: {consistency_score:.2%}")
        print(f"📊 平均偏移: {avg_dev:.3f}m，最大偏移: {max_dev:.3f}m，最小偏移: {min_dev:.3f}m")
        print(f"📊 超过阈值的点: {over}/{len(bound_points)}")

        return consistency_score

    def _get_all_json_points(self) -> List[Dict]:
        """获取JSON中所有的坐标点"""
        all_points = []

        # 从bounds中提取坐标点
        for bound in self.json_data.get('bounds', []):
            bound_id = bound['id']
            for pt in bound['pts']:
                point = {
                    'x': pt['x'],
                    'y': pt['y'],
                    'z': pt['z'],
                    'bound_id': bound_id,
                    'source': 'bound'
                }
                all_points.append(point)

        # 也可以从objects中提取坐标点
        for obj in self.json_data.get('objects', []):
            obj_id = obj['id']
            for pt in obj.get('outline', []):
                point = {
                    'x': pt['x'],
                    'y': pt['y'],
                    'z': pt['z'],
                    'object_id': obj_id,
                    'source': 'object'
                }
                all_points.append(point)

        return all_points

    def _sample_xodr_curves_and_lanes(self, step: float = 0.05):
        """
        采样参考线 + 车道(中心线/边界线)，并返回：
          - reference_lines: 扁平点集（兼容旧逻辑）
          - lane_boundaries: 扁平点集（注意：现在是“边界线”所有点，非中心线）
          - reference_polylines: 每条 road 一条参考线 polyline
          - lane_polylines: 每条车道的“中心线” polyline（绘图用，兼容旧字段名）
          - lane_center_polylines: 同上（显式命名）
          - lane_edge_polylines: 每条车道的内/外边界 polyline（绘图/匹配可用）
        """
        import xml.etree.ElementTree as ET
        import math
        import numpy as np

        tree = ET.parse(self.xodr_file)
        root = tree.getroot()

        out = {
            "reference_lines": [],
            "lane_boundaries": [],
            "reference_polylines": [],
            "lane_polylines": [],  # = center polylines（兼容旧字段名）
            "lane_center_polylines": [],
            "lane_edge_polylines": []  # [{"road_id","lane_id","side","kind":"inner"/"outer","points":[(x,y),...]}]
        }

        # ---------- 工具 ----------
        def local_to_global(x0, y0, hdg, u, v):
            ch, sh = math.cos(hdg), math.sin(hdg)
            return x0 + ch * u - sh * v, y0 + sh * u + ch * v

        def sample_planview_polyline(plan_view, default_step=step):
            ref_pts = []  # [(s_abs, x, y)]
            s_abs = 0.0
            geoms = list(plan_view.findall('geometry'))
            for g in geoms:
                x0 = float(g.attrib["x"])
                y0 = float(g.attrib["y"])
                hdg = float(g.attrib["hdg"])
                L = float(g.attrib["length"])
                child = list(g)[0]
                tag = child.tag

                n = max(2, int(L / default_step) + 1)
                s_vals = np.linspace(0.0, L, n)

                if tag == "line":
                    for s in s_vals:
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, s, 0.0)))

                elif tag == "arc":
                    k = float(child.attrib["curvature"])
                    R = 1.0 / k if k != 0 else 1e12
                    for s in s_vals:
                        ang = k * s
                        u = R * math.sin(ang)
                        v = R * (1.0 - math.cos(ang))
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, u, v)))

                elif tag == "spiral":
                    c0 = float(child.attrib["curvStart"])
                    c1 = float(child.attrib["curvEnd"])
                    for s in s_vals:
                        c = c0 + (c1 - c0) * (s / L)
                        theta = 0.5 * c * s  # 简化
                        u = s * math.cos(theta)
                        v = s * math.sin(theta)
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, u, v)))

                elif tag == "paramPoly3":
                    aU = float(child.attrib.get("aU", 0))
                    bU = float(child.attrib.get("bU", 1))
                    cU = float(child.attrib.get("cU", 0))
                    dU = float(child.attrib.get("dU", 0))
                    aV = float(child.attrib.get("aV", 0))
                    bV = float(child.attrib.get("bV", 0))
                    cV = float(child.attrib.get("cV", 0))
                    dV = float(child.attrib.get("dV", 0))
                    p_range = child.attrib.get("pRange", "normalized")
                    t_vals = s_vals if p_range == "arcLength" else np.linspace(0.0, 1.0, len(s_vals))
                    for i, t in enumerate(t_vals):
                        u = aU + bU * t + cU * t * t + dU * t * t * t
                        v = aV + bV * t + cV * t * t + dV * t * t * t
                        ref_pts.append((s_abs + s_vals[i], *local_to_global(x0, y0, hdg, u, v)))
                s_abs += L
            return ref_pts

        def finite_diff_heading(xs, ys):
            th = []
            n = len(xs)
            for i in range(n):
                if i == 0:
                    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
                elif i == n - 1:
                    dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
                else:
                    dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
                th.append(math.atan2(dy, dx) if dx * dx + dy * dy > 0 else 0.0)
            return th

        # laneWidth & laneOffset 多项式
        def width_poly_at(width_list, s_rel):
            if not width_list:
                return 0.0
            prev = width_list[0]
            for w in width_list:
                if w["sOffset"] <= s_rel:
                    prev = w
                else:
                    break
            ds = max(0.0, s_rel - prev["sOffset"])
            return prev["a"] + prev["b"] * ds + prev["c"] * ds * ds + prev["d"] * ds * ds * ds

        def parse_lane_sections(lanes_elem):
            sections = []
            for sec in lanes_elem.findall('laneSection'):
                s0 = float(sec.attrib['s'])
                left = sec.find('left')
                right = sec.find('right')
                one = {"s": s0, "left": [], "right": []}

                def collect(side_elem, side_name):
                    if side_elem is None:
                        return
                    for ln in side_elem.findall('lane'):
                        lid = ln.attrib.get('id', '')
                        widths = []
                        for w in ln.findall('width'):
                            widths.append({
                                "sOffset": float(w.attrib.get('sOffset', 0.0)),
                                "a": float(w.attrib.get('a', 0.0)),
                                "b": float(w.attrib.get('b', 0.0)),
                                "c": float(w.attrib.get('c', 0.0)),
                                "d": float(w.attrib.get('d', 0.0)),
                            })
                        widths.sort(key=lambda x: x["sOffset"])
                        one[side_name].append({"id": lid, "widths": widths})

                    # 从中线向外排序（left: +1,+2,...； right: -1,-2,... 的绝对值升序）
                    if side_name == "left":
                        one[side_name].sort(key=lambda e: int(e["id"]))
                    else:
                        one[side_name].sort(key=lambda e: abs(int(e["id"])))

                collect(left, "left");
                collect(right, "right")
                sections.append(one)
            sections.sort(key=lambda s: s["s"])
            return sections

        def parse_lane_offsets(lanes_elem):
            offsets = []
            for lo in lanes_elem.findall('laneOffset'):
                offsets.append({
                    "s": float(lo.attrib.get("s", 0.0)),
                    "a": float(lo.attrib.get("a", 0.0)),
                    "b": float(lo.attrib.get("b", 0.0)),
                    "c": float(lo.attrib.get("c", 0.0)),
                    "d": float(lo.attrib.get("d", 0.0)),
                })
            offsets.sort(key=lambda x: x["s"])
            return offsets

        def lane_offset_at(offsets, s_abs):
            if not offsets:
                return 0.0
            prev = offsets[0]
            for o in offsets:
                if o["s"] <= s_abs:
                    prev = o
                else:
                    break
            ds = max(0.0, s_abs - prev["s"])
            return prev["a"] + prev["b"] * ds + prev["c"] * ds * ds + prev["d"] * ds * ds * ds

        # ---------- 主流程：逐 road ----------
        for road in root.findall('.//road'):
            road_id = road.attrib.get('id', 'unknown')
            plan_view = road.find('planView')
            if plan_view is None:
                continue

            # 1) 参考线 polyline
            ref = sample_planview_polyline(plan_view)
            if len(ref) < 2:
                continue
            s_arr = [p[0] for p in ref]
            x_arr = [p[1] for p in ref]
            y_arr = [p[2] for p in ref]
            th_arr = finite_diff_heading(x_arr, y_arr)

            ref_poly = [(x_arr[i], y_arr[i]) for i in range(len(ref))]
            out["reference_polylines"].append(ref_poly)
            out["reference_lines"].extend(ref_poly)

            # 2) lanes：中心线 + 边界线
            lanes_elem = road.find('lanes')
            if lanes_elem is None:
                continue
            sections = parse_lane_sections(lanes_elem)
            offsets = parse_lane_offsets(lanes_elem)

            road_len = s_arr[-1]
            for si, sec in enumerate(sections):
                s0 = sec["s"]
                s1 = sections[si + 1]["s"] if si + 1 < len(sections) else road_len + 1e-6

                idxs = [i for i, sv in enumerate(s_arr) if (sv >= s0 and sv <= s1)]
                if len(idxs) < 2:
                    continue

                # 累积容器：key -> pts
                centers = {}  # f"{side}:{lane_id}" -> [(x,y)]
                edges = {}  # f"{side}:{lane_id}:inner/outer" -> [(x,y)]

                def ensure(dct, key):
                    if key not in dct:
                        dct[key] = []
                    return dct[key]

                for i in idxs:
                    s_here = s_arr[i]
                    s_rel = s_here - s0
                    nx = -math.sin(th_arr[i]);
                    ny = math.cos(th_arr[i])  # 左法向
                    base = lane_offset_at(offsets, s_here)

                    # ---- 左侧：从中线向外 ----
                    cum = 0.0
                    for ln in sec["left"]:
                        w = width_poly_at(ln["widths"], s_rel)
                        inner_off = base + cum  # 靠近中线的边界
                        outer_off = inner_off + w  # 远离中线的边界
                        center_off = inner_off + 0.5 * w

                        # 边界点
                        ensure(edges, f"left:{ln['id']}:inner").append(
                            (x_arr[i] + inner_off * nx, y_arr[i] + inner_off * ny))
                        ensure(edges, f"left:{ln['id']}:outer").append(
                            (x_arr[i] + outer_off * nx, y_arr[i] + outer_off * ny))
                        # 中心点
                        ensure(centers, f"left:{ln['id']}").append(
                            (x_arr[i] + center_off * nx, y_arr[i] + center_off * ny))

                        cum += w

                    # ---- 右侧：从中线向外（负方向）----
                    cum = 0.0
                    for ln in sec["right"]:
                        w = width_poly_at(ln["widths"], s_rel)
                        inner_off = base - cum  # 靠近中线的边界（负）
                        outer_off = inner_off - w  # 远离中线的边界（更负）
                        center_off = inner_off - 0.5 * w

                        ensure(edges, f"right:{ln['id']}:inner").append(
                            (x_arr[i] + inner_off * nx, y_arr[i] + inner_off * ny))
                        ensure(edges, f"right:{ln['id']}:outer").append(
                            (x_arr[i] + outer_off * nx, y_arr[i] + outer_off * ny))
                        ensure(centers, f"right:{ln['id']}").append(
                            (x_arr[i] + center_off * nx, y_arr[i] + center_off * ny))

                        cum += w

                # 写出本 section 的曲线
                for key, pts in centers.items():
                    side, lane_id = key.split(":")
                    out["lane_center_polylines"].append({
                        "road_id": road_id, "lane_id": lane_id, "side": side, "points": pts
                    })
                    out["lane_polylines"].append({  # 兼容别名
                        "road_id": road_id, "lane_id": lane_id, "side": side, "points": pts
                    })

                for key, pts in edges.items():
                    side, lane_id, kind = key.split(":")
                    out["lane_edge_polylines"].append({
                        "road_id": road_id, "lane_id": lane_id, "side": side, "kind": kind, "points": pts
                    })
                    # 扁平边界点（用于一致性计算/最近点搜索）
                    out["lane_boundaries"].extend(pts)

        return out

    def _get_all_xodr_points(self) -> List[Tuple[float, float]]:
        """把参考线与车道边界合并成一个 (x, y) 点列表"""
        data = self._sample_xodr_curves_and_lanes()
        points = []
        points.extend(data.get("reference_lines", []))
        points.extend(data.get("lane_boundaries", []))
        return points

    def _find_nearest_distance_to_xodr(self, json_point: Dict, xodr_points: List[Tuple[float, float]]) -> float:
        """找到JSON点到XODR采样点的最短距离"""
        jx, jy = json_point['x'], json_point['y']

        min_distance = float('inf')
        for xodr_x, xodr_y in xodr_points:
            distance = math.sqrt((jx - xodr_x) ** 2 + (jy - xodr_y) ** 2)
            min_distance = min(min_distance, distance)

        return min_distance

    def visualize_point_matching(self, save_path: str = None) -> str:
        """
        生成可视化图表，并记录到 self._viz_path，方便 HTML 报告引用
        """
        print("\n🎨 Generating visualization charts...")

        all_json_points = self._get_all_json_points()
        xodr_sample_points = self._sample_xodr_curves_and_lanes()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self._plot_overall_distribution(ax1, all_json_points, xodr_sample_points)
        self._plot_deviation_analysis(ax2, all_json_points, xodr_sample_points)
        plt.tight_layout()

        if save_path is None:
            save_path = self.json_file.parent / f"{self.json_file.stem}_visualization.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization chart saved: {save_path}")

        # 记录图片路径（新增）
        self._viz_path = str(save_path)

        try:
            plt.show()
        except:
            pass

        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points, xodr_data):
        """
        参考线(灰) + 每条车道中心线(彩色) + 边界(浅灰虚线) + JSON点
        并在每条车道中心线的中点标注 'R{road} L{lane}'
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) 参考线
        ref_polys = xodr_data.get("reference_polylines", [])
        for poly in ref_polys:
            if len(poly) >= 2:
                xs, ys = zip(*poly)
                ax.plot(xs, ys, color='0.4', linewidth=1.5, alpha=0.7)
        if ref_polys:
            ax.plot([], [], color='0.4', linewidth=1.5, label=f'XODR reference ({sum(len(p) for p in ref_polys)})')

        # 2) 车道边界（浅灰虚线，辅助对齐观察）
        edge_polys = xodr_data.get("lane_edge_polylines", [])
        for ed in edge_polys:
            pts = ed["points"]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color='0.75', linewidth=0.8, alpha=0.6, linestyle='--')
        if edge_polys:
            ax.plot([], [], color='0.75', linewidth=0.8, linestyle='--',
                    label=f'XODR lane edges ({sum(len(e["points"]) for e in edge_polys)})')

        # 3) 车道中心线（每条车道独立 polyline + 文本标注）
        lane_polys = xodr_data.get("lane_center_polylines", xodr_data.get("lane_polylines", []))
        n_lane = len(lane_polys)
        if n_lane:
            colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_lane)))  # 颜色足够多
        for i, lane in enumerate(lane_polys):
            pts = lane["points"]
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, linewidth=1.5, alpha=0.95, color=colors[i % len(colors)])

            # 在线中点处标注：R{road} L{lane_id}
            mid = len(pts) // 2
            tx, ty = pts[mid]
            txt = f"R{lane['road_id']} L{lane['lane_id']}"
            ax.text(tx, ty, txt, fontsize=8, alpha=0.85)

        if lane_polys:
            ax.plot([], [], linewidth=1.5, label=f'XODR lanes ({sum(len(l["points"]) for l in lane_polys)})')

        # 4) JSON 点
        bound_points = [p for p in json_points if p['source'] == 'bound']
        if bound_points:
            bx = [p['x'] for p in bound_points];
            by = [p['y'] for p in bound_points]
            ax.scatter(bx, by, s=30, alpha=0.85, label=f'JSON bounds ({len(bound_points)})')

        obj_points = [p for p in json_points if p['source'] == 'object']
        if obj_points:
            ox = [p['x'] for p in obj_points];
            oy = [p['y'] for p in obj_points]
            ax.scatter(ox, oy, s=30, alpha=0.85, label=f'JSON objects ({len(obj_points)})')

        ax.set_xlabel('X (m)');
        ax.set_ylabel('Y (m)')
        ax.set_title('JSON vs XODR Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 需要时也可以保留一个别名
    def _plot_distribution(self, ax, json_points, xodr_data):
        return self._plot_overall_distribution(ax, json_points, xodr_data)

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_data: Dict):
        """Plot deviation analysis（仅对 bounds 计算偏移；objects 置灰说明忽略）"""

        # 合并 XODR 点
        xodr_points = []
        if xodr_data["reference_lines"]:
            xodr_points.extend(xodr_data["reference_lines"])
        if xodr_data["lane_boundaries"]:
            xodr_points.extend(xodr_data["lane_boundaries"])

        # 分离 JSON
        bound_points = [p for p in json_points if p['source'] == 'bound']
        obj_points = [p for p in json_points if p['source'] == 'object']

        # 偏移（仅 bounds）
        deviations = []
        for p in bound_points:
            deviations.append(self._find_nearest_distance_to_xodr(p, xodr_points))

        # 背景 XODR
        if xodr_points:
            xp, yp = zip(*xodr_points)
            ax.scatter(xp, yp, c='lightgray', s=5, alpha=0.3, label='XODR samples')

        # bounds 以偏移着色
        if bound_points:
            bx = [p['x'] for p in bound_points];
            by = [p['y'] for p in bound_points]
            sc = ax.scatter(bx, by, c=deviations, s=50, cmap='RdYlGn_r',
                            alpha=0.9, edgecolors='black', linewidth=0.4, label=f'JSON bounds ({len(bound_points)})')
            cbar = plt.colorbar(sc, ax=ax);
            cbar.set_label('Deviation (m)')

            avg = np.mean(deviations) if deviations else 0.0
            stats_text = f"""Statistics:
        Total bound points: {len(bound_points)}
        Avg deviation: {avg:.3f}m
        Max deviation: {np.max(deviations) if deviations else 0:.3f}m
        Min deviation: {np.min(deviations) if deviations else 0:.3f}m
        Over threshold: {sum(1 for d in deviations if d > self.threshold)}/{len(deviations)}"""
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # objects 置灰说明忽略
        if obj_points:
            ox = [p['x'] for p in obj_points];
            oy = [p['y'] for p in obj_points]
            ax.scatter(ox, oy, s=30, alpha=0.6, c='gray', label=f'JSON objects (ignored: {len(obj_points)})')

        ax.set_xlabel('X (m)');
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Deviation Analysis (Avg: {np.mean(deviations) if deviations else 0:.3f}m)')
        ax.legend();
        ax.grid(True, alpha=0.3);
        ax.axis('equal')

    def analyze_matching_details(self, include_objects: bool = False) -> Dict:
        print("\n🔍 Analyzing matching details...")

        all_json_points = self._get_all_json_points()
        if include_objects:
            use_points = all_json_points
        else:
            use_points = [p for p in all_json_points if p.get('source') == 'bound']

        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_sample_points = []
        xodr_sample_points.extend(xodr_data.get("reference_lines", []))
        xodr_sample_points.extend(xodr_data.get("lane_boundaries", []))

        analysis_results = {
            'total_json_points': len(use_points),
            'total_xodr_points': len(xodr_sample_points),
            'detailed_matches': []
        }

        print(f"\n📊 Detailed matching analysis:")
        print(f"{'Index':<5} {'Type':<6} {'JSON Coord':<20} {'Nearest XODR':<20} {'Dev(m)':<8} {'Status'}")
        print("-" * 80)

        for i, json_point in enumerate(use_points):
            min_distance = float('inf');
            nearest = None
            for xp, yp in xodr_sample_points:
                d = math.hypot(json_point['x'] - xp, json_point['y'] - yp)
                if d < min_distance:
                    min_distance = d;
                    nearest = (xp, yp)

            status = "✅ Pass" if min_distance <= self.threshold else \
                ("⚠️ Warning" if min_distance <= self.threshold * 2 else "❌ Fail")

            analysis_results['detailed_matches'].append({
                'index': i, 'json_point': json_point,
                'nearest_xodr_point': nearest, 'deviation': min_distance, 'status': status
            })

            if i < 20:
                jc = f"({json_point['x']:.1f}, {json_point['y']:.1f})"
                xc = f"({nearest[0]:.1f}, {nearest[1]:.1f})" if nearest else "None"
                src = json_point['source']
                print(f"{i:<5} {src:<6} {jc:<20} {xc:<20} {min_distance:<8.3f} {status}")
            elif i == 20:
                print("...")

        return analysis_results

    def _count_xodr_lanes(self) -> int:
        """统计XODR中的车道数量"""
        count = 0
        for road in self.xodr_data.findall('.//road'):
            for lane in road.findall('.//lane'):
                if lane.get('type') == 'driving':  # 只统计行车道
                    count += 1
        return count

    def _count_xodr_road_marks(self) -> int:
        """统计XODR中的道路标记数量"""
        return len(self.xodr_data.findall('.//roadMark'))

    def _count_xodr_objects(self) -> int:
        """统计XODR中的物体数量"""
        return len(self.xodr_data.findall('.//object'))

    def _extract_xodr_curves(self) -> List[Dict]:
        """从XODR中提取几何曲线参数（保留用于其他用途）"""
        curves = []
        for road in self.xodr_data.findall('.//road'):
            road_id = road.get('id')
            for geometry in road.findall('.//geometry'):
                curve_data = {
                    'road_id': road_id,
                    's': float(geometry.get('s', 0)),
                    'x': float(geometry.get('x', 0)),
                    'y': float(geometry.get('y', 0)),
                    'hdg': float(geometry.get('hdg', 0)),
                    'length': float(geometry.get('length', 0))
                }

                # 提取paramPoly3参数
                param_poly3 = geometry.find('paramPoly3')
                if param_poly3 is not None:
                    curve_data.update({
                        'aU': float(param_poly3.get('aU', 0)),
                        'bU': float(param_poly3.get('bU', 1)),
                        'cU': float(param_poly3.get('cU', 0)),
                        'dU': float(param_poly3.get('dU', 0)),
                        'aV': float(param_poly3.get('aV', 0)),
                        'bV': float(param_poly3.get('bV', 0)),
                        'cV': float(param_poly3.get('cV', 0)),
                        'dV': float(param_poly3.get('dV', 0))
                    })

                curves.append(curve_data)
        return curves

    def _evaluate_param_poly3(self, curve: Dict, t: float) -> Tuple[float, float]:
        """计算paramPoly3曲线在参数t处的坐标"""
        # paramPoly3参数
        aU = curve.get('aU', 0)
        bU = curve.get('bU', 1)
        cU = curve.get('cU', 0)
        dU = curve.get('dU', 0)
        aV = curve.get('aV', 0)
        bV = curve.get('bV', 0)
        cV = curve.get('cV', 0)
        dV = curve.get('dV', 0)

        # 计算局部坐标
        u = aU + bU * t + cU * t * t + dU * t * t * t
        v = aV + bV * t + cV * t * t + dV * t * t * t

        # 转换到世界坐标系
        x0, y0 = curve['x'], curve['y']
        hdg = curve['hdg']

        cos_hdg = math.cos(hdg)
        sin_hdg = math.sin(hdg)

        world_x = x0 + u * cos_hdg - v * sin_hdg
        world_y = y0 + u * sin_hdg + v * cos_hdg

        return world_x, world_y

    def generate_report(self, max_detail_rows: int = 200) -> str:
        """生成中文 HTML 质检报告"""
        print("\n📝 生成质检报告...")

        # 只在 details 缺失时计算，避免重复
        if 'completeness' not in self.report.details:
            self.check_completeness()
        if 'consistency' not in self.report.details:
            self.check_curve_consistency()

        # 仅对 bounds 做匹配明细（忽略 objects）
        matching = self.analyze_matching_details(include_objects=False)

        # 确保可视化图片存在
        viz_path = getattr(self, "_viz_path", None)
        if not viz_path or not Path(viz_path).exists():
            viz_path = self.visualize_point_matching()

        # 组织给 HTML 的数据包
        data = {
            "json_file": str(self.json_file),
            "xodr_file": str(self.xodr_file),
            "threshold": self.threshold,
            "report": self.report,  # 包含 completeness/consistency 两个块
            "matching": matching,  # 匹配明细
            "viz_path": str(viz_path),
            "max_detail_rows": max_detail_rows,
        }

        html_report = self._generate_html_report(data)

        report_file = self.json_file.parent / f"{self.json_file.stem}_quality_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"✅ 质检报告已生成: {report_file}")
        return str(report_file)

    def _generate_html_report(self, data: Dict) -> str:
        """
        生成中文 HTML 报告（自包含样式）。
        data 包含：
          - json_file, xodr_file, threshold
          - report: QualityReport（含 details.completeness / details.consistency）
          - matching: analyze_matching_details() 的返回
          - viz_path: 可视化图片路径
          - max_detail_rows: 匹配明细最大展示行数
        """
        import html
        from datetime import datetime

        # 取字段
        json_file = html.escape(data["json_file"])
        xodr_file = html.escape(data["xodr_file"])
        threshold = data["threshold"]
        report: QualityReport = data["report"]
        matching = data["matching"]
        viz_path = data["viz_path"]
        max_rows = int(data.get("max_detail_rows", 200))

        # 完整性细节
        comp = report.details.get("completeness", {})
        lanes_info = comp.get("lanes", {})
        bounds_info = comp.get("bounds", {})
        objs_info = comp.get("objects", {})

        # 一致性细节
        cons = report.details.get("consistency", {})
        avg_dev = cons.get("average_deviation", 0.0)
        max_dev = cons.get("max_deviation", 0.0)
        min_dev = cons.get("min_deviation", 0.0)
        pt_cnt = cons.get("point_count", 0)
        warn_cnt = cons.get("warnings_count", 0)
        over_cnt = cons.get("points_over_threshold", 0)

        # 匹配明细表格（默认展示前 max_rows 条）
        rows_html = []
        for row in matching.get("detailed_matches", [])[:max_rows]:
            jp = row["json_point"]
            nx, ny = (row["nearest_xodr_point"] or (None, None))
            rows_html.append(
                f"<tr>"
                f"<td class='num'>{row['index']}</td>"
                f"<td>{html.escape(jp.get('source', ''))}</td>"
                f"<td>({jp['x']:.3f}, {jp['y']:.3f})</td>"
                f"<td>{'(' + f'{nx:.3f}, {ny:.3f}' + ')' if nx is not None else '—'}</td>"
                f"<td class='num'>{row['deviation']:.3f}</td>"
                f"<td>{html.escape(row['status'])}</td>"
                f"</tr>"
            )
        if not rows_html:
            rows_html.append("<tr><td colspan='6' style='text-align:center;'>暂无数据</td></tr>")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Warnings（截断展示）
        warn_list_html = ""
        if report.warnings:
            warn_items = report.warnings[:200]
            warn_list_html = "<ul class='warning-list'>" + "".join(
                f"<li>{html.escape(w)}</li>" for w in warn_items
            ) + "</ul>"
            if len(report.warnings) > 200:
                warn_list_html += f"<p class='muted'>（仅显示前 200 条，剩余 {len(report.warnings) - 200} 条已省略）</p>"
        else:
            warn_list_html = "<p class='muted'>无</p>"

        # 拼 HTML
        return f"""<!doctype html>
    <html lang="zh-CN">
    <head>
    <meta charset="utf-8" />
    <title>JSON→XODR 质检报告</title>
    <style>
      html,body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif; color:#222; }}
      .container {{ max-width: 1100px; margin: 32px auto; padding: 0 16px; }}
      h1 {{ font-size: 26px; margin: 0 0 6px; }}
      h2 {{ font-size: 20px; margin: 28px 0 12px; border-left:4px solid #555; padding-left:8px; }}
      h3 {{ font-size: 16px; margin: 18px 0 10px; }}
      .meta, .muted {{ color:#666; font-size: 13px; }}
      .grid {{ display:grid; grid-template-columns: repeat(3,1fr); gap:12px; }}
      .card {{ border:1px solid #e6e6e6; border-radius:10px; padding:12px 14px; background:#fff; }}
      .kpi {{ font-size: 22px; font-weight:600; }}
      .ok {{ color:#138000; }}
      .warn {{ color:#d97706; }}
      .bad {{ color:#b91c1c; }}
      table {{ width:100%; border-collapse: collapse; }}
      th, td {{ border-bottom:1px solid #eee; padding:8px 10px; text-align:left; }}
      th {{ background:#fafafa; }}
      td.num, th.num {{ text-align:right; }}
      .imgwrap {{ text-align:center; margin: 14px 0 6px; }}
      .warning-list {{ margin:6px 0 0 16px; }}
      .tag {{ display:inline-block; padding:2px 6px; border-radius:6px; background:#f2f2f2; font-size:12px; margin-left:6px; }}
      .legend {{ color:#555; font-size: 13px; }}
      .footer {{ color:#777; font-size: 12px; margin-top: 28px; }}
    </style>
    </head>
    <body>
    <div class="container">

      <h1>JSON → XODR 质检报告</h1>
      <div class="meta">生成时间：{now_str}</div>
      <div class="meta">JSON 文件：{json_file}</div>
      <div class="meta">XODR 文件：{xodr_file}</div>

      <h2>摘要</h2>
      <div class="grid">
        <div class="card">
          <div class="muted">完整性得分</div>
          <div class="kpi">{report.completeness_score:.1%}</div>
          <div class="legend">依据：车道/边界/物体的数量对比</div>
        </div>
        <div class="card">
          <div class="muted">一致性得分</div>
          <div class="kpi">{report.consistency_score:.1%}</div>
          <div class="legend">依据：仅以 JSON 边界点到 XODR（参考线+车道边界）最近距离的平均偏移与阈值（{threshold:.3f} m）比较</div>
        </div>
        <div class="card">
          <div class="muted">告警数量</div>
          <div class="kpi {'ok' if warn_cnt == 0 else 'warn' if warn_cnt < 10 else 'bad'}">{warn_cnt}</div>
          <div class="legend">超阈值点：{over_cnt}/{pt_cnt}</div>
        </div>
      </div>

      <h2>完整性检查</h2>
      <div class="card">
        <table>
          <thead>
            <tr><th>要素</th><th class="num">JSON 数量</th><th class="num">XODR 数量</th><th class="num">子分数</th></tr>
          </thead>
          <tbody>
            <tr><td>车道（driving）</td><td class="num">{lanes_info.get('json_count', '—')}</td><td class="num">{lanes_info.get('xodr_count', '—')}</td><td class="num">{lanes_info.get('score', 0):.2f}</td></tr>
            <tr><td>边界 / 标线</td><td class="num">{bounds_info.get('json_count', '—')}</td><td class="num">{bounds_info.get('xodr_count', '—')}</td><td class="num">{bounds_info.get('score', 0):.2f}</td></tr>
            <tr><td>物体（objects）</td><td class="num">{objs_info.get('json_count', '—')}</td><td class="num">{objs_info.get('xodr_count', '—')}</td><td class="num">{objs_info.get('score', 0):.2f}</td></tr>
          </tbody>
        </table>
        <div class="muted" style="margin-top:8px;">
            说明：XODR 的数量可能大于 JSON，这是因为 OpenDRIVE 会将车道、标线按段落或属性拆分；只要不少于 JSON 即视为完整。
        </div>
      </div>

      <h2>一致性检查</h2>
      <div class="grid">
        <div class="card">
          <div class="muted">平均偏移</div>
          <div class="kpi">{avg_dev:.3f} m</div>
        </div>
        <div class="card">
          <div class="muted">最大 / 最小偏移</div>
          <div class="kpi">{max_dev:.3f} / {min_dev:.3f} m</div>
        </div>
        <div class="card">
          <div class="muted">参与点数</div>
          <div class="kpi">{pt_cnt}</div>
          <div class="legend">仅包含 JSON 中的边界点（objects 已忽略）</div>
        </div>
      </div>

      <div class="card">
        <div class="muted">可视化（分布 & 偏移热力）</div>
        <div class="imgwrap">
          <img src="{html.escape(viz_path)}" alt="Visualization" style="max-width:100%;border:1px solid #eee;border-radius:8px;">
        </div>
        <div class="legend">左图：参考线 + 车道中心线/边界 + JSON 点；右图：仅边界点偏移着色</div>
      </div>

      <h2>匹配明细（前 {max_rows} 条）<span class="tag">仅 JSON 边界点</span></h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th class="num">序号</th>
              <th>类型</th>
              <th>JSON 坐标 (x, y)</th>
              <th>最近 XODR 坐标 (x, y)</th>
              <th class="num">偏移 (m)</th>
              <th>判定</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
        <div class="muted">总计 {matching.get('total_json_points', 0)} 条匹配记录；为避免过大，仅展示前 {max_rows} 条。</div>
      </div>

      <h2>告警列表（超过阈值的点）</h2>
      <div class="card">
        {warn_list_html}
      </div>

      <div class="footer">
        阈值（横向距离）：{threshold:.3f} m。<br/>
        注：一致性仅以 JSON 的边界点参与计算；JSON 中 objects 与道路无直接关系，已在统计和可视化中忽略。
      </div>

    </div>
    </body>
    </html>"""


# 使用示例
if __name__ == "__main__":
    # 创建质检器实例
    checker = QualityChecker(
        json_file="label3_2.json",
        xodr_file="label3_2.xodr",
        threshold=0.1  # 10cm阈值
    )

    # 执行质检
    completeness = checker.check_completeness()
    consistency = checker.check_curve_consistency()

    # 详细分析匹配情况
    analysis = checker.analyze_matching_details()

    # 生成可视化图表
    viz_file = checker.visualize_point_matching()

    # 生成报告
    report_file = checker.generate_report()

    print(f"\n📋 Quality check summary:")
    print(f"   Completeness score: {completeness:.1%}")
    print(f"   Consistency score: {consistency:.1%}")
    print(f"   Warning count: {len(checker.report.warnings)}")
    print(f"   Visualization chart: {viz_file}")
    print(f"   Report file: {report_file}")