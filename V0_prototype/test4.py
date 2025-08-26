#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker
质检合作方的JSON转XODR代码工具（参考线 + 车道边界采样，按 road 分组绘制）
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import math
import matplotlib.pyplot as plt


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
        """
        self.json_file = Path(json_file)
        self.xodr_file = Path(xodr_file)
        self.threshold = threshold

        # 加载数据
        self.json_data = self._load_json()
        self.xodr_data = self._load_xodr()

        # 初始化报告
        self.report = QualityReport()

    # ----------------------------
    # 基础加载
    # ----------------------------
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

    # ----------------------------
    # 完整性检查
    # ----------------------------
    def check_completeness(self) -> float:
        """
        检查元素完整性：对比JSON输入和XODR输出的关键元素数量
        """
        print("\n🔍 开始元素完整性检查...")

        completeness_details = {}
        total_score = 0
        check_count = 0

        # 1) 车道数量
        json_lanes = len(self.json_data.get('lanes', []))
        xodr_lanes = self._count_xodr_lanes()
        lane_score = min(xodr_lanes / json_lanes, 1.0) if json_lanes > 0 else 1.0
        completeness_details['lanes'] = {'json_count': json_lanes, 'xodr_count': xodr_lanes, 'score': lane_score}
        total_score += lane_score
        check_count += 1

        # 2) 边界数量（粗略用 roadMark 数量近似）
        json_bounds = len(self.json_data.get('bounds', []))
        xodr_road_marks = self._count_xodr_road_marks()
        bound_score = min(xodr_road_marks / (json_bounds * 2), 1.0) if json_bounds > 0 else 1.0
        completeness_details['bounds'] = {'json_count': json_bounds, 'xodr_count': xodr_road_marks, 'score': bound_score}
        total_score += bound_score
        check_count += 1

        # 3) 物体数量
        json_objects = len(self.json_data.get('objects', []))
        xodr_objects = self._count_xodr_objects()
        object_score = min(xodr_objects / json_objects, 1.0) if json_objects > 0 else 1.0
        completeness_details['objects'] = {'json_count': json_objects, 'xodr_count': xodr_objects, 'score': object_score}
        total_score += object_score
        check_count += 1

        # 总体得分
        self.report.completeness_score = total_score / check_count if check_count > 0 else 0
        self.report.details['completeness'] = completeness_details
        print(f"📊 完整性检查完成，总体得分: {self.report.completeness_score:.2%}")
        return self.report.completeness_score

    # ----------------------------
    # 一致性检查
    # ----------------------------
    def check_curve_consistency(self) -> float:
        """
        检查曲线一致性：对比XODR参考线/车道边界与JSON坐标点的偏移
        """
        print("\n🔍 开始曲线一致性检查...")

        # JSON 点
        all_json_points = self._get_all_json_points()

        # XODR 点（参考线 + 车道边界）
        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_points: List[Tuple[float, float]] = []
        # 合并参考线
        xodr_points.extend(xodr_data.get("reference_lines", []))
        # 合并所有边界多折线
        for pts in xodr_data.get("lane_boundaries", {}).values():
            xodr_points.extend(pts)

        print(f"   📍 JSON总坐标点数: {len(all_json_points)}")
        print(f"   📍 XODR采样点数: {len(xodr_points)} "
              f"(ref: {sum(len(v) for v in xodr_data.get('reference_by_road', {}).values())}, "
              f"lanes polylines: {len(xodr_data.get('lane_boundaries', {}))})")

        if not xodr_points:
            print("   ❌ 没有找到XODR采样点")
            return 0.0

        # 计算每个 JSON 点到 XODR 的最近距离
        deviations = []
        warnings = []
        for i, json_point in enumerate(all_json_points):
            min_distance = self._find_nearest_distance_to_xodr(json_point, xodr_points)
            deviations.append(min_distance)
            if min_distance > self.threshold:
                warnings.append(
                    f"⚠️ 坐标点 ({json_point['x']:.1f}, {json_point['y']:.1f}) 偏移超过阈值: {min_distance:.3f}m > {self.threshold}m"
                )
            if (i + 1) % 100 == 0 or i == len(all_json_points) - 1:
                print(f"   📊 已处理 {i + 1}/{len(all_json_points)} 个点")

        # 统计与得分
        avg_deviation = float(np.mean(deviations)) if deviations else 0.0
        max_deviation = float(np.max(deviations)) if deviations else 0.0
        min_deviation = float(np.min(deviations)) if deviations else 0.0
        point_count = len(deviations)
        consistency_score = max(0, 1 - (avg_deviation / self.threshold)) if self.threshold > 0 else 0

        self.report.consistency_score = consistency_score
        self.report.warnings.extend(warnings)
        self.report.details['consistency'] = {
            'average_deviation': avg_deviation,
            'max_deviation': max_deviation,
            'min_deviation': min_deviation,
            'point_count': point_count,
            'threshold': self.threshold,
            'warnings_count': len(warnings),
            'points_over_threshold': sum(1 for d in deviations if d > self.threshold)
        }

        print(f"📊 一致性检查完成，得分: {consistency_score:.2%}")
        print(f"📊 平均偏移: {avg_deviation:.3f}m，最大偏移: {max_deviation:.3f}m，最小偏移: {min_deviation:.3f}m")
        print(f"📊 超过阈值的点: {self.report.details['consistency']['points_over_threshold']}/{point_count}")
        return consistency_score

    # ----------------------------
    # JSON / XODR 采样与几何
    # ----------------------------
    def _get_all_json_points(self) -> List[Dict]:
        """获取JSON中所有的坐标点"""
        all_points = []

        # bounds
        for bound in self.json_data.get('bounds', []):
            bound_id = bound.get('id', '')
            for pt in bound.get('pts', []):
                all_points.append({
                    'x': pt['x'],
                    'y': pt['y'],
                    'z': pt.get('z', 0),
                    'bound_id': bound_id,
                    'source': 'bound'
                })

        # objects（可选）
        for obj in self.json_data.get('objects', []):
            obj_id = obj.get('id', '')
            for pt in obj.get('outline', []):
                all_points.append({
                    'x': pt['x'],
                    'y': pt['y'],
                    'z': pt.get('z', 0),
                    'object_id': obj_id,
                    'source': 'object'
                })

        return all_points

    # ====== 参考线 + 车道边界统采（按 road 分组）======
    def _sample_xodr_curves_and_lanes(self, samples_per_geometry: int = 200) -> Dict:
        """
        返回:
        {
            "reference_by_road": {road_id: [(x,y), ...], ...},
            "reference_lines": [(x,y), ...],         # 扁平化合集，便于快速使用
            "lane_boundaries": {"road{rid}_L1": [(x,y),...], "road{rid}_R1": [...], ...}
        }
        """
        root = self.xodr_data
        out_ref_by_road: Dict[str, List[Tuple[float, float]]] = {}
        out_lane_boundaries: Dict[str, List[Tuple[float, float]]] = {}

        # 遍历每条 road
        for road in root.findall('./road'):
            rid = road.get('id', '?')

            # ---- 1) 采样参考线（带 s、heading）----
            ref_samples = self._sample_road_reference_with_heading(road, samples_per_geometry)
            ref_pts = [(p['x'], p['y']) for p in ref_samples]
            out_ref_by_road[rid] = ref_pts

            # ---- 2) laneOffset piecewise 多项式 ----
            lane_offsets = []
            for lo in road.findall('./laneOffset'):
                lane_offsets.append({
                    's': float(lo.get('s', 0)),
                    'a': float(lo.get('a', 0)),
                    'b': float(lo.get('b', 0)),
                    'c': float(lo.get('c', 0)),
                    'd': float(lo.get('d', 0)),
                })
            lane_offsets.sort(key=lambda e: e['s'])

            # ---- 3) laneSection & lanes ----
            lanes_node = road.find('./lanes')
            if lanes_node is None or not ref_samples:
                continue

            lane_sections = lanes_node.findall('./laneSection')
            lane_sections.sort(key=lambda s: float(s.get('s', 0.0)))
            road_end_s = ref_samples[-1]['s']

            # 为了连续绘制边界，创建一个“边界轨迹缓存”字典（仅在当前 road 内）
            boundary_traces: Dict[str, List[Tuple[float, float]]] = {}

            for si, sec in enumerate(lane_sections):
                s_start = float(sec.get('s', 0.0))
                s_end = float(lane_sections[si + 1].get('s', road_end_s)) if si + 1 < len(lane_sections) else road_end_s

                # section 内参考点
                sec_samples = [p for p in ref_samples if s_start - 1e-6 <= p['s'] <= s_end + 1e-6]
                if not sec_samples:
                    continue

                # 左右车道
                left_node = sec.find('./left')
                right_node = sec.find('./right')

                # 解析 lane -> widths piecewise
                def _collect_lanes(node, side: str):
                    lanes = []
                    if node is None:
                        return lanes
                    for ln in node.findall('./lane'):
                        lane_id = int(ln.get('id', 0))
                        widths = []
                        for w in ln.findall('./width'):
                            widths.append({
                                'sOffset': float(w.get('sOffset', 0)),
                                'a': float(w.get('a', 0)),
                                'b': float(w.get('b', 0)),
                                'c': float(w.get('c', 0)),
                                'd': float(w.get('d', 0)),
                            })
                        widths.sort(key=lambda e: e['sOffset'])
                        if (side == 'L' and lane_id > 0) or (side == 'R' and lane_id < 0):
                            lanes.append({'id': abs(lane_id), 'widths': widths})
                    lanes.sort(key=lambda L: L['id'])
                    return lanes

                left_lanes = _collect_lanes(left_node, 'L')   # id: 1,2,3...
                right_lanes = _collect_lanes(right_node, 'R') # id: 1,2,3...

                for p in sec_samples:
                    s = p['s']; x_c, y_c, hdg = p['x'], p['y'], p['hdg']
                    lane_off = self._eval_piecewise_poly(lane_offsets, s)

                    # 左侧：从中心线向外累加
                    cum = lane_off
                    for lane in left_lanes:
                        w = self._eval_lane_width_at(lane['widths'], s - s_start)
                        boundary_offset = cum + w
                        bx, by = self._offset_point_normal(x_c, y_c, hdg, boundary_offset)
                        key = f"road{rid}_L{lane['id']}"
                        boundary_traces.setdefault(key, []).append((bx, by))
                        cum += w

                    # 右侧：从中心线向外累加（负向）
                    cum = lane_off
                    for lane in right_lanes:
                        w = self._eval_lane_width_at(lane['widths'], s - s_start)
                        boundary_offset = cum - w
                        bx, by = self._offset_point_normal(x_c, y_c, hdg, boundary_offset)
                        key = f"road{rid}_R{lane['id']}"
                        boundary_traces.setdefault(key, []).append((bx, by))
                        cum -= w

            # 将当前 road 的边界加入总输出（不同 road 的同名车道不再被串接）
            for key, pts in boundary_traces.items():
                if len(pts) >= 2:
                    out_lane_boundaries[key] = pts

        # 扁平化参考线合集（仅用于最近邻/背景散点）
        ref_all = [pt for pts in out_ref_by_road.values() for pt in pts]

        return {
            "reference_by_road": out_ref_by_road,
            "reference_lines": ref_all,
            "lane_boundaries": out_lane_boundaries
        }

    # ====== 参考线采样：返回包含 s, x, y, hdg 的列表 ======
    def _sample_road_reference_with_heading(self, road: ET.Element, samples_per_geometry: int) -> List[Dict]:
        pts = []
        plan = road.find('./planView')
        if plan is None:
            return pts

        for geom in plan.findall('./geometry'):
            s0 = float(geom.get('s', 0))
            x0 = float(geom.get('x', 0))
            y0 = float(geom.get('y', 0))
            hdg0 = float(geom.get('hdg', 0))
            length = float(geom.get('length', 0))

            children = list(geom)
            if not children:
                continue
            g = children[0]
            tag = g.tag

            def local_to_global(u, v, hdg_base):
                x = x0 + math.cos(hdg_base) * u - math.sin(hdg_base) * v
                y = y0 + math.sin(hdg_base) * u + math.cos(hdg_base) * v
                return x, y

            # 均匀按弧长采样
            for i in range(samples_per_geometry + 1):
                u = length * (i / samples_per_geometry)  # 局部弧长
                s = s0 + u

                if tag == 'line':
                    x, y = local_to_global(u, 0.0, hdg0)
                    hdg = hdg0

                elif tag == 'arc':
                    curvature = float(g.get('curvature', 0))
                    angle = curvature * u
                    R = 1.0 / curvature if abs(curvature) > 1e-12 else 1e12
                    lx = R * math.sin(angle)
                    lv = R * (1 - math.cos(angle))
                    x, y = local_to_global(lx, lv, hdg0)
                    hdg = hdg0 + angle

                elif tag == 'spiral':
                    k0 = float(g.get('curvStart', 0))
                    k1 = float(g.get('curvEnd', 0))
                    dk = (k1 - k0) / length if length > 0 else 0.0
                    theta = k0 * u + 0.5 * dk * (u ** 2)  # ∫k(s)ds
                    lx = u * math.cos(theta / 2.0)
                    lv = u * math.sin(theta / 2.0)
                    x, y = local_to_global(lx, lv, hdg0)
                    hdg = hdg0 + theta

                elif tag == 'paramPoly3':
                    aU = float(g.get('aU', 0)); bU = float(g.get('bU', 1)); cU = float(g.get('cU', 0)); dU = float(g.get('dU', 0))
                    aV = float(g.get('aV', 0)); bV = float(g.get('bV', 0)); cV = float(g.get('cV', 0)); dV = float(g.get('dV', 0))
                    p_range = g.get('pRange', 'normalized')
                    t = (i / samples_per_geometry) if p_range != 'arcLength' else (u / length if length > 0 else 0.0)
                    uu = aU + bU*t + cU*t*t + dU*t*t*t
                    vv = aV + bV*t + cV*t*t + dV*t*t*t
                    x, y = local_to_global(uu, vv, hdg0)
                    du = bU + 2*cU*t + 3*dU*t*t
                    dv = bV + 2*cV*t + 3*dV*t*t
                    hdg = hdg0 + math.atan2(dv, du if abs(du) > 1e-12 else 1e-12)

                else:
                    x, y = local_to_global(u, 0.0, hdg0)
                    hdg = hdg0

                pts.append({'s': s, 'x': x, 'y': y, 'hdg': hdg})

        pts.sort(key=lambda d: d['s'])
        return pts

    # ====== piecewise polynomial & 偏移 ======
    @staticmethod
    def _eval_piecewise_poly(segments: List[Dict], s: float) -> float:
        if not segments:
            return 0.0
        seg = None
        for cand in segments:
            if s + 1e-9 >= cand['s']:
                seg = cand
            else:
                break
        if seg is None:
            seg = segments[0]
        ds = max(0.0, s - seg['s'])
        return seg['a'] + seg['b']*ds + seg['c']*ds*ds + seg['d']*ds*ds*ds

    @staticmethod
    def _eval_lane_width_at(width_segs: List[Dict], s_rel: float) -> float:
        if not width_segs:
            return 0.0
        seg = None
        for w in width_segs:
            if s_rel + 1e-9 >= w['sOffset']:
                seg = w
            else:
                break
        if seg is None:
            seg = width_segs[0]
        ds = max(0.0, s_rel - seg['sOffset'])
        return seg['a'] + seg['b']*ds + seg['c']*ds*ds + seg['d']*ds*ds*ds

    @staticmethod
    def _offset_point_normal(x: float, y: float, hdg: float, v: float) -> Tuple[float, float]:
        nx = -math.sin(hdg)
        ny =  math.cos(hdg)
        return x + v * nx, y + v * ny

    # ----------------------------
    # 距离/可视化
    # ----------------------------
    @staticmethod
    def _find_nearest_distance_to_xodr(json_point: Dict, xodr_points: List[Tuple[float, float]]) -> float:
        jx, jy = json_point['x'], json_point['y']
        min_d2 = float('inf')
        for xx, yy in xodr_points:
            dx = jx - xx; dy = jy - yy
            d2 = dx*dx + dy*dy
            if d2 < min_d2:
                min_d2 = d2
        return math.sqrt(min_d2) if min_d2 < float('inf') else float('inf')

    def visualize_point_matching(self, save_path: str = None) -> str:
        """绘制：左-总体分布；右-偏移分析"""
        print("\n🎨 Generating visualization charts...")

        all_json_points = self._get_all_json_points()
        xodr_data = self._sample_xodr_curves_and_lanes()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：总体分布（参考线画线段；车道边界按各自折线画）
        self._plot_overall_distribution(ax1, all_json_points, xodr_data)

        # 右图：偏移分析（参考线+所有边界作为对比基准）
        xodr_points = []
        xodr_points.extend(xodr_data.get("reference_lines", []))
        for pts in xodr_data.get("lane_boundaries", {}).values():
            xodr_points.extend(pts)
        self._plot_deviation_analysis(ax2, all_json_points, xodr_points)

        plt.tight_layout()
        if save_path is None:
            save_path = self.json_file.parent / f"{self.json_file.stem}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization chart saved: {save_path}")
        try:
            plt.show()
        except Exception:
            pass
        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points: List[Dict], xodr_data: Dict):
        """Plot overall point distribution"""

        # 1) 参考线：按 road 分别连线
        refs_by_road = xodr_data.get("reference_by_road", {})
        for i, (rid, pts) in enumerate(refs_by_road.items()):
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color='skyblue', linewidth=1.4, alpha=0.9,
                        label='XODR reference' if i == 0 else None)

        # 2) 车道边界（多条折线；不同 road + 车道独立）
        lane_bd = xodr_data.get("lane_boundaries", {})
        if lane_bd:
            colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(lane_bd))))
            for idx, (key, pts) in enumerate(lane_bd.items()):
                if len(pts) < 2:
                    continue
                xs, ys = zip(*pts)
                # 只让第一条进入图例，避免过长
                ax.plot(xs, ys, linewidth=1.6, color=colors[idx % len(colors)], alpha=0.95,
                        label='XODR lane R/L' if idx == 0 else None)

        # 3) JSON 边界点
        bound_points = [p for p in json_points if p['source'] == 'bound']
        if bound_points:
            bx = [p['x'] for p in bound_points]; by = [p['y'] for p in bound_points]
            ax.scatter(bx, by, c='red', s=28, alpha=0.9, label=f'JSON bounds ({len(bound_points)})')

        # 4) JSON 物体点
        obj_points = [p for p in json_points if p['source'] == 'object']
        if obj_points:
            ox = [p['x'] for p in obj_points]; oy = [p['y'] for p in obj_points]
            ax.scatter(ox, oy, c='orange', s=28, alpha=0.9, label=f'JSON objects ({len(obj_points)})')

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('JSON vs XODR Distribution')
        ax.grid(True, alpha=0.3); ax.axis('equal')
        ax.legend()

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_points: List[Tuple[float, float]]):
        """Plot deviation analysis（以所有 XODR 点作为基准）"""
        deviations = [self._find_nearest_distance_to_xodr(p, xodr_points) for p in json_points]
        json_x = [p['x'] for p in json_points]
        json_y = [p['y'] for p in json_points]

        # 背景：XODR 密集点
        if xodr_points:
            xp, yp = zip(*xodr_points)
            ax.scatter(xp, yp, c='lightgray', s=3, alpha=0.35, label='XODR samples')

        # JSON 点按偏移着色
        scatter = ax.scatter(json_x, json_y, c=deviations, s=50,
                             cmap='RdYlGn_r', alpha=0.9, edgecolors='black', linewidth=0.4)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deviation (m)')

        stats_text = f"""Statistics:
Total points: {len(json_points)}
Avg deviation: {np.mean(deviations):.3f}m
Max deviation: {np.max(deviations):.3f}m
Min deviation: {np.min(deviations):.3f}m
Over threshold: {sum(1 for d in deviations if d > self.threshold)}/{len(deviations)}"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Deviation Analysis (Avg: {np.mean(deviations):.3f}m)')
        ax.grid(True, alpha=0.3); ax.axis('equal')
        ax.legend()

    # ----------------------------
    # 统计计数
    # ----------------------------
    def _count_xodr_lanes(self) -> int:
        """统计XODR中的车道数量（行车道）"""
        count = 0
        for road in self.xodr_data.findall('./road'):
            lanes_node = road.find('./lanes')
            if lanes_node is None:
                continue
            for lane in lanes_node.findall('.//lane'):
                if lane.get('type') == 'driving':
                    count += 1
        return count

    def _count_xodr_road_marks(self) -> int:
        """统计XODR中的道路标记数量（粗略）"""
        return len(self.xodr_data.findall('.//roadMark'))

    def _count_xodr_objects(self) -> int:
        """统计XODR中的物体数量"""
        return len(self.xodr_data.findall('.//object'))

    # ----------------------------
    # 报告（占位）
    # ----------------------------
    def generate_report(self) -> str:
        """生成质检报告（占位实现）"""
        print("\n📝 生成质检报告...")
        if self.report.completeness_score == 0:
            self.check_completeness()
        if self.report.consistency_score == 0:
            self.check_curve_consistency()
        html_report = self._generate_html_report()
        report_file = self.json_file.parent / f"{self.json_file.stem}_quality_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"✅ 质检报告已生成: {report_file}")
        return str(report_file)

    @staticmethod
    def _generate_html_report() -> str:
        """生成HTML报告（占位）"""
        return "<html><body><h1>质检报告生成中...</h1></body></html>"


# 使用示例
if __name__ == "__main__":
    checker = QualityChecker(
        json_file="../src/sample_objects.json",
        xodr_file="../src/sample_objects.xodr",
        threshold=0.1  # 10cm阈值
    )

    # 执行质检
    completeness = checker.check_completeness()
    consistency = checker.check_curve_consistency()

    # 可视化
    viz_file = checker.visualize_point_matching()

    # 报告
    report_file = checker.generate_report()

    print(f"\n📋 Quality check summary:")
    print(f"   Completeness score: {completeness:.1%}")
    print(f"   Consistency score: {consistency:.1%}")
    print(f"   Warning count: {len(checker.report.warnings)}")
    print(f"   Visualization chart: {viz_file}")
    print(f"   Report file: {report_file}")
