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
import matplotlib.cm as cm
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
        self.json_data = self._load_json()
        self.xodr_data = self._load_xodr()
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

    def _evaluate_polynomial(self, s: float, a: float, b: float, c: float, d: float) -> float:
        """计算三次多项式 a + bs + cs^2 + ds^3 的值"""
        return a + b * s + c * s ** 2 + d * s ** 3

    def check_completeness(self) -> float:
        """检查元素完整性"""
        print("\n🔍 开始元素完整性检查...")
        completeness_details = {}
        total_score = 0
        check_count = 0

        json_lanes = len(self.json_data.get('lanes', []))
        xodr_lanes = self._count_xodr_lanes()
        lane_score = min(xodr_lanes / json_lanes, 1.0) if json_lanes > 0 else 1.0
        completeness_details['lanes'] = {'json_count': json_lanes, 'xodr_count': xodr_lanes, 'score': lane_score}
        total_score += lane_score
        check_count += 1

        json_bounds = len(self.json_data.get('bounds', []))
        xodr_road_marks = self._count_xodr_road_marks()
        bound_score = min(xodr_road_marks / (json_bounds * 2), 1.0) if json_bounds > 0 else 1.0
        completeness_details['bounds'] = {'json_count': json_bounds, 'xodr_count': xodr_road_marks,
                                          'score': bound_score}
        total_score += bound_score
        check_count += 1

        json_objects = len(self.json_data.get('objects', []))
        xodr_objects = self._count_xodr_objects()
        object_score = min(xodr_objects / json_objects, 1.0) if json_objects > 0 else 1.0
        completeness_details['objects'] = {'json_count': json_objects, 'xodr_count': xodr_objects,
                                           'score': object_score}
        total_score += object_score
        check_count += 1

        self.report.completeness_score = total_score / check_count if check_count > 0 else 0
        self.report.details['completeness'] = completeness_details
        print(f"📊 完整性检查完成，总体得分: {self.report.completeness_score:.2%}")
        return self.report.completeness_score

    def check_curve_consistency(self) -> float:
        """检查曲线一致性"""
        print("\n🔍 开始曲线一致性检查...")
        warnings = []
        all_json_points = self._get_all_json_points()
        xodr_points = self._get_all_xodr_points()

        print(f"   📍 JSON总坐标点数: {len(all_json_points)}")
        print(f"   📍 XODR总采样点数: {len(xodr_points)}")

        if not xodr_points:
            print("   ❌ 没有找到XODR采样点")
            self.report.consistency_score = 0.0
            return 0.0

        deviations = [self._find_nearest_distance_to_xodr(p, xodr_points) for p in all_json_points]
        for i, dist in enumerate(deviations):
            if dist > self.threshold:
                warnings.append(
                    f"⚠️ 坐标点 ({all_json_points[i]['x']:.1f}, {all_json_points[i]['y']:.1f}) 偏移超过阈值: {dist:.3f}m > {self.threshold}m")

        avg_deviation = float(np.mean(deviations)) if deviations else 0.0
        consistency_score = max(0, 1 - (avg_deviation / self.threshold)) if self.threshold > 0 else 0
        self.report.consistency_score = consistency_score
        self.report.warnings.extend(warnings)
        self.report.details['consistency'] = {
            'average_deviation': avg_deviation,
            'max_deviation': float(np.max(deviations)) if deviations else 0.0,
            'min_deviation': float(np.min(deviations)) if deviations else 0.0,
            'points_over_threshold': sum(1 for d in deviations if d > self.threshold)
        }
        print(f"📊 一致性检查完成，得分: {consistency_score:.2%}")
        print(f"📊 平均偏移: {avg_deviation:.3f}m")
        return consistency_score

    def _get_all_json_points(self) -> List[Dict]:
        """获取JSON中所有的坐标点"""
        all_points = []
        for bound in self.json_data.get('bounds', []):
            for pt in bound['pts']:
                all_points.append({'x': pt['x'], 'y': pt['y'], 'source': 'bound'})
        for obj in self.json_data.get('objects', []):
            for pt in obj.get('outline', []):
                all_points.append({'x': pt['x'], 'y': pt['y'], 'source': 'object'})
        return all_points

    def _sample_xodr_curves_and_lanes(self, num_points_per_road=2000):
        """
        [修复] 通过累加<width>元素来解析车道边界。
        """
        results = {"reference_lines": {}, "lane_boundaries": {}}
        for road in self.xodr_data.findall('.//road'):
            road_id = road.get("id")
            road_length = float(road.get("length"))
            road_reference_data = []

            # 1. 采样道路参考线
            for geometry in road.findall('.//planView/geometry'):
                s0, x0, y0, hdg, length = [float(geometry.get(k, 0)) for k in ['s', 'x', 'y', 'hdg', 'length']]
                segment_points = max(2, int((
                                                        length / road_length) * num_points_per_road)) if road_length > 0 else num_points_per_road
                s_vals = np.linspace(0, length, segment_points)

                if not list(geometry): continue  # 跳过空的geometry标签
                tag = list(geometry)[0].tag

                for s in s_vals:
                    x, y, new_hdg = x0, y0, hdg
                    if tag == "line":
                        x, y = x0 + s * math.cos(hdg), y0 + s * math.sin(hdg)
                    elif tag == "arc":
                        curvature = float(list(geometry)[0].get("curvature", 0))
                        if abs(curvature) > 1e-9:
                            radius, angle = 1.0 / curvature, s * curvature
                            x_loc, y_loc = radius * math.sin(angle), radius * (1 - math.cos(angle))
                            x = x0 + (x_loc * math.cos(hdg) - y_loc * math.sin(hdg))
                            y = y0 + (x_loc * math.sin(hdg) + y_loc * math.cos(hdg))
                            new_hdg = hdg + angle
                    road_reference_data.append({'s': s0 + s, 'point': (x, y), 'hdg': new_hdg})

            road_reference_data.sort(key=lambda p: p['s'])
            if road_reference_data:
                results["reference_lines"][f"road_{road_id}"] = [p['point'] for p in road_reference_data]

            # 2. 解析车道边界 (通过累加宽度)
            for i, section in enumerate(road.findall('.//lanes/laneSection')):
                s_start_section = float(section.get('s', 0))

                next_sections = road.findall('.//lanes/laneSection')
                s_end_section = road_length
                if i + 1 < len(next_sections):
                    s_end_section = float(next_sections[i + 1].get('s', 0))

                ref_data_in_section = [d for d in road_reference_data if s_start_section <= d['s'] < s_end_section]
                if not ref_data_in_section: continue

                # 处理右侧车道 (id: -1, -2, ...)
                cumulative_width_poly_right = [0.0] * 4  # a,b,c,d
                right_lanes = sorted(section.findall('.//right/lane'), key=lambda l: int(l.get('id')), reverse=True)
                for lane in right_lanes:
                    if lane.get('type') != 'driving': continue
                    lane_id = int(lane.attrib["id"])
                    width_elem = lane.find('width')
                    if width_elem is None: continue

                    width_poly = [float(width_elem.get(k, 0)) for k in ['a', 'b', 'c', 'd']]

                    # 累加当前车道宽度到总偏移多项式
                    for j in range(4): cumulative_width_poly_right[j] += width_poly[j]

                    border_pts = []
                    for ref_data in ref_data_in_section:
                        s_rel = ref_data['s'] - s_start_section
                        ref_pt, ref_hdg = ref_data['point'], ref_data['hdg']
                        offset = self._evaluate_polynomial(s_rel, *cumulative_width_poly_right)

                        perp_angle = ref_hdg + math.pi / 2
                        offset_x = offset * math.cos(perp_angle) * -1  # 右侧偏移为负
                        offset_y = offset * math.sin(perp_angle) * -1
                        border_pts.append((ref_pt[0] + offset_x, ref_pt[1] + offset_y))
                    if border_pts:
                        results["lane_boundaries"][f"road_{road_id}_lane_{lane_id}"] = border_pts

                # 处理左侧车道 (id: 1, 2, ...)
                cumulative_width_poly_left = [0.0] * 4  # a,b,c,d
                left_lanes = sorted(section.findall('.//left/lane'), key=lambda l: int(l.get('id')))
                for lane in left_lanes:
                    if lane.get('type') != 'driving': continue
                    lane_id = int(lane.attrib["id"])
                    width_elem = lane.find('width')
                    if width_elem is None: continue

                    width_poly = [float(width_elem.get(k, 0)) for k in ['a', 'b', 'c', 'd']]
                    for j in range(4): cumulative_width_poly_left[j] += width_poly[j]

                    border_pts = []
                    for ref_data in ref_data_in_section:
                        s_rel = ref_data['s'] - s_start_section
                        ref_pt, ref_hdg = ref_data['point'], ref_data['hdg']
                        offset = self._evaluate_polynomial(s_rel, *cumulative_width_poly_left)

                        perp_angle = ref_hdg + math.pi / 2
                        offset_x = offset * math.cos(perp_angle)
                        offset_y = offset * math.sin(perp_angle)
                        border_pts.append((ref_pt[0] + offset_x, ref_pt[1] + offset_y))
                    if border_pts:
                        results["lane_boundaries"][f"road_{road_id}_lane_{lane_id}"] = border_pts
        return results

    def _get_all_xodr_points(self) -> List[Tuple[float, float]]:
        """合并所有XODR采样点"""
        data = self._sample_xodr_curves_and_lanes()
        points = []
        for ref_pts in data.get("reference_lines", {}).values(): points.extend(ref_pts)
        for lane_pts in data.get("lane_boundaries", {}).values(): points.extend(lane_pts)
        return points

    def _find_nearest_distance_to_xodr(self, json_point: Dict, xodr_points: List[Tuple[float, float]]) -> float:
        """计算JSON点到XODR点的最短距离"""
        jx, jy = json_point['x'], json_point['y']
        xodr_pts_arr = np.array(xodr_points)
        distances = np.sqrt(np.sum((xodr_pts_arr - (jx, jy)) ** 2, axis=1))
        return np.min(distances)

    def visualize_point_matching(self, save_path: str = None) -> str:
        """可视化JSON点与XODR曲线的匹配情况"""
        print("\n🎨 生成可视化图表...")
        all_json_points = self._get_all_json_points()
        xodr_sample_data = self._sample_xodr_curves_and_lanes()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self._plot_overall_distribution(ax1, all_json_points, xodr_sample_data)
        self._plot_deviation_analysis(ax2, all_json_points, xodr_sample_data)
        plt.tight_layout()
        if save_path is None:
            save_path = self.json_file.parent / f"{self.json_file.stem}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存: {save_path}")
        try:
            plt.show()
        except Exception as e:
            print(f"ℹ️  无法显示图表 (可能在无GUI环境下运行): {e}")
        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points: List[Dict], xodr_data: Dict):
        """
        [修复] 绘制整体分布图，参考线使用散点，车道边界使用曲线。
        """
        # 1. 绘制参考线 (使用散点图以观察趋势)
        reference_lines = xodr_data.get("reference_lines", {})
        total_ref_points = sum(len(pts) for pts in reference_lines.values())
        is_first_ref_plot = True
        for road_id, ref_pts in reference_lines.items():
            if ref_pts:
                ref_x, ref_y = zip(*ref_pts)
                ax.scatter(ref_x, ref_y, c='skyblue', s=2, alpha=0.7,
                           label=f'XODR reference ({total_ref_points})' if is_first_ref_plot else "")
                is_first_ref_plot = False

        # 2. 绘制车道边界 (每条独立绘制)
        lane_boundaries = xodr_data.get("lane_boundaries", {})
        colors = cm.get_cmap('gist_rainbow', len(lane_boundaries) + 1)
        for i, (lane_id, lane_pts) in enumerate(lane_boundaries.items()):
            if lane_pts:
                lane_x, lane_y = zip(*lane_pts)
                ax.plot(lane_x, lane_y, c=colors(i), linewidth=1.5, label=f'{lane_id} ({len(lane_x)})')

        # 3. 绘制 JSON 点
        for source_type, color in [('bound', 'red'), ('object', 'orange')]:
            points = [p for p in json_points if p['source'] == source_type]
            if points:
                px, py = [p['x'] for p in points], [p['y'] for p in points]
                ax.scatter(px, py, c=color, s=30, alpha=0.8, edgecolors='k', linewidth=0.5,
                           label=f'JSON {source_type}s ({len(points)})')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('JSON vs XODR Distribution')
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='box')

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_data: Dict):
        """绘制偏移分析图"""
        xodr_points = []
        for pts_list in xodr_data.get("reference_lines", {}).values(): xodr_points.extend(pts_list)
        for pts_list in xodr_data.get("lane_boundaries", {}).values(): xodr_points.extend(pts_list)

        if not json_points or not xodr_points:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
            return

        deviations = [self._find_nearest_distance_to_xodr(p, xodr_points) for p in json_points]
        json_x, json_y = [p['x'] for p in json_points], [p['y'] for p in json_points]

        vmax = self.threshold * 3
        scatter = ax.scatter(json_x, json_y, c=deviations, s=50, cmap='RdYlGn_r', alpha=0.9, edgecolors='black',
                             linewidth=0.5, vmin=0, vmax=vmax)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deviation (m)')

        stats_text = f"Statistics:\n  Total points: {len(json_points)}\n  Avg deviation: {np.mean(deviations):.3f}m\n  Max deviation: {np.max(deviations):.3f}m\n  Over threshold: {sum(1 for d in deviations if d > self.threshold)}/{len(deviations)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Deviation Analysis (Threshold: {self.threshold}m)')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='box')

    def _count_xodr_lanes(self) -> int:
        return len(self.xodr_data.findall('.//lane[@type="driving"]'))

    def _count_xodr_road_marks(self) -> int:
        return len(self.xodr_data.findall('.//roadMark'))

    def _count_xodr_objects(self) -> int:
        return len(self.xodr_data.findall('.//object'))

    def generate_report(self) -> str:
        """生成质检报告"""
        print("\n📝 生成质检报告...")
        # ... (implementation unchanged)
        return "report.html"


if __name__ == "__main__":
#     # 创建虚拟测试文件
#     sample_json_content = {
#         "bounds": [{"id": "b1", "pts": [{"x": i, "y": 5.25, "z": 0} for i in range(50)]}],
#         "objects": [{"id": "o1", "outline": [{"x": i, "y": -3.75, "z": 0} for i in range(50)]}],
#         "lanes": [{"id": "l1"}, {"id": "l2"}]
#     }
#     with open("sample_data.json", "w") as f:
#         json.dump(sample_json_content, f)
#
#     sample_xodr_content = """<?xml version="1.0" standalone="yes"?>
# <OpenDRIVE>
#     <road name="road 1" length="50.0" id="1" junction="-1">
#         <planView><geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="50.0"><line/></geometry></planView>
#         <lanes>
#             <laneSection s="0.0">
#                 <left>
#                     <lane id="1" type="driving" level="false">
#                         <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
#                     </lane>
#                     <lane id="2" type="driving" level="false">
#                         <width sOffset="0.0" a="1.75" b="0.0" c="0.0" d="0.0"/>
#                     </lane>
#                 </left>
#                 <center><lane id="0" type="driving" level="false"/></center>
#                 <right>
#                     <lane id="-1" type="driving" level="false">
#                         <width sOffset="0.0" a="3.75" b="0.0" c="0.0" d="0.0"/>
#                     </lane>
#                 </right>
#             </laneSection>
#         </lanes>
#     </road>
# </OpenDRIVE>
#     """
#     with open("sample_data.xodr", "w") as f:
#         f.write(sample_xodr_content)

    checker = QualityChecker(json_file="../src/sample_objects.json", xodr_file="../src/sample_objects.xodr", threshold=0.2)
    checker.check_completeness()
    checker.check_curve_consistency()
    checker.visualize_point_matching()
