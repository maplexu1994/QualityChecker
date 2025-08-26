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
        检查曲线一致性：对比XODR曲线/边界与JSON点的偏移
        """
        print("\n🔍 开始曲线一致性检查...")

        warnings = []

        # JSON 点
        all_json_points = self._get_all_json_points()

        # XODR 点（参考线+车道边界）
        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_points = []
        xodr_points.extend(xodr_data.get("reference_lines", []))
        xodr_points.extend(xodr_data.get("lane_boundaries", []))

        print(f"   📍 JSON总坐标点数: {len(all_json_points)}")
        print(f"   📍 XODR采样点数: {len(xodr_points)} "
              f"(ref: {len(xodr_data.get('reference_lines', []))}, "
              f"lanes: {len(xodr_data.get('lane_boundaries', []))})")

        if not xodr_points:
            print("   ❌ 没有找到XODR采样点")
            return 0.0

        # 计算每个 JSON 点到 XODR 的最近距离
        deviations = []
        for i, json_point in enumerate(all_json_points):
            min_distance = self._find_nearest_distance_to_xodr(json_point, xodr_points)
            deviations.append(min_distance)

            if min_distance > self.threshold:
                warnings.append(
                    f"⚠️ 坐标点 ({json_point['x']:.1f}, {json_point['y']:.1f}) "
                    f"偏移超过阈值: {min_distance:.3f}m > {self.threshold}m"
                )

            if (i + 1) % 100 == 0 or i == len(all_json_points) - 1:
                print(f"   📊 已处理 {i + 1}/{len(all_json_points)} 个点")

        # 统计&得分
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

    def _sample_xodr_curves_and_lanes(self, num_points=100):
        """
        采样 XODR 文件中的几何曲线 (参考线) 和车道边界 (lanes)
        返回 dict，包含 reference line 和 lanes 的点集
        """
        import xml.etree.ElementTree as ET
        import math
        import numpy as np

        xodr_path = self.xodr_file
        tree = ET.parse(xodr_path)
        root = tree.getroot()

        results = {
            "reference_lines": [],
            "lane_boundaries": []
        }

        # -----------------------------
        # 1. 解析参考线 (planView/geometry)
        # -----------------------------
        for geometry in root.findall('.//planView/geometry'):
            x0 = float(geometry.attrib["x"])
            y0 = float(geometry.attrib["y"])
            hdg = float(geometry.attrib["hdg"])
            length = float(geometry.attrib["length"])

            geom_elem = list(geometry)[0]
            tag = geom_elem.tag

            def local_to_global(u, v):
                x = x0 + math.cos(hdg) * u - math.sin(hdg) * v
                y = y0 + math.sin(hdg) * u + math.cos(hdg) * v
                return x, y

            if tag == "line":
                s_vals = np.linspace(0, length, num_points)
                for s in s_vals:
                    results["reference_lines"].append(local_to_global(s, 0))

            elif tag == "arc":
                curvature = float(geom_elem.attrib["curvature"])
                radius = 1.0 / curvature if curvature != 0 else 1e6
                angle_vals = np.linspace(0, length * curvature, num_points)
                for angle in angle_vals:
                    u = radius * math.sin(angle)
                    v = radius * (1 - math.cos(angle))
                    results["reference_lines"].append(local_to_global(u, v))

            elif tag == "spiral":
                curv_start = float(geom_elem.attrib["curvStart"])
                curv_end = float(geom_elem.attrib["curvEnd"])
                s_vals = np.linspace(0, length, num_points)
                for s in s_vals:
                    curvature = curv_start + (curv_end - curv_start) * (s / length)
                    theta = curvature * s / 2.0
                    u = s * math.cos(theta)
                    v = s * math.sin(theta)
                    results["reference_lines"].append(local_to_global(u, v))

            elif tag == "paramPoly3":
                aU, bU, cU, dU = [float(geom_elem.attrib.get(k, 0)) for k in ["aU", "bU", "cU", "dU"]]
                aV, bV, cV, dV = [float(geom_elem.attrib.get(k, 0)) for k in ["aV", "bV", "cV", "dV"]]
                p_range = geom_elem.attrib.get("pRange", "normalized")
                if p_range == "arcLength":
                    t_vals = np.linspace(0, length, num_points)
                else:
                    t_vals = np.linspace(0, 1, num_points)
                for t in t_vals:
                    u = aU + bU * t + cU * t ** 2 + dU * t ** 3
                    v = aV + bV * t + cV * t ** 2 + dV * t ** 3
                    results["reference_lines"].append(local_to_global(u, v))

        # -----------------------------
        # 2. 解析车道边界 (lanes)
        # -----------------------------
        for lane_boundary in root.findall('.//lane/roadMark'):
            # NOTE: 这里我们简化，只用 roadMark 的 sOffset 和宽度来近似车道边界
            # 如果需要完整几何，可以进一步解析 <border> / <width>
            s_offset = float(lane_boundary.attrib.get("sOffset", 0))
            width = float(lane_boundary.attrib.get("width", 0))

            # 车道边界点 (这里只给简单的平移示例，真实情况需要跟随 reference line 曲线计算)
            for ref_pt in results["reference_lines"]:
                x, y = ref_pt
                results["lane_boundaries"].append((x, y + width))

        return results

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
        Visualize JSON points and XODR sampling points matching
        """
        print("\n🎨 Generating visualization charts...")

        # Get data
        all_json_points = self._get_all_json_points()
        xodr_sample_points = self._sample_xodr_curves_and_lanes()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Overall distribution
        self._plot_overall_distribution(ax1, all_json_points, xodr_sample_points)

        # Right plot: Deviation analysis
        self._plot_deviation_analysis(ax2, all_json_points, xodr_sample_points)

        plt.tight_layout()

        # Save image
        if save_path is None:
            save_path = self.json_file.parent / f"{self.json_file.stem}_visualization.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization chart saved: {save_path}")

        # Show image (if in interactive environment)
        try:
            plt.show()
        except:
            pass

        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points: List[Dict], xodr_data: Dict):
        """Plot overall point distribution"""

        # ---------------- XODR参考线 ----------------
        if xodr_data["reference_lines"]:
            ref_x, ref_y = zip(*xodr_data["reference_lines"])
            ax.scatter(ref_x, ref_y, c='skyblue', s=8, alpha=0.7,
                       label=f'XODR reference ({len(ref_x)})')

        # ---------------- XODR车道边界 ----------------
        if xodr_data["lane_boundaries"]:
            lane_x, lane_y = zip(*xodr_data["lane_boundaries"])
            ax.scatter(lane_x, lane_y, c='lime', s=8, alpha=0.7,
                       label=f'XODR lanes ({len(lane_x)})')

        # ---------------- JSON边界点 ----------------
        bound_points = [p for p in json_points if p['source'] == 'bound']
        if bound_points:
            bx, by = [p['x'] for p in bound_points], [p['y'] for p in bound_points]
            ax.scatter(bx, by, c='red', s=30, alpha=0.8,
                       label=f'JSON bounds ({len(bound_points)})')

        # ---------------- JSON物体点 ----------------
        obj_points = [p for p in json_points if p['source'] == 'object']
        if obj_points:
            ox, oy = [p['x'] for p in obj_points], [p['y'] for p in obj_points]
            ax.scatter(ox, oy, c='orange', s=30, alpha=0.8,
                       label=f'JSON objects ({len(obj_points)})')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('JSON vs XODR Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_data: Dict):
        """Plot deviation analysis"""

        # 合并参考线 + 车道边界，作为采样点全集
        xodr_points = []
        if xodr_data["reference_lines"]:
            xodr_points.extend(xodr_data["reference_lines"])
        if xodr_data["lane_boundaries"]:
            xodr_points.extend(xodr_data["lane_boundaries"])

        # 计算每个 JSON 点的偏移
        deviations = []
        for json_point in json_points:
            deviation = self._find_nearest_distance_to_xodr(json_point, xodr_points)
            deviations.append(deviation)

        # JSON 点绘制，颜色表示偏移大小
        json_x = [p['x'] for p in json_points]
        json_y = [p['y'] for p in json_points]
        scatter = ax.scatter(json_x, json_y, c=deviations, s=50,
                             cmap='RdYlGn_r', alpha=0.8, edgecolors='black', linewidth=0.5)

        # 背景画出 XODR 点
        if xodr_points:
            xp, yp = zip(*xodr_points)
            ax.scatter(xp, yp, c='lightgray', s=5, alpha=0.3, label='XODR samples')

        # colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deviation (m)')

        # 文本统计
        stats_text = f"""Statistics:
    Total points: {len(json_points)}
    Avg deviation: {np.mean(deviations):.3f}m
    Max deviation: {np.max(deviations):.3f}m
    Min deviation: {np.min(deviations):.3f}m
    Over threshold: {sum(1 for d in deviations if d > self.threshold)}/{len(deviations)}"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Deviation Analysis (Avg: {np.mean(deviations):.3f}m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def analyze_matching_details(self) -> Dict:
        print("\n🔍 Analyzing matching details...")

        all_json_points = self._get_all_json_points()
        # 旧函数名 _sample_xodr_curves 已不存在，改用新函数并合并
        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_sample_points = []
        xodr_sample_points.extend(xodr_data.get("reference_lines", []))
        xodr_sample_points.extend(xodr_data.get("lane_boundaries", []))

        analysis_results = {
            'total_json_points': len(all_json_points),
            'total_xodr_points': len(xodr_sample_points),
            'detailed_matches': []
        }

        print(f"\n📊 Detailed matching analysis:")
        print(f"{'Index':<5} {'Type':<6} {'JSON Coord':<20} {'Nearest XODR':<20} {'Dev(m)':<8} {'Status'}")
        print("-" * 80)

        for i, json_point in enumerate(all_json_points):
            min_distance = float('inf')
            nearest_xodr_point = None
            for xodr_point in xodr_sample_points:
                distance = math.sqrt((json_point['x'] - xodr_point[0]) ** 2 +
                                     (json_point['y'] - xodr_point[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_xodr_point = xodr_point

            status = "✅ Pass" if min_distance <= self.threshold else \
                ("⚠️ Warning" if min_distance <= self.threshold * 2 else "❌ Fail")

            analysis_results['detailed_matches'].append({
                'index': i,
                'json_point': json_point,
                'nearest_xodr_point': nearest_xodr_point,
                'deviation': min_distance,
                'status': status
            })

            if i < 20:
                json_coord = f"({json_point['x']:.1f}, {json_point['y']:.1f})"
                xodr_coord = f"({nearest_xodr_point[0]:.1f}, {nearest_xodr_point[1]:.1f})" if nearest_xodr_point else "None"
                source = json_point['source']
                print(f"{i:<5} {source:<6} {json_coord:<20} {xodr_coord:<20} {min_distance:<8.3f} {status}")
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

    def generate_report(self) -> str:
        """生成质检报告"""
        print("\n📝 生成质检报告...")

        # 执行所有检查
        if self.report.completeness_score == 0:
            self.check_completeness()
        if self.report.consistency_score == 0:
            self.check_curve_consistency()

        # 生成HTML报告（下一步实现）
        html_report = self._generate_html_report()

        # 保存报告
        report_file = self.json_file.parent / f"{self.json_file.stem}_quality_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"✅ 质检报告已生成: {report_file}")
        return str(report_file)

    def _generate_html_report(self) -> str:
        """生成HTML报告（待实现）"""
        return "<html><body><h1>质检报告生成中...</h1></body></html>"


# 使用示例
if __name__ == "__main__":
    # 创建质检器实例
    checker = QualityChecker(
        json_file="../src/sample_objects.json",
        xodr_file="../src/sample_objects.xodr",
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