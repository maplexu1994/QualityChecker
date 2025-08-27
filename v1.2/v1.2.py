#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker — v1.2 merged final
- 主体逻辑基于：Qa Json2xodr V1.2 Merged Fixed（完整性 + 一致性 + 匹配明细 + 可视化）
- HTML 报告样式：沿用 v1.0 的美化模板与结构（自包含 CSS），并修正 Py3.11 下 f-string 的嵌套问题
- 新增/保留：
  1) 边界统计口径：JSON 唯一边界ID  vs  XODR 拓扑边界（laneSection 左/右车道数，并在左右同时存在时 +1 记中央分隔）
  2) sign → signal 完整性检验
  3) 共有边界识别（≥2 车道引用的边界）
注：一致性仅以 JSON 边界点参与计算；objects 不参与一致性，仅在完整性计数。
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import math
import matplotlib.pyplot as plt
import html as _html
from datetime import datetime


@dataclass
class QualityReport:
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
    def __init__(self, json_file: str, xodr_file: str, threshold: float = 0.1):
        self.json_file = Path(json_file)
        self.xodr_file = Path(xodr_file)
        self.threshold = threshold
        self.json_data = self._load_json()
        self.xodr_root = self._load_xodr()
        self.report = QualityReport()
        self._viz_path = None

    # ---------------- IO ----------------
    def _load_json(self) -> Dict:
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ 成功加载JSON:", self.json_file)
        return data

    def _load_xodr(self) -> ET.Element:
        tree = ET.parse(self.xodr_file)
        root = tree.getroot()
        print("✅ 成功加载XODR:", self.xodr_file)
        return root

    # ---------------- JSON helpers ----------------
    def _json_signs(self) -> List[Dict]:
        return list(self.json_data.get('sign', []))

    def _detect_shared_bounds(self) -> Tuple[List[int], List[int]]:
        lanes = self.json_data.get('lanes', [])
        usage: Dict[int, int] = {}
        for ln in lanes:
            for key in ('left_bound_id', 'right_bound_id'):
                bid = ln.get(key)
                if bid is None:
                    continue
                usage[bid] = usage.get(bid, 0) + 1
        if usage:
            unique_ids = sorted(usage.keys())
            shared_ids = sorted([bid for bid, c in usage.items() if c >= 2])
            return unique_ids, shared_ids
        # 回退：直接用 bounds[*].id
        b_ids = sorted({int(b.get('id')) for b in self.json_data.get('bounds', []) if 'id' in b})
        return b_ids, []

    # ---------------- XODR counters ----------------
    def _count_xodr_lanes(self) -> int:
        cnt = 0
        for lane in self.xodr_root.iter('lane'):
            t_attr = (lane.get('type') or '').lower()
            if t_attr == 'driving':
                cnt += 1
                continue
            t_node = lane.find('type')
            if t_node is not None and (t_node.text or '').strip().lower() == 'driving':
                cnt += 1
        return cnt

    def _count_xodr_topo_boundaries(self) -> int:
        total = 0
        for road in self.xodr_root.findall('road'):
            lanes = road.find('lanes')
            if lanes is None:
                continue
            for ls in lanes.findall('laneSection'):
                left = ls.find('left'); right = ls.find('right')
                l_cnt = len(left.findall('lane')) if left is not None else 0
                r_cnt = len(right.findall('lane')) if right is not None else 0
                if l_cnt > 0:
                    total += l_cnt
                if r_cnt > 0:
                    total += r_cnt
                if l_cnt > 0 and r_cnt > 0:
                    total += 1  # 中央分隔
        return total

    def _count_xodr_objects(self) -> int:
        return sum(1 for _ in self.xodr_root.iter('object'))

    def _count_xodr_signals(self) -> int:
        return sum(1 for _ in self.xodr_root.iter('signal'))

    # ---------------- Completeness ----------------
    def check_completeness(self) -> float:
        print("\n🔍 开始完整性检查…")
        details = {}
        total = 0.0
        n = 0

        # 车道（driving）
        json_lanes = [ln for ln in self.json_data.get('lanes', []) if (ln.get('type') or '').lower() == 'driving']
        json_lane_count = len(json_lanes)
        xodr_lanes = self._count_xodr_lanes()
        lane_score = min(xodr_lanes / json_lane_count, 1.0) if json_lane_count > 0 else 1.0
        details['lanes'] = {'json_count': json_lane_count, 'xodr_count': xodr_lanes, 'score': lane_score}
        total += lane_score; n += 1

        # 边界（唯一ID vs 拓扑边界）+ 共有边界
        unique_ids, shared_ids = self._detect_shared_bounds()
        json_bounds_unique = len(unique_ids)
        xodr_topo = self._count_xodr_topo_boundaries()
        bound_score = min(xodr_topo / json_bounds_unique, 1.0) if json_bounds_unique > 0 else 1.0
        details['bounds'] = {
            'json_count': json_bounds_unique,
            'xodr_count': xodr_topo,
            'score': bound_score,
            'unique_ids': unique_ids,
            'shared_ids': shared_ids,
        }
        self.report.details['bounds_unique_ids'] = unique_ids
        self.report.details['bounds_shared_ids'] = shared_ids
        total += bound_score; n += 1

        # 物体
        json_objs = len(self.json_data.get('objects', []))
        xodr_objs = self._count_xodr_objects()
        obj_score = min(xodr_objs / json_objs, 1.0) if json_objs > 0 else 1.0
        details['objects'] = {'json_count': json_objs, 'xodr_count': xodr_objs, 'score': obj_score}
        total += obj_score; n += 1

        # 标识（sign → signal）
        json_signs = len(self._json_signs())
        xodr_signals = self._count_xodr_signals()
        sign_score = min(xodr_signals / json_signs, 1.0) if json_signs > 0 else 1.0
        details['signs'] = {'json_count': json_signs, 'xodr_count': xodr_signals, 'score': sign_score}
        total += sign_score; n += 1

        self.report.completeness_score = total / n if n else 0.0
        self.report.details['completeness'] = details
        print("📊 完整性得分:", f"{self.report.completeness_score:.2%}")
        return self.report.completeness_score

    # ---------------- Consistency ----------------
    def _get_all_json_points(self) -> List[Dict]:
        pts = []
        for b in self.json_data.get('bounds', []):
            bid = b.get('id')
            for p in b.get('pts', []):
                pts.append({'x': p['x'], 'y': p['y'], 'z': p['z'], 'bound_id': bid, 'source': 'bound'})
        for obj in self.json_data.get('objects', []):
            oid = obj.get('id')
            for p in obj.get('outline', []):
                pts.append({'x': p['x'], 'y': p['y'], 'z': p['z'], 'object_id': oid, 'source': 'object'})
        return pts

    def _sample_xodr_curves_and_lanes(self, step: float = 0.05):
        tree = ET.parse(self.xodr_file); root = tree.getroot()
        out = {"reference_lines": [], "lane_boundaries": [], "reference_polylines": [], "lane_polylines": [], "lane_center_polylines": [], "lane_edge_polylines": []}

        def local_to_global(x0, y0, hdg, u, v):
            ch, sh = math.cos(hdg), math.sin(hdg)
            return x0 + ch * u - sh * v, y0 + sh * u + ch * v

        def sample_planview_polyline(plan_view, default_step=step):
            ref_pts = []; s_abs = 0.0; geoms = list(plan_view.findall('geometry'))
            for g in geoms:
                x0 = float(g.attrib.get('x', 0)); y0 = float(g.attrib.get('y', 0)); hdg = float(g.attrib.get('hdg', 0)); L = float(g.attrib.get('length', 0))
                if L <= 0:
                    continue
                child = list(g)[0]; tag = child.tag
                n = max(2, int(L / default_step) + 1)
                s_vals = np.linspace(0.0, L, n)
                if tag == 'line':
                    for s in s_vals:
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, s, 0.0)))
                elif tag == 'arc':
                    k = float(child.attrib.get('curvature', 0)); R = 1.0 / k if k != 0 else 1e12
                    for s in s_vals:
                        ang = k * s; u = R * math.sin(ang); v = R * (1.0 - math.cos(ang))
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, u, v)))
                elif tag == 'spiral':
                    c0 = float(child.attrib.get('curvStart', 0)); c1 = float(child.attrib.get('curvEnd', 0))
                    for s in s_vals:
                        c = c0 + (c1 - c0) * (s / L); theta = 0.5 * c * s
                        u = s * math.cos(theta); v = s * math.sin(theta)
                        ref_pts.append((s_abs + s, *local_to_global(x0, y0, hdg, u, v)))
                elif tag == 'paramPoly3':
                    aU = float(child.attrib.get('aU', 0)); bU = float(child.attrib.get('bU', 1)); cU = float(child.attrib.get('cU', 0)); dU = float(child.attrib.get('dU', 0))
                    aV = float(child.attrib.get('aV', 0)); bV = float(child.attrib.get('bV', 0)); cV = float(child.attrib.get('cV', 0)); dV = float(child.attrib.get('dV', 0))
                    p_range = child.attrib.get('pRange', 'normalized')
                    t_vals = s_vals if p_range == 'arcLength' else np.linspace(0.0, 1.0, len(s_vals))
                    for i, t in enumerate(t_vals):
                        u = aU + bU*t + cU*t*t + dU*t*t*t
                        v = aV + bV*t + cV*t*t + dV*t*t*t
                        ref_pts.append((s_abs + s_vals[i], *local_to_global(x0, y0, hdg, u, v)))
                s_abs += L
            return ref_pts

        def finite_diff_heading(xs, ys):
            th = []; n = len(xs)
            for i in range(n):
                if i == 0:
                    dx, dy = xs[1]-xs[0], ys[1]-ys[0]
                elif i == n-1:
                    dx, dy = xs[-1]-xs[-2], ys[-1]-ys[-2]
                else:
                    dx, dy = xs[i+1]-xs[i-1], ys[i+1]-ys[i-1]
                th.append(math.atan2(dy, dx) if (dx*dx + dy*dy) > 0 else 0.0)
            return th

        def width_poly_at(width_list, s_rel):
            if not width_list:
                return 0.0
            prev = width_list[0]
            for w in width_list:
                if w['sOffset'] <= s_rel:
                    prev = w
                else:
                    break
            ds = max(0.0, s_rel - prev['sOffset'])
            return prev['a'] + prev['b']*ds + prev['c']*ds*ds + prev['d']*ds*ds*ds

        def parse_lane_sections(lanes_elem):
            sections = []
            for sec in lanes_elem.findall('laneSection'):
                s0 = float(sec.attrib.get('s', 0.0))
                left = sec.find('left'); right = sec.find('right')
                one = {'s': s0, 'left': [], 'right': []}
                def collect(side_elem, side_name):
                    if side_elem is None:
                        return
                    for ln in side_elem.findall('lane'):
                        lid = ln.attrib.get('id', '')
                        widths = []
                        for w in ln.findall('width'):
                            widths.append({
                                'sOffset': float(w.attrib.get('sOffset', 0.0)),
                                'a': float(w.attrib.get('a', 0.0)),
                                'b': float(w.attrib.get('b', 0.0)),
                                'c': float(w.attrib.get('c', 0.0)),
                                'd': float(w.attrib.get('d', 0.0)),
                            })
                        widths.sort(key=lambda x: x['sOffset'])
                        one[side_name].append({'id': lid, 'widths': widths})
                    if side_name == 'left':
                        one[side_name].sort(key=lambda e: int(e['id']))
                    else:
                        one[side_name].sort(key=lambda e: abs(int(e['id'])))
                collect(left, 'left'); collect(right, 'right')
                sections.append(one)
            sections.sort(key=lambda s: s['s'])
            return sections

        def parse_lane_offsets(lanes_elem):
            offsets = []
            for lo in lanes_elem.findall('laneOffset'):
                offsets.append({
                    's': float(lo.attrib.get('s', 0.0)),
                    'a': float(lo.attrib.get('a', 0.0)),
                    'b': float(lo.attrib.get('b', 0.0)),
                    'c': float(lo.attrib.get('c', 0.0)),
                    'd': float(lo.attrib.get('d', 0.0)),
                })
            offsets.sort(key=lambda x: x['s'])
            return offsets

        def lane_offset_at(offsets, s_abs):
            if not offsets:
                return 0.0
            prev = offsets[0]
            for o in offsets:
                if o['s'] <= s_abs:
                    prev = o
                else:
                    break
            ds = max(0.0, s_abs - prev['s'])
            return prev['a'] + prev['b']*ds + prev['c']*ds*ds + prev['d']*ds*ds*ds

        # 主流程
        for road in root.findall('road'):
            road_id = road.attrib.get('id', 'unknown')
            plan_view = road.find('planView')
            if plan_view is None:
                continue
            ref = sample_planview_polyline(plan_view)
            if len(ref) < 2:
                continue
            s_arr = [p[0] for p in ref]
            x_arr = [p[1] for p in ref]
            y_arr = [p[2] for p in ref]
            th_arr = finite_diff_heading(x_arr, y_arr)
            ref_poly = [(x_arr[i], y_arr[i]) for i in range(len(ref))]
            out['reference_polylines'].append(ref_poly)
            out['reference_lines'].extend(ref_poly)

            lanes_elem = road.find('lanes')
            if lanes_elem is None:
                continue
            sections = parse_lane_sections(lanes_elem)
            offsets = parse_lane_offsets(lanes_elem)
            road_len = s_arr[-1]

            for si, sec in enumerate(sections):
                s0 = sec['s']
                s1 = sections[si+1]['s'] if si+1 < len(sections) else road_len + 1e-6
                idxs = [i for i, sv in enumerate(s_arr) if (sv >= s0 and sv <= s1)]
                if len(idxs) < 2:
                    continue
                centers = {}; edges = {}
                def ensure(d, k):
                    if k not in d:
                        d[k] = []
                    return d[k]
                for i in idxs:
                    s_here = s_arr[i]; s_rel = s_here - s0
                    nx = -math.sin(th_arr[i]); ny = math.cos(th_arr[i])
                    base = lane_offset_at(offsets, s_here)
                    # 左侧
                    cum = 0.0
                    for ln in sec['left']:
                        w = width_poly_at(ln['widths'], s_rel)
                        inner_off = base + cum
                        outer_off = inner_off + w
                        center_off = inner_off + 0.5*w
                        ensure(edges, f"left:{ln['id']}:inner").append((x_arr[i] + inner_off*nx, y_arr[i] + inner_off*ny))
                        ensure(edges, f"left:{ln['id']}:outer").append((x_arr[i] + outer_off*nx, y_arr[i] + outer_off*ny))
                        ensure(centers, f"left:{ln['id']}").append((x_arr[i] + center_off*nx, y_arr[i] + center_off*ny))
                        cum += w
                    # 右侧
                    cum = 0.0
                    for ln in sec['right']:
                        w = width_poly_at(ln['widths'], s_rel)
                        inner_off = base - cum
                        outer_off = inner_off - w
                        center_off = inner_off - 0.5*w
                        ensure(edges, f"right:{ln['id']}:inner").append((x_arr[i] + inner_off*nx, y_arr[i] + inner_off*ny))
                        ensure(edges, f"right:{ln['id']}:outer").append((x_arr[i] + outer_off*nx, y_arr[i] + outer_off*ny))
                        ensure(centers, f"right:{ln['id']}").append((x_arr[i] + center_off*nx, y_arr[i] + center_off*ny))
                        cum += w
                for key, pts in centers.items():
                    side, lane_id = key.split(':')
                    out['lane_center_polylines'].append({'road_id': road_id, 'lane_id': lane_id, 'side': side, 'points': pts})
                    out['lane_polylines'].append({'road_id': road_id, 'lane_id': lane_id, 'side': side, 'points': pts})
                for key, pts in edges.items():
                    side, lane_id, kind = key.split(':')
                    out['lane_edge_polylines'].append({'road_id': road_id, 'lane_id': lane_id, 'side': side, 'kind': kind, 'points': pts})
                    out['lane_boundaries'].extend(pts)
        return out

    def _find_nearest_distance_to_xodr(self, json_point: Dict, xodr_points: List[Tuple[float, float]]) -> float:
        jx, jy = json_point['x'], json_point['y']
        md = float('inf')
        for xp, yp in xodr_points:
            d = math.hypot(jx - xp, jy - yp)
            if d < md:
                md = d
        return md

    def check_curve_consistency(self) -> float:
        print("\n🔍 开始曲线一致性检查…")
        warnings = []
        all_json_points = self._get_all_json_points()
        bound_points = [p for p in all_json_points if p.get('source') == 'bound']

        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_points = []
        xodr_points.extend(xodr_data.get('reference_lines', []))
        xodr_points.extend(xodr_data.get('lane_boundaries', []))

        if not bound_points or not xodr_points:
            self.report.consistency_score = 0.0
            self.report.details['consistency'] = {
                'average_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0,
                'point_count': len(bound_points), 'threshold': self.threshold,
                'warnings_count': 0, 'points_over_threshold': 0
            }
            return 0.0

        devs = []
        for i, jp in enumerate(bound_points):
            d = self._find_nearest_distance_to_xodr(jp, xodr_points)
            devs.append(d)
            if d > self.threshold:
                warnings.append("⚠️ 偏移超阈值点: (%.3f, %.3f) → %.3fm" % (jp['x'], jp['y'], d))
            if (i+1) % 200 == 0 or i == len(bound_points) - 1:
                print("  已处理 %d/%d" % (i+1, len(bound_points)))

        avg_dev = float(np.mean(devs)) if devs else 0.0
        max_dev = float(np.max(devs)) if devs else 0.0
        min_dev = float(np.min(devs)) if devs else 0.0
        over = sum(1 for d in devs if d > self.threshold)

        score = max(0.0, 1.0 - (avg_dev / self.threshold)) if self.threshold > 0 else 0.0
        self.report.consistency_score = score
        self.report.warnings.extend(warnings)
        self.report.details['consistency'] = {
            'average_deviation': avg_dev, 'max_deviation': max_dev, 'min_deviation': min_dev,
            'point_count': len(bound_points), 'threshold': self.threshold,
            'warnings_count': len(warnings), 'points_over_threshold': over
        }
        print("📊 一致性得分:", "%.2f%%" % (score*100))
        return score

    # ---------------- Visualization ----------------
    def visualize_point_matching(self, save_path: str = None) -> str:
        print("\n🎨 生成可视化图表…")
        all_json_points = self._get_all_json_points()
        xodr_data = self._sample_xodr_curves_and_lanes()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self._plot_overall_distribution(ax1, all_json_points, xodr_data)
        self._plot_deviation_analysis(ax2, all_json_points, xodr_data)
        plt.tight_layout()
        if save_path is None:
            save_path = self.json_file.parent / (self.json_file.stem + "_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._viz_path = str(save_path)
        print("✅ Visualization chart saved:", save_path)
        try:
            plt.show()
        except Exception:
            pass
        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points, xodr_data):
        # 参考线
        ref_polys = xodr_data.get('reference_polylines', [])
        for poly in ref_polys:
            if len(poly) >= 2:
                xs, ys = zip(*poly)
                ax.plot(xs, ys, color='0.4', linewidth=1.5, alpha=0.7)
        # 车道边界（浅灰虚线）
        edge_polys = xodr_data.get('lane_edge_polylines', [])
        for ed in edge_polys:
            pts = ed['points']
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color='0.75', linewidth=0.8, alpha=0.6, linestyle='--')
        # 车道中心线
        lane_polys = xodr_data.get('lane_center_polylines', xodr_data.get('lane_polylines', []))
        for lane in lane_polys:
            pts = lane['points']
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, linewidth=1.5, alpha=0.95)
        # JSON 边界点
        bpts = [p for p in json_points if p['source'] == 'bound']
        if bpts:
            bx = [p['x'] for p in bpts]; by = [p['y'] for p in bpts]
            ax.scatter(bx, by, s=30, alpha=0.85)
        ax.set_title('JSON vs XODR Distribution')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_data: Dict):
        xodr_points = []
        xodr_points.extend(xodr_data.get('reference_lines', []))
        xodr_points.extend(xodr_data.get('lane_boundaries', []))
        bound_points = [p for p in json_points if p['source'] == 'bound']
        devs = []
        for p in bound_points:
            devs.append(self._find_nearest_distance_to_xodr(p, xodr_points))
        if xodr_points:
            xp, yp = zip(*xodr_points)
            ax.scatter(xp, yp, c='lightgray', s=5, alpha=0.3)
        if bound_points:
            bx = [p['x'] for p in bound_points]
            by = [p['y'] for p in bound_points]
            sc = ax.scatter(bx, by, c=devs, s=50, cmap='RdYlGn_r', alpha=0.9, edgecolors='black', linewidth=0.4)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Deviation (m)')
        ax.set_title('Deviation Analysis')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)

    # ---------------- Matching details ----------------
    def analyze_matching_details(self, include_objects: bool = False) -> Dict:
        all_json_points = self._get_all_json_points()
        use_points = all_json_points if include_objects else [p for p in all_json_points if p.get('source') == 'bound']
        xodr_data = self._sample_xodr_curves_and_lanes()
        xodr_sample_points = []
        xodr_sample_points.extend(xodr_data.get('reference_lines', []))
        xodr_sample_points.extend(xodr_data.get('lane_boundaries', []))
        results = {
            'total_json_points': len(use_points),
            'total_xodr_points': len(xodr_sample_points),
            'detailed_matches': []
        }
        for i, jp in enumerate(use_points):
            md = float('inf'); nearest = None
            jx, jy = jp['x'], jp['y']
            for xp, yp in xodr_sample_points:
                d = math.hypot(jx - xp, jy - yp)
                if d < md:
                    md = d; nearest = (xp, yp)
            status = '✅ Pass' if md <= self.threshold else ('⚠️ Warning' if md <= 2*self.threshold else '❌ Fail')
            results['detailed_matches'].append({'index': i, 'json_point': jp, 'nearest_xodr_point': nearest, 'deviation': md, 'status': status})
        return results

    # ---------------- HTML（v1.0 风格，修正 Py3.11 嵌套 f-string） ----------------
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
        # 取字段
        json_file = _html.escape(data['json_file'])
        xodr_file = _html.escape(data['xodr_file'])
        threshold = data['threshold']
        report: QualityReport = data['report']
        matching = data['matching']
        viz_path = data['viz_path']
        max_rows = int(data.get('max_detail_rows', 200))

        # 完整性细节
        comp = report.details.get('completeness', {})
        lanes_info = comp.get('lanes', {})
        bounds_info = comp.get('bounds', {})
        objs_info = comp.get('objects', {})
        signs_info = comp.get('signs', {})

        # 一致性细节
        cons = report.details.get('consistency', {})
        avg_dev = cons.get('average_deviation', 0.0)
        max_dev = cons.get('max_deviation', 0.0)
        min_dev = cons.get('min_deviation', 0.0)
        pt_cnt = cons.get('point_count', 0)
        warn_cnt = cons.get('warnings_count', 0)
        over_cnt = cons.get('points_over_threshold', 0)

        # 匹配明细表格（默认展示前 max_rows 条）——避免 f-string 嵌套
        detail_rows = []
        for row in matching.get('detailed_matches', [])[:max_rows]:
            jp = row['json_point']
            nearest = row.get('nearest_xodr_point')
            if nearest is not None:
                nx, ny = nearest
                nearest_str = f"({nx:.3f}, {ny:.3f})"
            else:
                nearest_str = '—'
            detail_rows.append(
                "<tr>"
                + f"<td class='num'>{row['index']}</td>"
                + f"<td>{_html.escape(str(jp.get('source','')))}</td>"
                + f"<td>({jp['x']:.3f}, {jp['y']:.3f})</td>"
                + f"<td>{nearest_str}</td>"
                + f"<td class='num'>{row['deviation']:.3f}</td>"
                + f"<td>{_html.escape(str(row['status']))}</td>"
                + "</tr>"
            )
        if not detail_rows:
            detail_rows.append("<tr><td colspan='6' style='text-align:center;'>暂无数据</td></tr>")
        rows_html = "".join(detail_rows)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Warnings（截断展示）
        if report.warnings:
            warn_items = report.warnings[:200]
            warn_list_html = "<ul class='warning-list'>" + "".join(
                f"<li>{_html.escape(w)}</li>" for w in warn_items
            ) + "</ul>"
            if len(report.warnings) > 200:
                warn_list_html += f"<p class='muted'>（仅显示前 200 条，剩余 {len(report.warnings) - 200} 条已省略）</p>"
        else:
            warn_list_html = "<p class='muted'>无</p>"

        # KPI 颜色 class
        if warn_cnt == 0:
            kpi_class = 'ok'
        elif warn_cnt < 10:
            kpi_class = 'warn'
        else:
            kpi_class = 'bad'

        # 可视化图片
        viz_img_html = ""
        if viz_path:
            viz_img_html = f"<img src=\"{_html.escape(viz_path)}\" alt=\"Visualization\" style=\"max-width:100%;border:1px solid #eee;border-radius:8px;\">"

        # 拼 HTML（单层 f-string，CSS 花括号翻倍）
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
          <div class="legend">依据：车道/边界/物体/标识的数量对比</div>
        </div>
        <div class="card">
          <div class="muted">一致性得分</div>
          <div class="kpi">{report.consistency_score:.1%}</div>
          <div class="legend">依据：仅以 JSON 边界点到 XODR（参考线+车道边界）最近距离的平均偏移与阈值（{threshold:.3f} m）比较</div>
        </div>
        <div class="card">
          <div class="muted">告警数量</div>
          <div class="kpi {kpi_class}">{warn_cnt}</div>
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
            <tr><td>边界（唯一ID vs 拓扑边界）</td><td class="num">{bounds_info.get('json_count', '—')}</td><td class="num">{bounds_info.get('xodr_count', '—')}</td><td class="num">{bounds_info.get('score', 0):.2f}</td></tr>
            <tr><td>标识（sign → signal）</td><td class="num">{signs_info.get('json_count', '—')}</td><td class="num">{signs_info.get('xodr_count', '—')}</td><td class="num">{signs_info.get('score', 0):.2f}</td></tr>
            <tr><td>物体（objects）</td><td class="num">{objs_info.get('json_count', '—')}</td><td class="num">{objs_info.get('xodr_count', '—')}</td><td class="num">{objs_info.get('score', 0):.2f}</td></tr>
          </tbody>
        </table>
        <div class="legend" style="margin-top:8px;">
          边界备注：唯一边界ID数 {len(bounds_info.get('unique_ids', []))}；共有边界 {len(bounds_info.get('shared_ids', []))} 个（ID: {bounds_info.get('shared_ids', [])}）
        </div>
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
          {viz_img_html}
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
            {rows_html}
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

    # ---------------- Report orchestrator ----------------
    def generate_report(self, max_detail_rows: int = 200) -> str:
        print("\n📝 生成质检报告…")
        if 'completeness' not in self.report.details:
            self.check_completeness()
        if 'consistency' not in self.report.details:
            self.check_curve_consistency()
        matching = self.analyze_matching_details(include_objects=False)
        viz_path = self._viz_path
        if not viz_path or not Path(viz_path).exists():
            viz_path = self.visualize_point_matching()

        data = {
            'json_file': str(self.json_file),
            'xodr_file': str(self.xodr_file),
            'threshold': self.threshold,
            'report': self.report,
            'matching': matching,
            'viz_path': str(viz_path),
            'max_detail_rows': max_detail_rows,
        }
        html_report = self._generate_html_report(data)
        report_file = self.json_file.parent / f"{self.json_file.stem}_quality_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print("✅ 质检报告已生成:", report_file)
        return str(report_file)


if __name__ == '__main__':
    # 示例：如需命令行参数，可在外部脚本包装；此处直接用同目录样例名，便于开箱即用
    checker = QualityChecker(
        json_file="sample_objects.json",
        xodr_file="sample_objects.xodr",
        threshold=0.1
    )
    checker.check_completeness()
    checker.check_curve_consistency()
    checker.visualize_point_matching()
    checker.analyze_matching_details()
    checker.generate_report()
    print("\nDone.")
