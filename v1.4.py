#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker — v1.4.2

变更摘要（相对 v1.3）：
- 功能：采用“车道中心线分组匹配 + 跨 road 拼接边界 + 2×2 指派”的一致性计算，避免整段平移或跨段截断导致的误判（v1.4.1 方案并入）。
- HTML：严格增量，不删除旧版模块：
  · 摘要：保留 v1.3 的 4 张卡，并新增“对象一致性 / 标识一致性”两张独立摘要卡；
  · 恢复 v1.2 的“完整性检查表格”（要素/JSON 数量/XODR 数量/子分数）；
  · 恢复“outline_tol 判定规则”说明；
  · 轮廓叠加图仍独立展示，不混入摘要图。
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import math
import matplotlib.pyplot as plt
import html as _html
from datetime import datetime
from bisect import bisect_left

# ===== 采样与阈值参数 =====
DEFAULT_SAMPLE_STEP = 0.05
CHAMFER_SAMPLE_CAP = 2000
BOUND_HEATMAP_CMAP = 'RdYlGn_r'

JSON_CENTER_RESAMPLE_STEP = 0.2   # JSON 车道中心线重采样步长（米）
BOUND_RESAMPLE_STEP       = 0.2   # JSON 边界折线重采样步长（米）

LANE_CENTER_MATCH_GATE    = 6.0   # 车道中心线匹配代价门限（Chamfer 均值，米）
LANE_MATCH_OVERLAP_MARGIN = 2.0   # 中心线重叠裁剪 bbox 裕度（米）


@dataclass
class QualityReport:
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    warnings: List[str] = None
    details: Dict[str, Any] = None
    def __post_init__(self):
        if self.warnings is None: self.warnings = []
        if self.details is None: self.details = {}


class QualityChecker:
    def __init__(self, json_file: str, xodr_file: str,
                 threshold: float = 0.1,    # 曲线一致性判定阈值（边界点 m）
                 outline_tol: float = 0.20   # 轮廓一致性判定阈值（Chamfer 均值 m）
                 ):
        self.json_file = Path(json_file)
        self.xodr_file = Path(xodr_file)
        self.threshold = float(threshold)
        self.outline_tol = float(outline_tol)

        self.json_data = self._load_json()
        self.xodr_root = self._load_xodr()

        self.report = QualityReport()
        self._viz_path: Optional[str] = None
        self._viz_outline_path: Optional[str] = None

        # 参考线缓存
        self._road_cache: Dict[str, Dict[str, List[float]]] = {}
        self._build_road_ref_cache()

        # v1.4 缓存：匹配结果
        self._v14_lane_matches: Dict[int, Dict[str, Any]] = {}        # json_lane_id -> {'xodr': {...}, 'cost': float}（主候选）
        self._v14_lane_allcands: Dict[int, List[Dict[str, Any]]] = {} # json_lane_id -> 多候选列表（v1.4.1）
        self._v14_bound2xodr: Dict[int, List[Tuple[float,float]]] = {}  # json_bound_id -> 指派后的 xodr 折线点列

    # ---------- IO ----------
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

    # ---------- Completeness ----------
    def _count_xodr_lanes(self) -> int:
        cnt = 0
        for lane in self.xodr_root.iter('lane'):
            t_attr = (lane.get('type') or '').lower()
            if t_attr == 'driving':
                cnt += 1; continue
            t_node = lane.find('type')
            if t_node is not None and (t_node.text or '').strip().lower() == 'driving':
                cnt += 1
        return cnt

    def _detect_shared_bounds(self) -> Tuple[List[int], List[int]]:
        lanes = self.json_data.get('lanes', [])
        usage: Dict[int, int] = {}
        for ln in lanes:
            for key in ('left_bound_id', 'right_bound_id'):
                bid = ln.get(key)
                if bid is None: continue
                usage[bid] = usage.get(bid, 0) + 1
        if usage:
            unique_ids = sorted(usage.keys())
            shared_ids = sorted([bid for bid, c in usage.items() if c >= 2])
            return unique_ids, shared_ids
        b_ids = sorted({int(b.get('id')) for b in self.json_data.get('bounds', []) if 'id' in b})
        return b_ids, []

    def _count_xodr_topo_boundaries(self) -> int:
        total = 0
        for road in self.xodr_root.findall('road'):
            lanes = road.find('lanes')
            if lanes is None: continue
            for ls in lanes.findall('laneSection'):
                left = ls.find('left'); right = ls.find('right')
                l = len(left.findall('lane')) if left is not None else 0
                r = len(right.findall('lane')) if right is not None else 0
                if l>0: total += l
                if r>0: total += r
                if l>0 and r>0: total += 1  # 中央分隔
        return total

    def _count_xodr_objects(self) -> int:
        return sum(1 for _ in self.xodr_root.iter('object'))

    def _count_xodr_signals(self) -> int:
        return sum(1 for _ in self.xodr_root.iter('signal'))

    def check_completeness(self) -> float:
        print("\n🔍 开始完整性检查…")
        details = {}
        total = 0.0; n = 0

        json_lanes = [ln for ln in self.json_data.get('lanes', []) if (ln.get('type') or '').lower() == 'driving']
        xodr_lane_cnt = self._count_xodr_lanes()
        lane_score = min(xodr_lane_cnt / max(1,len(json_lanes)), 1.0) if json_lanes else 1.0
        details['lanes'] = {'json_count': len(json_lanes), 'xodr_count': xodr_lane_cnt, 'score': lane_score}
        total += lane_score; n += 1

        unique_ids, shared_ids = self._detect_shared_bounds()
        x_topo = self._count_xodr_topo_boundaries()
        bound_score = min(x_topo / max(1,len(unique_ids)), 1.0) if unique_ids else 1.0
        details['bounds'] = {'json_count': len(unique_ids), 'xodr_count': x_topo, 'score': bound_score,
                             'unique_ids': unique_ids, 'shared_ids': shared_ids}
        total += bound_score; n += 1

        json_objs = len(self.json_data.get('objects', []))
        x_objs = self._count_xodr_objects()
        obj_score = min(x_objs / max(1,json_objs), 1.0) if json_objs else 1.0
        details['objects'] = {'json_count': json_objs, 'xodr_count': x_objs, 'score': obj_score}
        total += obj_score; n += 1

        json_signs = len(self.json_data.get('sign', []))
        x_sigs = self._count_xodr_signals()
        sign_score = min(x_sigs / max(1,json_signs), 1.0) if json_signs else 1.0
        details['signs'] = {'json_count': json_signs, 'xodr_count': x_sigs, 'score': sign_score}
        total += sign_score; n += 1

        self.report.completeness_score = total / n if n else 0.0
        self.report.details['completeness'] = details
        print("📊 完整性得分:", f"{self.report.completeness_score:.2%}")
        return self.report.completeness_score

    # ---------- 参考线采样 & st→xy ----------
    @staticmethod
    def _finite_diff_heading(xs, ys):
        th=[]; n=len(xs)
        for i in range(n):
            if i==0: dx,dy=xs[1]-xs[0], ys[1]-ys[0]
            elif i==n-1: dx,dy=xs[-1]-xs[-2], ys[-1]-ys[-2]
            else: dx,dy=xs[i+1]-xs[i-1], ys[i+1]-ys[i-1]
            th.append(math.atan2(dy,dx) if (dx*dx+dy*dy)>0 else 0.0)
        return th

    @staticmethod
    def _local_to_global(x0, y0, hdg, u, v):
        ch, sh = math.cos(hdg), math.sin(hdg)
        return x0 + ch*u - sh*v, y0 + sh*u + ch*v

    def _sample_planview_polyline(self, plan_view, step=DEFAULT_SAMPLE_STEP):
        ref_pts=[]; s_abs=0.0; geoms=list(plan_view.findall('geometry'))
        for g in geoms:
            x0=float(g.get('x',0)); y0=float(g.get('y',0))
            hdg=float(g.get('hdg',0)); L=float(g.get('length',0))
            if L<=0: continue
            child=list(g)[0]; tag=child.tag
            n=max(2, int(L/step)+1); s_vals=np.linspace(0.0, L, n)
            if tag=='line':
                for s in s_vals: ref_pts.append((s_abs+s, *self._local_to_global(x0,y0,hdg,s,0.0)))
            elif tag=='arc':
                k=float(child.get('curvature',0)); R=1.0/k if k!=0 else 1e12
                for s in s_vals:
                    ang=k*s; u=R*math.sin(ang); v=R*(1.0-math.cos(ang))
                    ref_pts.append((s_abs+s, *self._local_to_global(x0,y0,hdg,u,v)))
            elif tag=='spiral':
                c0=float(child.get('curvStart',0)); c1=float(child.get('curvEnd',0))
                for s in s_vals:
                    c=c0+(c1-c0)*(s/L); theta=0.5*c*s
                    u=s*math.cos(theta); v=s*math.sin(theta)
                    ref_pts.append((s_abs+s, *self._local_to_global(x0,y0,hdg,u,v)))
            elif tag=='paramPoly3':
                aU=float(child.get('aU',0)); bU=float(child.get('bU',1))
                cU=float(child.get('cU',0)); dU=float(child.get('dU',0))
                aV=float(child.get('aV',0)); bV=float(child.get('bV',0))
                cV=float(child.get('cV',0)); dV=float(child.get('dV',0))
                p_range=child.get('pRange','normalized')
                t_vals = s_vals if p_range=='arcLength' else np.linspace(0.0,1.0,len(s_vals))
                for i,t in enumerate(t_vals):
                    u=aU+bU*t+cU*t*t+dU*t*t*t; v=aV+bV*t+cV*t*t+dV*t*t*t
                    ref_pts.append((s_abs+s_vals[i], *self._local_to_global(x0,y0,hdg,u,v)))
            s_abs += L
        return ref_pts

    def _build_road_ref_cache(self):
        self._road_cache.clear()
        for road in self.xodr_root.findall('road'):
            rid=road.get('id','unknown'); pv=road.find('planView')
            if pv is None: continue
            ref=self._sample_planview_polyline(pv, step=DEFAULT_SAMPLE_STEP)
            if len(ref)<2: continue
            s=[p[0] for p in ref]; x=[p[1] for p in ref]; y=[p[2] for p in ref]
            hdg=self._finite_diff_heading(x,y)
            self._road_cache[rid]={'s':s,'x':x,'y':y,'hdg':hdg}

    def _interp_ref_at(self, road_id: str, s: float) -> Optional[Tuple[float,float,float]]:
        d=self._road_cache.get(road_id);
        if not d: return None
        s_arr=d['s']; x_arr=d['x']; y_arr=d['y']; h_arr=d['hdg']
        if not s_arr: return None
        if s<=s_arr[0]: i=0
        elif s>=s_arr[-1]: i=len(s_arr)-2
        else: i=max(0,min(len(s_arr)-2, bisect_left(s_arr,s)-1))
        s0,s1=s_arr[i],s_arr[i+1]; t=0.0 if s1==s0 else (s-s0)/(s1-s0)
        x=x_arr[i]+(x_arr[i+1]-x_arr[i])*t; y=y_arr[i]+(y_arr[i+1]-y_arr[i])*t
        c0,s0h=math.cos(h_arr[i]),math.sin(h_arr[i]); c1,s1h=math.cos(h_arr[i+1]),math.sin(h_arr[i+1])
        c=c0+(c1-c0)*t; s_=s0h+(s1h-s0h)*t; hdg=math.atan2(s_,c)
        return x,y,hdg

    def _st_to_world(self, road_id: str, s: float, t_off: float) -> Optional[Tuple[float,float]]:
        ref=self._interp_ref_at(road_id,s)
        if ref is None: return None
        x,y,hdg=ref; nx,ny=-math.sin(hdg), math.cos(hdg)
        return x + t_off*nx, y + t_off*ny

    # ---------- Polyline 工具 ----------
    @staticmethod
    def _polyline_length(pts: List[Tuple[float,float]]) -> float:
        if len(pts)<2: return 0.0
        return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))

    @staticmethod
    def _resample_polyline(pts: List[Tuple[float,float]], step: float) -> List[Tuple[float,float]]:
        if not pts: return []
        if len(pts)==1: return pts[:]
        segs=[0.0]
        for i in range(1,len(pts)):
            segs.append(segs[-1]+math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
        L=segs[-1]
        if L==0.0: return [pts[0]]*max(2,int(1/step)+1)
        n=max(2, int(L/step)+1)
        targets=np.linspace(0.0, L, n)
        res=[]
        j=0
        for t in targets:
            while j+1<len(segs) and segs[j+1]<t: j+=1
            if j+1>=len(segs):
                res.append(pts[-1]); continue
            t0=segs[j]; t1=segs[j+1]; ratio=0.0 if t1==t0 else (t-t0)/(t1-t0)
            x=pts[j][0]+(pts[j+1][0]-pts[j][0])*ratio
            y=pts[j][1]+(pts[j+1][1]-pts[j][1])*ratio
            res.append((x,y))
        return res

    @staticmethod
    def _chamfer_mean(A: List[Tuple[float,float]], B: List[Tuple[float,float]]) -> float:
        if not A or not B: return float('inf')
        def nearest(a, arr):
            ax,ay=a; md=float('inf')
            for bx,by in arr:
                d=math.hypot(ax-bx, ay-by)
                if d<md: md=d
            return md
        def sample(arr):
            if len(arr)<=CHAMFER_SAMPLE_CAP: return arr
            idxs=np.linspace(0,len(arr)-1,CHAMFER_SAMPLE_CAP).astype(int)
            return [arr[i] for i in idxs]
        A_=sample(A); B_=sample(B)
        d1=[nearest(a,B_) for a in A_]; d2=[nearest(b,A_) for b in B_]
        return float(np.mean(d1+d2))

    @staticmethod
    def _maybe_reverse_to_match(P: List[Tuple[float,float]], Q: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
        if not P or not Q: return Q
        d_forward = math.hypot(P[0][0]-Q[0][0], P[0][1]-Q[0][1]) + math.hypot(P[-1][0]-Q[-1][0], P[-1][1]-Q[-1][1])
        d_reverse = math.hypot(P[0][0]-Q[-1][0],P[0][1]-Q[-1][1]) + math.hypot(P[-1][0]-Q[0][0], P[-1][1]-Q[0][1])
        return Q if d_forward <= d_reverse else list(reversed(Q))

    # ---------- JSON 结构：bounds / lane centers ----------
    def _json_bound_polylines(self) -> Dict[int, List[Tuple[float,float]]]:
        out={}
        for b in self.json_data.get('bounds', []):
            bid = int(b.get('id'))
            pts=[(float(p['x']), float(p['y'])) for p in b.get('pts', [])]
            if len(pts)>=2:
                out[bid] = self._resample_polyline(pts, BOUND_RESAMPLE_STEP)
            elif pts:
                out[bid] = pts
        return out

    def _json_lane_centers(self, bound_polys: Dict[int,List[Tuple[float,float]]]) -> List[Dict]:
        lanes = [ln for ln in self.json_data.get('lanes', []) if (ln.get('type') or '').lower()=='driving']
        out=[]
        for ln in lanes:
            lb=ln.get('left_bound_id'); rb=ln.get('right_bound_id')
            if lb is None or rb is None: continue
            P = bound_polys.get(int(lb)); Q = bound_polys.get(int(rb))
            if not P or not Q: continue
            P_ = self._resample_polyline(P, JSON_CENTER_RESAMPLE_STEP)
            Q_ = self._resample_polyline(Q, JSON_CENTER_RESAMPLE_STEP)
            Q_ = self._maybe_reverse_to_match(P_, Q_)
            n=min(len(P_), len(Q_))
            center=[((P_[i][0]+Q_[i][0])*0.5, (P_[i][1]+Q_[i][1])*0.5) for i in range(n)]
            out.append({'json_lane_id': int(ln.get('id')), 'left_bound_id': int(lb), 'right_bound_id': int(rb), 'points': center})
        return out

    # ---------- 采样 XODR 曲线 ----------
    def _sample_xodr_curves_and_lanes(self, step: float = DEFAULT_SAMPLE_STEP):
        root=self.xodr_root
        out={"reference_lines": [], "lane_boundaries": [], "reference_polylines": [],
             "lane_polylines": [], "lane_center_polylines": [], "lane_edge_polylines": []}
        for road in root.findall('road'):
            road_id=road.get('id','unknown')
            pv=road.find('planView')
            if pv is None: continue
            ref=self._sample_planview_polyline(pv, step=step)
            if len(ref)<2: continue
            s_arr=[p[0] for p in ref]; x_arr=[p[1] for p in ref]; y_arr=[p[2] for p in ref]
            th_arr=self._finite_diff_heading(x_arr,y_arr)
            ref_poly=[(x_arr[i],y_arr[i]) for i in range(len(ref))]
            out['reference_polylines'].append(ref_poly); out['reference_lines'].extend(ref_poly)
            lanes=road.find('lanes')
            if lanes is None: continue

            def parse_lane_sections(le):
                sections=[]
                for sec in le.findall('laneSection'):
                    s0=float(sec.get('s',0.0)); left=sec.find('left'); right=sec.find('right')
                    one={'s':s0,'left':[],'right':[]}
                    def collect(side_elem,name):
                        if side_elem is None: return
                        for ln in side_elem.findall('lane'):
                            lid=ln.get('id',''); widths=[]
                            for w in ln.findall('width'):
                                widths.append({'sOffset':float(w.get('sOffset',0.0)),
                                               'a':float(w.get('a',0.0)),'b':float(w.get('b',0.0)),
                                               'c':float(w.get('c',0.0)),'d':float(w.get('d',0.0))})
                            widths.sort(key=lambda x:x['sOffset'])
                            one[name].append({'id':lid,'widths':widths})
                        if name=='left': one[name].sort(key=lambda e:int(e['id']))
                        else: one[name].sort(key=lambda e:abs(int(e['id'])))
                    collect(left,'left'); collect(right,'right'); sections.append(one)
                sections.sort(key=lambda s:s['s']); return sections

            def parse_lane_offsets(le):
                offs=[]
                for lo in le.findall('laneOffset'):
                    offs.append({'s':float(lo.get('s',0.0)),'a':float(lo.get('a',0.0)),'b':float(lo.get('b',0.0)),
                                 'c':float(lo.get('c',0.0)),'d':float(lo.get('d',0.0))})
                offs.sort(key=lambda x:x['s']); return offs

            def lane_offset_at(offs,s_abs):
                if not offs: return 0.0
                prev=offs[0]
                for o in offs:
                    if o['s']<=s_abs: prev=o
                    else: break
                ds=max(0.0, s_abs-prev['s'])
                return prev['a']+prev['b']*ds+prev['c']*ds*ds+prev['d']*ds*ds*ds

            def width_poly_at(widths,s_rel):
                if not widths: return 0.0
                prev=widths[0]
                for w in widths:
                    if w['sOffset']<=s_rel: prev=w
                    else: break
                ds=max(0.0, s_rel-prev['sOffset'])
                return prev['a']+prev['b']*ds+prev['c']*ds*ds+prev['d']*ds*ds*ds

            sections=parse_lane_sections(lanes); offsets=parse_lane_offsets(lanes)
            road_len=s_arr[-1]
            for si,sec in enumerate(sections):
                s0=sec['s']; s1=sections[si+1]['s'] if si+1<len(sections) else road_len+1e-6
                idxs=[i for i,sv in enumerate(s_arr) if (sv>=s0 and sv<=s1)]
                if len(idxs)<2: continue
                centers={}; edges={}
                def ensure(d,k):
                    if k not in d: d[k]=[]
                    return d[k]
                for i in idxs:
                    s_here=s_arr[i]; s_rel=s_here-s0
                    nx=-math.sin(th_arr[i]); ny=math.cos(th_arr[i]); base=lane_offset_at(offsets,s_here)
                    cum=0.0
                    for ln in sec['left']:
                        w=width_poly_at(ln['widths'], s_rel)
                        inner=base+cum; outer=inner+w; center=inner+0.5*w
                        ensure(edges,f"left:{ln['id']}:inner").append((x_arr[i]+inner*nx, y_arr[i]+inner*ny))
                        ensure(edges,f"left:{ln['id']}:outer").append((x_arr[i]+outer*nx, y_arr[i]+outer*ny))
                        ensure(centers,f"left:{ln['id']}").append((x_arr[i]+center*nx, y_arr[i]+center*ny))
                        cum+=w
                    cum=0.0
                    for ln in sec['right']:
                        w=width_poly_at(ln['widths'], s_rel)
                        inner=base-cum; outer=inner-w; center=inner-0.5*w
                        ensure(edges,f"right:{ln['id']}:inner").append((x_arr[i]+inner*nx, y_arr[i]+inner*ny))
                        ensure(edges,f"right:{ln['id']}:outer").append((x_arr[i]+outer*nx, y_arr[i]+outer*ny))
                        ensure(centers,f"right:{ln['id']}").append((x_arr[i]+center*nx, y_arr[i]+center*ny))
                        cum+=w
                for key,pts in centers.items():
                    side,lane_id=key.split(':')
                    out['lane_center_polylines'].append({'road_id':road_id,'lane_id':lane_id,'side':side,'points':pts})
                    out['lane_polylines'].append({'road_id':road_id,'lane_id':lane_id,'side':side,'points':pts})
                for key,pts in edges.items():
                    side,lane_id,kind=key.split(':')
                    out['lane_edge_polylines'].append({'road_id':road_id,'lane_id':lane_id,'side':side,'kind':kind,'points':pts})
                    out['lane_boundaries'].extend(pts)
        return out

    # ---------- v1.4.1：重叠裁剪 + 多候选 + 跨路段拼接 ----------
    def _chamfer_mean_overlap(self, A: List[Tuple[float,float]], B: List[Tuple[float,float]],
                              margin: float = LANE_MATCH_OVERLAP_MARGIN) -> float:
        if not A or not B:
            return float('inf')
        minx = min(x for x,_ in B) - margin; maxx = max(x for x,_ in B) + margin
        miny = min(y for _,y in B) - margin; maxy = max(y for _,y in B) + margin
        A_clip = [p for p in A if (minx <= p[0] <= maxx and miny <= p[1] <= maxy)]
        if not A_clip:
            return self._chamfer_mean(A, B)  # 无重叠则退回普通 Chamfer
        return self._chamfer_mean(A_clip, B)

    def _match_json_lanes_to_xodr_multi(self,
                                        j_centers: List[Dict],
                                        x_centers: List[Dict]) -> Dict[int, List[Dict]]:
        """
        多候选中心线匹配：{ json_lane_id: [ { 'xodr': x_info, 'cost': float }, ... ] }
        """
        lane2cands: Dict[int, List[Dict]] = {}
        for j in j_centers:
            jpts = j['points']
            cands=[]
            for x in x_centers:
                cost = self._chamfer_mean_overlap(jpts, x['points'], margin=LANE_MATCH_OVERLAP_MARGIN)
                if cost <= LANE_CENTER_MATCH_GATE:
                    cands.append({'xodr': x, 'cost': float(cost)})
            cands.sort(key=lambda c: c['cost'])
            lane2cands[j['json_lane_id']] = cands
        return lane2cands

    def _assign_bounds_2x2(self,
                           bound_left: List[Tuple[float,float]],
                           bound_right: List[Tuple[float,float]],
                           cand_A: List[Tuple[float,float]],
                           cand_B: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
        """把 JSON 的左右 bound 分配到两条候选边界（A/B）。返回 (left->poly, right->poly)"""
        if not cand_A and not cand_B:
            return [], []
        if cand_A and not cand_B:
            return cand_A, cand_A
        if cand_B and not cand_A:
            return cand_B, cand_B
        c_LA = self._chamfer_mean(bound_left,  cand_A)
        c_LB = self._chamfer_mean(bound_left,  cand_B)
        c_RA = self._chamfer_mean(bound_right, cand_A)
        c_RB = self._chamfer_mean(bound_right, cand_B)
        cost1 = c_LA + c_RB   # left->A, right->B
        cost2 = c_LB + c_RA   # left->B, right->A
        return (cand_A, cand_B) if cost1 <= cost2 else (cand_B, cand_A)

    def _build_v14_bound_mapping(self):
        """构建 v1.4 的 bound->xodr 曲线映射（跨 road 拼接）"""
        j_bound_polylines = self._json_bound_polylines()
        j_centers = self._json_lane_centers(j_bound_polylines)
        xodr = self._sample_xodr_curves_and_lanes()
        x_centers = xodr.get('lane_center_polylines', [])
        x_edges = xodr.get('lane_edge_polylines', [])

        # 1) 多候选中心线
        lane2cands = self._match_json_lanes_to_xodr_multi(j_centers, x_centers)
        self._v14_lane_allcands = lane2cands
        self._v14_lane_matches = {lid: (cands[0] if cands else None) for lid, cands in lane2cands.items()}

        # 2) (road_id, side, lane_id) -> {'inner': pts, 'outer': pts}
        edge_map: Dict[Tuple[str, str, str], Dict[str, List[Tuple[float, float]]]] = {}
        for ed in x_edges:
            key = (ed['road_id'], ed['side'], ed['lane_id'])
            edge_map.setdefault(key, {})[ed['kind']] = ed['points']

        # 3) 对每个 JSON 车道：按 (side,lane_id) 把不同 road 的 inner/outer 拼接
        bound2x: Dict[int, List[Tuple[float, float]]] = {}
        for ln in j_centers:
            j_lane_id = ln['json_lane_id']
            lb = ln['left_bound_id'];
            rb = ln['right_bound_id']
            P_left = j_bound_polylines.get(lb, [])
            P_right = j_bound_polylines.get(rb, [])
            cands = lane2cands.get(j_lane_id, [])

            if not P_left or not P_right or not cands:
                continue

            # (side,lane_id) -> {'inner':[...], 'outer':[...]}
            group_map: Dict[Tuple[str, str], Dict[str, List[Tuple[float, float]]]] = {}
            for c in cands:
                xinfo = c['xodr']
                ek = (xinfo['road_id'], xinfo['side'], xinfo['lane_id'])
                ed = edge_map.get(ek, {})
                if not ed:
                    continue
                g = group_map.setdefault((xinfo['side'], xinfo['lane_id']), {'inner': [], 'outer': []})
                for kind in ('inner', 'outer'):
                    pts = ed.get(kind, [])
                    if pts:
                        g[kind].extend(pts)

            if not group_map:
                continue

            # 4) 对每个组做 2×2 指派，选成本最小的组
            best = None
            for (_side, _lid), pair in group_map.items():
                A = pair.get('inner', [])
                B = pair.get('outer', [])
                if not A and not B:
                    continue
                c_LA = self._chamfer_mean(P_left, A) if A else float('inf')
                c_LB = self._chamfer_mean(P_left, B) if B else float('inf')
                c_RA = self._chamfer_mean(P_right, A) if A else float('inf')
                c_RB = self._chamfer_mean(P_right, B) if B else float('inf')
                cost1 = c_LA + c_RB  # left->inner, right->outer
                cost2 = c_LB + c_RA  # left->outer, right->inner
                if best is None or min(cost1, cost2) < best['cost']:
                    best = {'cost': cost1, 'left': A, 'right': B} if cost1 <= cost2 else \
                        {'cost': cost2, 'left': B, 'right': A}

            if best:
                if best['left']:  bound2x[int(lb)] = best['left']
                if best['right']: bound2x[int(rb)] = best['right']

        self._v14_bound2xodr = bound2x

    # ---------- 旧版“点对全局最近邻”工具 ----------
    def _find_nearest_distance_to_xodr(self, jp: Dict, xodr_pts: List[Tuple[float,float]]) -> float:
        jx,jy=jp['x'],jp['y']; md=float('inf')
        for xp,yp in xodr_pts:
            d=math.hypot(jx-xp, jy-yp)
            if d<md: md=d
        return md

    # ---------- 曲线一致性（v1.4.2：按 bound 指派；失败回退全局） ----------
    def check_curve_consistency(self) -> float:
        print("\n🔍 开始曲线一致性检查（v1.4 分组 + 跨段拼接）…")
        warnings = []
        all_json_points = self._get_all_json_points()
        bound_points = [p for p in all_json_points if p.get('source') == 'bound']

        # 构建映射（bound -> 指派的 XODR 曲线）
        self._build_v14_bound_mapping()
        use_v14 = len(self._v14_bound2xodr) > 0

        # 预备全局最近邻点集（仅在需要时）
        xodr_all_points = None

        # 明细：每个点的记录（用于 HTML 渲染）
        self._eval_points_v14 = []

        if not bound_points:
            self.report.consistency_score = 0.0
            self.report.details['consistency'] = {'average_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0,
                                                  'point_count': 0, 'threshold': self.threshold,
                                                  'warnings_count': 0, 'points_over_threshold': 0,
                                                  'v14_grouped': use_v14,
                                                  'v14_matched_lanes': 0, 'v14_mapped_bounds': 0}
            self.report.details["curve_eval_points_v14"] = self._eval_points_v14
            return 0.0

        devs = []  # 用于整体统计的“实际采用”的距离（assign 若可，否则 nn）
        for i, jp in enumerate(bound_points):
            bid = int(jp.get('bound_id'))
            jx, jy = jp['x'], jp['y']

            # 1) 指派曲线距离（若存在）
            assigned_poly = self._v14_bound2xodr.get(bid)
            d_assign = None
            if assigned_poly:
                d_assign = min(math.hypot(jx - xp, jy - yp) for (xp, yp) in assigned_poly)

            # 2) 全局最近邻（求距离 + 最近邻点坐标，仅用于展示/回退）
            if xodr_all_points is None:
                xodr = self._sample_xodr_curves_and_lanes()
                xodr_all_points = xodr.get('reference_lines', []) + xodr.get('lane_boundaries', [])
            d_nn = float('inf');
            nn_pt = None
            for (xp, yp) in xodr_all_points:
                d = math.hypot(jx - xp, jy - yp)
                if d < d_nn:
                    d_nn = d;
                    nn_pt = (xp, yp)

            # 3) 实际用于统计/告警的距离（有指派用指派，否则回退最近邻）
            if d_assign is not None:
                d_use = d_assign
            else:
                d_use = d_nn

            devs.append(d_use)
            if d_use > self.threshold:
                warnings.append("⚠️ 边界点偏移超阈值: (%.3f, %.3f) → %.3fm" % (jx, jy, d_use))

            status = ("通过" if d_use <= self.threshold
                      else "警告" if d_use <= 2 * self.threshold
            else "失败")

            # 记录明细（type 固定为 bound；偏移列显示 d_assign，没有则显示 —）
            self._eval_points_v14.append({
                "type": "bound",
                "bound_id": bid,
                "x": jx, "y": jy,
                "nn_point": nn_pt,  # 最近 XODR 坐标（示意）
                "d_assign": (None if d_assign is None else float(d_assign)),
                "d_use": float(d_use),  # 用于排序/统计
                "status": status,  # 通过/警告/失败（与告警一致）
            })

            if (i + 1) % 200 == 0 or i == len(bound_points) - 1:
                print(f"  已处理 {i + 1}/{len(bound_points)}")

        # 4) 统计
        avg_dev = float(np.mean(devs)) if devs else 0.0
        max_dev = float(np.max(devs)) if devs else 0.0
        min_dev = float(np.min(devs)) if devs else 0.0
        over = sum(1 for d in devs if d > self.threshold)
        score = max(0.0, 1.0 - (avg_dev / self.threshold)) if self.threshold > 0 else 0.0

        self.report.consistency_score = score
        self.report.warnings.extend(warnings)
        self.report.details['consistency'] = {'average_deviation': avg_dev, 'max_deviation': max_dev,
                                              'min_deviation': min_dev,
                                              'point_count': len(bound_points), 'threshold': self.threshold,
                                              'warnings_count': len(warnings), 'points_over_threshold': over,
                                              'v14_grouped': use_v14,
                                              'v14_matched_lanes': sum(
                                                  1 for v in getattr(self, "_v14_lane_allcands", {}).values() if v),
                                              'v14_mapped_bounds': len(self._v14_bound2xodr)}
        # 5) 存入报告详情，供 HTML 明细使用
        self.report.details["curve_eval_points_v14"] = self._eval_points_v14

        print("📊 一致性得分:", "%.2f%%" % (score * 100))
        return score

    # ---------- objects/signals 一致性（Chamfer） ----------
    @staticmethod
    def _centroid_xy(pts: List[Tuple[float,float]]) -> Tuple[float,float]:
        if not pts: return 0.0,0.0
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    @staticmethod
    def _chamfer_metrics(A: List[Tuple[float,float]], B: List[Tuple[float,float]]) -> Dict[str,float]:
        if not A or not B: return {'mean':None,'max':None,'nA':len(A),'nB':len(B)}
        def nearest(a, arr):
            ax,ay=a; md=float('inf')
            for bx,by in arr:
                d=math.hypot(ax-bx, ay-by)
                if d<md: md=d
            return md
        def sample(arr):
            if len(arr)<=CHAMFER_SAMPLE_CAP: return arr
            idxs=np.linspace(0,len(arr)-1,CHAMFER_SAMPLE_CAP).astype(int)
            return [arr[i] for i in idxs]
        A_=sample(A); B_=sample(B)
        d1=[nearest(a,B_) for a in A_]; d2=[nearest(b,A_) for b in B_]
        all_d=d1+d2
        return {'mean':float(np.mean(all_d)), 'max':float(np.max(all_d)), 'nA':len(A), 'nB':len(B)}

    def _json_objects(self) -> List[Dict]:
        out=[]
        for obj in self.json_data.get('objects', []):
            oid=obj.get('id'); pts=[(p['x'],p['y']) for p in obj.get('outline',[])]
            cx,cy=self._centroid_xy(pts)
            out.append({'json_id':oid,'center':(cx,cy),'outline':pts,'raw':obj})
        return out

    def _json_signs(self) -> List[Dict]:
        out=[]
        for sg in self.json_data.get('sign', []):
            sid=sg.get('id'); pts=[(p['x'],p['y']) for p in sg.get('outline',[])]
            cx,cy=self._centroid_xy(pts)
            out.append({'json_id':sid,'center':(cx,cy),'outline':pts,'raw':sg})
        return out

    def _object_outline_world(self, road_id: str, obj: ET.Element) -> List[Tuple[float,float]]:
        outlines_parent=obj.find('outlines')
        outlines=outlines_parent.findall('outline') if outlines_parent is not None else obj.findall('outline')
        pts_all=[]
        if not outlines: return pts_all
        s_obj=float(obj.get('s',0.0)); t_obj=float(obj.get('t',0.0)); hdg_obj=float(obj.get('hdg',0.0) or 0.0)
        ref=self._interp_ref_at(road_id, s_obj); hdg_abs=0.0
        if ref is not None: _,_,hdg_ref=ref; hdg_abs=hdg_ref+hdg_obj
        center=self._st_to_world(road_id, s_obj, t_obj)
        for ol in outlines:
            cr=ol.findall('cornerRoad'); cl=ol.findall('cornerLocal'); ring=[]
            if cr:
                for c in cr:
                    s=float(c.get('s',s_obj)); t=float(c.get('t',t_obj))
                    w=self._st_to_world(road_id,s,t)
                    if w is not None: ring.append(w)
            elif cl and center is not None:
                x0,y0=center
                for c in cl:
                    u=float(c.get('u',0.0)); v=float(c.get('v',0.0))
                    ring.append(self._local_to_global(x0,y0,hdg_abs,u,v))
            pts_all.extend(ring)
        return pts_all

    def _xodr_objects_world(self) -> List[Dict]:
        out=[]
        for road in self.xodr_root.findall('road'):
            rid=road.get('id','unknown')
            objs_parent=road.find('objects'); objs=[]
            if objs_parent is not None: objs.extend(objs_parent.findall('object'))
            objs.extend([n for n in road.findall('object')])
            for obj in objs:
                oid=obj.get('id'); s=float(obj.get('s',0.0)); t=float(obj.get('t',0.0))
                world_center=self._st_to_world(rid,s,t)
                outline=self._object_outline_world(rid,obj)
                cen=self._centroid_xy(outline) if outline else (world_center if world_center else (0.0,0.0))
                out.append({'xodr_id':oid,'road_id':rid,'center':cen,'outline':outline})
        return out

    def _signal_outline_world(self, road_id: str, sig: ET.Element) -> List[Tuple[float,float]]:
        outlines_parent=sig.find('outlines')
        outlines=outlines_parent.findall('outline') if outlines_parent is not None else sig.findall('outline')
        pts_all=[]
        if not outlines: return pts_all
        s_sig=float(sig.get('s',0.0)); t_sig=float(sig.get('t',0.0)); hdg_sig=float(sig.get('hdg',0.0) or 0.0)
        ref=self._interp_ref_at(road_id, s_sig); hdg_abs=0.0
        if ref is not None: _,_,hdg_ref=ref; hdg_abs=hdg_ref+hdg_sig
        center=self._st_to_world(road_id, s_sig, t_sig)
        for ol in outlines:
            cr=ol.findall('cornerRoad'); cl=ol.findall('cornerLocal'); ring=[]
            if cr:
                for c in cr:
                    s=float(c.get('s',s_sig)); t=float(c.get('t',t_sig))
                    w=self._st_to_world(road_id,s,t)
                    if w is not None: ring.append(w)
            elif cl and center is not None:
                x0,y0=center
                for c in cl:
                    u=float(c.get('u',0.0)); v=float(c.get('v',0.0))
                    ring.append(self._local_to_global(x0,y0,hdg_abs,u,v))
            pts_all.extend(ring)
        return pts_all

    def _xodr_signals_world(self) -> List[Dict]:
        out=[]
        for road in self.xodr_root.findall('road'):
            rid=road.get('id','unknown')
            sigs_parent=road.find('signals'); sigs=[]
            if sigs_parent is not None: sigs.extend(sigs_parent.findall('signal'))
            sigs.extend([n for n in road.findall('signal')])
            for sg in sigs:
                sid=sg.get('id'); s=float(sg.get('s',0.0)); t=float(sg.get('t',0.0))
                world_center=self._st_to_world(rid,s,t)
                outline=self._signal_outline_world(rid, sg)
                cen=self._centroid_xy(outline) if outline else (world_center if world_center else (0.0,0.0))
                out.append({'xodr_id':sid,'road_id':rid,'center':cen,'outline':outline})
        return out

    def _match_by_nearest_center(self, js: List[Dict], xs: List[Dict]) -> List[Tuple[Dict, Optional[Dict], float]]:
        if not js: return []
        if not xs: return [(j,None,float('inf')) for j in js]
        used=set(); pairs=[]
        for j in js:
            jx,jy=j['center']; best=None; best_d=float('inf'); best_k=None
            for k,x in enumerate(xs):
                if k in used: continue
                xx,xy=x['center']; d=math.hypot(jx-xx, jy-xy)
                if d<best_d: best_d=d; best=x; best_k=k
            if best_k is not None: used.add(best_k)
            pairs.append((j,best,best_d))
        return pairs

    def _per_item_score(self, chamfer_mean: Optional[float]) -> float:
        if chamfer_mean is None: return 0.0
        return max(0.0, 1.0 - (chamfer_mean / (2.0 * self.outline_tol)))

    def _judge_outline_status(self, chamfer_mean: Optional[float]) -> str:
        if chamfer_mean is None: return '缺失'
        if chamfer_mean <= self.outline_tol: return '通过'
        if chamfer_mean <= 2*self.outline_tol: return '警告'
        return '失败'

    def _objects_signals_consistency(self):
        # objects
        j_objs=self._json_objects(); x_objs=self._xodr_objects_world()
        pairs_obj=self._match_by_nearest_center(j_objs, x_objs)
        items_obj=[]; chamfer_means_obj=[]; chamfer_maxs_obj=[]; per_scores_obj=[]
        cnt_pass_o=cnt_warn_o=cnt_fail_o=cnt_missing_o=0
        for j,x,_ in pairs_obj:
            A=j['outline']; B=(x['outline'] if (x and x.get('outline')) else [])
            if not A or not B:
                ch={'mean':None,'max':None,'nA':len(A),'nB':len(B)}
            else:
                ch=self._chamfer_metrics(A,B); chamfer_means_obj.append(ch['mean']); chamfer_maxs_obj.append(ch['max'])
            status=self._judge_outline_status(ch['mean']); score_i=self._per_item_score(ch['mean']); per_scores_obj.append(score_i)
            if   status=='通过': cnt_pass_o+=1
            elif status=='警告': cnt_warn_o+=1
            elif status=='失败': cnt_fail_o+=1
            else: cnt_missing_o+=1
            items_obj.append({'json_id':j['json_id'],'xodr_id':(x['xodr_id'] if x else None),
                              'json_pts':len(A),'xodr_pts':(ch['nB'] if B else 0),
                              'chamfer_mean':ch['mean'],'chamfer_max':ch['max'],
                              'outline_status':status,'item_score':score_i})
        obj_score=float(np.mean(per_scores_obj)) if per_scores_obj else None
        self.report.details['object_consistency_v14']={'json_count':len(j_objs),'xodr_count':len(x_objs),'matched':len(pairs_obj),
            'chamfer_mean_avg':(float(np.mean(chamfer_means_obj)) if chamfer_means_obj else None),
            'chamfer_max_max':(float(np.max(chamfer_maxs_obj)) if chamfer_maxs_obj else None),
            'outline_pass':cnt_pass_o,'outline_warn':cnt_warn_o,'outline_fail':cnt_fail_o,'outline_missing_count':cnt_missing_o,
            'score':obj_score,'items':items_obj}

        # signals
        j_sigs=self._json_signs(); x_sigs=self._xodr_signals_world()
        pairs_sig=self._match_by_nearest_center(j_sigs, x_sigs)
        items_sig=[]; chamfer_means_sig=[]; chamfer_maxs_sig=[]; per_scores_sig=[]
        cnt_pass_s=cnt_warn_s=cnt_fail_s=cnt_missing_s=0
        for j,x,_ in pairs_sig:
            A=j['outline']; B=(x['outline'] if (x and x.get('outline')) else [])
            if not A or not B:
                ch={'mean':None,'max':None,'nA':len(A),'nB':len(B)}
            else:
                ch=self._chamfer_metrics(A,B); chamfer_means_sig.append(ch['mean']); chamfer_maxs_sig.append(ch['max'])
            status=self._judge_outline_status(ch['mean']); score_i=self._per_item_score(ch['mean']); per_scores_sig.append(score_i)
            if   status=='通过': cnt_pass_s+=1
            elif status=='警告': cnt_warn_s+=1
            elif status=='失败': cnt_fail_s+=1
            else: cnt_missing_s+=1
            items_sig.append({'json_id':j['json_id'],'xodr_id':(x['xodr_id'] if x else None),
                              'json_pts':len(A),'xodr_pts':(ch['nB'] if B else 0),
                              'chamfer_mean':ch['mean'],'chamfer_max':ch['max'],
                              'outline_status':status,'item_score':score_i})
        sig_score=float(np.mean(per_scores_sig)) if per_scores_sig else None
        self.report.details['signal_consistency_v14']={'json_count':len(j_sigs),'xodr_count':len(x_sigs),'matched':len(pairs_sig),
            'chamfer_mean_avg':(float(np.mean(chamfer_means_sig)) if chamfer_means_sig else None),
            'chamfer_max_max':(float(np.max(chamfer_maxs_sig)) if chamfer_maxs_sig else None),
            'outline_pass':cnt_pass_s,'outline_warn':cnt_warn_s,'outline_fail':cnt_fail_s,'outline_missing_count':cnt_missing_s,
            'score':sig_score,'items':items_sig}

        # 综合（按样本数加权）
        n_obj=len(self.report.details['object_consistency_v14']['items'])
        n_sig=len(self.report.details['signal_consistency_v14']['items'])
        w_obj = n_obj if obj_score is not None else 0
        w_sig = n_sig if sig_score is not None else 0
        if (w_obj + w_sig) > 0:
            comb = ((obj_score or 0.0)*w_obj + (sig_score or 0.0)*w_sig) / (w_obj + w_sig)
        else:
            comb = None
        self.report.details['objects_signals_score']=comb

    # ---------- 可视化 ----------
    def _get_all_json_points(self) -> List[Dict]:
        pts=[]
        for b in self.json_data.get('bounds', []):
            bid=b.get('id')
            for p in b.get('pts', []):
                pts.append({'x':p['x'],'y':p['y'],'z':p['z'],'bound_id':bid,'source':'bound'})
        for obj in self.json_data.get('objects', []):
            oid=obj.get('id')
            for p in obj.get('outline', []):
                pts.append({'x':p['x'],'y':p['y'],'z':p['z'],'object_id':oid,'source':'object'})
        return pts

    def visualize_point_matching(self, save_path: str = None) -> str:
        print("\n🎨 生成可视化图表…")
        all_json_points=self._get_all_json_points()
        xodr_data=self._sample_xodr_curves_and_lanes()
        fig,(ax1,ax2)=plt.subplots(1,2, figsize=(16,8))
        self._plot_overall_distribution(ax1, all_json_points, xodr_data)
        self._plot_deviation_analysis(ax2, all_json_points, xodr_data)
        plt.tight_layout()
        if save_path is None:
            save_path=self.json_file.parent/(self.json_file.stem+"_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); self._viz_path=str(save_path)
        print("✅ Visualization chart saved:", save_path)
        try: plt.close(fig)
        except Exception: pass
        return str(save_path)

    def _plot_overall_distribution(self, ax, json_points, xodr_data):
        for poly in xodr_data.get('reference_polylines', []):
            if len(poly)>=2: xs,ys=zip(*poly); ax.plot(xs,ys, color='0.4', linewidth=1.5, alpha=0.7)
        for ed in xodr_data.get('lane_edge_polylines', []):
            pts=ed['points'];
            if len(pts)>=2: xs,ys=zip(*pts); ax.plot(xs,ys, color='0.75', linewidth=0.8, alpha=0.6, linestyle='--')
        for lane in xodr_data.get('lane_center_polylines', xodr_data.get('lane_polylines', [])):
            pts=lane['points'];
            if len(pts)>=2: xs,ys=zip(*pts); ax.plot(xs,ys, linewidth=1.5, alpha=0.95)
        bpts=[p for p in json_points if p['source']=='bound']
        if bpts: bx=[p['x'] for p in bpts]; by=[p['y'] for p in bpts]; ax.scatter(bx,by, s=30, alpha=0.85)
        ax.set_title('JSON vs XODR Distribution'); ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)

    def _plot_deviation_analysis(self, ax, json_points: List[Dict], xodr_data: Dict):
        # 为了可视化易读，这里仍用“全局最近邻”的偏移上色（不影响实际得分）
        xodr_points=[]; xodr_points.extend(xodr_data.get('reference_lines', [])); xodr_points.extend(xodr_data.get('lane_boundaries', []))
        bound_points=[p for p in json_points if p['source']=='bound']
        devs=[]
        for p in bound_points:
            devs.append(self._find_nearest_distance_to_xodr(p, xodr_points))
        if xodr_points:
            xp,yp=zip(*xodr_points); ax.scatter(xp,yp, c='lightgray', s=5, alpha=0.3)
        if bound_points:
            bx=[p['x'] for p in bound_points]; by=[p['y'] for p in bound_points]
            sc=ax.scatter(bx,by, c=devs, s=50, cmap=BOUND_HEATMAP_CMAP, alpha=0.9, edgecolors='black', linewidth=0.4)
            cbar=plt.colorbar(sc, ax=ax); cbar.set_label('Deviation (m)')
        ax.set_title('Deviation Analysis (viz)'); ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)

    def visualize_outline_overlay(self, save_path: str = None) -> str:
        print("\n🎨 生成对象/标识 Outline 叠加图…")
        j_objs=self._json_objects(); x_objs=self._xodr_objects_world()
        pairs_obj=self._match_by_nearest_center(j_objs, x_objs)
        j_sigs=self._json_signs(); x_sigs=self._xodr_signals_world()
        pairs_sig=self._match_by_nearest_center(j_sigs, x_sigs)
        fig,ax=plt.subplots(1,1, figsize=(10,10))
        ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)
        ax.set_title('Objects/Signals Outline Overlay (JSON solid, XODR dashed)')
        for j,x,_ in pairs_obj:
            if j.get('outline'): xs,ys=zip(*j['outline']); ax.plot(xs,ys, linewidth=1.6, alpha=0.95)
            if x and x.get('outline'): xs,ys=zip(*x['outline']); ax.plot(xs,ys, linewidth=1.2, alpha=0.95, linestyle='--')
        for j,x,_ in pairs_sig:
            if j.get('outline'): xs,ys=zip(*j['outline']); ax.plot(xs,ys, linewidth=1.6, alpha=0.95)
            if x and x.get('outline'): xs,ys=zip(*x['outline']); ax.plot(xs,ys, linewidth=1.2, alpha=0.95, linestyle='--')
        if save_path is None:
            save_path=self.json_file.parent/(self.json_file.stem+"_outline_overlay.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); self._viz_outline_path=str(save_path)
        print("✅ Outline overlay saved:", save_path)
        try: plt.close(fig)
        except Exception: pass
        return str(save_path)

    # ---------- HTML ----------
    def _generate_html_report(self, data: Dict) -> str:
        json_file = _html.escape(data['json_file'])
        xodr_file = _html.escape(data['xodr_file'])
        threshold = data['threshold']
        outline_tol = data['outline_tol']
        report: QualityReport = data['report']
        viz_path = data['viz_path']
        viz_outline = data['viz_outline']
        max_rows = int(data.get('max_detail_rows', 200))
        max_outline_rows = int(data.get('max_outline_rows', 200))

        comp = report.details.get('completeness', {})
        lanes_info = comp.get('lanes', {})
        bounds_info = comp.get('bounds', {})
        objs_info = comp.get('objects', {})
        signs_info = comp.get('signs', {})

        cons = report.details.get('consistency', {})
        avg_dev = cons.get('average_deviation', 0.0)
        max_dev = cons.get('max_deviation', 0.0)
        min_dev = cons.get('min_deviation', 0.0)
        pt_cnt = cons.get('point_count', 0)
        warn_cnt = cons.get('warnings_count', 0)
        over_cnt = cons.get('points_over_threshold', 0)
        v14_grouped = cons.get('v14_grouped', False)
        matched_lanes = cons.get('v14_matched_lanes', 0)
        mapped_bounds = cons.get('v14_mapped_bounds', 0)

        objc = report.details.get('object_consistency_v14', {})
        sigc = report.details.get('signal_consistency_v14', {})
        eval_points = report.details.get('curve_eval_points_v14', [])  # 明细（按 bound 指派，用于打分）

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        warn_list_html = "<p class='muted'>无</p>"
        if report.warnings:
            warn_items = report.warnings[:200]
            warn_list_html = "<ul class='warning-list'>" + "".join(
                f"<li>{_html.escape(w)}</li>" for w in warn_items) + "</ul>"
            if len(report.warnings) > 200:
                warn_list_html += f"<p class='muted'>（仅显示前 200 条，剩余 {len(report.warnings) - 200} 条已省略）</p>"

        def fmt(v, pct=False, nd=3):
            if v is None: return '—'
            return f"{v:.1%}" if pct else f"{v:.{nd}f}"

        viz_img_html = f"<img src=\"{_html.escape(viz_path)}\" style=\"max-width:100%;border:1px solid #eee;border-radius:8px;\">" if viz_path else ""
        viz_outline_html = f"<img src=\"{_html.escape(viz_outline)}\" style=\"max-width:100%;border:1px solid #eee;border-radius:8px;\">" if viz_outline else ""

        # 完整性表格行
        comp_table_rows = f"""
          <tr><td>车道（driving）</td><td class="num">{lanes_info.get('json_count', '—')}</td><td class="num">{lanes_info.get('xodr_count', '—')}</td><td class="num">{lanes_info.get('score', 0):.2f}</td></tr>
          <tr><td>边界（唯一ID vs 拓扑边界）</td><td class="num">{bounds_info.get('json_count', '—')}</td><td class="num">{bounds_info.get('xodr_count', '—')}</td><td class="num">{bounds_info.get('score', 0):.2f}</td></tr>
          <tr><td>物体（objects）</td><td class="num">{objs_info.get('json_count', '—')}</td><td class="num">{objs_info.get('xodr_count', '—')}</td><td class="num">{objs_info.get('score', 0):.2f}</td></tr>
          <tr><td>标识（sign → signal）</td><td class="num">{signs_info.get('json_count', '—')}</td><td class="num">{signs_info.get('xodr_count', '—')}</td><td class="num">{signs_info.get('score', 0):.2f}</td></tr>
        """

        # 对象/标识摘要卡数据
        obj_kpi_score = fmt(objc.get('score'), pct=True)
        sig_kpi_score = fmt(sigc.get('score'), pct=True)
        obj_legend = f"均值: {fmt(objc.get('chamfer_mean_avg'))} m；最大: {fmt(objc.get('chamfer_max_max'))} m；通过/警告/失败/缺失: {objc.get('outline_pass', 0)}/{objc.get('outline_warn', 0)}/{objc.get('outline_fail', 0)}/{objc.get('outline_missing_count', 0)}"
        sig_legend = f"均值: {fmt(sigc.get('chamfer_mean_avg'))} m；最大: {fmt(sigc.get('chamfer_max_max'))} m；通过/警告/失败/缺失: {sigc.get('outline_pass', 0)}/{sigc.get('outline_warn', 0)}/{sigc.get('outline_fail', 0)}/{sigc.get('outline_missing_count', 0)}"

        return f"""<!doctype html>
    <html lang="zh-CN">
    <head>
    <meta charset="utf-8" />
    <title>JSON→XODR 质检报告</title>
    <style>
      html,body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif;color:#222; }}
      .container {{ max-width: 1100px; margin: 32px auto; padding: 0 16px; }}
      h1 {{ font-size: 26px; margin: 0 0 6px; }}
      h2 {{ font-size: 20px; margin: 28px 0 12px; border-left:4px solid #555; padding-left:8px; }}
      h3 {{ font-size: 16px; margin: 18px 0 10px; }}
      .meta,.muted {{ color:#666; font-size: 13px; }}
      .grid3 {{ display:grid; grid-template-columns: repeat(3,1fr); gap:12px; }}
      .grid2 {{ display:grid; grid-template-columns: repeat(2,1fr); gap:12px; }}
      .card {{ border:1px solid #e6e6e6; border-radius:10px; padding:12px 14px; background:#fff; }}
      .kpi {{ font-size: 22px; font-weight:600; }}
      .ok {{ color:#138000; }} .warn {{ color:#d97706; }} .bad {{ color:#b91c1c; }}
      table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
      th,td {{ border-bottom:1px solid #eee; padding:8px 10px; text-align:left; }}
      th {{ background:#fafafa; }}
      td.num,th.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
      .imgwrap {{ text-align:center; margin: 14px 0 6px; }}
      .warning-list {{ margin:6px 0 0 16px; }}
      .legend {{ color:#555; font-size: 13px; }}
      .footer {{ color:#777; font-size: 12px; margin-top: 28px; }}
      .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid transparent; }}
      .badge.tag {{ background:#eef; color:#225; border-color:#dde; }}
    </style>
    </head>
    <body>
    <div class="container">

      <h1>JSON → XODR 质检报告</h1>
      <div class="meta">生成时间：{now_str}</div>
      <div class="meta">JSON 文件：{json_file}</div>
      <div class="meta">XODR 文件：{xodr_file}</div>

      <h2>摘要</h2>
      <div class="grid3">
        <div class="card">
          <div class="muted">完整性得分</div>
          <div class="kpi">{report.completeness_score:.1%}</div>
          <div class="legend">依据：车道/边界/物体/标识数量对比</div>
        </div>
        <div class="card">
          <div class="muted">一致性得分（曲线，v1.4）</div>
          <div class="kpi">{report.consistency_score:.1%}</div>
          <div class="legend">按车道分组 + 跨路段拼接 + 边界指派；阈值 {threshold:.3f} m</div>
        </div>
        <div class="card">
          <div class="muted">告警数量</div>
          <div class="kpi {'ok' if warn_cnt == 0 else ('warn' if warn_cnt < 10 else 'bad')}">{warn_cnt}</div>
          <div class="legend">超阈值点：{over_cnt}/{pt_cnt}</div>
        </div>
      </div>

      <div class="grid2" style="margin-top:12px;">
        <div class="card">
          <div class="muted">对象一致性（轮廓）</div>
          <div class="kpi">{obj_kpi_score}</div>
          <div class="legend">{obj_legend}</div>
        </div>
        <div class="card">
          <div class="muted">标识一致性（轮廓）</div>
          <div class="kpi">{sig_kpi_score}</div>
          <div class="legend">{sig_legend}</div>
        </div>
      </div>

      <h2>完整性检查</h2>
      <div class="card">
        <table>
          <thead>
            <tr><th>要素</th><th class="num">JSON 数量</th><th class="num">XODR 数量</th><th class="num">子分数</th></tr>
          </thead>
          <tbody>
            {comp_table_rows}
          </tbody>
        </table>
        <div class="muted">说明：XODR 数量可能大于 JSON（例如拓扑边界拆分、道路拼接等）；子分数按 min(XODR/JSON, 1.0) 计算。</div>
      </div>

      <h2>一致性检查（曲线） <span class="badge tag">{'v1.4 分组启用' if v14_grouped else '回退：全局最近邻'}</span></h2>
      <div class="grid3">
        <div class="card">
          <div class="muted">平均偏移</div><div class="kpi">{avg_dev:.3f} m</div>
          <div class="legend">匹配车道：{matched_lanes}；映射边界：{mapped_bounds}</div>
        </div>
        <div class="card">
          <div class="muted">最大 / 最小偏移</div><div class="kpi">{max_dev:.3f} / {min_dev:.3f} m</div>
        </div>
        <div class="card">
          <div class="muted">参与点数</div><div class="kpi">{pt_cnt}</div>
          <div class="legend">仅包含 JSON 边界点</div>
        </div>
      </div>
      <div class="card"><div class="muted">分布 & 偏移热力（示意）</div><div class="imgwrap">{viz_img_html}</div></div>

      <h2>对象/标识一致性（轮廓）</h2>
      <div class="card">
        <div class="imgwrap">{viz_outline_html}</div>
        <div class="muted">判定规则：阈值 <b>outline_tol = {outline_tol:.3f} m</b>；均值 ≤ tol → <b>通过</b>；≤ 2·tol → <b>警告</b>；其余 → <b>失败</b>；任一侧缺失 → <b>缺失</b>。</div>
      </div>

      <!-- 精简后的：按 bound 指派（用于打分）的明细表 -->
      <h2>边界匹配明细（按 bound 指派，用于打分，前 {max_rows} 条）</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th class="num">序号</th>
              <th>类型</th>
              <th>Bound ID</th>
              <th>JSON 坐标 (x, y)</th>
              <th>最近 XODR 坐标 (x, y)（示意）</th>
              <th class="num">偏移 (m)</th>
              <th>判定</th>
            </tr>
          </thead>
          <tbody>
            {self._build_assign_detail_rows(eval_points, max_rows)}
          </tbody>
        </table>
        <div class="muted">注：偏移 (m) 为按 bound 指派的距离；若某点未能指派显示 “—”。最近 XODR 坐标仅用于可视化示意，不参与打分。</div>
      </div>

      <h2>对象轮廓匹配明细（前 {max_outline_rows} 条）</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th class="num">序号</th>
              <th>JSON 对象ID</th>
              <th>XODR 对象ID</th>
              <th class="num">JSON 点数</th>
              <th class="num">XODR 点数</th>
              <th class="num">Chamfer 均值 (m)</th>
              <th class="num">最大 (m)</th>
              <th>判定</th>
              <th class="num">得分</th>
            </tr>
          </thead>
          <tbody>
            {self._build_outline_detail_rows(objc.get('items', []), max_outline_rows)}
          </tbody>
        </table>
        <div class="muted">注：对象轮廓匹配基于 Chamfer 距离；判定与阈值规则同上。</div>
      </div>

      <h2>标识轮廓匹配明细（前 {max_outline_rows} 条）</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th class="num">序号</th>
              <th>JSON 标识ID</th>
              <th>XODR 标识ID</th>
              <th class="num">JSON 点数</th>
              <th class="num">XODR 点数</th>
              <th class="num">Chamfer 均值 (m)</th>
              <th class="num">最大 (m)</th>
              <th>判定</th>
              <th class="num">得分</th>
            </tr>
          </thead>
          <tbody>
            {self._build_outline_detail_rows(sigc.get('items', []), max_outline_rows)}
          </tbody>
        </table>
        <div class="muted">注：标识轮廓匹配基于 Chamfer 距离；判定与阈值规则同上。</div>
      </div>

      <h2>告警列表</h2>
      <div class="card">{warn_list_html}</div>

      <div class="footer">
        曲线一致性阈值：{threshold:.3f} m；对象/标识轮廓阈值：{outline_tol:.3f} m。<br/>
        本报告采用 v1.4 的“车道中心多候选→跨路段拼接→2×2 指派”，无法指派时回退到全局最近邻用于统计与明细展示。
      </div>

    </div>
    </body>
    </html>"""

    def _build_match_detail_rows(self, matching: Dict, max_rows: int) -> str:
        rows=[]
        for row in matching.get('detailed_matches', [])[:max_rows]:
            jp=row['json_point']; nearest=row.get('nearest_xodr_point')
            nearest_str=f"({nearest[0]:.3f}, {nearest[1]:.3f})" if nearest else '—'
            rows.append("<tr>"
                + f"<td class='num'>{row['index']}</td>"
                + f"<td>{_html.escape(str(jp.get('source','')))}</td>"
                + f"<td>({jp['x']:.3f}, {jp['y']:.3f})</td>"
                + f"<td>{nearest_str}</td>"
                + f"<td class='num'>{row['deviation']:.3f}</td>"
                + f"<td>{_html.escape(str(row['status']))}</td>"
                + "</tr>")
        if not rows: rows.append("<tr><td colspan='6' style='text-align:center;'>暂无数据</td></tr>")
        return "".join(rows)

    def _build_assign_detail_rows(self, items: List[Dict], max_rows: int) -> str:
        def fmt_pt(pt):
            if not pt: return "—"
            return f"({pt[0]:.3f}, {pt[1]:.3f})"

        def fmt_m(v):
            return "—" if (v is None) else f"{v:.3f}"

        if not items:
            return "<tr><td colspan='7' style='text-align:center;'>暂无数据</td></tr>"

        # 按实际采用距离 d_use 从大到小排序，越大的越靠前便于排查
        rows_sorted = sorted(items, key=lambda r: (-(r.get("d_use") or -1e9)))

        rows = []
        for i, r in enumerate(rows_sorted[:max_rows]):
            icon = "✅" if r.get("status") == "通过" else ("⚠️" if r.get("status") == "警告" else "❌")
            status_txt = f"{icon} {r.get('status', '—')}"
            rows.append(
                "<tr>"
                f"<td class='num'>{i}</td>"
                f"<td>{r.get('type', 'bound')}</td>"
                f"<td>{r.get('bound_id', '—')}</td>"
                f"<td>({r['x']:.3f}, {r['y']:.3f})</td>"
                f"<td>{fmt_pt(r.get('nn_point'))}</td>"
                f"<td class='num'>{fmt_m(r.get('d_assign'))}</td>"
                f"<td>{status_txt}</td>"
                "</tr>"
            )
        return "".join(rows)

    def _build_outline_detail_rows(self, items: List[Dict], max_rows: int) -> str:
        def fmt(v, nd=3):
            if v is None:
                return '—'
            return f"{v:.{nd}f}"

        rows = []
        for i, it in enumerate(items[:max_rows]):
            rows.append(
                "<tr>"
                f"<td class='num'>{i}</td>"
                f"<td>{str(it.get('json_id', ''))}</td>"
                f"<td>{str(it.get('xodr_id', '—'))}</td>"
                f"<td class='num'>{it.get('json_pts', 0)}</td>"
                f"<td class='num'>{it.get('xodr_pts', 0)}</td>"
                f"<td class='num'>{fmt(it.get('chamfer_mean', None))}</td>"
                f"<td class='num'>{fmt(it.get('chamfer_max', None))}</td>"
                f"<td>{it.get('outline_status', '—')}</td>"
                f"<td class='num'>{(it.get('item_score') or 0.0):.0%}</td>"
                "</tr>"
            )
        if not rows:
            rows.append("<tr><td colspan='9' style='text-align:center;'>暂无数据</td></tr>")
        return "".join(rows)

    # ---------- 明细（用于表格，可视化示意，不影响打分） ----------
    def analyze_matching_details(self, include_objects: bool=False) -> Dict:
        all_json_points=self._get_all_json_points()
        use_points=all_json_points if include_objects else [p for p in all_json_points if p.get('source')=='bound']
        xodr_data=self._sample_xodr_curves_and_lanes()
        xodr_pts=[]; xodr_pts.extend(xodr_data.get('reference_lines',[])); xodr_pts.extend(xodr_data.get('lane_boundaries',[]))
        results={'total_json_points':len(use_points),'total_xodr_points':len(xodr_pts),'detailed_matches':[]}
        for i,jp in enumerate(use_points):
            md=float('inf'); nearest=None; jx,jy=jp['x'],jp['y']
            for xp,yp in xodr_pts:
                d=math.hypot(jx-xp, jy-yp)
                if d<md: md=d; nearest=(xp,yp)
            status='✅ Pass' if md<=self.threshold else ('⚠️ Warning' if md<=2*self.threshold else '❌ Fail')
            results['detailed_matches'].append({'index':i,'json_point':jp,'nearest_xodr_point':nearest,'deviation':md,'status':status})
        return results

    # ---------- 编排 ----------
    def generate_report(self, max_detail_rows: int = 200, max_outline_rows: int = 200) -> str:
        print("\n📝 生成质检报告…")
        if 'completeness' not in self.report.details: self.check_completeness()
        if 'consistency' not in self.report.details: self.check_curve_consistency()
        self._objects_signals_consistency()
        matching = self.analyze_matching_details(include_objects=False)
        viz_path = self._viz_path
        if not viz_path or not Path(viz_path).exists(): viz_path = self.visualize_point_matching()
        viz_outline = self._viz_outline_path
        if not viz_outline or not Path(viz_outline).exists(): viz_outline = self.visualize_outline_overlay()
        data = {'json_file': str(self.json_file), 'xodr_file': str(self.xodr_file),
                'threshold': self.threshold, 'outline_tol': self.outline_tol,
                'report': self.report, 'matching': matching,
                'viz_path': str(viz_path), 'viz_outline': str(viz_outline),
                'max_detail_rows': max_detail_rows,
                'max_outline_rows': max_outline_rows}
        html_report = self._generate_html_report(data)
        report_file = self.json_file.parent / f"{self.json_file.stem}_quality_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print("✅ 质检报告已生成:", report_file)
        return str(report_file)


if __name__ == '__main__':
    checker = QualityChecker(
        json_file="label3.json",
        xodr_file="label3.xodr",
        threshold=0.1,     # 曲线一致性（边界点）位置阈值
        outline_tol=0.20   # 轮廓一致性（Chamfer 均值）阈值
    )
    checker.check_completeness()
    checker.check_curve_consistency()
    checker.visualize_point_matching()
    checker.visualize_outline_overlay()
    checker.generate_report()
    print("\nDone.")
