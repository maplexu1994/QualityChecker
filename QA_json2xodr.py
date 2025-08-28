#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to XODR Quality Checker — v1.3.6

变更（相对 v1.3.5）：
- 将“轮廓叠加”可视化，从『对象/标识一致性』摘要卡片中移除，
  单独作为一个章节放在 Objects/Signals 明细表格之前，便于放大查看。
- 其余：算法、打分、可视化生成逻辑均与 v1.3.5 保持一致（仅 HTML 布局调整）。
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

DEFAULT_SAMPLE_STEP = 0.05
CHAMFER_SAMPLE_CAP = 2000
BOUND_HEATMAP_CMAP = 'RdYlGn_r'


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
    def __init__(self, json_file: str, xodr_file: str, threshold: float = 0.1, outline_tol: float = 0.20):
        self.json_file = Path(json_file)
        self.xodr_file = Path(xodr_file)
        self.threshold = float(threshold)     # 曲线一致性（边界点）位置阈值
        self.outline_tol = float(outline_tol) # 轮廓一致性（Chamfer 均值）阈值

        self.json_data = self._load_json()
        self.xodr_root = self._load_xodr()

        self.report = QualityReport()
        self._viz_path: Optional[str] = None
        self._viz_outline_path: Optional[str] = None

        self._road_cache: Dict[str, Dict[str, List[float]]] = {}
        self._build_road_ref_cache()

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
        lane_score = min(self._count_xodr_lanes() / max(1,len(json_lanes)), 1.0) if json_lanes else 1.0
        details['lanes'] = {'json_count': len(json_lanes), 'xodr_count': self._count_xodr_lanes(), 'score': lane_score}
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

    # ---------- 参考线缓存 & s,t→(x,y) ----------
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

    # ---------- JSON points ----------
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

    # ---------- 曲线一致性 ----------
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
            sections=parse_lane_sections(lanes); offsets=parse_lane_offsets(lanes); road_len=s_arr[-1]
            for si,sec in enumerate(sections):
                s0=sec['s']; s1=sections[si+1]['s'] if si+1<len(sections) else road_len+1e-6
                idxs=[i for i,sv in enumerate(s_arr) if (sv>=s0 and sv<=s1)]
                if len(idxs)<2: continue
                centers={}; edges={}
                def ensure(d,k):
                    if k not in d: d[k]=[]
                    return d[k]
                for i in idxs:
                    s_here=s_arr[i]; s_rel=s_here-s0; nx=-math.sin(th_arr[i]); ny=math.cos(th_arr[i]); base=lane_offset_at(offsets,s_here)
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

    def _find_nearest_distance_to_xodr(self, jp: Dict, xodr_pts: List[Tuple[float,float]]) -> float:
        jx,jy=jp['x'],jp['y']; md=float('inf')
        for xp,yp in xodr_pts:
            d=math.hypot(jx-xp, jy-yp)
            if d<md: md=d
        return md

    def check_curve_consistency(self) -> float:
        print("\n🔍 开始曲线一致性检查…")
        warnings=[]; all_json_points=self._get_all_json_points()
        bound_points=[p for p in all_json_points if p.get('source')=='bound']
        xodr_data=self._sample_xodr_curves_and_lanes()
        xodr_points=[]; xodr_points.extend(xodr_data.get('reference_lines',[])); xodr_points.extend(xodr_data.get('lane_boundaries',[]))
        if not bound_points or not xodr_points:
            self.report.consistency_score=0.0
            self.report.details['consistency']={'average_deviation':0.0,'max_deviation':0.0,'min_deviation':0.0,
                                                'point_count':len(bound_points),'threshold':self.threshold,
                                                'warnings_count':0,'points_over_threshold':0}
            return 0.0
        devs=[]
        for i,jp in enumerate(bound_points):
            d=self._find_nearest_distance_to_xodr(jp, xodr_points)
            devs.append(d)
            if d>self.threshold:
                warnings.append("⚠️ 边界点偏移超阈值: (%.3f, %.3f) → %.3fm"%(jp['x'],jp['y'],d))
            if (i+1)%200==0 or i==len(bound_points)-1:
                print(f"  已处理 {i+1}/{len(bound_points)}")
        avg_dev=float(np.mean(devs)) if devs else 0.0
        max_dev=float(np.max(devs)) if devs else 0.0
        min_dev=float(np.min(devs)) if devs else 0.0
        over=sum(1 for d in devs if d>self.threshold)
        score=max(0.0, 1.0-(avg_dev/self.threshold)) if self.threshold>0 else 0.0
        self.report.consistency_score=score
        self.report.warnings.extend(warnings)
        self.report.details['consistency']={'average_deviation':avg_dev,'max_deviation':max_dev,'min_deviation':min_dev,
                                            'point_count':len(bound_points),'threshold':self.threshold,
                                            'warnings_count':len(warnings),'points_over_threshold':over}
        print("📊 一致性得分:", "%.2f%%"%(score*100))
        return score

    # ---------- objects/signals 一致性（仅 outline） ----------
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
        items_obj=[]; chamfer_means=[]; chamfer_maxs=[]; per_scores=[]
        cnt_pass=cnt_warn=cnt_fail=cnt_missing=0
        for j,x,_ in pairs_obj:
            A=j['outline']; B=(x['outline'] if (x and x.get('outline')) else [])
            if not A or not B:
                ch={'mean':None,'max':None,'nA':len(A),'nB':len(B)}
            else:
                ch=self._chamfer_metrics(A,B); chamfer_means.append(ch['mean']); chamfer_maxs.append(ch['max'])
            status=self._judge_outline_status(ch['mean']); score_i=self._per_item_score(ch['mean']); per_scores.append(score_i)
            if   status=='通过': cnt_pass+=1
            elif status=='警告': cnt_warn+=1
            elif status=='失败': cnt_fail+=1
            else: cnt_missing+=1
            items_obj.append({'json_id':j['json_id'],'xodr_id':(x['xodr_id'] if x else None),
                              'json_pts':len(A),'xodr_pts':(ch['nB'] if B else 0),
                              'chamfer_mean':ch['mean'],'chamfer_max':ch['max'],
                              'outline_status':status,'item_score':score_i})
        obj_score=float(np.mean(per_scores)) if per_scores else None
        self.report.details['object_consistency_v136']={'json_count':len(j_objs),'xodr_count':len(x_objs),'matched':len(pairs_obj),
            'chamfer_mean_avg':(float(np.mean(chamfer_means)) if chamfer_means else None),
            'chamfer_max_max':(float(np.max(chamfer_maxs)) if chamfer_maxs else None),
            'outline_pass':cnt_pass,'outline_warn':cnt_warn,'outline_fail':cnt_fail,'outline_missing_count':cnt_missing,
            'score':obj_score,'items':items_obj}

        # signals
        j_sigs=self._json_signs(); x_sigs=self._xodr_signals_world()
        pairs_sig=self._match_by_nearest_center(j_sigs, x_sigs)
        items_sig=[]; chamfer_means=[]; chamfer_maxs=[]; per_scores=[]
        cnt_pass=cnt_warn=cnt_fail=cnt_missing=0
        for j,x,_ in pairs_sig:
            A=j['outline']; B=(x['outline'] if (x and x.get('outline')) else [])
            if not A or not B:
                ch={'mean':None,'max':None,'nA':len(A),'nB':len(B)}
            else:
                ch=self._chamfer_metrics(A,B); chamfer_means.append(ch['mean']); chamfer_maxs.append(ch['max'])
            status=self._judge_outline_status(ch['mean']); score_i=self._per_item_score(ch['mean']); per_scores.append(score_i)
            if   status=='通过': cnt_pass+=1
            elif status=='警告': cnt_warn+=1
            elif status=='失败': cnt_fail+=1
            else: cnt_missing+=1
            items_sig.append({'json_id':j['json_id'],'xodr_id':(x['xodr_id'] if x else None),
                              'json_pts':len(A),'xodr_pts':(ch['nB'] if B else 0),
                              'chamfer_mean':ch['mean'],'chamfer_max':ch['max'],
                              'outline_status':status,'item_score':score_i})
        sig_score=float(np.mean(per_scores)) if per_scores else None
        self.report.details['signal_consistency_v136']={'json_count':len(j_sigs),'xodr_count':len(x_sigs),'matched':len(pairs_sig),
            'chamfer_mean_avg':(float(np.mean(chamfer_means)) if chamfer_means else None),
            'chamfer_max_max':(float(np.max(chamfer_maxs)) if chamfer_maxs else None),
            'outline_pass':cnt_pass,'outline_warn':cnt_warn,'outline_fail':cnt_fail,'outline_missing_count':cnt_missing,
            'score':sig_score,'items':items_sig}

        # 汇总得分（按样本数加权）
        n_obj=len(self.report.details['object_consistency_v136']['items'])
        n_sig=len(self.report.details['signal_consistency_v136']['items'])
        w_obj = n_obj if obj_score is not None else 0
        w_sig = n_sig if sig_score is not None else 0
        if (w_obj + w_sig) > 0:
            comb = ((obj_score or 0.0)*w_obj + (sig_score or 0.0)*w_sig) / (w_obj + w_sig)
        else:
            comb = None
        self.report.details['objects_signals_score']=comb

    # ---------- 可视化 ----------
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
        xodr_points=[]; xodr_points.extend(xodr_data.get('reference_lines', [])); xodr_points.extend(xodr_data.get('lane_boundaries', []))
        bound_points=[p for p in json_points if p['source']=='bound']; devs=[self._find_nearest_distance_to_xodr(p,xodr_points) for p in bound_points]
        if xodr_points:
            xp,yp=zip(*xodr_points); ax.scatter(xp,yp, c='lightgray', s=5, alpha=0.3)
        if bound_points:
            bx=[p['x'] for p in bound_points]; by=[p['y'] for p in bound_points]
            sc=ax.scatter(bx,by, c=devs, s=50, cmap=BOUND_HEATMAP_CMAP, alpha=0.9, edgecolors='black', linewidth=0.4)
            cbar=plt.colorbar(sc, ax=ax); cbar.set_label('Deviation (m)')
        ax.set_title('Deviation Analysis'); ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)

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
        json_file=_html.escape(data['json_file']); xodr_file=_html.escape(data['xodr_file'])
        threshold=data['threshold']; outline_tol=data['outline_tol']
        report:QualityReport=data['report']; matching=data['matching']
        viz_path=data['viz_path']; viz_outline=data['viz_outline']; max_rows=int(data.get('max_detail_rows',200))

        comp=report.details.get('completeness',{})
        lanes_info=comp.get('lanes',{}); bounds_info=comp.get('bounds',{}); objs_info=comp.get('objects',{}); signs_info=comp.get('signs',{})

        cons=report.details.get('consistency',{})
        avg_dev=cons.get('average_deviation',0.0); max_dev=cons.get('max_deviation',0.0); min_dev=cons.get('min_deviation',0.0)
        pt_cnt=cons.get('point_count',0); warn_cnt=cons.get('warnings_count',0); over_cnt=cons.get('points_over_threshold',0)

        objc=report.details.get('object_consistency_v136',{}); sigc=report.details.get('signal_consistency_v136',{})
        comb_score=report.details.get('objects_signals_score', None)

        now_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        warn_list_html="<p class='muted'>无</p>"
        if report.warnings:
            warn_items=report.warnings[:200]
            warn_list_html="<ul class='warning-list'>"+"".join(f"<li>{_html.escape(w)}</li>" for w in warn_items)+"</ul>"
            if len(report.warnings)>200:
                warn_list_html+=f"<p class='muted'>（仅显示前 200 条，剩余 {len(report.warnings)-200} 条已省略）</p>"

        def badge_html(status: str) -> str:
            cls={'通过':'pass','警告':'warn','失败':'fail'}.get(status,'miss')
            return f"<span class='badge {cls}'>{_html.escape(status)}</span>"

        def fmt(v, pct=False):
            if v is None: return '—'
            return f"{v:.1%}" if pct else f"{v:.3f}"

        def make_items_table(items: List[Dict]) -> str:
            rows=[]
            for it in items[:max_rows]:
                ch_mean='—' if it['chamfer_mean'] is None else f"{it['chamfer_mean']:.3f}"
                ch_max ='—' if it['chamfer_max']  is None else f"{it['chamfer_max']:.3f}"
                rows.append(
                    "<tr>"
                    + f"<td>{_html.escape(str(it['json_id']))}</td>"
                    + f"<td>{_html.escape(str(it.get('xodr_id','—') or '—'))}</td>"
                    + f"<td class='num'>{it['json_pts']}</td>"
                    + f"<td class='num'>{it['xodr_pts']}</td>"
                    + f"<td class='num'>{ch_mean}</td>"
                    + f"<td class='num'>{ch_max}</td>"
                    + f"<td class='num'>{fmt(it['item_score'], pct=True)}</td>"
                    + f"<td>{badge_html(it['outline_status'])}</td>"
                    + "</tr>"
                )
            if not rows:
                rows.append("<tr><td colspan='8' style='text-align:center;'>暂无数据</td></tr>")
            return "".join(rows)

        objects_rows = make_items_table(objc.get('items', []))
        signals_rows = make_items_table(sigc.get('items', []))

        viz_img_html     = f"<img src=\"{_html.escape(viz_path)}\" style=\"max-width:100%;border:1px solid #eee;border-radius:8px;\">" if viz_path else ""
        viz_outline_html = f"<img src=\"{_html.escape(viz_outline)}\" style=\"max-width:100%;border:1px solid #eee;border-radius:8px;\">" if viz_outline else ""

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
  .grid4 {{ display:grid; grid-template-columns: repeat(4,1fr); gap:12px; }}
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
  .badge.pass {{ background:#E6F4EA; color:#137333; border-color:#CFE9D6; }}
  .badge.warn {{ background:#FEF7E0; color:#B06000; border-color:#F3D89C; }}
  .badge.fail {{ background:#FDE7E9; color:#B3261E; border-color:#F4C7C9; }}
  .badge.miss {{ background:#E7F3FF; color:#01579B; border-color:#CDE7FF; }}
</style>
</head>
<body>
<div class="container">

  <h1>JSON → XODR 质检报告</h1>
  <div class="meta">生成时间：{now_str}</div>
  <div class="meta">JSON 文件：{json_file}</div>
  <div class="meta">XODR 文件：{xodr_file}</div>

  <h2>摘要</h2>
  <div class="grid4">
    <div class="card">
      <div class="muted">完整性得分</div>
      <div class="kpi">{report.completeness_score:.1%}</div>
      <div class="legend">依据：车道/边界/物体/标识数量对比</div>
    </div>
    <div class="card">
      <div class="muted">一致性得分（曲线）</div>
      <div class="kpi">{report.consistency_score:.1%}</div>
      <div class="legend">依据：JSON 边界点 ↔ XODR（参考线+车道边界）最近距离均值 vs 阈值（{threshold:.3f} m）</div>
    </div>
    <div class="card">
      <div class="muted">告警数量</div>
      <div class="kpi {'ok' if warn_cnt==0 else ('warn' if warn_cnt<10 else 'bad')}">{warn_cnt}</div>
      <div class="legend">曲线超阈值点：{over_cnt}/{pt_cnt}</div>
    </div>
    <div class="card">
      <div class="muted">对象/标识一致性得分</div>
      <div class="kpi">{fmt(comb_score, pct=True)}</div>
      <div class="legend">单项得分：max(0, 1 - d/(2·tol))；缺失记 0；tol = {outline_tol:.3f} m</div>
    </div>
  </div>

  <h2>一致性检查（曲线）</h2>
  <div class="grid3">
    <div class="card">
      <div class="muted">平均偏移</div><div class="kpi">{avg_dev:.3f} m</div>
    </div>
    <div class="card">
      <div class="muted">最大 / 最小偏移</div><div class="kpi">{max_dev:.3f} / {min_dev:.3f} m</div>
    </div>
    <div class="card">
      <div class="muted">参与点数</div><div class="kpi">{pt_cnt}</div>
      <div class="legend">仅包含 JSON 边界点</div>
    </div>
  </div>
  <div class="card"><div class="muted">分布 & 偏移热力</div><div class="imgwrap">{viz_img_html}</div></div>

  <h2>对象/标识一致性（v1.3.6，仅轮廓）</h2>
  <div class="grid2">
    <div class="card">
      <h3>Objects</h3>
      <div class="legend">匹配：{objc.get('matched',0)}/{objc.get('json_count',0)}（XODR {objc.get('xodr_count',0)}）</div>
      <div class="legend">Chamfer 均值（平均）：{fmt(objc.get('chamfer_mean_avg'))} m；最大：{fmt(objc.get('chamfer_max_max'))} m</div>
      <div class="legend">一致性得分：{fmt(objc.get('score'), pct=True)}</div>
      <div class="legend">判定统计：<span class='badge pass'>通过</span> {objc.get('outline_pass',0)}；<span class='badge warn'>警告</span> {objc.get('outline_warn',0)}；<span class='badge fail'>失败</span> {objc.get('outline_fail',0)}；<span class='badge miss'>缺失</span> {objc.get('outline_missing_count',0)}</div>
    </div>
    <div class="card">
      <h3>Signals</h3>
      <div class="legend">匹配：{sigc.get('matched',0)}/{sigc.get('json_count',0)}（XODR {sigc.get('xodr_count',0)}）</div>
      <div class="legend">Chamfer 均值（平均）：{fmt(sigc.get('chamfer_mean_avg'))} m；最大：{fmt(sigc.get('chamfer_max_max'))} m</div>
      <div class="legend">一致性得分：{fmt(sigc.get('score'), pct=True)}</div>
      <div class="legend">判定统计：<span class='badge pass'>通过</span> {sigc.get('outline_pass',0)}；<span class='badge warn'>警告</span> {sigc.get('outline_warn',0)}；<span class='badge fail'>失败</span> {sigc.get('outline_fail',0)}；<span class='badge miss'>缺失</span> {sigc.get('outline_missing_count',0)}</div>
    </div>
  </div>

  <h2>对象/标识 轮廓叠加</h2>
  <div class="card">
    <div class="muted">实线：JSON；虚线：XODR。重合度越高越好。</div>
    <div class="imgwrap">{viz_outline_html}</div>
    <div class="legend">阈值 outline_tol = {outline_tol:.3f} m；判定：均值 ≤ tol → 通过；≤ 2·tol → 警告；其余 → 失败；任一侧缺失 → 缺失。</div>
  </div>

  <h2>Objects 综述明细（前 {max_rows} 条）</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>JSON ID</th><th>XODR ID</th>
          <th class="num">JSON点数</th><th class="num">XODR点数</th>
          <th class="num">Chamfer均值(m)</th><th class="num">Chamfer最大(m)</th>
          <th class="num">单项得分</th><th>轮廓判定</th>
        </tr>
      </thead>
      <tbody>
        {objects_rows}
      </tbody>
    </table>
  </div>

  <h2>Signals 综述明细（前 {max_rows} 条）</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>JSON ID</th><th>XODR ID</th>
          <th class="num">JSON点数</th><th class="num">XODR点数</th>
          <th class="num">Chamfer均值(m)</th><th class="num">Chamfer最大(m)</th>
          <th class="num">单项得分</th><th>轮廓判定</th>
        </tr>
      </thead>
      <tbody>
        {signals_rows}
      </tbody>
    </table>
  </div>

  <h2>边界匹配明细（前 {max_rows} 条）</h2>
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
        {self._build_match_detail_rows(matching, max_rows)}
      </tbody>
    </table>
  </div>

  <h2>告警列表</h2>
  <div class="card">{warn_list_html}</div>

  <div class="footer">
    曲线一致性阈值：{threshold:.3f} m。<br/>
    对象/标识一致性：基于对称 Chamfer 距离（双向最近邻的均值/最大值）；单项得分＝max(0, 1 - d/(2·tol))，缺失 outline 记 0。<br/>
    本报告仅用于 JSON→XODR 转换结果的自动化质检与可视化分析。
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

    # ---------- 编排 ----------
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

    def generate_report(self, max_detail_rows: int = 200) -> str:
        print("\n📝 生成质检报告…")
        if 'completeness' not in self.report.details: self.check_completeness()
        if 'consistency' not in self.report.details: self.check_curve_consistency()
        self._objects_signals_consistency()
        matching=self.analyze_matching_details(include_objects=False)
        viz_path=self._viz_path
        if not viz_path or not Path(viz_path).exists(): viz_path=self.visualize_point_matching()
        viz_outline=self._viz_outline_path
        if not viz_outline or not Path(viz_outline).exists(): viz_outline=self.visualize_outline_overlay()
        data={'json_file':str(self.json_file),'xodr_file':str(self.xodr_file),
              'threshold':self.threshold,'outline_tol':self.outline_tol,
              'report':self.report,'matching':matching,
              'viz_path':str(viz_path),'viz_outline':str(viz_outline),
              'max_detail_rows':max_detail_rows}
        html_report=self._generate_html_report(data)
        report_file=self.json_file.parent/f"{self.json_file.stem}_quality_report.html"
        with open(report_file,'w',encoding='utf-8') as f: f.write(html_report)
        print("✅ 质检报告已生成:", report_file)
        return str(report_file)


if __name__ == '__main__':
    checker = QualityChecker(
        json_file="sample_objects.json",
        xodr_file="sample_objects.xodr",
        threshold=0.1,     # 曲线一致性（边界点）位置阈值
        outline_tol=0.20   # 轮廓一致性（Chamfer 均值）阈值
    )
    checker.check_completeness()
    checker.check_curve_consistency()
    checker.visualize_point_matching()
    checker.visualize_outline_overlay()
    checker.generate_report()
    print("\nDone.")
