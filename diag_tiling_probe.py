# diag_tiling_probe.py — THROWAWAY diagnostic (do not commit). Run from repo root on Colab.
import argparse, math, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.detection import eval_stratified as E

def native_side(x1,y1,x2,y2): return math.sqrt((x2-x1)*(y2-y1))

def per_gt_best(preds, gbox):
    best_conf, best_overlap = None, 0.0
    for px1,py1,px2,py2,pc in preds:
        v = E.iou_xyxy((px1,py1,px2,py2), gbox[:4])
        if v > 0: best_overlap = max(best_overlap, pc)
        if v >= E.MATCH_IOU and (best_conf is None or pc > best_conf): best_conf = pc
    return best_conf, best_overlap

ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); args = ap.parse_args()
small, medium, _ = E.load_thresholds()
stems = [Path(l).stem for l in E.TEST_LIST.read_text().splitlines() if l.strip()]
gt = E.load_gt_d40(stems, small, medium)
preds640 = E.predict_baseline(args.weights, stems, E.CONF_AP, E.NMS_IOU, 640)

# PART 1 — miss analysis on the 67 small GT
found_both=conf_fail=loc_fail=0; ex={"conf_fail":[],"loc_fail":[],"found_both":[]}
for stem,boxes in gt.items():
    for g in boxes:
        if g[4]!="small": continue
        bc,boc = per_gt_best(preds640.get(stem,[]), g); area=(g[2]-g[0])*(g[3]-g[1])
        if bc is None:
            loc_fail+=1
            if len(ex["loc_fail"])<4: ex["loc_fail"].append((stem,round(area),round(boc,3)))
        elif bc < E.CONF_OP:
            conf_fail+=1
            if len(ex["conf_fail"])<4: ex["conf_fail"].append((stem,round(area),round(bc,3)))
        else:
            found_both+=1
            if len(ex["found_both"])<4: ex["found_both"].append((stem,round(area),round(bc,3)))
print("\n=== PART 1: small-D40 miss analysis (67 GT) ===")
print(f"found @0.001 AND @0.25            : {found_both}")
print(f"found @0.001, lost by 0.25 (CONF) : {conf_fail}")
print(f"missed even @0.001 (LOCALIZATION) : {loc_fail}")
print("examples (stem, native_area_px, matched/overlap_conf):", ex)

# PART 2 — finer small split by native side
def recall_sub(cond, conf):
    tp=tot=0
    for stem,boxes in gt.items():
        preds=[p for p in preds640.get(stem,[]) if p[4]>=conf]
        for g in boxes:
            if g[4]=="small" and cond(native_side(*g[:4])):
                tot+=1; bc,_=per_gt_best(preds,g); tp+= 1 if bc is not None else 0
    return tp,tot
print("\n=== PART 2: finer small split (native-720 side) ===")
print(f"{'sub-bucket':<16}{'GT':>5}{'recall@0.001':>14}{'recall@0.25':>13}")
for name,cond in (("very-small<16",lambda s:s<16),("small 16-32",lambda s:16<=s<32),("residual>=32",lambda s:s>=32)):
    tp_lo,tot=recall_sub(cond,E.CONF_AP); tp_hi,_=recall_sub(cond,E.CONF_OP)
    r_lo=tp_lo/tot if tot else float("nan"); r_hi=tp_hi/tot if tot else float("nan")
    print(f"{name:<16}{tot:>5}{r_lo:>14.3f}{r_hi:>13.3f}")

# PART 3 — resolution probe (640 vs 1280 inference, no retrain)
print("\n=== PART 3: resolution probe ===")
print(f"{'imgsz':<8}{'overall D40 recall@0.25':>26}{'small recall@0.25':>20}")
for imgsz,preds in (("640",preds640),("1280",None)):
    if preds is None: preds=E.predict_baseline(args.weights, stems, E.CONF_AP, E.NMS_IOU, 1280)
    gc,tp,_=E._match_one_conf(preds, gt, E.CONF_OP, small, medium)
    overall=sum(tp.values())/sum(gc.values()); sm=tp["small"]/gc["small"] if gc["small"] else float("nan")
    print(f"{imgsz:<8}{overall:>26.3f}{sm:>20.3f}")