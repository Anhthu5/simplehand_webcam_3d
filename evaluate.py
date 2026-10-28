"""PA metrics for paired arrays, with FreiHAND-style similarity alignment.

No webcam labels are fabricated. Generate paired predictions with infer_freihand.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def align_similarity(predicted, target, proper_rotation=False):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Expected matching (N,3) arrays")
    if not (np.isfinite(predicted).all() and np.isfinite(target).all()):
        raise ValueError("Non-finite coordinates")
    p_mean, t_mean = predicted.mean(0), target.mean(0)
    p, t = predicted-p_mean, target-t_mean
    variance = np.sum(p*p)
    if variance < 1e-20 or np.sum(t*t) < 1e-20:
        raise ValueError("Degenerate point cloud; alignment undefined")
    u, singular, vt = np.linalg.svd(p.T @ t)
    correction = np.ones(3)
    if proper_rotation and np.linalg.det(u @ vt) < 0:
        correction[-1] = -1
    rotation = (u * correction) @ vt
    scale = np.dot(singular, correction) / variance
    return scale * p @ rotation + t_mean


def pa_error(predictions, targets, proper_rotation=False):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or predictions.ndim != 3 or len(predictions) == 0:
        raise ValueError("Expected matching nonempty (B,N,3) arrays")
    return np.array([np.linalg.norm(align_similarity(p, t, proper_rotation)-t, axis=1).mean()
                     for p, t in zip(predictions, targets)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", help="NPZ containing pred_joints, gt_joints, pred_vertices, gt_vertices")
    parser.add_argument("--unit", choices=["m", "mm"], required=True, help="Unit of ALL arrays")
    parser.add_argument("--proper-rotation", action="store_true",
                        help="Forbid reflections; differs from upstream orthogonal alignment")
    parser.add_argument("--output-json", type=Path, help="Save measured metrics for the report")
    args = parser.parse_args()
    factor = 1000 if args.unit == "m" else 1
    measured = {"input": str(args.arrays), "input_unit": args.unit,
                "proper_rotation_only": args.proper_rotation,
                "protocol": "Independent per-sample similarity alignment for joints and vertices"}
    with np.load(args.arrays, allow_pickle=False) as data:
        for suffix, label, count in [("joints", "PA-MPJPE", 21), ("vertices", "PA-MPVPE", 778)]:
            pred, gt = data[f"pred_{suffix}"], data[f"gt_{suffix}"]
            if pred.shape[1:] != (count, 3):
                raise ValueError(f"{suffix} must have shape (B,{count},3)")
            errors = pa_error(pred, gt, args.proper_rotation)*factor
            print(f"{label}: {errors.mean():.4f} mm (N={len(errors)})")
            measured[label+"_mm"] = float(errors.mean())
            measured["samples_"+suffix] = len(errors)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(measured, stream, indent=2)
    print("Independent per-sample joint/mesh alignment. No unaligned absolute-camera error reported.")


if __name__ == "__main__":
    main()
