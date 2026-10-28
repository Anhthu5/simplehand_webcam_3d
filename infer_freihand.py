"""Inference on author's preprocessed image/JSON pairs with known labels.

Matches official evaluation perspective crop and inverse 3D rotation, without TTA.
Not an implementation of the public leaderboard submission service.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from utils.inference import SimpleHandPredictor

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="JSON list of image paths")
    p.add_argument("--data-root", type=Path, required=True, help="Root for relative paths in list")
    p.add_argument("--checkpoint", type=Path, default=ROOT/"weights/simplehand_fastvit_ma36.pth")
    p.add_argument("--upstream", type=Path, default=ROOT/"third_party/simpleHand")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output", type=Path, default=ROOT/"captures/freihand_predictions.npz")
    p.add_argument("--limit", type=int, default=0, help="0 evaluates full provided list")
    args = p.parse_args()
    helper = args.upstream/"kp_preprocess.py"
    spec = importlib.util.spec_from_file_location("official_kp_preprocess", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = SimpleHandPredictor(args.checkpoint, args.upstream/"models/MANO_RIGHT_C.pkl", args.device)
    with args.images.open(encoding="utf-8") as stream:
        paths = json.load(stream)
    if not isinstance(paths, list) or not paths:
        raise ValueError("Expected nonempty JSON image-path list")
    if args.limit > 0:
        paths = paths[:args.limit]
    output = {k: [] for k in ("pred_joints", "gt_joints", "pred_vertices", "gt_vertices")}
    for index, name in enumerate(paths):
        path = Path(name)
        path = path if path.is_absolute() else args.data_root/path
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {path}. Fix paths in eval.json for Windows.")
        with path.with_suffix(".json").open(encoding="utf-8") as stream:
            ann = json.load(stream)
        uv, xyz, vertices, k = [np.asarray(ann[key], dtype=np.float32)
                               for key in ("uv", "xyz", "vertices", "K")]
        if uv.shape != (21, 2) or xyz.shape != (21, 3) or vertices.shape != (778, 3):
            raise ValueError(f"Invalid annotation dimensions: {path}")
        h, w = image.shape[:2]
        norm = uv / [w, h]
        valid = ((norm > 0) & (norm < 1)).all(1)
        points = uv[valid] if valid.sum() > 1 else uv
        low, high = points.min(0), points.max(0)
        center, scale = (low+high)/2, (high-low)*1.25
        _, transform2d, transform3d = module.get_2d3d_perspective_transform(k, center, scale, 0, 224)
        crop = cv2.warpPerspective(image, transform2d, (224, 224))
        result = predictor.predict(crop)
        root = result["joints"][9].copy()
        inverse3d = np.linalg.inv(transform3d)
        output["pred_joints"].append((result["joints"]-root) @ inverse3d.T)
        output["pred_vertices"].append((result["vertices"]-root) @ inverse3d.T)
        output["gt_joints"].append(xyz)
        output["gt_vertices"].append(vertices)
        if (index+1) % 50 == 0:
            print(f"{index+1}/{len(paths)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in output.items()},
                        image_paths=np.asarray(paths))
    print(f"Saved {len(paths)} pairs to {args.output}. Check annotation units before evaluate.py.")


if __name__ == "__main__":
    main()
