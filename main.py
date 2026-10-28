"""Webcam ROI detection with MediaPipe + genuine SimpleHand mesh inference."""
import argparse
import logging
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from utils.detector import HandDetector
from utils.geometry import square_crop, uv_to_frame
from utils.inference import SimpleHandPredictor
from utils.drawing import skeleton
from utils.mesh_viewer import MeshViewer, export_obj
from utils.session import SessionStats, DemoRecorder

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "weights/simplehand_fastvit_ma36.pth")
    p.add_argument("--mano", type=Path, default=ROOT / "third_party/simpleHand/models/MANO_RIGHT_C.pkl")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--padding", type=float, default=1.35)
    p.add_argument("--confidence", type=float, default=.6)
    p.add_argument("--amp", action="store_true", help="Optional CUDA fp16; compare with fp32 first")
    p.add_argument("--mirror", action="store_true", help="Mirror display only, after inference")
    p.add_argument("--canonicalize-left", action="store_true", help="Experimental left-hand reflection")
    p.add_argument("--view3d", action="store_true", default=True, help="3D mesh is enabled by default")
    p.add_argument("--no-view3d", action="store_false", dest="view3d", help="Diagnostics only: disable mesh window")
    p.add_argument("--show-joints", action="store_true", help="Optional 2D skeleton on input webcam")
    p.add_argument("--save-dir", type=Path, default=ROOT / "captures")
    p.add_argument("--video", type=Path, help="Optional prerecorded input instead of webcam")
    p.add_argument("--record", type=Path, help="Save webcam + actual 3D mesh side by side as MP4")
    p.add_argument("--record-fps", type=float, default=20.)
    args = p.parse_args()
    if args.padding < 1 or not 0 < args.confidence <= 1 or min(args.width, args.height) < 1:
        p.error("padding >= 1, confidence in (0,1], dimensions > 0 required")
    if not 1 <= args.record_fps <= 60:
        p.error("record-fps must be between 1 and 60")
    if args.record and not args.view3d:
        p.error("--record requires the mesh window, remove --no-view3d")
    if args.video and not args.video.is_file():
        p.error(f"Input video not found: {args.video}")
    if args.record and args.record.exists():
        p.error(f"Recording file already exists; choose a new name: {args.record}")
    if args.video and args.record and args.video.resolve() == args.record.resolve():
        p.error("Input and output video paths must differ")
    return args


def main():
    args = arguments()
    cv2.setNumThreads(1)
    predictor = SimpleHandPredictor(args.checkpoint, args.mano, args.device, args.amp)
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[args.backend]
    camera = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(args.camera, backend)
    detector = None
    viewer = None
    recorder = None
    stats = SessionStats()
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam. Close other camera apps or try --camera 1 --backend msmf.")
        if not args.video:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # backend may ignore this hint
        detector = HandDetector(args.confidence)
        if args.view3d:
            viewer = MeshViewer(predictor.faces)
            print(f"3D renderer: {viewer.renderer}")
        if args.record:
            recorder = DemoRecorder(args.record, args.record_fps)
        if viewer is not None:
            print("LIVE 3D mesh enabled. Drag: rotate; scroll: zoom; W: wireframe; R: reset.")
        print("Q/ESC: quit; S: save actual predicted mesh as OBJ and NPZ.")
        last_time, fps = perf_counter(), 0.0
        failures = 0
        while True:
            frame_start = perf_counter()
            actions = viewer.events() if viewer is not None else set()
            if "quit" in actions:
                break
            ok, frame = camera.read()
            if not ok:
                if args.video:
                    print("End of input video.")
                    break
                failures += 1
                if failures >= 5:
                    raise RuntimeError("Webcam read failed five consecutive times.")
                continue
            failures = 0
            view = frame.copy()
            detect_start = perf_counter()
            detection = detector.detect(frame)
            detection_ms = (perf_counter()-detect_start)*1000
            prediction, points, crop = None, None, None
            inference_ms = 0.0
            status = "No hand - SimpleHand skipped"
            if detection is not None:
                crop = square_crop(frame, detection.points, args.padding,
                                   flip=args.canonicalize_left and detection.label == "Left")
                if crop is not None:
                    start = perf_counter()
                    prediction = predictor.predict(crop.image)
                    inference_ms = (perf_counter()-start)*1000
                    points = uv_to_frame(prediction["uv"], crop)
                    if args.show_joints:
                        skeleton(view, points)
                    # An optional reflected crop requires reflected X in relative 3D.
                    if args.canonicalize_left and detection.label == "Left":
                        prediction["joints"][:, 0] *= -1
                        prediction["vertices"][:, 0] *= -1
                    x1, y1, x2, y2 = map(int, crop.box)
                    cv2.rectangle(view, (x1, y1), (x2, y2), (80, 190, 80), 1)
                    status = f"SimpleHand 21 joints / 778 vertices | {detection.label}"
            if viewer is not None:
                render_start = perf_counter()
                viewer.update(prediction["vertices"] if prediction is not None else None)
                viewer.render(fps, inference_ms)
                render_ms = (perf_counter()-render_start)*1000
            else:
                render_ms = 0.
            if args.mirror:
                view = cv2.flip(view, 1)
            cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.putText(view, f"Loop {fps:.1f} FPS | MP {detection_ms:.1f} ms | SH {inference_ms:.1f} ms",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            if recorder is not None:
                recorder.write(view, viewer.capture_bgr(), perf_counter())
            cv2.imshow("SimpleHand webcam", view)
            key = cv2.waitKey(1) & 0xFF
            if viewer is not None:
                if key == ord("w"):
                    viewer.wireframe = not viewer.wireframe
                elif key == ord("r"):
                    viewer.reset()
            now = perf_counter()
            instantaneous = 1 / max(now-last_time, 1e-6)
            fps = instantaneous if fps == 0 else .9*fps + .1*instantaneous
            last_time = now
            stats.add(now-frame_start, detection_ms, inference_ms, render_ms, prediction is not None)
            if key in (27, ord("q")):
                break
            if (key == ord("s") or "save" in actions) and prediction is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                path = args.save_dir / f"prediction_{perf_counter():.6f}.npz"
                mesh_faces = predictor.faces
                if args.canonicalize_left and detection.label == "Left":
                    mesh_faces = mesh_faces[:, [0, 2, 1]]
                np.savez_compressed(path, uv=points, uv_crop=prediction["uv"],
                                    joints=prediction["joints"], vertices=prediction["vertices"],
                                    faces=mesh_faces, label=detection.label, inverse_crop=crop.inverse)
                export_obj(path.with_suffix(".obj"), prediction["vertices"], mesh_faces)
                cv2.imwrite(str(path.with_suffix(".jpg")), frame)
                if viewer is not None:
                    viewer.save_screenshot(path.with_name(path.stem+"_mesh.png"))
                print(f"Saved actual prediction: {path}, {path.with_suffix('.obj')}")
    finally:
        camera.release()
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()
        path = stats.save(args.save_dir, {
            "checkpoint": str(args.checkpoint), "device": args.device, "amp": args.amp,
            "source": str(args.video) if args.video else f"webcam:{args.camera}",
            "mesh": args.view3d, "recording": str(args.record) if args.record else None,
            "canonicalize_left": args.canonicalize_left,
        })
        if path:
            print(f"Measured session statistics: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Stopped by user; camera and render resources released.")
    except (OSError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
