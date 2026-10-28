from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import main as app
from utils.mesh_viewer import vertex_normals, export_obj
from utils.session import SessionStats, DemoRecorder, compose_demo


class ApplicationTests(unittest.TestCase):
    def args(self):
        with patch.object(sys, "argv", ["main.py"]):
            return app.arguments()

    def test_mesh_is_default(self):
        args = self.args()
        self.assertTrue(args.view3d)
        self.assertFalse(args.show_joints)

    def test_normals_and_export(self):
        vertices = np.array([[0.,0,0], [1.,0,0], [0.,1,0]], np.float32)
        faces = np.array([[0,1,2]], np.uint32)
        np.testing.assert_allclose(vertex_normals(vertices, faces), [[0,0,1]]*3)
        with TemporaryDirectory() as folder:
            path = Path(folder)/"sample.obj"
            export_obj(path, vertices, faces)
            lines = path.read_text().splitlines()
            self.assertEqual(sum(line.startswith("v ") for line in lines), 3)
            self.assertIn("f 1 2 3", lines)

    def test_no_hand_clears_mesh_and_skips_inference(self):
        args = self.args()
        camera, detector, viewer = [MagicMock() for _ in range(3)]
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((480,640,3), np.uint8))
        detector.detect.return_value = None
        viewer.events.return_value = set()
        with TemporaryDirectory() as folder:
            args.save_dir = Path(folder)
            with patch.object(app, "arguments", return_value=args), \
                 patch.object(app, "SimpleHandPredictor") as predictor, \
                 patch.object(app, "HandDetector", return_value=detector), \
                 patch.object(app, "MeshViewer", return_value=viewer), \
                 patch.object(app.cv2, "VideoCapture", return_value=camera), \
                 patch.object(app.cv2, "imshow"), \
                 patch.object(app.cv2, "waitKey", return_value=ord("q")), \
                 patch.object(app.cv2, "destroyAllWindows"):
                app.main()
            predictor.return_value.predict.assert_not_called()
            viewer.update.assert_called_once_with(None)
            viewer.close.assert_called_once()
            camera.release.assert_called_once()
            self.assertEqual(len(list(Path(folder).glob("session_*.json"))), 1)

    def test_metrics_separate_active_and_empty_frames(self):
        stats = SessionStats()
        stats.add(.1, 5., 20., 2., True)
        stats.add(.01, 5., 0., 2., False)
        summary = stats.summary()
        self.assertAlmostEqual(summary["loop_fps_when_prediction_present"], 10.)
        self.assertEqual(summary["mean_simplehand_ms_when_present"], 20.)
        self.assertEqual(summary["tracked_frame_fraction"], .5)

    def test_recording_preserves_time_and_has_both_views(self):
        webcam = np.full((480,640,3), 40, np.uint8)
        mesh = np.full((620,720,3), 180, np.uint8)
        composed = compose_demo(webcam, mesh)
        self.assertEqual(composed.shape, (620,1360,3))
        self.assertGreater(composed[:,640:].mean(), composed[:,:640].mean())
        with TemporaryDirectory() as folder:
            path = Path(folder)/"demo.mp4"
            record = DemoRecorder(path, fps=20.)
            record.write(webcam, mesh, 100.)
            record.write(webcam, mesh, 100.25)
            record.close()
            video = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
            finally:
                video.release()
            with self.assertRaises(FileExistsError):
                DemoRecorder(path)

    def test_recording_needs_mesh(self):
        with patch.object(sys, "argv", ["main.py", "--no-view3d", "--record", "new.mp4"]):
            with self.assertRaises(SystemExit):
                app.arguments()


if __name__ == "__main__":
    unittest.main()
