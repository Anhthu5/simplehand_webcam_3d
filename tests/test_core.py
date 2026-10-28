"""Run python -m unittest discover -s tests -v. No camera/checkpoint needed."""
import unittest
import cv2
import numpy as np
from evaluate import align_similarity, pa_error
from utils.geometry import square_crop, uv_to_frame


class CoreTests(unittest.TestCase):
    def test_inverse_crop_at_boundary_and_reflection(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        points = np.array([[0, 0], [30, 45]], dtype=np.float32)
        for flip in (False, True):
            crop = square_crop(image, points, flip=flip)
            forward = cv2.invertAffineTransform(crop.inverse)
            in_crop = np.column_stack((points, np.ones(2))) @ forward.T
            np.testing.assert_allclose(uv_to_frame(in_crop/224, crop), points, atol=1e-5)
            self.assertEqual(crop.image.shape, (224, 224, 3))

    def test_degenerate_crop(self):
        image = np.zeros((100, 100, 3), np.uint8)
        self.assertIsNone(square_crop(image, np.ones((21, 2))))

    def test_pa_known_similarity(self):
        rng = np.random.default_rng(13)
        gt = rng.normal(size=(21, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        pred = 2.7*gt@q + [4, -5, 6]
        np.testing.assert_allclose(align_similarity(pred, gt), gt, atol=1e-12)
        np.testing.assert_allclose(align_similarity(pred, gt, True), gt, atol=1e-12)
        self.assertLess(pa_error(pred[None], gt[None])[0], 1e-12)

    def test_reflections_and_zero_spread(self):
        rng = np.random.default_rng(4)
        gt = rng.normal(size=(21, 3))
        pred = gt * [-1, 1, 1]
        np.testing.assert_allclose(align_similarity(pred, gt), gt, atol=1e-12)
        self.assertGreater(np.linalg.norm(align_similarity(pred, gt, True)-gt), .1)
        with self.assertRaises(ValueError):
            align_similarity(np.zeros_like(gt), gt)


if __name__ == "__main__":
    unittest.main()
