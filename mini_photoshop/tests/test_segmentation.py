"""Jalankan: python -m unittest discover -s mini_photoshop/tests -v"""

import unittest

import numpy as np

from mini_photoshop.modules import segmentation as seg


class SegmentationTests(unittest.TestCase):
    def test_threshold_and_inverse(self):
        image = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        expected = np.array([[0, 0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(seg.global_threshold(image), expected)
        np.testing.assert_array_equal(seg.global_threshold(image, invert=True), 255 - expected)

    def test_otsu_separates_two_levels(self):
        image = np.tile(np.array([20, 20, 200, 200], dtype=np.uint8), (8, 1))
        expected = np.where(image == 20, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(seg.otsu_threshold(image), expected)
        np.testing.assert_array_equal(seg.otsu_threshold(image, invert=True), 255 - expected)

    def test_adaptive_follows_local_brightness(self):
        image = np.zeros((21, 42), dtype=np.uint8)
        image[:, 21:] = 200
        for method in ("mean", "gaussian"):
            result = seg.adaptive_threshold(image, method, block_size=5)
            self.assertEqual(result[10, 5], 255)
            self.assertEqual(result[10, 30], 255)
            np.testing.assert_array_equal(
                seg.adaptive_threshold(image, method, block_size=5, invert=True),
                255 - result,
            )

    def test_mask_uses_rgb_and_wrapped_red_hue(self):
        image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 85]]], dtype=np.uint8)
        original = image.copy()
        mask, selected = seg.color_masking(image, (170, 100, 100), (10, 255, 255))
        np.testing.assert_array_equal(mask, [[255, 0, 0, 255]])
        np.testing.assert_array_equal(selected[:, 1:3], 0)
        np.testing.assert_array_equal(selected[:, [0, 3]], image[:, [0, 3]])
        np.testing.assert_array_equal(image, original)
        full_mask, full_image = seg.color_masking(image)
        np.testing.assert_array_equal(full_mask, 255)
        np.testing.assert_array_equal(full_image, image)

    def test_morphology_changes_binary_shapes(self):
        point = np.zeros((9, 9), dtype=np.uint8)
        point[4, 4] = 255
        expanded = seg.morphology(point, "dilation")
        self.assertEqual(np.count_nonzero(expanded), 9)
        np.testing.assert_array_equal(seg.morphology(expanded, "erosion"), point)
        self.assertFalse(seg.morphology(point, "opening").any())
        square = np.zeros((9, 9), dtype=np.uint8)
        square[2:7, 2:7] = 255
        square[4, 4] = 0
        self.assertEqual(seg.morphology(square, "closing")[4, 4], 255)
        np.testing.assert_array_equal(point[4, 4], 255)

    def test_edges_detect_boundary_and_leave_flat_region(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, 16:] = 255
        original = image.copy()
        for method in ("sobel", "prewitt", "laplacian", "canny"):
            edges = seg.edge_detection(image, method)
            self.assertEqual(edges.dtype, np.uint8)
            self.assertEqual(edges.shape, (32, 32))
            self.assertTrue(edges[:, 14:18].any())
            self.assertFalse(edges[:, :10].any())
        np.testing.assert_array_equal(image, original)

    def test_sketch_uniform_images_are_white(self):
        for level in (0, 100, 255):
            image = np.full((16, 16, 3), level, dtype=np.uint8)
            np.testing.assert_array_equal(seg.sketch(image), 255)

    def test_logo_opacity_composes_with_alpha(self):
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = image.copy()
        logo = np.full((2, 2, 4), 255, dtype=np.uint8)
        logo[:, :, 3] = 128
        result = seg.logo_watermark(image, logo, opacity=0.5, position="top-left", margin=1)
        np.testing.assert_array_equal(result[1:3, 1:3], 139)
        np.testing.assert_array_equal(result[0], image[0])
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(seg.logo_watermark(image, logo, opacity=0, margin=0), image)

    def test_text_watermark_changes_only_output(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = seg.text_watermark(image, "PicaLab", opacity=1, position="center")
        self.assertTrue(result.any())
        self.assertFalse(image.any())
        np.testing.assert_array_equal(seg.text_watermark(image, "PicaLab", opacity=0), image)

    def test_invalid_inputs_raise_value_error(self):
        image = np.zeros((20, 20), dtype=np.uint8)
        invalid_calls = [
            lambda: seg.global_threshold(image.astype(float)),
            lambda: seg.global_threshold(image, 256),
            lambda: seg.global_threshold(np.zeros((0, 0), dtype=np.uint8)),
            lambda: seg.adaptive_threshold(image, block_size=4),
            lambda: seg.adaptive_threshold(image, method="invalid"),
            lambda: seg.color_masking(image),
            lambda: seg.morphology(image + 1),
            lambda: seg.morphology(image, iterations=0),
            lambda: seg.edge_detection(image, method="invalid"),
            lambda: seg.edge_detection(image, low_threshold=200, high_threshold=100),
            lambda: seg.sketch(image, blur_size=2),
            lambda: seg.text_watermark(image, ""),
            lambda: seg.text_watermark(image, "too large for image"),
            lambda: seg.logo_watermark(image, np.zeros((2, 2, 4), dtype=np.uint8), opacity=2),
        ]
        for call in invalid_calls:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
