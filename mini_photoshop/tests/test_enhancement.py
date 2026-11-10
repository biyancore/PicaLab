
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from modules import enhancement as enh  # noqa: E402


def sample(gray=False):
    image = np.full((120, 160, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 90), (200, 50, 50), -1)
    cv2.circle(image, (120, 60), 30, (40, 160, 220), -1)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if gray else image


def upload(label, **kwargs):
    output = BytesIO()
    Image.fromarray(sample()).save(output, format="PNG")
    output.seek(0)
    return output


class EnhancementFunctionTests(unittest.TestCase):
    def test_all_functions_keep_shape_and_dtype(self):
        for image in (sample(), sample(gray=True)):
            results = [
                enh.brightness_contrast(image, 20, 1.3), enh.gamma_correction(image, 1.5),
                enh.histogram_equalization(image), enh.clahe(image),
                enh.comprehensive_enhancement(image), enh.wiener_deblur(image),
                enh.wiener_deblur(image, "motion", 15, angle=30), enh.deblur_unsharp(image),
                enh.inpaint_damage(image), enh.inpaint_damage(image, "dark"),
            ]
            results += [enh.blur(image, m) for m in ("gaussian", "averaging", "median", "bilateral")]
            results += [enh.sharpen(image, m) for m in ("laplacian", "high_boost", "unsharp")]
            results += [enh.denoise(image, m) for m in ("nlm", "gaussian", "median", "bilateral")]
            for result in results:
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_input_is_not_modified(self):
        image = sample()
        original = image.copy()
        enh.comprehensive_enhancement(image)
        enh.histogram_equalization(image)
        enh.clahe(image)
        np.testing.assert_array_equal(image, original)

    def test_identity_values(self):
        image = sample()
        np.testing.assert_array_equal(enh.brightness_contrast(image, 0, 1.0), image)
        np.testing.assert_array_equal(enh.gamma_correction(image, 1.0), image)

    def test_median_removes_salt_and_pepper(self):
        clean = sample()
        noisy = clean.copy()
        mask = np.random.default_rng(0).random(clean.shape[:2])
        noisy[mask < 0.03], noisy[mask > 0.97] = 0, 255
        error = lambda a: np.abs(a.astype(int) - clean).mean()
        self.assertLess(error(enh.median_blur(noisy, 3)), error(noisy))

    def test_wiener_restores_gaussian_blur(self):
        clean = sample()
        blurred = cv2.GaussianBlur(clean, (0, 0), 2.0)
        restored = enh.wiener_deblur(blurred, "gaussian", 13, 2.0, k=0.001)
        error = lambda a: np.abs(a.astype(int) - clean).mean()
        self.assertLess(error(restored), error(blurred))
    def test_histogram_counts_all_pixels(self):
        image = sample()
        self.assertEqual(enh.histogram(image).shape, (256,))
        self.assertEqual(enh.histogram(image).sum(), image.shape[0] * image.shape[1])
        self.assertEqual(enh.channel_histograms(image).shape, (3, 256))
        self.assertTrue((enh.channel_histograms(image).sum(axis=1) == image.shape[0] * image.shape[1]).all())

    def test_histogram_figure_modes(self):
        image = sample()
        for mode in ("gray", "rgb"):
            figure = enh.histogram_figure(image, enh.histogram_equalization(image), mode)
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(figure.axes[0].get_xlabel(), "Intensitas piksel (0–255)")
        with self.assertRaises(ValueError):
            enh.histogram_figure(image, image, "x")

    def test_equalization_flattens_histogram(self):
        dull = np.clip(sample().astype(int) // 4 + 80, 0, 255).astype(np.uint8)
        gray = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        self.assertGreater(gray(enh.histogram_equalization(dull)).std(), gray(dull).std())

    def test_invalid_arguments_raise_value_error(self):
        image = sample()
        for call in (lambda: enh.gaussian_blur(image, 4), lambda: enh.clahe(image, 0),
                     lambda: enh.brightness_contrast(image, 999), lambda: enh.sharpen(image, "x"),
                     lambda: enh.blur(image, "x"), lambda: enh.denoise(image, "x"),
                     lambda: enh.wiener_deblur(image, "x"), lambda: enh.gamma_correction(image, 0),
                     lambda: enh.brightness_contrast(image.astype(float))):
            with self.assertRaises(ValueError):
                call()


class EnhancementAppTests(unittest.TestCase):
    @patch("streamlit.file_uploader", side_effect=upload)
    def test_all_enhancement_features_produce_download(self, _upload):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        for feature in ("Brightness dan Kontras", "Histogram Equalization", "Enhancement Komprehensif",
                        "Denoising", "Blur Suite", "Sharpening", "Deblurring dan Restorasi"):
            app.selectbox[0].select(feature).run()
            self.assertFalse(app.exception, feature)
            self.assertFalse(app.error, feature)
            self.assertEqual(len(app.get("download_button")), 1, feature)

    @patch("streamlit.file_uploader", side_effect=upload)
    def test_method_variants(self, _upload):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        variants = {
            "Blur Suite": ("enh_bl_method", ["Gaussian Blur", "Averaging Blur", "Median Blur", "Bilateral Blur"]),
            "Denoising": ("enh_dn_method", ["Non-Local Means", "Gaussian", "Median (salt & pepper)", "Bilateral"]),
            "Sharpening": ("enh_sh_method", ["Kernel Laplacian", "Kernel High-boost", "Unsharp Mask"]),
            "Deblurring dan Restorasi": ("enh_db_method", ["Wiener (blur umum)", "Wiener (blur gerak)",
                                                           "Unsharp deblur", "Inpainting goresan"]),
        }
        for feature, (key, methods) in variants.items():
            app.selectbox[0].select(feature).run()
            for method in methods:
                app.selectbox(key=key).select(method).run()
                self.assertFalse(app.exception, (feature, method))
                self.assertFalse(app.error, (feature, method))
                self.assertEqual(len(app.get("download_button")), 1, (feature, method))
        app.selectbox[0].select("Histogram Equalization").run()
        app.radio(key="enh_he_method").set_value("CLAHE").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    @patch("streamlit.file_uploader", side_effect=upload)
    def test_histogram_shown_for_equalization_and_optional_for_brightness(self, _upload):
        def has_histogram(app):
            return any(h.value == "Histogram sebelum dan sesudah" for h in app.subheader)
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        app.selectbox[0].select("Histogram Equalization").run()
        self.assertFalse(app.exception)
        self.assertTrue(has_histogram(app))
        app.radio(key="enh_hist_mode").set_value("Per kanal RGB").run()
        self.assertFalse(app.exception)
        app.selectbox[0].select("Brightness dan Kontras").run()
        self.assertTrue(has_histogram(app))
        app.checkbox(key="enh_show_hist").uncheck().run()
        self.assertFalse(has_histogram(app))
        app.selectbox[0].select("Blur Suite").run()
        self.assertFalse(has_histogram(app))


if __name__ == "__main__":
    unittest.main()