"""Uji integrasi Streamlit menggunakan gambar unggahan sintetis."""

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


def upload(label, **kwargs):
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    image[40:120, 60:180] = (255, 100, 40)
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    output.seek(0)
    return output


class AppTests(unittest.TestCase):
    def test_no_upload_prompts_for_image(self):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertIn("Upload gambar", app.info[0].value)

    @patch("streamlit.file_uploader", side_effect=upload)
    def test_all_features_produce_download(self, _upload):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        for feature in ("Thresholding", "Color masking", "Morfologi", "Deteksi tepi", "Sketch", "Watermark"):
            app.selectbox[0].select(feature).run()
            self.assertFalse(app.exception, feature)
            self.assertFalse(app.error, feature)
            self.assertEqual(len(app.get("download_button")), 1, feature)
        app.radio[0].set_value("Logo").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("download_button")), 1)

    @patch("streamlit.file_uploader", side_effect=upload)
    def test_threshold_and_edge_method_controls(self, _upload):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        for method in ("Adaptive", "Otsu"):
            app.selectbox(key="threshold_method").select(method).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.get("download_button")), 1)
        app.selectbox[0].select("Deteksi tepi").run()
        for method in ("sobel", "prewitt", "laplacian", "canny"):
            app.selectbox[1].select(method).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)

    @patch("streamlit.file_uploader", return_value=BytesIO(b"invalid image"))
    def test_invalid_upload_shows_error(self, _upload):
        app = AppTest.from_file(str(APP_DIR / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertIn("Gambar tidak dapat dibaca", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
