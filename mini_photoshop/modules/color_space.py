"""
Modul pengolahan ruang warna untuk PicaLab.

Fitur:
1. Grayscale          : Mengubah citra RGB menjadi grayscale.
2. Saturation         : Mengatur tingkat saturasi warna.
3. Hue                : Menggeser corak warna.
4. RGB Decomposition  : Memisahkan channel Red, Green, dan Blue.
5. Negative           : Membalik nilai intensitas piksel.

Standar input:
- Array NumPy dengan tipe data uint8.
- Format grayscale (H, W) atau RGB (H, W, 3).

Standar output:
- Array NumPy uint8 baru.
- Citra masukan tidak diubah.
"""

import cv2
import numpy as np


def _image(image):
    """Memvalidasi format dan tipe data citra."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Citra harus berupa array NumPy uint8.")

    if image.size == 0 or not (
        image.ndim == 2 or
        (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError(
            "Citra harus grayscale (H, W) atau RGB (H, W, 3)."
        )

    return image


def _rgb_image(image):
    """Memastikan citra memiliki format RGB."""
    image = _image(image)

    if image.ndim != 3:
        raise ValueError("Operasi ini membutuhkan citra RGB.")

    return image


def grayscale(image):
    """
    Mengubah citra RGB menjadi grayscale.

    Parameter:
        image: Citra RGB uint8.

    Return:
        Citra grayscale uint8 dengan ukuran (H, W).
    """
    image = _image(image)

    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def adjust_saturation(image, value=1.0):
    """
    Mengatur tingkat saturasi warna menggunakan ruang warna HSV.

    Parameter:
        image: Citra RGB uint8.
        value: Faktor saturasi.
               0.0 = tanpa saturasi (grayscale).
               1.0 = saturasi asli.
               Lebih dari 1.0 = warna lebih pekat.

    Return:
        Citra RGB uint8.
    """
    image = _rgb_image(image)

    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("Nilai saturasi harus berupa angka >= 0.")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1].astype(np.float32)
    saturation = np.clip(saturation * float(value), 0, 255)

    hsv[:, :, 1] = saturation.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_hue(image, value=0):
    """
    Menggeser corak warna menggunakan ruang warna HSV.

    Parameter:
        image: Citra RGB uint8.
        value: Besar pergeseran hue dalam satuan OpenCV.
               Rentang yang digunakan -179 sampai 179.
               Nilai 0 = warna asli.

    Return:
        Citra RGB uint8.
    """
    image = _rgb_image(image)

    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or not -179 <= value <= 179
    ):
        raise ValueError("Nilai hue harus bilangan bulat -179 sampai 179.")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hue = hsv[:, :, 0].astype(np.int16)
    hue = (hue + int(value)) % 180

    hsv[:, :, 0] = hue.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb_decomposition(image):
    """
    Memisahkan citra RGB menjadi tiga channel.

    Parameter:
        image: Citra RGB uint8.

    Return:
        Tuple berisi:
        - red   : Channel merah dalam grayscale.
        - green : Channel hijau dalam grayscale.
        - blue  : Channel biru dalam grayscale.
    """
    image = _rgb_image(image)

    red = image[:, :, 0].copy()
    green = image[:, :, 1].copy()
    blue = image[:, :, 2].copy()

    return red, green, blue


def negative(image):
    """
    Membuat efek negatif dengan membalik intensitas piksel.

    Rumus:
        Negative = 255 - nilai piksel

    Parameter:
        image: Citra RGB atau grayscale uint8.

    Return:
        Citra negatif uint8.
    """
    image = _image(image)

    return 255 - image