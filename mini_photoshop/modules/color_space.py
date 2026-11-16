"""
Modul pengolahan ruang warna untuk PicaLab.
(Format masukan: BGR OpenCV, sesuai pipeline app.py)

Fitur:
1. to_grayscale       : Mengubah citra BGR menjadi grayscale.
2. adjust_saturation  : Mengatur tingkat saturasi warna.
3. adjust_hue         : Menggeser corak warna (satuan OpenCV, 1 unit = 2 derajat).
4. adjust_hue_degrees : Menggeser corak warna dalam satuan derajat (-180..180).
5. decompose_rgb      : Memisahkan channel Red, Green, dan Blue.
6. channel_view       : Menampilkan satu channel (grayscale atau berwarna/tinted).
7. rgb_channel_grid   : Grid 2x2 (Asli | R | G | B) untuk visualisasi dekomposisi.
8. invert_colors      : Membalik nilai intensitas piksel (Negatif).
"""

import cv2
import numpy as np

# Indeks channel pada array BGR OpenCV
_CHANNEL_INDEX = {"B": 0, "G": 1, "R": 2}


def _image(image):
    """Memvalidasi format dan tipe data citra."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Citra harus berupa array NumPy uint8.")

    if image.size == 0 or not (
        image.ndim == 2 or
        (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("Citra harus grayscale (H, W) atau BGR (H, W, 3).")

    return image


def _color_image(image):
    """Memastikan citra memiliki format warna (3 Channel)."""
    image = _image(image)

    if image.ndim != 3:
        raise ValueError("Operasi ini membutuhkan citra berwarna (3 Channel).")

    return image


def to_grayscale(image):
    """
    Mengubah citra BGR menjadi grayscale.
    """
    image = _image(image)

    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_saturation(image, factor=1.0):
    """
    Mengatur tingkat saturasi warna menggunakan ruang warna HSV.
    factor 0 = grayscale, 1 = tanpa perubahan, > 1 = lebih jenuh.
    """
    image = _color_image(image)

    if (
        not isinstance(factor, (int, float, np.integer, np.floating))
        or not np.isfinite(factor)
        or factor < 0
    ):
        raise ValueError("Nilai saturasi harus berupa angka >= 0.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1].astype(np.float32)
    hsv[:, :, 1] = np.clip(saturation * float(factor), 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_hue(image, shift=0):
    """
    Menggeser corak warna menggunakan ruang warna HSV.
    shift dalam satuan OpenCV (Hue 0..179, 1 unit = 2 derajat), rentang -180..180.
    """
    image = _color_image(image)

    if isinstance(shift, (bool, np.bool_)) or not isinstance(
        shift, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Nilai hue harus berupa angka.")
    shift = int(round(float(shift)))
    if not -180 <= shift <= 180:
        raise ValueError("Nilai hue harus -180 sampai 180 (satuan OpenCV).")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0].astype(np.int16)
    hsv[:, :, 0] = ((hue + shift) % 180).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_hue_degrees(image, degrees=0):
    """
    Menggeser corak warna dalam satuan derajat roda warna (-180..180).
    OpenCV menyimpan Hue 0..179 (setengah derajat), sehingga dibagi 2.
    """
    if not -180 <= float(degrees) <= 180:
        raise ValueError("Pergeseran hue harus -180 sampai 180 derajat.")
    return adjust_hue(image, int(round(float(degrees) / 2.0)))


def decompose_rgb(image):
    """
    Memisahkan citra BGR menjadi tiga channel.
    Mengembalikan Dictionary {"R", "G", "B"} berisi array grayscale per channel.
    """
    image = _color_image(image)

    return {
        "R": image[:, :, 2].copy(),
        "G": image[:, :, 1].copy(),
        "B": image[:, :, 0].copy(),
    }


def channel_view(image, channel="R", tinted=True):
    """
    Menampilkan satu channel saja.
    tinted=True  -> channel lain di-nol-kan sehingga tampil berwarna (merah/hijau/biru).
    tinted=False -> intensitas channel sebagai citra grayscale.
    """
    image = _color_image(image)
    if channel not in _CHANNEL_INDEX:
        raise ValueError("Channel harus 'R', 'G', atau 'B'.")

    idx = _CHANNEL_INDEX[channel]
    if not tinted:
        return image[:, :, idx].copy()

    out = np.zeros_like(image)
    out[:, :, idx] = image[:, :, idx]
    return out


def rgb_channel_grid(image, max_side=900):
    """
    Membuat grid 2x2 berlabel: [Asli | Red] / [Green | Blue] untuk visualisasi dekomposisi.
    Diperkecil hingga max_side agar ringan ditampilkan.
    """
    image = _color_image(image)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)

    tiles = [
        ("Asli", image.copy()),
        ("Red", channel_view(image, "R")),
        ("Green", channel_view(image, "G")),
        ("Blue", channel_view(image, "B")),
    ]
    font_scale = max(0.5, image.shape[1] / 600)
    for label, tile in tiles:
        cv2.rectangle(tile, (0, 0), (int(110 * font_scale), int(34 * font_scale)), (20, 20, 20), -1)
        cv2.putText(tile, label, (int(8 * font_scale), int(24 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (255, 255, 255),
                    max(1, int(font_scale * 1.5)), cv2.LINE_AA)

    top = cv2.hconcat([tiles[0][1], tiles[1][1]])
    bottom = cv2.hconcat([tiles[2][1], tiles[3][1]])
    return cv2.vconcat([top, bottom])


def invert_colors(image):
    """
    Membuat efek negatif dengan membalik intensitas piksel.
    """
    image = _image(image)

    return 255 - image
