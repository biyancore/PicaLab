"""
Modul efek kreatif untuk PicaLab (format masukan BGR/grayscale uint8).

Fitur:
1. cartoon_filter : Efek kartun/komik (penghalusan warna + garis tepi tebal).
2. vignette_effect: Bayangan gelap di sudut-sudut gambar (fokus ke tengah).
3. sepia_effect   : Nuansa foto tua kecokelatan.
4. pixelate_effect: Efek mozaik/piksel besar (retro / sensor area).
5. emboss_effect  : Efek timbul (relief) seperti ukiran logam.
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
        raise ValueError("Citra harus grayscale (H, W) atau BGR (H, W, 3).")

    return image


def cartoon_filter(image, line_size=9, blur_value=5, smooth_passes=3):
    """
    Memberikan efek kartun pada gambar.
    Cara kerja:
    1. Ekstrak garis tepi (edges) menggunakan Adaptive Thresholding.
    2. Haluskan warna gambar asli menggunakan Bilateral Filter agar tampak seperti lukisan.
    3. Gabungkan warna halus dengan garis tepi yang tajam.

    line_size     : ukuran blok adaptive threshold (ganjil, makin besar makin tebal garis).
    blur_value    : median blur sebelum deteksi tepi (ganjil, mengurangi noise garis).
    smooth_passes : jumlah iterasi bilateral (makin banyak makin "flat").
    """
    image = _image(image)
    line_size = max(3, int(line_size) | 1)
    blur_value = max(3, int(blur_value) | 1)
    smooth_passes = max(1, int(smooth_passes))

    is_gray = image.ndim == 2
    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 1. Garis tepi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, line_size, 9
    )

    # 2. Penghalusan warna. Optimasi: bilateral dijalankan pada resolusi setengah
    #    untuk citra besar (>1 MP) lalu diperbesar lagi — hasil visual hampir sama,
    #    tetapi 3-4x lebih cepat.
    h, w = image.shape[:2]
    small = image if h * w <= 1_000_000 else cv2.resize(image, (w // 2, h // 2),
                                                          interpolation=cv2.INTER_AREA)
    color = small
    for _ in range(smooth_passes):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. Gabungkan warna halus dengan garis tepi
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    if is_gray:
        return cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
    return cartoon


def vignette_effect(image, strength=1.0):
    """
    Memberikan efek vignette (gelap di ujung, terang di tengah).
    Cara kerja: membuat kernel Gaussian 2D seukuran gambar, lalu mengalikannya
    dengan nilai piksel gambar asli.

    strength: 0 (tanpa efek) .. 2 (sangat kuat). 1.0 = perilaku versi awal.
    """
    image = _image(image)
    strength = float(np.clip(strength, 0.0, 2.0))
    if strength == 0:
        return image.copy()

    rows, cols = image.shape[:2]

    # Sigma makin kecil -> vignette makin kuat
    sigma_x = cols / (1.0 + strength * 1.5)
    sigma_y = rows / (1.0 + strength * 1.5)
    kernel_x = cv2.getGaussianKernel(cols, sigma_x)
    kernel_y = cv2.getGaussianKernel(rows, sigma_y)
    kernel_2d = kernel_y * kernel_x.T
    mask = (kernel_2d / kernel_2d.max()).astype(np.float32)

    # Vektorisasi (tanpa loop per channel)
    if image.ndim == 3:
        vignette = image.astype(np.float32) * mask[:, :, None]
    else:
        vignette = image.astype(np.float32) * mask

    return np.clip(vignette, 0, 255).astype(np.uint8)


def sepia_effect(image, intensity=1.0):
    """
    Efek sepia klasik menggunakan matriks transformasi warna (dalam urutan BGR).
    intensity 0..1 mencampur hasil sepia dengan citra asli.
    """
    image = _image(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Matriks sepia standar (RGB) dikonversi ke urutan BGR
    kernel = np.array([
        [0.131, 0.534, 0.272],  # B' = f(B, G, R)
        [0.168, 0.686, 0.349],  # G'
        [0.189, 0.769, 0.393],  # R'
    ], dtype=np.float32)
    sepia = np.clip(cv2.transform(image.astype(np.float32), kernel), 0, 255).astype(np.uint8)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return cv2.addWeighted(sepia, intensity, image, 1.0 - intensity, 0)


def pixelate_effect(image, block_size=12):
    """
    Efek pixelate/mozaik: perkecil citra dengan INTER_AREA lalu perbesar dengan
    INTER_NEAREST sehingga terbentuk blok-blok piksel.
    """
    image = _image(image)
    block_size = max(2, int(block_size))
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def emboss_effect(image, strength=1.0):
    """
    Efek emboss (relief) menggunakan kernel konvolusi arah diagonal.
    Hasil berupa citra grayscale bernuansa logam.
    """
    image = _image(image)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s = float(np.clip(strength, 0.1, 3.0))
    kernel = np.array([[-2 * s, -s, 0], [-s, 1, s], [0, s, 2 * s]], dtype=np.float32)
    embossed = cv2.filter2D(gray.astype(np.float32), -1, kernel) + 128
    return np.clip(embossed, 0, 255).astype(np.uint8)
