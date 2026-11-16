"""
Modul transformasi geometris & ukuran untuk PicaLab.

Semua fungsi menerima array NumPy uint8 (grayscale (H, W) atau BGR (H, W, 3))
dan mengembalikan array baru tanpa mengubah masukan.
"""

import cv2
import numpy as np

# Mapping metode interpolasi agar mudah dipanggil dari dropdown Streamlit
INTERPOLATION_METHODS = {
    "Nearest Neighbor": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
    "Area": cv2.INTER_AREA,  # Terbaik untuk memperkecil (downscale)
}

# Batas aman agar resize/resample tidak menghabiskan RAM server
MAX_OUTPUT_SIDE = 8000


def _check_output_size(width: int, height: int) -> None:
    if width > MAX_OUTPUT_SIDE or height > MAX_OUTPUT_SIDE:
        raise ValueError(
            f"Ukuran hasil {width}×{height}px melebihi batas {MAX_OUTPUT_SIDE}px per sisi."
        )


def resize_image(image: np.ndarray, width: int, height: int, method: str = "Bilinear") -> np.ndarray:
    """
    Mengubah dimensi ukuran lebar dan tinggi citra dengan pilihan metode interpolasi.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return image.copy()
    _check_output_size(width, height)

    interp = INTERPOLATION_METHODS.get(method, cv2.INTER_LINEAR)
    return cv2.resize(image, (width, height), interpolation=interp)


def resample_scale(image: np.ndarray, scale_factor: float, method: str = "Bicubic") -> np.ndarray:
    """
    Peningkatan/Perubahan Resolusi (Resampling / Interpolasi) berdasarkan rasio skala.
    Contoh: scale_factor = 2.0 (perbesar 2x), scale_factor = 0.5 (perkecil separuh).
    """
    if scale_factor <= 0:
        return image.copy()

    h, w = image.shape[:2]
    new_w, new_h = max(1, round(w * scale_factor)), max(1, round(h * scale_factor))
    _check_output_size(new_w, new_h)

    interp = INTERPOLATION_METHODS.get(method, cv2.INTER_CUBIC)
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def crop_image(image: np.ndarray, x_start: int, y_start: int, x_end: int, y_end: int) -> np.ndarray:
    """
    Memotong area citra (Cropping) berdasarkan koordinat piksel (x_start, y_start) hingga (x_end, y_end).
    Dilengkapi pengaman batas (boundary clamping) dan koordinat terbalik otomatis ditukar.
    """
    h, w = image.shape[:2]

    # Tukar otomatis bila pengguna memasukkan koordinat akhir < awal
    x_start, x_end = sorted((int(x_start), int(x_end)))
    y_start, y_end = sorted((int(y_start), int(y_end)))

    # Pastikan koordinat berada di dalam rentang dimensi gambar
    x1 = max(0, min(x_start, w - 1))
    y1 = max(0, min(y_start, h - 1))
    x2 = max(x1 + 1, min(x_end, w))
    y2 = max(y1 + 1, min(y_end, h))

    return image[y1:y2, x1:x2].copy()


def flip_image(image: np.ndarray, mode: str = "Horizontal") -> np.ndarray:
    """
    Pencerminan citra (Flipping).
    Pilihan mode: 'Horizontal', 'Vertical', atau 'Both' (Keduanya).
    """
    codes = {"Horizontal": 1, "Vertical": 0, "Both": -1}
    if mode not in codes:
        return image.copy()
    return cv2.flip(image, codes[mode])


def rotate_image(image: np.ndarray, angle: float, keep_full_canvas: bool = True,
                 border_value=(0, 0, 0)) -> np.ndarray:
    """
    Memutar citra berdasarkan sudut derajat.
    Sudut positif = berlawanan arah jarum jam (konvensi OpenCV).
    Jika keep_full_canvas=True, ukuran kanvas otomatis menyesuaikan agar ujung gambar tidak terpotong.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Matriks rotasi 2D
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)

    if keep_full_canvas:
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(round((h * sin) + (w * cos)))
        new_h = int(round((h * cos) + (w * sin)))

        # Sesuaikan pergeseran titik tengah pada matriks translasi
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]
        size = (new_w, new_h)
    else:
        size = (w, h)

    return cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Fungsi internal untuk mengurutkan 4 titik koordinat secara otomatis:
    [0: Kiri-Atas, 1: Kanan-Atas, 2: Kanan-Bawah, 3: Kiri-Bawah]
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Kiri-Atas memiliki jumlah (x+y) terkecil, Kanan-Bawah memiliki jumlah terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Kanan-Atas memiliki selisih (y-x) terkecil, Kiri-Bawah memiliki selisih terbesar
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(image: np.ndarray, pts: list) -> np.ndarray:
    """
    Koreksi Geometrik Perspektif (meluruskan objek/dokumen yang miring).
    Parameter `pts` menerima list berisi 4 pasang koordinat [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
    """
    pts_array = np.array(pts, dtype="float32")
    if pts_array.shape != (4, 2):
        raise ValueError("Koreksi perspektif membutuhkan tepat 4 titik koordinat (x, y).")

    rect = _order_points(pts_array)
    (tl, tr, br, bl) = rect

    # Pengaman: titik duplikat membuat matriks perspektif singular
    if len({tuple(p) for p in rect.tolist()}) < 4:
        raise ValueError("Keempat titik perspektif harus berbeda satu sama lain.")

    # Hitung lebar & tinggi maksimal objek baru menggunakan jarak Euclidean
    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    if max_width < 2 or max_height < 2:
        raise ValueError("Area perspektif terlalu kecil. Atur ulang posisi titik.")
    _check_output_size(max_width, max_height)

    # Koordinat tujuan dari sudut pandang lurus (Bird's Eye View)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def draw_perspective_points(image: np.ndarray, pts: list) -> np.ndarray:
    """
    Fungsi bantu visual (Preview): Menggambar titik sudut, label, dan garis bantu
    di atas citra asli saat pengguna mengatur titik perspektif di Streamlit.
    """
    preview = _to_bgr(image)
    pts_array = np.array(pts, dtype="float32")
    rect = _order_points(pts_array).astype(np.int32)

    thickness = max(2, int(min(preview.shape[:2]) / 300))
    radius = thickness * 4

    # Area terpilih diberi overlay transparan agar mudah terlihat
    overlay = preview.copy()
    cv2.fillPoly(overlay, [rect], (89, 122, 255))
    preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)

    cv2.polylines(preview, [rect], isClosed=True, color=(0, 220, 255), thickness=thickness)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), rect):
        cv2.circle(preview, (int(x), int(y)), radius, (89, 122, 255), -1)
        cv2.circle(preview, (int(x), int(y)), radius, (255, 255, 255), max(1, thickness // 2))
        cv2.putText(preview, label, (int(x) + radius + 4, int(y) - radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, (255, 255, 255), max(1, thickness // 2),
                    cv2.LINE_AA)

    return preview


def draw_crop_box(image: np.ndarray, x_start: int, y_start: int, x_end: int, y_end: int) -> np.ndarray:
    """
    Fungsi bantu visual: menggelapkan area di luar kotak crop dan menggambar bingkai seleksi.
    """
    preview = _to_bgr(image)
    h, w = preview.shape[:2]
    x1, x2 = sorted((int(np.clip(x_start, 0, w)), int(np.clip(x_end, 0, w))))
    y1, y2 = sorted((int(np.clip(y_start, 0, h)), int(np.clip(y_end, 0, h))))

    dimmed = (preview * 0.35).astype(np.uint8)
    dimmed[y1:y2, x1:x2] = preview[y1:y2, x1:x2]

    thickness = max(2, int(min(h, w) / 300))
    cv2.rectangle(dimmed, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 220, 255), thickness)

    # Garis bantu rule-of-thirds di dalam kotak crop
    for i in (1, 2):
        gx = x1 + (x2 - x1) * i // 3
        gy = y1 + (y2 - y1) * i // 3
        cv2.line(dimmed, (gx, y1), (gx, y2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dimmed, (x1, gy), (x2, gy), (255, 255, 255), 1, cv2.LINE_AA)

    return dimmed
