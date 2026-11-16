"""
Utilitas I/O & konversi citra untuk PicaLab (tanpa logika UI).

Revisi: versi lama memakai cv2.imdecode yang mengabaikan orientasi EXIF
(foto HP tampil miring) dan transparansi PNG, serta tidak dipakai oleh app.py.
Sekarang semua pemuatan gambar lewat modul ini.
"""

import base64
import hashlib
import pickle
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

# Lindungi server dari "decompression bomb" (gambar ratusan megapiksel)
Image.MAX_IMAGE_PIXELS = 120_000_000

EXPORT_FORMATS = {
    "PNG": {"ext": ".png", "mime": "image/png", "lossy": False},
    "JPEG": {"ext": ".jpg", "mime": "image/jpeg", "lossy": True},
    "WEBP": {"ext": ".webp", "mime": "image/webp", "lossy": True},
}


# ---------------------------------------------------------------------------
# Pemuatan gambar
# ---------------------------------------------------------------------------
def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Konversi PIL (mode apa pun) ke BGR uint8; transparansi diratakan ke putih."""
    if pil_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(pil_img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(background, rgba)

    rgb = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Perkecil citra agar sisi terpanjang <= max_side (tidak pernah memperbesar)."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def load_image_with_meta(uploaded_file, max_side: int = None):
    """
    Baca file upload Streamlit -> (citra BGR, metadata).
    Menangani orientasi EXIF, PNG transparan, palet, CMYK, dan gambar 16-bit.
    Raise ValueError dengan pesan ramah jika file rusak / bukan gambar.
    """
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not data:
        raise ValueError("File kosong.")
    try:
        with Image.open(BytesIO(data)) as src:
            fmt, mode = src.format, src.mode
            has_exif = bool(src.getexif())
            pil_img = ImageOps.exif_transpose(src)
            pil_img.load()
            orig_w, orig_h = pil_img.size
            image_bgr = _pil_to_bgr(pil_img)
    except Image.DecompressionBombError as exc:
        raise ValueError("Resolusi gambar terlalu besar untuk diproses dengan aman.") from exc
    except Exception as exc:
        raise ValueError(f"File tidak dapat dibaca sebagai gambar ({exc}).") from exc

    image_bgr = resize_max_side(image_bgr, max_side)
    meta = {
        "name": getattr(uploaded_file, "name", "gambar"),
        "format": fmt or "?",
        "mode": mode,
        "file_size": len(data),
        "original_size": (orig_w, orig_h),
        "working_size": (image_bgr.shape[1], image_bgr.shape[0]),
        "exif": has_exif,
        "downscaled": (image_bgr.shape[1], image_bgr.shape[0]) != (orig_w, orig_h),
    }
    return image_bgr, meta


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Membaca file gambar yang diunggah melalui st.file_uploader Streamlit
    dan mengonversinya menjadi matriks NumPy berformat BGR (standar OpenCV).
    """
    return load_image_with_meta(uploaded_file)[0]


def decode_logo(uploaded_file) -> np.ndarray:
    """
    Baca logo watermark sebagai RGBA uint8 (kontrak modul segmentation).
    Mendukung PNG transparan, JPG/WEBP, palet, grayscale, dan orientasi EXIF.
    """
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    try:
        with Image.open(BytesIO(data)) as src:
            return np.array(ImageOps.exif_transpose(src).convert("RGBA"))
    except Exception as exc:
        raise ValueError(f"Logo tidak dapat dibaca ({exc}).") from exc


def source_id(uploaded_file) -> str:
    """ID stabil untuk file upload (file_id Streamlit, fallback nama+ukuran)."""
    fid = getattr(uploaded_file, "file_id", None)
    if fid:
        return str(fid)
    return f"{getattr(uploaded_file, 'name', 'file')}:{getattr(uploaded_file, 'size', 0)}"


# ---------------------------------------------------------------------------
# Konversi warna
# ---------------------------------------------------------------------------
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Mengonversi matriks citra dari BGR (OpenCV) ke RGB (Streamlit/Web).
    Wajib dipanggil sebelum menampilkan gambar menggunakan st.image().
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Mengonversi matriks citra dari RGB ke BGR.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Pastikan citra 3 channel BGR uint8 (grayscale dinaikkan, BGRA diratakan)."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def hex_to_rgb(hex_color: str) -> tuple:
    """'#FF6A4D' -> (255, 106, 77)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(hex_color: str) -> tuple:
    """'#FF6A4D' -> (77, 106, 255)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ---------------------------------------------------------------------------
# Ekspor
# ---------------------------------------------------------------------------
def encode_image_for_download(image: np.ndarray, format_ext: str = ".png", quality: int = 95) -> bytes:
    """
    Mengonversi matriks citra (NumPy array, BGR/grayscale) menjadi bytes file gambar.
    Fungsi ini dipakai untuk tombol unduh (st.download_button).
    quality berlaku untuk JPEG/WEBP (1-100).
    """
    if not format_ext.startswith('.'):
        format_ext = f".{format_ext}"
    format_ext = format_ext.lower()
    quality = int(np.clip(quality, 1, 100))

    params = []
    if format_ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    elif format_ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif format_ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, encoded_image = cv2.imencode(format_ext, image, params)
    if not success:
        raise ValueError("Gagal melakukan encoding pada gambar.")
    return encoded_image.tobytes()


def image_to_data_uri(image: np.ndarray, max_side: int = 1400, quality: int = 85) -> str:
    """Citra -> data URI JPEG (diperkecil) untuk komponen HTML seperti slider pembanding."""
    small = resize_max_side(ensure_bgr(image), max_side)
    payload = base64.b64encode(encode_image_for_download(small, ".jpg", quality)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def params_fingerprint(params: dict) -> str:
    """Hash stabil dari parameter filter (termasuk array logo) untuk kunci cache."""
    def normalize(value):
        if isinstance(value, np.ndarray):
            return ("nd", value.shape, str(value.dtype), hashlib.md5(value.tobytes()).hexdigest())
        if isinstance(value, dict):
            return tuple(sorted((k, normalize(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(normalize(v) for v in value)
        return value

    return hashlib.md5(pickle.dumps(normalize(params))).hexdigest()


# ---------------------------------------------------------------------------
# Gambar contoh (untuk mencoba aplikasi tanpa file)
# ---------------------------------------------------------------------------
def generate_sample_image(width: int = 1280, height: int = 853) -> np.ndarray:
    """Pola uji berwarna: gradien langit, matahari, bukit, rumput, papan uji warna & teks."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = yy / height
    sky_top, sky_bottom = np.array([180, 90, 40]), np.array([150, 190, 255])  # BGR
    img = (sky_top[None, None, :] * (1 - t[..., None]) + sky_bottom[None, None, :] * t[..., None])

    sun = (xx - width * 0.72) ** 2 + (yy - height * 0.32) ** 2
    img[sun < (height * 0.11) ** 2] = (90, 215, 255)
    glow = np.exp(-sun / (2 * (height * 0.2) ** 2))[..., None]
    img = img * (1 - glow * 0.35) + np.array([120, 230, 255]) * glow * 0.35

    for i, (amp, base, color) in enumerate([(40, 0.62, (95, 120, 70)), (55, 0.72, (60, 140, 60))]):
        ridge = height * base + amp * np.sin(xx / (110 + 40 * i) + i)
        img[yy > ridge] = color
    img[yy > height * 0.84] = (50, 165, 95)

    img = np.clip(img, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))

    palette = [(60, 60, 230), (60, 200, 240), (80, 200, 80), (230, 160, 50), (200, 80, 160), (240, 240, 240)]
    sw = width // 22
    for i, col in enumerate(palette):
        x0 = int(width * 0.05) + i * (sw + 8)
        cv2.rectangle(img, (x0, int(height * 0.07)), (x0 + sw, int(height * 0.07) + sw), col, -1)
    cv2.putText(img, "PicaLab", (int(width * 0.05), int(height * 0.95)), cv2.FONT_HERSHEY_DUPLEX,
                width / 700, (255, 255, 255), max(2, width // 500), cv2.LINE_AA)
    return img
