"""Fungsi pengolahan citra bagian Andini, tanpa dependensi Streamlit.

Masukan: array NumPy uint8, grayscale (H, W) atau RGB (H, W, 3).
Keluaran selalu array uint8 baru; gambar masukan tidak diubah. Mask dan
hasil threshold memakai 0 untuk latar dan 255 untuk objek.

Catatan revisi: kontrak RGB modul ini DIPERTAHANKAN (sesuai tests/test_segmentation.py).
Bug lama ada di app.py yang mengirim citra BGR langsung ke sini sehingga color
masking memilih warna tertukar (merah <-> biru) dan logo watermark berubah warna.
Sekarang utils/pipeline.py mengonversi BGR -> RGB sebelum memanggil modul ini.
Fitur baru (auto_binarize, fit, scale, stroke) bersifat opsional; nilai default
mempertahankan perilaku & validasi asli.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Citra harus berupa array NumPy uint8.")
    if image.size == 0 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("Citra harus grayscale (H, W) atau RGB (H, W, 3).")
    return image


def _gray(image):
    image = _image(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def _odd_size(value, minimum=3):
    if not isinstance(value, (int, np.integer)) or value < minimum or value % 2 == 0:
        raise ValueError(f"Ukuran kernel harus bilangan ganjil minimal {minimum}.")
    return int(value)


def global_threshold(image, threshold=127, invert=False):
    """Threshold global; piksel > threshold menjadi putih, kecuali invert."""
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold harus dalam rentang 0–255.")
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(_gray(image), threshold, 255, mode)[1]


def adaptive_threshold(image, method="gaussian", block_size=11, c=2, invert=False):
    """Threshold lokal: method 'mean' atau 'gaussian'; c mengurangi ambang lokal."""
    methods = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C,
               "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
    if method not in methods:
        raise ValueError("Metode adaptif harus 'mean' atau 'gaussian'.")
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        _gray(image), 255, methods[method], mode, _odd_size(block_size), c
    )


def otsu_threshold(image, invert=False):
    """Menghasilkan citra biner dengan ambang otomatis Otsu."""
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(_gray(image), 0, 255, mode | cv2.THRESH_OTSU)[1]


def otsu_value(image):
    """Nilai ambang yang dipilih Otsu (berguna untuk ditampilkan ke pengguna)."""
    return float(cv2.threshold(_gray(image), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])


def color_masking(image, lower_hsv=(0, 0, 0), upper_hsv=(179, 255, 255)):
    """Kembalikan (mask biner, RGB terseleksi).

    HSV mengikuti OpenCV: H=0–179, S/V=0–255. H bawah > H atas
    memilih rentang yang melintasi nol, misalnya merah H=170 hingga 10.
    """
    image = _image(image)
    if image.ndim != 3:
        raise ValueError("Color masking membutuhkan citra RGB.")
    lower, upper = np.asarray(lower_hsv), np.asarray(upper_hsv)
    limit = np.array([179, 255, 255])
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("Batas HSV harus memiliki tiga komponen.")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all((lower >= 0) & (lower <= limit))
            and np.all((upper >= 0) & (upper <= limit))
            and np.all(lower[1:] <= upper[1:])):
        raise ValueError("Batas HSV tidak valid: H=0–179, S/V=0–255.")
    lower, upper = lower.astype(np.uint8), upper.astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower, np.array([179, *upper[1:]], dtype=np.uint8)),
            cv2.inRange(hsv, np.array([0, *lower[1:]], dtype=np.uint8), upper),
        )
    return mask, cv2.bitwise_and(image, image, mask=mask)


def to_binary_mask(image, invert=False):
    """Ubah citra apa pun menjadi mask biner 0/255 (Otsu) bila belum biner."""
    image = _image(image)
    if image.ndim == 2 and np.all((image == 0) | (image == 255)):
        return 255 - image if invert else image.copy()
    return otsu_threshold(image, invert=invert)


def morphology(image, operation="opening", kernel_size=3, iterations=1, shape="rect",
               auto_binarize=False):
    """Erosion/dilation/opening/closing/gradient/tophat/blackhat pada mask biner 0/255.

    Bentuk kernel: 'rect', 'ellipse', atau 'cross'. Threshold citra dahulu
    agar objek yang diproses berwarna putih dan latarnya hitam — atau gunakan
    auto_binarize=True agar citra berwarna/grayscale di-threshold Otsu otomatis
    (dipakai app; sebelumnya fitur ini selalu gagal karena app mengirim foto berwarna).
    """
    image = _image(image)
    if auto_binarize:
        image = to_binary_mask(image)
    elif image.ndim != 2 or not np.all((image == 0) | (image == 255)):
        raise ValueError("Morfologi membutuhkan mask biner dua dimensi bernilai 0/255.")
    operations = {"erosion": cv2.MORPH_ERODE, "dilation": cv2.MORPH_DILATE,
                  "opening": cv2.MORPH_OPEN, "closing": cv2.MORPH_CLOSE,
                  "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
                  "blackhat": cv2.MORPH_BLACKHAT}
    shapes = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE,
              "cross": cv2.MORPH_CROSS}
    if operation not in operations or shape not in shapes:
        raise ValueError("Operasi atau bentuk kernel morfologi tidak dikenali.")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("Iterations harus bilangan bulat positif.")
    size = _odd_size(kernel_size, minimum=1)
    kernel = cv2.getStructuringElement(shapes[shape], (size, size))
    return cv2.morphologyEx(image, operations[operation], kernel, iterations=int(iterations))


def edge_detection(image, method="canny", kernel_size=3, low_threshold=100, high_threshold=200):
    """Peta tepi grayscale: 'sobel', 'prewitt', 'laplacian', atau 'canny'.

    Sobel/Prewitt menggabungkan gradien X/Y menjadi magnitudo 0–255.
    Prewitt selalu memakai kernel 3x3; kernel_size berlaku pada metode lain.
    """
    gray = _gray(image)
    size = _odd_size(kernel_size)
    if method == "canny":
        if size not in (3, 5, 7) or not 0 <= low_threshold <= high_threshold:
            raise ValueError("Canny: kernel 3/5/7 dan 0 <= ambang bawah <= ambang atas.")
        return cv2.Canny(gray, low_threshold, high_threshold, apertureSize=size)
    if method == "laplacian":
        if size > 31:
            raise ValueError("Kernel Laplacian maksimal 31.")
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=size))
    if method == "sobel":
        if size > 31:
            raise ValueError("Kernel Sobel maksimal 31.")
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=size)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=size)
    elif method == "prewitt":
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        gx = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gy = cv2.filter2D(gray, cv2.CV_64F, kernel.T)
    else:
        raise ValueError("Metode tepi harus sobel, prewitt, laplacian, atau canny.")
    return np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)


def sketch(image, blur_size=21):
    """Efek pensil grayscale menggunakan inverse blur dan color dodge."""
    gray = _gray(image)
    size = _odd_size(blur_size)
    blurred = cv2.GaussianBlur(255 - gray, (size, size), 0)
    denominator = 255 - blurred.astype(np.float32)
    result = np.full(gray.shape, 255, dtype=np.float32)
    np.divide(gray.astype(np.float32) * 255, denominator, out=result, where=denominator > 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def _fit_mark(mark, base_w, base_h, margin, scale=None, fit=False):
    """Skalakan watermark ke `scale` x lebar citra; bila fit=True perkecil agar muat di kanvas."""
    if scale is not None:
        if not 0 < scale <= 1:
            raise ValueError("Skala watermark harus di antara 0 dan 1.")
        target_w = max(1, int(base_w * scale))
        ratio = target_w / mark.width
        mark = mark.resize((target_w, max(1, int(mark.height * ratio))), Image.LANCZOS)
    if not fit:
        return mark
    max_w, max_h = base_w - 2 * margin, base_h - 2 * margin
    if max_w < 1 or max_h < 1:
        raise ValueError("Margin watermark terlalu besar untuk citra ini.")
    if mark.width > max_w or mark.height > max_h:
        ratio = min(max_w / mark.width, max_h / mark.height)
        mark = mark.resize((max(1, int(mark.width * ratio)), max(1, int(mark.height * ratio))),
                           Image.LANCZOS)
    return mark


def _watermark(image, mark, opacity, position, margin, scale=None, fit=False):
    image = _image(image)
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity harus dalam rentang 0–1.")
    if not isinstance(margin, (int, np.integer)) or margin < 0:
        raise ValueError("Margin harus bilangan bulat nonnegatif.")
    base = Image.fromarray(image).convert("RGBA")
    mark = _fit_mark(mark, base.width, base.height, int(margin), scale, fit)
    positions = {
        "top-left": (margin, margin),
        "top-right": (base.width - mark.width - margin, margin),
        "bottom-left": (margin, base.height - mark.height - margin),
        "bottom-right": (base.width - mark.width - margin, base.height - mark.height - margin),
        "center": ((base.width - mark.width) // 2, (base.height - mark.height) // 2),
    }
    if position not in positions:
        raise ValueError("Posisi harus top-left, top-right, bottom-left, bottom-right, atau center.")
    x, y = (int(v) for v in positions[position])
    if x < 0 or y < 0 or x + mark.width > base.width or y + mark.height > base.height:
        raise ValueError("Watermark dan margin terlalu besar untuk citra.")
    mark = mark.copy()
    mark.putalpha(mark.getchannel("A").point(lambda alpha: round(alpha * opacity)))
    base.alpha_composite(mark, (x, y))
    return np.array(base.convert("RGB"))


def _load_font(font_size, font_path=None):
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    # Coba font TrueType umum (Windows/Linux/macOS) agar teks tajam, lalu fallback.
    for candidate in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                      "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default(size=font_size)


def text_watermark(image, text, opacity=0.5, position="bottom-right",
                   font_size=24, color=(255, 255, 255), margin=10, font_path=None,
                   stroke_width=0, stroke_color=(0, 0, 0), fit=False):
    """Tambahkan teks; font_path opsional menunjuk berkas font TTF/OTF.

    Hasil RGB, termasuk untuk masukan grayscale. stroke_width > 0 menambah outline
    agar teks terbaca di latar terang/gelap; fit=True mengecilkan teks yang
    terlalu besar (default: raise ValueError seperti versi asli).
    Font default membutuhkan Pillow >= 10.1 untuk parameter ukuran.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Teks watermark tidak boleh kosong.")
    if not isinstance(font_size, (int, np.integer)) or font_size < 1:
        raise ValueError("Ukuran font harus bilangan bulat positif.")
    for c in (color, stroke_color):
        if len(c) != 3 or any(not isinstance(v, (int, np.integer)) or not 0 <= v <= 255 for v in c):
            raise ValueError("Warna teks harus tiga bilangan bulat 0–255.")
    font = _load_font(int(font_size), font_path)
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0), text, font=font, stroke_width=stroke_width)
    mark = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(mark).text((-left, -top), text, font=font, fill=tuple(int(v) for v in color) + (255,),
                              stroke_width=stroke_width,
                              stroke_fill=tuple(int(v) for v in stroke_color) + (200,))
    return _watermark(image, mark, opacity, position, margin, fit=fit)


def logo_watermark(image, logo, opacity=0.5, position="bottom-right", margin=10, scale=None, fit=False):
    """Tambahkan logo array uint8 RGB/RGBA; kanal alpha logo dipertahankan.

    `scale` (0–1) mengatur lebar logo relatif terhadap lebar citra; fit=True
    mengecilkan logo yang terlalu besar agar muat (default: raise ValueError).
    Untuk PNG transparan: np.array(Image.open(path).convert('RGBA')).
    """
    if (not isinstance(logo, np.ndarray) or logo.dtype != np.uint8 or logo.size == 0
            or logo.ndim != 3 or logo.shape[2] not in (3, 4)):
        raise ValueError("Logo harus array uint8 RGB atau RGBA.")
    # Pillow menebak mode dari bentuk array: (H,W,3)->RGB, (H,W,4)->RGBA
    return _watermark(image, Image.fromarray(np.ascontiguousarray(logo)).convert("RGBA"),
                      opacity, position, margin, scale, fit)
