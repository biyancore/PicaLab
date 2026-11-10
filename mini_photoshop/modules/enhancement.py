
import cv2
import numpy as np
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Citra harus berupa array NumPy uint8.")
    if image.size == 0 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("Citra harus grayscale (H, W) atau RGB (H, W, 3).")
    return image


def _odd_size(value, minimum=3):
    if not isinstance(value, (int, np.integer)) or value < minimum or value % 2 == 0:
        raise ValueError(f"Ukuran kernel harus bilangan ganjil minimal {minimum}.")
    return int(value)


def _luminance_apply(image, func):
    """Terapkan func (gray uint8 -> gray uint8) pada luminansi saja agar warna tetap."""
    if image.ndim == 2:
        return func(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = func(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# 1. Brightness & Contrast
# ---------------------------------------------------------------------------
def brightness_contrast(image, brightness=0, contrast=1.0):
    """Transformasi linier g = contrast * f + brightness (hasil dipotong 0–255).

    brightness: -255..255, contrast: 0..5 (1.0 = tanpa perubahan).
    """
    image = _image(image)
    if not -255 <= brightness <= 255:
        raise ValueError("Brightness harus dalam rentang -255 hingga 255.")
    if not 0 <= contrast <= 5:
        raise ValueError("Contrast harus dalam rentang 0–5.")
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def gamma_correction(image, gamma=1.0):
    """Koreksi gamma (non-linier). gamma > 1 mencerahkan, < 1 menggelapkan."""
    image = _image(image)
    if gamma <= 0:
        raise ValueError("Gamma harus lebih besar dari 0.")
    table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(image, table)


# ---------------------------------------------------------------------------
# 2. Histogram Equalization & CLAHE
# ---------------------------------------------------------------------------
def histogram_equalization(image):
    """Equalization global. Citra RGB diproses pada kanal luminansi (Y)."""
    image = _image(image)
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe(image, clip_limit=2.0, tile_size=8):
    """CLAHE (equalization adaptif). Citra RGB diproses pada kanal L (LAB)."""
    image = _image(image)
    if clip_limit <= 0:
        raise ValueError("Clip limit harus lebih besar dari 0.")
    if not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
        raise ValueError("Ukuran tile harus bilangan bulat positif.")
    engine = cv2.createCLAHE(clipLimit=float(clip_limit),
                             tileGridSize=(int(tile_size), int(tile_size)))
    return _luminance_apply(image, engine.apply)


# ---------------------------------------------------------------------------
# 3. Blur Suite
# ---------------------------------------------------------------------------
def gaussian_blur(image, kernel_size=5, sigma=0):
    """Gaussian blur. sigma=0 berarti dihitung otomatis dari ukuran kernel."""
    size = _odd_size(kernel_size, minimum=1)
    if sigma < 0:
        raise ValueError("Sigma tidak boleh negatif.")
    return cv2.GaussianBlur(_image(image), (size, size), sigma)


def averaging_blur(image, kernel_size=5):
    """Averaging (box) blur."""
    size = _odd_size(kernel_size, minimum=1)
    return cv2.blur(_image(image), (size, size))


def median_blur(image, kernel_size=5):
    """Median blur, efektif untuk noise salt-and-pepper."""
    return cv2.medianBlur(_image(image), _odd_size(kernel_size, minimum=3))


def bilateral_blur(image, diameter=9, sigma_color=75, sigma_space=75):
    """Bilateral filter: meratakan area datar, tepi objek tetap tajam."""
    if not isinstance(diameter, (int, np.integer)) or diameter < 1:
        raise ValueError("Diameter harus bilangan bulat positif.")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Sigma color dan sigma space harus lebih besar dari 0.")
    return cv2.bilateralFilter(_image(image), int(diameter), sigma_color, sigma_space)


def blur(image, method="gaussian", kernel_size=5, sigma=0,
         diameter=9, sigma_color=75, sigma_space=75):
    """Pintu masuk blur: 'gaussian', 'averaging', 'median', atau 'bilateral'."""
    if method == "gaussian":
        return gaussian_blur(image, kernel_size, sigma)
    if method == "averaging":
        return averaging_blur(image, kernel_size)
    if method == "median":
        return median_blur(image, kernel_size)
    if method == "bilateral":
        return bilateral_blur(image, diameter, sigma_color, sigma_space)
    raise ValueError("Metode blur harus gaussian, averaging, median, atau bilateral.")


# ---------------------------------------------------------------------------
# 4. Sharpening
# ---------------------------------------------------------------------------
_SHARPEN_KERNELS = {
    "laplacian": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "high_boost": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32),
}


def unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5):
    """Unsharp mask: asli + amount * (asli - blur)."""
    size = _odd_size(kernel_size, minimum=1)
    if sigma <= 0 or amount < 0:
        raise ValueError("Sigma harus > 0 dan amount tidak boleh negatif.")
    image = _image(image)
    blurred = cv2.GaussianBlur(image, (size, size), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def sharpen(image, method="laplacian", strength=1.0, kernel_size=5, sigma=1.0, amount=1.5):
    """Penajaman: 'laplacian' / 'high_boost' (kernel konvolusi) atau 'unsharp'.

    strength (0–1) mencampur hasil kernel dengan citra asli.
    """
    image = _image(image)
    if method == "unsharp":
        return unsharp_mask(image, kernel_size, sigma, amount)
    if method not in _SHARPEN_KERNELS:
        raise ValueError("Metode sharpen harus laplacian, high_boost, atau unsharp.")
    if not 0 <= strength <= 1:
        raise ValueError("Strength harus dalam rentang 0–1.")
    sharpened = cv2.filter2D(image, -1, _SHARPEN_KERNELS[method])
    return cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)


# ---------------------------------------------------------------------------
# 5. Smoothing & Denoising
# ---------------------------------------------------------------------------
def denoise(image, method="nlm", strength=10, kernel_size=5):
    """Pengurangan noise.

    method: 'nlm' (Non-Local Means; strength = h), 'gaussian', 'median'
    (cocok untuk salt-and-pepper), atau 'bilateral' (strength mengatur sigma).
    """
    image = _image(image)
    if strength <= 0:
        raise ValueError("Strength harus lebih besar dari 0.")
    if method == "nlm":
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    if method == "gaussian":
        return gaussian_blur(image, kernel_size)
    if method == "median":
        return median_blur(image, kernel_size)
    if method == "bilateral":
        return bilateral_blur(image, kernel_size, strength * 5, strength * 5)
    raise ValueError("Metode denoise harus nlm, gaussian, median, atau bilateral.")


# ---------------------------------------------------------------------------
# 6. Deblurring & Restoration
# ---------------------------------------------------------------------------
def _gaussian_psf(size, sigma):
    size = _odd_size(size)
    if sigma <= 0:
        raise ValueError("Sigma PSF harus lebih besar dari 0.")
    axis = np.arange(size) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (psf / psf.sum()).astype(np.float32)


def _motion_psf(length, angle):
    length = _odd_size(length)
    psf = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cv2.line(psf, (0, center), (length - 1, center), 1.0, 1)
    matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    psf = cv2.warpAffine(psf, matrix, (length, length))
    return psf / psf.sum()


def _wiener_channel(channel, psf, k):
    pad = psf.shape[0] // 2
    data = cv2.copyMakeBorder(channel.astype(np.float32) / 255.0,
                              pad, pad, pad, pad, cv2.BORDER_REFLECT)
    kernel = np.zeros(data.shape, dtype=np.float32)
    kernel[:psf.shape[0], :psf.shape[1]] = psf
    kernel = np.roll(kernel, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    h = np.fft.rfft2(kernel)
    restored = np.fft.irfft2(np.fft.rfft2(data) * np.conj(h) / (np.abs(h) ** 2 + k),
                             s=data.shape)
    restored = restored[pad:-pad, pad:-pad]
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def wiener_deblur(image, psf_type="gaussian", size=7, sigma=2.0, angle=0, k=0.01):
    """Restorasi citra buram dengan Wiener deconvolution.

    psf_type: 'gaussian' (blur umum, pakai size & sigma) atau 'motion'
    (blur gerak, pakai size sebagai panjang & angle dalam derajat).
    k: konstanta regularisasi. Kecil = lebih tajam tetapi noise/ringing
    bertambah; besar = lebih halus.
    """
    image = _image(image)
    if k <= 0:
        raise ValueError("Konstanta K harus lebih besar dari 0.")
    if psf_type == "gaussian":
        psf = _gaussian_psf(size, sigma)
    elif psf_type == "motion":
        psf = _motion_psf(size, angle)
    else:
        raise ValueError("Tipe PSF harus gaussian atau motion.")
    if image.ndim == 2:
        return _wiener_channel(image, psf, k)
    return np.dstack([_wiener_channel(image[:, :, i], psf, k) for i in range(3)])


def deblur_unsharp(image, amount=2.0, sigma=2.0):
    """Deblur ringan memakai unsharp mask kuat (tanpa perlu menebak PSF)."""
    return unsharp_mask(image, kernel_size=_odd_size(int(sigma * 6) | 1, 3),
                        sigma=sigma, amount=amount)


def inpaint_damage(image, damage="bright", threshold=240, radius=3):
    """Restorasi goresan/bercak dengan inpainting.

    damage: 'bright' (piksel > threshold dianggap rusak, misalnya goresan putih)
    atau 'dark' (piksel < 255 - threshold dianggap rusak, misalnya bercak hitam).
    """
    image = _image(image)
    if damage not in ("bright", "dark"):
        raise ValueError("Jenis kerusakan harus bright atau dark.")
    if not 0 <= threshold <= 255 or radius < 1:
        raise ValueError("Threshold harus 0–255 dan radius minimal 1.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if damage == "bright":
        mask = (gray > threshold).astype(np.uint8) * 255
    else:
        mask = (gray < 255 - threshold).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# 7. Image Enhancement komprehensif
# ---------------------------------------------------------------------------
def comprehensive_enhancement(image, denoise_strength=5, use_clahe=True, clip_limit=2.0,
                              tile_size=8, brightness=0, contrast=1.0, sharpen_amount=0.8):
    """Pipeline: denoise -> CLAHE -> brightness/contrast -> sharpen.

    Setiap tahap bisa dimatikan (denoise_strength=0, use_clahe=False,
    brightness=0 & contrast=1, sharpen_amount=0). Koreksi geometrik
    perspektif ada di modules/geometry.py dan dijalankan terpisah.
    """
    result = _image(image).copy()
    if denoise_strength > 0:
        result = denoise(result, "nlm", denoise_strength)
    if use_clahe:
        result = clahe(result, clip_limit, tile_size)
    if brightness != 0 or contrast != 1.0:
        result = brightness_contrast(result, brightness, contrast)
    if sharpen_amount > 0:
        result = unsharp_mask(result, 5, 1.0, sharpen_amount)
    return result

# ---------------------------------------------------------------------------
# 8. Histogram (visualisasi sebelum dan sesudah)
# ---------------------------------------------------------------------------
def histogram(image):
    """Jumlah piksel untuk tiap intensitas 0–255 (array panjang 256).

    Citra RGB dikonversi ke grayscale (luminansi) terlebih dahulu.
    """
    image = _image(image)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.bincount(gray.ravel(), minlength=256)


def channel_histograms(image):
    """Histogram per kanal R, G, B: array bentuk (3, 256)."""
    image = _image(image)
    if image.ndim != 3:
        raise ValueError("Histogram per kanal membutuhkan citra RGB.")
    return np.stack([np.bincount(image[:, :, i].ravel(), minlength=256) for i in range(3)])


def histogram_figure(before, after, mode="gray"):
    """Figure matplotlib: histogram 'Sebelum' dan 'Sesudah' berdampingan.

    Sumbu X = intensitas piksel (0–255), sumbu Y = jumlah piksel.
    mode: 'gray' (intensitas grayscale) atau 'rgb' (tiga kanal ditumpuk).
    """
    if mode not in ("gray", "rgb"):
        raise ValueError("Mode histogram harus gray atau rgb.")
    figure = Figure(figsize=(10, 3.4), layout="constrained")
    axes = figure.subplots(1, 2)
    x = np.arange(256)
    for ax, image, title in ((axes[0], before, "Sebelum"), (axes[1], after, "Sesudah")):
        image = _image(image)
        if mode == "gray":
            ax.bar(x, histogram(image), width=1.0, color="#555555")
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            title += f"  (rata-rata {gray.mean():.0f}, std {gray.std():.0f})"
        else:
            for counts, color, name in zip(channel_histograms(image),
                                           ("#d62728", "#2ca02c", "#1f77b4"), "RGB"):
                ax.fill_between(x, counts, color=color, alpha=0.35)
                ax.plot(x, counts, color=color, linewidth=0.8, label=name)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensitas piksel (0–255)")
        ax.set_ylabel("Jumlah piksel")
    return figure