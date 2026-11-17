"""
Pipeline pemrosesan PicaLab: router filter + statistik citra.

Dipisah dari app.py agar:
- dapat dipakai oleh tab Foto, Video, dan Live Camera (thread WebRTC) tanpa
  import Streamlit UI;
- setiap alat diidentifikasi dengan KODE (bukan parsing teks label). Versi lama
  memotong label seperti "Kernel High-boost".split()[1] -> "high-boost" yang
  tidak dikenali modul, sehingga filter diam-diam tidak berjalan.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from modules import beauty, color_space, effects, enhancement, geometry, segmentation
from utils.helpers import ensure_bgr

# ---------------------------------------------------------------------------
# Registry kategori & alat (dipakai juga oleh sidebar)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "geometry": {
        "icon": "📐", "label": "Geometri & Ukuran",
        "tools": {"resize": "Resize", "resample": "Resample (Skala)", "crop": "Crop",
                  "flip": "Flip / Cermin", "rotate": "Rotasi", "perspective": "Koreksi Perspektif"},
    },
    "color": {
        "icon": "🎨", "label": "Warna & Dekomposisi",
        "tools": {"grayscale": "Grayscale", "saturation": "Saturasi", "hue": "Hue Shift",
                  "channels": "Dekomposisi RGB", "negative": "Efek Negatif"},
    },
    "enhance": {
        "icon": "✨", "label": "Peningkatan & Restorasi",
        "tools": {"bc": "Brightness & Kontras", "gamma": "Koreksi Gamma",
                  "histeq": "Histogram Equalization", "denoise": "Denoising",
                  "blur": "Blur Suite", "sharpen": "Sharpening",
                  "deblur": "Deblurring & Restorasi", "auto": "Auto Enhance (Pipeline)"},
    },
    "segment": {
        "icon": "🧩", "label": "Segmentasi & Morfologi",
        "tools": {"threshold": "Thresholding", "colormask": "Color Masking",
                  "morph": "Morfologi", "edges": "Deteksi Tepi"},
    },
    "effects": {
        "icon": "🖌️", "label": "Efek Kreatif",
        "tools": {"sketch": "Sketch Pensil", "cartoon": "Cartoon", "vignette": "Vignette",
                  "sepia": "Sepia", "pixelate": "Pixelate", "emboss": "Emboss",
                  "watermark": "Watermark"},
    },
    "beauty": {
        "icon": "💄", "label": "Beauty (AI)",
        "tools": {"teeth": "Teeth Whitening", "blush": "Blush On", "lipstick": "Lipstick",
                  "bokeh": "Background Blur (Bokeh)", "mesh": "Face Mesh (Deteksi AI)"},
    },
}

# Kategori yang tidak mendukung slider "Kekuatan Filter" (alpha blending)
NO_INTENSITY_CATEGORIES = {"geometry", "beauty"}
# Alat yang hasilnya selalu berubah ukuran (blending tidak mungkin)
GEOMETRY_CHANGES_SIZE = {"resize", "resample", "crop", "rotate", "perspective"}
# Hasil segmentasi sudah berupa mask atau citra biner; mencampurnya dengan foto
# asli membuat ambang kembali menjadi gradasi, sehingga slider kekuatan dimatikan.
NO_INTENSITY_TOOLS = {("segment", "threshold"), ("segment", "colormask"), ("segment", "edges")}


def tool_label(params: dict) -> str:
    cat = CATEGORIES.get(params.get("category"), {})
    tool = cat.get("tools", {}).get(params.get("tool"), "Tanpa filter")
    return f"{cat.get('icon', '')} {tool}".strip()


@dataclass
class ProcessResult:
    image: np.ndarray                       # selalu BGR uint8 3 channel
    aux: dict = field(default_factory=dict)  # citra pendukung (nama -> BGR)
    info: list = field(default_factory=list)
    warning: str = None
    error: str = None
    elapsed_ms: float = 0.0
    label: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Handler per kategori. Masing-masing: (image_bgr, params, ctx) -> result
# ctx = {"aux": {}, "info": [], "strict": bool}
# ---------------------------------------------------------------------------
def _geometry(img, p, ctx):
    t = p["tool"]
    if t == "resize":
        return geometry.resize_image(img, p["width"], p["height"], p["method"])
    if t == "resample":
        return geometry.resample_scale(img, p["scale"], p["method"])
    if t == "crop":
        if ctx["strict"]:
            ctx["aux"]["Panduan Crop"] = geometry.draw_crop_box(
                img, p["x_start"], p["y_start"], p["x_end"], p["y_end"])
        return geometry.crop_image(img, p["x_start"], p["y_start"], p["x_end"], p["y_end"])
    if t == "flip":
        return geometry.flip_image(img, p["mode"])
    if t == "rotate":
        return geometry.rotate_image(img, p["angle"], p["keep_canvas"], p.get("border", (0, 0, 0)))
    if t == "perspective":
        if ctx["strict"]:
            ctx["aux"]["Panduan Titik"] = geometry.draw_perspective_points(img, p["pts"])
        return geometry.perspective_transform(img, p["pts"])
    return img


def _color(img, p, ctx):
    t = p["tool"]
    if t == "grayscale":
        return color_space.to_grayscale(img)
    if t == "saturation":
        return color_space.adjust_saturation(img, p["factor"])
    if t == "hue":
        return color_space.adjust_hue_degrees(img, p["degrees"])
    if t == "channels":
        if ctx["strict"]:
            ctx["aux"]["Grid Channel RGB"] = color_space.rgb_channel_grid(img)
        return color_space.channel_view(img, p["channel"], tinted=p["tinted"])
    if t == "negative":
        return color_space.invert_colors(img)
    return img


def _enhance(img, p, ctx):
    t = p["tool"]
    if t == "bc":
        return enhancement.brightness_contrast(img, p["brightness"], p["contrast"])
    if t == "gamma":
        return enhancement.gamma_correction(img, p["gamma"])
    if t == "histeq":
        if p["method"] == "clahe":
            return enhancement.clahe(img, p["clip_limit"], p["tile_size"])
        return enhancement.histogram_equalization(img)
    if t == "denoise":
        return enhancement.denoise(img, p["method"], strength=p["strength"], kernel_size=p["kernel"])
    if t == "blur":
        if p["method"] == "bilateral":
            return enhancement.blur(img, "bilateral", diameter=min(p["kernel"], 25),
                                    sigma_color=p["sigma_color"], sigma_space=p["sigma_color"])
        return enhancement.blur(img, p["method"], kernel_size=p["kernel"])
    if t == "sharpen":
        return enhancement.sharpen(img, p["method"], strength=p["strength"], amount=p["amount"])
    if t == "deblur":
        m = p["method"]
        if m == "wiener_gaussian":
            return enhancement.wiener_deblur(img, "gaussian", size=p["size"], sigma=p["sigma"], k=p["k"])
        if m == "wiener_motion":
            return enhancement.wiener_deblur(img, "motion", size=p["size"], angle=p["angle"], k=p["k"])
        if m == "unsharp":
            return enhancement.deblur_unsharp(img, amount=p["amount"], sigma=p["sigma"])
        if m == "inpaint":
            return enhancement.inpaint_damage(img, p["damage"], p["threshold"], p["radius"])
    if t == "auto":
        return enhancement.comprehensive_enhancement(
            img, denoise_strength=p["denoise"], use_clahe=p["use_clahe"], clip_limit=p["clip_limit"],
            brightness=p["brightness"], contrast=p["contrast"], sharpen_amount=p["sharpen"])
    return img


def _rgb(img):
    """Modul segmentation (Andini) berkontrak RGB -> konversi di perbatasan."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _from_rgb(out):
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 else out


def _segment(img, p, ctx):
    t = p["tool"]
    rgb = _rgb(img)
    if t == "threshold":
        m = p["method"]
        if m == "global":
            return segmentation.global_threshold(rgb, p["value"], invert=p["invert"])
        if m == "otsu":
            if ctx["strict"]:
                ctx["info"].append(f"Ambang otomatis Otsu: {segmentation.otsu_value(rgb):.0f}")
            return segmentation.otsu_threshold(rgb, invert=p["invert"])
        return segmentation.adaptive_threshold(rgb, p["adaptive_method"], p["block_size"],
                                               c=p["c"], invert=p["invert"])
    if t == "colormask":
        mask, selected_rgb = segmentation.color_masking(rgb, p["lower"], p["upper"])
        selected = _from_rgb(selected_rgb)
        if ctx["strict"]:
            ctx["aux"]["Mask Biner"] = mask
            ctx["info"].append(f"Cakupan mask: {100.0 * np.count_nonzero(mask) / mask.size:.1f}% piksel")
        if p["output"] == "mask":
            return mask
        if p["output"] == "highlight":
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            return np.where(mask[:, :, None] > 0, img, gray)
        return selected
    if t == "morph":
        if ctx["strict"]:
            ctx["aux"]["Mask Biner (Input Otsu)"] = segmentation.to_binary_mask(rgb)
        return segmentation.morphology(rgb, p["method"], p["kernel"], p["iterations"], p["shape"],
                                       auto_binarize=True)
    if t == "edges":
        edges = segmentation.edge_detection(rgb, p["method"], kernel_size=p["kernel"],
                                            low_threshold=p.get("low", 100),
                                            high_threshold=p.get("high", 200))
        if ctx["strict"]:
            ctx["info"].append(f"Kepadatan tepi: {100.0 * np.count_nonzero(edges > 40) / edges.size:.1f}%")
        return edges
    return img


def _effects(img, p, ctx):
    t = p["tool"]
    if t == "sketch":
        return segmentation.sketch(_rgb(img), p["blur_size"])
    if t == "cartoon":
        return effects.cartoon_filter(img, p["line_size"], p["blur_value"], p["passes"])
    if t == "vignette":
        return effects.vignette_effect(img, p["strength"])
    if t == "sepia":
        return effects.sepia_effect(img, p["intensity"])
    if t == "pixelate":
        return effects.pixelate_effect(img, p["block"])
    if t == "emboss":
        return effects.emboss_effect(img, p["strength"])
    if t == "watermark":
        w = img.shape[1]
        if p["wm_type"] == "text":
            font_px = max(8, int(w * p["font_pct"] / 100.0))
            return _from_rgb(segmentation.text_watermark(
                _rgb(img), p["text"], opacity=p["opacity"], position=p["pos"], font_size=font_px,
                color=p["color"], margin=max(4, w // 60), stroke_width=max(1, font_px // 18), fit=True))
        if p.get("logo") is None:
            ctx["info"].append("Unggah logo PNG di sidebar untuk menerapkan watermark logo.")
            return img
        return _from_rgb(segmentation.logo_watermark(
            _rgb(img), p["logo"], opacity=p["opacity"], position=p["pos"], margin=max(4, w // 60),
            scale=p["logo_pct"] / 100.0, fit=True))
    return img


def _beauty(img, p, ctx):
    t, strict = p["tool"], ctx["strict"]
    if t == "teeth":
        return beauty.teeth_whitening(img, p["intensity"], strict=strict)
    if t == "blush":
        return beauty.apply_blush(img, color_bgr=p["color"], intensity=p["intensity"], strict=strict)
    if t == "lipstick":
        return beauty.apply_lipstick(img, color_bgr=p["color"], intensity=p["intensity"], strict=strict)
    if t == "bokeh":
        return beauty.background_blur(img, blur_kernel=p["kernel"], strict=strict)
    if t == "mesh":
        return beauty.face_mesh_overlay(img, strict=strict)
    return img


_HANDLERS = {"geometry": _geometry, "color": _color, "enhance": _enhance,
             "segment": _segment, "effects": _effects, "beauty": _beauty}


# ---------------------------------------------------------------------------
# API publik
# ---------------------------------------------------------------------------
def blend_intensity(original_image, processed_image, intensity_pct):
    """Alpha blending hasil filter dengan citra asli (0..100%)."""
    if intensity_pct >= 100:
        return processed_image
    if intensity_pct <= 0:
        return original_image.copy()
    if original_image.shape[:2] != processed_image.shape[:2]:
        return processed_image  # dimensi berubah (mis. geometri) -> tidak bisa di-blend
    alpha = intensity_pct / 100.0
    return cv2.addWeighted(ensure_bgr(processed_image), alpha, ensure_bgr(original_image), 1 - alpha, 0)


def apply_processing(image: np.ndarray, params: dict, strict: bool = True) -> ProcessResult:
    """
    Terapkan filter sesuai params ke citra BGR.

    strict=True  (tab Foto) : citra pendukung & info dihitung, error dilaporkan
                              ke pengguna (bukan ditelan diam-diam seperti versi lama).
    strict=False (Video/Cam): mode cepat; bila gagal, frame asli dikembalikan.
    """
    start = time.perf_counter()
    image = ensure_bgr(image)
    ctx = {"aux": {}, "info": [], "strict": strict}
    result = ProcessResult(image=image, label=tool_label(params))

    handler = _HANDLERS.get(params.get("category"))
    if handler is None or not params.get("tool"):
        result.image = image.copy()
        return result

    try:
        out = handler(image, params, ctx)
        out = ensure_bgr(out)
        if (params.get("category") not in NO_INTENSITY_CATEGORIES
                and (params.get("category"), params.get("tool")) not in NO_INTENSITY_TOOLS):
            out = blend_intensity(image, out, params.get("intensity_pct", 100))
        result.image = out
    except beauty.SubjectNotFoundError as exc:
        result.warning = str(exc)
        result.image = image.copy()
    except beauty.BeautyUnavailableError as exc:
        result.error = f"Fitur AI tidak tersedia: {exc}"
        result.image = image.copy()
    except ValueError as exc:
        result.error = str(exc)
        result.image = image.copy()
    except cv2.error as exc:
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
        result.error = f"OpenCV gagal memproses: {msg}"
        result.image = image.copy()
    except MemoryError:
        result.error = "Memori tidak cukup. Kecilkan resolusi kerja gambar lalu coba lagi."
        result.image = image.copy()
    except Exception as exc:  # pengaman terakhir — tetap dilaporkan, tidak ditelan
        result.error = f"{type(exc).__name__}: {exc}"
        result.image = image.copy()

    result.aux = {k: ensure_bgr(v) for k, v in ctx["aux"].items()}
    result.info = ctx["info"]
    result.elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result


def apply_processing_frame(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Versi ringan untuk video/kamera: selalu mengembalikan frame BGR."""
    return apply_processing(frame_bgr, params, strict=False).image


# ---------------------------------------------------------------------------
# Statistik & histogram
# ---------------------------------------------------------------------------
def image_stats(image: np.ndarray) -> dict:
    """Statistik citra untuk metric cards (dihitung pada versi diperkecil agar cepat)."""
    image = ensure_bgr(image)
    h, w = image.shape[:2]
    scale = min(1.0, 640 / max(h, w))
    small = image if scale >= 1 else cv2.resize(image, (int(w * scale), int(h * scale)),
                                                 interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist[hist > 0] / hist.sum()

    b, g, r = (small[:, :, i].astype(np.float32) for i in range(3))
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return {
        "width": w, "height": h, "megapixels": w * h / 1e6,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": float(-(prob * np.log2(prob)).sum()),
        "colorfulness": colorfulness,
        "clipped_pct": float(100.0 * (hist[0] + hist[255]) / hist.sum()),
    }


def histogram_table(image: np.ndarray):
    """Histogram R, G, B, dan Luminansi (bentuk 'lebar') sebagai dict list (siap DataFrame)."""
    image = ensure_bgr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "Intensitas": np.arange(256),
        "Red": cv2.calcHist([image], [2], None, [256], [0, 256]).ravel(),
        "Green": cv2.calcHist([image], [1], None, [256], [0, 256]).ravel(),
        "Blue": cv2.calcHist([image], [0], None, [256], [0, 256]).ravel(),
        "Luminansi": cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel(),
    }
