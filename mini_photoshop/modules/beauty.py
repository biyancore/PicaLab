"""
Modul Beauty (AI) untuk PicaLab — berbasis MediaPipe.

Revisi penting:
1. Kompatibel dua generasi MediaPipe:
   - Legacy `mp.solutions` (mediapipe <= 0.10.2x)
   - MediaPipe Tasks API (mediapipe >= 0.10.30 / 1.x — `mp.solutions` sudah DIHAPUS,
     sehingga kode lama selalu gagal dengan AttributeError).
   Model Tasks (.task/.tflite) diunduh otomatis sekali ke folder `models/`.
2. Model AI di-cache (singleton) — versi lama membuat model baru di SETIAP
   pemanggilan, sehingga Live Camera/Video sangat lambat.
3. Akses model dijaga threading.Lock karena graph MediaPipe tidak thread-safe
   (WebRTC memproses frame di thread terpisah).
4. Blending lebih natural: mask lembut (feathering), lipstik & pemutih gigi di
   ruang warna LAB sehingga tekstur asli tetap terlihat, blush tanpa "halo" gelap.

Semua fungsi menerima & mengembalikan citra BGR uint8.
"""

import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:  # Import dibungkus agar app tetap jalan walau mediapipe gagal ter-install
    import mediapipe as mp
    _IMPORT_ERROR = None
except Exception as exc:  # ImportError, atau DLL error di Windows
    mp = None
    _IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Konfigurasi model
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get(
    "PICALAB_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
}

MAX_FACES = 3
INFERENCE_MAX_SIDE = 1280  # Deteksi pada citra diperkecil -> jauh lebih cepat

# Indeks landmark Face Mesh (478 titik, sama untuk legacy & Tasks)
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
LOWER_LIP_IDX = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_CHEEK, RIGHT_CHEEK = 205, 425
LEFT_TEMPLE, RIGHT_TEMPLE = 234, 454
MOUTH_LEFT, MOUTH_RIGHT = 61, 291


class BeautyUnavailableError(RuntimeError):
    """MediaPipe / model tidak tersedia di lingkungan ini."""


class SubjectNotFoundError(ValueError):
    """Tidak ada wajah / orang yang terdeteksi pada citra."""


# Nama lama tetap tersedia agar kode lain yang mungkin memakainya tidak patah
NoFaceDetectedError = SubjectNotFoundError

_LOCK = threading.RLock()
_BACKEND = None


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def _odd(value, minimum=3):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _to_bgr(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Citra harus berupa array NumPy uint8.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("Citra harus grayscale atau BGR 3 channel.")


def _inference_rgb(image_bgr):
    """Perkecil citra untuk inferensi (landmark ternormalisasi, jadi aman)."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, INFERENCE_MAX_SIDE / max(h, w))
    small = image_bgr if scale >= 1.0 else cv2.resize(
        image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def ensure_model(name):
    """Pastikan file model ada; unduh otomatis bila belum. Mengembalikan Path."""
    path = MODEL_DIR / name
    if path.exists() and path.stat().st_size > 10_000:
        return path
    url = MODEL_URLS[name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        if tmp.stat().st_size < 10_000:
            raise OSError("ukuran file model tidak wajar")
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise BeautyUnavailableError(
            f"Gagal mengunduh model '{name}' ({exc}). Unduh manual dari {url} "
            f"lalu simpan ke folder: {MODEL_DIR}") from exc
    return path


# ---------------------------------------------------------------------------
# Backend MediaPipe
# ---------------------------------------------------------------------------
class _LegacyBackend:
    """mp.solutions (FaceMesh + SelfieSegmentation)."""

    name = "MediaPipe Solutions (legacy)"

    def __init__(self):
        self._mesh = None
        self._seg = None

    def face_landmarks(self, rgb):
        if self._mesh is None:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=MAX_FACES,
                refine_landmarks=True, min_detection_confidence=0.5)
        res = self._mesh.process(rgb)
        return [np.array([[p.x, p.y] for p in face.landmark], dtype=np.float32)
                for face in (res.multi_face_landmarks or [])]

    def person_mask(self, rgb):
        if self._seg is None:
            self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        return np.asarray(self._seg.process(rgb).segmentation_mask, dtype=np.float32)


class _TasksBackend:
    """MediaPipe Tasks API (FaceLandmarker + ImageSegmenter)."""

    name = "MediaPipe Tasks"

    def __init__(self):
        self._landmarker = None
        self._segmenter = None

    @staticmethod
    def _mp_image(rgb):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def face_landmarks(self, rgb):
        vision = mp.tasks.vision
        if self._landmarker is None:
            options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(ensure_model("face_landmarker.task"))),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=MAX_FACES)
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        res = self._landmarker.detect(self._mp_image(rgb))
        return [np.array([[p.x, p.y] for p in face], dtype=np.float32)
                for face in (res.face_landmarks or [])]

    def person_mask(self, rgb):
        vision = mp.tasks.vision
        if self._segmenter is None:
            options = vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(ensure_model("selfie_segmenter.tflite"))),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True)
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
        masks = self._segmenter.segment(self._mp_image(rgb)).confidence_masks
        first = np.squeeze(np.asarray(masks[0].numpy_view(), dtype=np.float32))
        # selfie_segmenter: 1 kategori (orang). Model multi-kelas: indeks 0 = latar.
        return first if len(masks) == 1 else 1.0 - first


def _has_legacy_solutions():
    try:
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    except Exception:
        return False


def get_backend():
    """Ambil backend MediaPipe (dibuat sekali, dipakai ulang)."""
    global _BACKEND
    if mp is None:
        raise BeautyUnavailableError(
            f"MediaPipe tidak dapat di-import: {_IMPORT_ERROR}. "
            "Jalankan: pip install mediapipe")
    with _LOCK:
        if _BACKEND is None:
            _BACKEND = _LegacyBackend() if _has_legacy_solutions() else _TasksBackend()
        return _BACKEND


def backend_status():
    """Info ringkas untuk ditampilkan di UI (tidak memicu unduhan model)."""
    if mp is None:
        return {"available": False, "name": "MediaPipe tidak terpasang",
                "detail": str(_IMPORT_ERROR)}
    version = getattr(mp, "__version__", "?")
    if _has_legacy_solutions():
        return {"available": True, "name": f"MediaPipe {version} · Solutions",
                "detail": "Model bawaan paket"}
    missing = [n for n in MODEL_URLS if not (MODEL_DIR / n).exists()]
    detail = ("Model siap" if not missing
              else "Model akan diunduh otomatis saat pertama dipakai (~4 MB)")
    return {"available": True, "name": f"MediaPipe {version} · Tasks", "detail": detail}


def detect_faces(image):
    """Kembalikan list array (478, 2) koordinat piksel landmark untuk tiap wajah."""
    image = _to_bgr(image)
    h, w = image.shape[:2]
    rgb = _inference_rgb(image)
    backend = get_backend()
    with _LOCK:
        faces = backend.face_landmarks(rgb)
    scale = np.array([w, h], dtype=np.float32)
    return [face * scale for face in faces if len(face) >= 468]


def person_mask(image):
    """Mask probabilitas orang (float32 0..1) seukuran citra."""
    image = _to_bgr(image)
    h, w = image.shape[:2]
    rgb = _inference_rgb(image)
    backend = get_backend()
    with _LOCK:
        mask = backend.person_mask(rgb)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def _require_faces(image, strict):
    faces = detect_faces(image)
    if not faces and strict:
        raise SubjectNotFoundError(
            "Tidak ada wajah terdeteksi. Gunakan foto wajah menghadap kamera dengan pencahayaan cukup.")
    return faces


# ---------------------------------------------------------------------------
# Fitur Beauty
# ---------------------------------------------------------------------------
def background_blur(image: np.ndarray, blur_kernel: int = 51, strict: bool = False) -> np.ndarray:
    """
    Efek Bokeh / Portrait Mode.
    Memisahkan subjek manusia dari latar belakang (selfie segmentation),
    lalu memburamkan latar dengan transisi tepi yang halus (feathering).
    """
    image = _to_bgr(image)
    blur_kernel = _odd(blur_kernel)
    h, w = image.shape[:2]

    mask = person_mask(image)
    if strict and float(mask.max(initial=0.0)) < 0.5:
        raise SubjectNotFoundError("Tidak ada orang terdeteksi untuk efek bokeh.")

    # Alpha lembut: transisi 0.35..0.65 lalu di-feather
    alpha = np.clip((mask - 0.35) / 0.30, 0.0, 1.0)
    feather = _odd(min(h, w) * 0.01)
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)[:, :, None]

    # Optimasi: blur kernel besar dikerjakan pada resolusi setengah
    if blur_kernel > 31 and min(h, w) > 200:
        small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        k = _odd(blur_kernel // 2)
        blurred = cv2.resize(cv2.GaussianBlur(small, (k, k), 0), (w, h),
                             interpolation=cv2.INTER_LINEAR)
    else:
        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    out = image.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def teeth_whitening(image: np.ndarray, intensity: float = 1.0, strict: bool = False) -> np.ndarray:
    """
    Pemutih Gigi.
    Mengunci area rongga mulut via landmark, lalu di ruang LAB: menurunkan
    komponen kuning (b*), sedikit kemerahan (a*), dan menaikkan kecerahan (L).
    Bobot efek dibatasi pada piksel terang agar lidah/rongga gelap tidak ikut memutih.
    """
    image = _to_bgr(image)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    faces = _require_faces(image, strict)
    if not faces or intensity == 0:
        return image.copy()

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mouth_w = 0.0
    for pts in faces:
        cv2.fillPoly(mask, [pts[INNER_LIP_IDX].astype(np.int32)], 255)
        mouth_w = max(mouth_w, float(np.linalg.norm(pts[MOUTH_RIGHT] - pts[MOUTH_LEFT])))
    if not mask.any():
        return image.copy()

    k = _odd(mouth_w * 0.06)
    soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    bright = np.clip((L - 60.0) / 70.0, 0.0, 1.0)  # gigi cenderung terang
    weight = soft * bright * intensity

    lab[:, :, 0] = L + weight * (255.0 - L) * 0.30
    lab[:, :, 1] = 128.0 + (lab[:, :, 1] - 128.0) * (1.0 - 0.45 * weight)
    lab[:, :, 2] = 128.0 + (lab[:, :, 2] - 128.0) * (1.0 - 0.85 * weight)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_blush(image: np.ndarray, color_bgr: tuple = (147, 105, 255), intensity: float = 0.5,
                strict: bool = False) -> np.ndarray:
    """
    Digital Makeup (Blush On).
    Mengunci titik tulang pipi kiri & kanan, membuat mask elips yang di-blur,
    lalu alpha-blend dengan warna blush. (Versi lama memakai mask biner sehingga
    muncul cincin gelap di tepi blush.)
    """
    image = _to_bgr(image)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    faces = _require_faces(image, strict)
    if not faces or intensity == 0:
        return image.copy()

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    max_radius = 2
    for pts in faces:
        face_width = float(np.linalg.norm(pts[RIGHT_TEMPLE] - pts[LEFT_TEMPLE]))
        radius = max(2, int(face_width * 0.12))
        max_radius = max(max_radius, radius)
        for idx in (LEFT_CHEEK, RIGHT_CHEEK):
            center = tuple(int(v) for v in pts[idx])
            cv2.ellipse(mask, center, (int(radius * 1.25), radius), 0, 0, 360, 255, -1)

    k = _odd(max_radius * 1.8)
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    alpha = (alpha * intensity * 0.6)[:, :, None]

    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_lipstick(image: np.ndarray, color_bgr: tuple = (30, 15, 170), intensity: float = 0.5,
                   strict: bool = False) -> np.ndarray:
    """
    Pewarna Bibir (Lipstick).
    Mask bibir atas & bawah di-feather, lalu warna diterapkan di ruang LAB:
    kroma (a*, b*) mengikuti warna lipstik, sedangkan detail kecerahan (L)
    bibir asli dipertahankan agar tekstur & kilau tetap natural.
    """
    image = _to_bgr(image)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    faces = _require_faces(image, strict)
    if not faces or intensity == 0:
        return image.copy()

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    lip_w = 0.0
    for pts in faces:
        cv2.fillPoly(mask, [pts[UPPER_LIP_IDX].astype(np.int32)], 255)
        cv2.fillPoly(mask, [pts[LOWER_LIP_IDX].astype(np.int32)], 255)
        lip_w = max(lip_w, float(np.linalg.norm(pts[MOUTH_RIGHT] - pts[MOUTH_LEFT])))
    if not mask.any():
        return image.copy()

    k = _odd(lip_w * 0.05)
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0 * intensity

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[list(color_bgr)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    lip_l_mean = float((lab[:, :, 0] * (mask > 0)).sum() / max(1, (mask > 0).sum()))
    lab[:, :, 0] += alpha * (target[0] - lip_l_mean) * 0.6
    lab[:, :, 1] += alpha * (target[1] - lab[:, :, 1])
    lab[:, :, 2] += alpha * (target[2] - lab[:, :, 2])
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def face_mesh_overlay(image: np.ndarray, strict: bool = False) -> np.ndarray:
    """
    Visualisasi deteksi AI: menggambar 478 titik landmark, kotak wajah,
    dan kontur bibir. Berguna untuk demo/presentasi cara kerja fitur Beauty.
    """
    image = _to_bgr(image).copy()
    faces = _require_faces(image, strict)
    h, w = image.shape[:2]
    dot = max(1, int(min(h, w) / 500))
    for i, pts in enumerate(faces, start=1):
        for x, y in pts.astype(np.int32):
            cv2.circle(image, (int(x), int(y)), dot, (120, 255, 170), -1, cv2.LINE_AA)
        cv2.polylines(image, [pts[UPPER_LIP_IDX].astype(np.int32),
                              pts[LOWER_LIP_IDX].astype(np.int32)], True, (89, 122, 255), dot + 1,
                      cv2.LINE_AA)
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), dot + 1)
        cv2.putText(image, f"Wajah {i}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 + dot * 0.2, (0, 220, 255), dot + 1, cv2.LINE_AA)
    return image
