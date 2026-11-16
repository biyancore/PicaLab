"""
Penanganan media: render video file & streaming kamera WebRTC.

Revisi penting dibanding versi lama:
1. Encoding video memakai PyAV (libx264) — cv2.VideoWriter dengan codec 'avc1'
   GAGAL DIBUKA pada paket opencv-python(-headless) dari pip (tidak membawa
   encoder H.264), sehingga video hasil kosong/tidak bisa diputar.
2. File sementara ditutup sebelum dibaca/dihapus. Versi lama memanggil
   os.unlink() saat file masih terbuka -> PermissionError [WinError 32] di Windows,
   dan input belum di-flush sehingga video bisa terbaca terpotong.
3. Dimensi frame dipaksa genap (syarat yuv420p/H.264); FPS 0/tak terbaca ditangani.
4. Histogram overlay digambar dengan cv2.polylines (vektor) — versi lama
   memanggil cv2.line 765x per frame.
5. Prosesor WebRTC thread-safe (Lock) + STUN server agar bisa diakses dari jaringan lain.
"""

import fractions
import os
import tempfile
import threading
import time
from dataclasses import dataclass

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from utils import helpers
from utils.pipeline import apply_processing_frame

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
SEPARATOR_PX = 6
HIST_HEIGHT = 150
_ENCODERS = ("libx264", "h264", "mpeg4")  # urutan prioritas


def handle_image_upload(uploaded_file) -> np.ndarray:
    """
    Menangani input berupa file gambar statis.
    Memanfaatkan helper untuk membaca bit mentah menjadi array BGR.
    """
    if uploaded_file is not None:
        return helpers.load_image_from_upload(uploaded_file)
    return None


# ---------------------------------------------------------------------------
# Komposisi frame (dipakai video & kamera)
# ---------------------------------------------------------------------------
def append_histogram_below(frame_bgr: np.ndarray, hist_h: int = HIST_HEIGHT) -> np.ndarray:
    """Tempel histogram RGB (digambar OpenCV) persis di bawah frame."""
    frame_bgr = helpers.ensure_bgr(frame_bgr)
    h, w = frame_bgr.shape[:2]
    canvas = np.full((hist_h, w, 3), (24, 20, 18), dtype=np.uint8)
    xs = np.linspace(0, w - 1, 256).astype(np.int32)

    for ch, color in ((2, (80, 80, 255)), (1, (90, 220, 120)), (0, (255, 160, 70))):
        hist = cv2.calcHist([frame_bgr], [ch], None, [256], [0, 256]).ravel()
        if hist.max() > 0:
            hist = hist / hist.max() * (hist_h - 28)
        ys = (hist_h - 6 - hist).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "RGB Histogram", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.vconcat([frame_bgr, canvas])


def _fit_into(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox: muatkan citra ke ukuran (width, height) tanpa merusak rasio."""
    image = helpers.ensure_bgr(image)
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def compose_frames(original: np.ndarray, processed: np.ndarray, layout: str = "side",
                   show_hist: bool = False) -> np.ndarray:
    """
    Gabungkan frame asli & hasil.
    layout: 'side' (berdampingan), 'result' (hasil saja), 'stack' (atas-bawah).
    """
    original = helpers.ensure_bgr(original)
    h, w = original.shape[:2]
    processed = _fit_into(processed, w, h)

    if show_hist:
        original = append_histogram_below(original)
        processed = append_histogram_below(processed)

    if layout == "result":
        return processed
    if layout == "stack":
        sep = np.full((SEPARATOR_PX, processed.shape[1], 3), 255, dtype=np.uint8)
        return cv2.vconcat([original, sep, processed])
    sep = np.full((processed.shape[0], SEPARATOR_PX, 3), 255, dtype=np.uint8)
    return cv2.hconcat([original, sep, processed])


def _make_even(image: np.ndarray) -> np.ndarray:
    """H.264 + yuv420p mewajibkan lebar & tinggi genap."""
    h, w = image.shape[:2]
    pad_b, pad_r = h % 2, w % 2
    if pad_b or pad_r:
        image = cv2.copyMakeBorder(image, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
    return image


# ---------------------------------------------------------------------------
# Video file
# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    codec: str


def save_upload_to_temp(uploaded_file) -> str:
    """Simpan upload ke file sementara (handle DITUTUP -> aman di Windows)."""
    suffix = os.path.splitext(getattr(uploaded_file, "name", "") or "")[1] or ".mp4"
    fd, path = tempfile.mkstemp(prefix="picalab_in_", suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read())
    return path


def safe_unlink(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass  # file masih dipakai proses lain (Windows) — biarkan dibersihkan OS


def probe_video(path: str) -> VideoInfo:
    """Baca metadata video. Raise ValueError bila bukan video valid."""
    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError("File tidak memiliki stream video.")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0) or 25.0
            duration = (float(stream.duration * stream.time_base) if stream.duration
                        else (container.duration or 0) / av.time_base)
            frames = int(stream.frames or round(duration * fps))
            return VideoInfo(stream.codec_context.width, stream.codec_context.height,
                             fps, frames, duration, stream.codec_context.name)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Video tidak dapat dibaca ({exc}).") from exc


def _open_encoder(container, fps: float, width: int, height: int):
    rate = fractions.Fraction(fps).limit_denominator(1001)
    last_error = None
    for codec in _ENCODERS:
        try:
            stream = container.add_stream(codec, rate=rate)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            if codec == "libx264":
                stream.options = {"crf": "23", "preset": "veryfast"}
            return stream, codec
        except Exception as exc:  # encoder tidak tersedia di build FFmpeg ini
            last_error = exc
    raise RuntimeError(f"Tidak ada encoder video yang tersedia: {last_error}")


def render_video(input_path: str, frame_fn, *, layout: str = "side", show_hist: bool = False,
                 max_seconds: float = None, process_scale: float = 1.0, progress_cb=None) -> dict:
    """
    Proses video frame-by-frame dan simpan sebagai MP4 (H.264 bila tersedia).

    frame_fn(frame_bgr) -> frame_bgr hasil filter.
    progress_cb(done_frames, total_frames, elapsed_seconds) dipanggil ~4x per detik.
    Return dict: output_path, frames, seconds, codec, fps, size.
    """
    info = probe_video(input_path)
    fps = min(max(info.fps, 1.0), 120.0)
    max_frames = int(max_seconds * fps) if max_seconds else None
    total = min(info.frames, max_frames) if (max_frames and info.frames) else (info.frames or max_frames or 0)

    fd, output_path = tempfile.mkstemp(prefix="picalab_out_", suffix=".mp4")
    os.close(fd)

    start, last_report, done = time.perf_counter(), 0.0, 0
    out_stream, codec = None, None
    try:
        with av.open(input_path) as src, av.open(output_path, "w") as dst:
            vstream = src.streams.video[0]
            vstream.thread_type = "AUTO"
            for frame in src.decode(vstream):
                if max_frames and done >= max_frames:
                    break
                bgr = frame.to_ndarray(format="bgr24")
                if process_scale < 1.0:
                    h, w = bgr.shape[:2]
                    bgr = cv2.resize(bgr, (max(2, int(w * process_scale)), max(2, int(h * process_scale))),
                                     interpolation=cv2.INTER_AREA)
                composed = _make_even(compose_frames(bgr, frame_fn(bgr), layout, show_hist))

                if out_stream is None:
                    out_stream, codec = _open_encoder(dst, fps, composed.shape[1], composed.shape[0])
                elif composed.shape[:2] != (out_stream.height, out_stream.width):
                    composed = cv2.resize(composed, (out_stream.width, out_stream.height))

                for packet in out_stream.encode(av.VideoFrame.from_ndarray(composed, format="bgr24")):
                    dst.mux(packet)
                done += 1

                now = time.perf_counter()
                if progress_cb and now - last_report > 0.25:
                    progress_cb(done, total, now - start)
                    last_report = now

            if out_stream is None:
                raise ValueError("Video tidak berisi frame yang dapat dibaca.")
            for packet in out_stream.encode():  # flush encoder
                dst.mux(packet)
    except Exception:
        safe_unlink(output_path)
        raise

    if progress_cb:
        progress_cb(done, done, time.perf_counter() - start)
    return {"output_path": output_path, "frames": done, "seconds": time.perf_counter() - start,
            "codec": codec, "fps": fps, "size": os.path.getsize(output_path)}


def process_video(uploaded_file, processing_function, **kwargs):
    """
    API lama (dipertahankan untuk kompatibilitas): terapkan processing_function
    ke tiap frame dan kembalikan path video hasil (tanpa tampilan berdampingan).
    """
    input_path = save_upload_to_temp(uploaded_file)
    try:
        result = render_video(input_path, lambda f: processing_function(f, **kwargs), layout="result")
    finally:
        safe_unlink(input_path)
    return result["output_path"]


# ---------------------------------------------------------------------------
# Live Camera (WebRTC)
# ---------------------------------------------------------------------------
class FilterVideoProcessor(VideoProcessorBase):
    """
    Memproses frame kamera di thread WebRTC. Parameter filter dikirim dari
    thread Streamlit lewat update(); akses dijaga Lock agar tidak race condition.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.params = {}
        self.options = {"layout": "side", "show_hist": False, "mirror": True, "show_fps": True}
        self.last_original_frame = None
        self.last_filtered_frame = None
        self.fps = 0.0
        self.latency_ms = 0.0
        self._last_t = time.perf_counter()

    def update(self, params: dict, options: dict = None):
        with self._lock:
            self.params = dict(params)
            if options:
                self.options.update(options)

    def snapshot(self):
        with self._lock:
            if self.last_original_frame is None:
                return None, None
            return self.last_original_frame.copy(), self.last_filtered_frame.copy()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            params, opts = self.params, dict(self.options)

        if opts.get("mirror"):
            img = cv2.flip(img, 1)

        t0 = time.perf_counter()
        processed = helpers.ensure_bgr(apply_processing_frame(img, params))
        latency = (time.perf_counter() - t0) * 1000.0

        now = time.perf_counter()
        inst_fps = 1.0 / max(now - self._last_t, 1e-6)
        self._last_t = now

        with self._lock:
            self.last_original_frame = img
            self.last_filtered_frame = processed
            self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps else inst_fps
            self.latency_ms = 0.9 * self.latency_ms + 0.1 * latency if self.latency_ms else latency
            fps, lat = self.fps, self.latency_ms

        composed = compose_frames(img, processed, opts.get("layout", "side"), opts.get("show_hist", False))
        if opts.get("show_fps"):
            text = f"{fps:4.1f} FPS | {lat:5.1f} ms"
            cv2.rectangle(composed, (8, 8), (230, 38), (20, 20, 20), -1)
            cv2.putText(composed, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 170), 1,
                        cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(composed, format="bgr24")


def start_webrtc_camera(processing_function=None, key: str = "picalab-live", processor_factory=None, **kwargs):
    """
    Menangani streaming kamera real-time menggunakan WebRTC.
    - processor_factory: class prosesor (default FilterVideoProcessor).
    - processing_function: (API lama) fungsi frame -> frame sederhana.
    """
    if processor_factory is None and processing_function is not None:
        class _SimpleProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = helpers.ensure_bgr(processing_function(img, **kwargs))
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        processor_factory = _SimpleProcessor

    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=processor_factory or FilterVideoProcessor,
        media_stream_constraints={"video": {"width": {"ideal": 960}, "height": {"ideal": 540}},
                                  "audio": False},
        async_processing=True,
    )
