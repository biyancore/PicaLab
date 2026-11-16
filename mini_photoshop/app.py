"""
PicaLab — Mini Photoshop berbasis Streamlit + OpenCV.

Jalankan dari folder mini_photoshop:
    streamlit run app.py

Arsitektur (modular, tiap modul tetap milik anggota tim masing-masing):
    modules/   -> algoritma pengolahan citra (geometry, color_space, enhancement,
                  segmentation, effects, beauty)
    utils/     -> state_manager (undo/redo), helpers (I/O), pipeline (router filter),
                  media_handler (video & kamera), sidebar (menu), ui (tema & komponen)
    app.py     -> hanya orkestrasi halaman (file ini)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import helpers, media_handler, pipeline, sidebar, state_manager, ui

st.set_page_config(page_title="PicaLab — Mini Photoshop", page_icon="🎨", layout="wide",
                   initial_sidebar_state="expanded")

PHOTO_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
WORKING_SIZES = {1280: "1280 px · cepat", 2048: "2048 px · seimbang", 4096: "4096 px · detail",
                 0: "Resolusi asli"}
MODES = {"photo": ":material/image: Editor Foto", "video": ":material/movie: Video",
         "camera": ":material/videocam: Live Camera"}


# ---------------------------------------------------------------------------
# Cache (kunci = id sesi + versi gambar + sidik jari parameter)
# Argumen berawalan "_" tidak di-hash Streamlit -> tidak ada overhead hashing array besar.
# ---------------------------------------------------------------------------
@st.cache_data(max_entries=24, show_spinner=False)
def cached_process(cache_key: str, _image, _params):
    return pipeline.apply_processing(_image, _params, strict=True)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_stats(cache_key: str, _image):
    return pipeline.image_stats(_image)


@st.cache_data(max_entries=12, show_spinner=False)
def cached_encode(cache_key: str, _image, ext: str, quality: int):
    return helpers.encode_image_for_download(_image, ext, quality)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
def _apply_to_history(image, label):
    state_manager.update_image(image, label=label)
    st.toast(f"Diterapkan: {label}", icon="✅")


def _load_sample():
    img = helpers.generate_sample_image()
    state_manager.load_new_image(img, source_id="sample", meta={
        "name": "contoh_picalab.png", "format": "Generated", "mode": "RGB", "file_size": img.nbytes,
        "original_size": (img.shape[1], img.shape[0]), "working_size": (img.shape[1], img.shape[0]),
        "exif": False, "downscaled": False})
    st.toast("Gambar contoh dimuat", icon="🖼️")


def _close_image():
    state_manager.clear_all_states()
    st.toast("Gambar ditutup", icon="🗑️")


def _send_capture_to_editor(which: str):
    cap = st.session_state.get("cam_capture")
    if not cap:
        return
    img = cap["original"] if which == "original" else cap["filtered"]
    state_manager.load_new_image(img, source_id=f"camera:{cap['time']}:{which}", meta={
        "name": f"kamera_{cap['time']}.png", "format": "Kamera", "mode": "RGB", "file_size": img.nbytes,
        "original_size": (img.shape[1], img.shape[0]), "working_size": (img.shape[1], img.shape[0]),
        "exif": False, "downscaled": False})
    st.session_state["nav_mode"] = "photo"
    st.toast("Gambar kamera dikirim ke Editor Foto", icon="📤")


# ---------------------------------------------------------------------------
# Workspace: FOTO
# ---------------------------------------------------------------------------
def _handle_photo_upload():
    with st.container(border=True):
        c1, c2, c3 = st.columns([3.2, 1.3, 1.2], vertical_alignment="bottom")
        uploaded = c1.file_uploader("Unggah foto", type=PHOTO_TYPES, key="photo_upload",
                                    help="JPG, PNG, WEBP, BMP, TIFF. Orientasi EXIF & transparansi ditangani otomatis.")
        max_side = c2.selectbox("Resolusi kerja", list(WORKING_SIZES), index=1, format_func=WORKING_SIZES.get,
                                key="working_size",
                                help="Foto besar diperkecil agar filter tetap responsif. Mengubah opsi ini memuat ulang foto.")
        with c3:
            st.button("Gambar contoh", icon=":material/auto_awesome:", on_click=_load_sample, width="stretch",
                      help="Coba semua filter tanpa mengunggah file")
            st.button("Tutup gambar", icon=":material/close:", on_click=_close_image, width="stretch",
                      disabled=not state_manager.has_image())

    if uploaded is None:
        return
    sid = f"{helpers.source_id(uploaded)}:{max_side}"
    if st.session_state.get("last_upload_sid") == sid:
        return  # upload ini sudah dimuat (mungkin kemudian diganti contoh/kamera)
    st.session_state["last_upload_sid"] = sid
    try:
        with st.spinner("Membaca foto…"):
            image, meta = helpers.load_image_with_meta(uploaded, max_side=max_side or None)
        state_manager.load_new_image(image, source_id=sid, meta=meta)
        note = " (diperkecil ke resolusi kerja)" if meta["downscaled"] else ""
        st.toast(f"Foto dimuat: {meta['name']}{note}", icon="🖼️")
    except ValueError as exc:
        st.error(f"Gagal memuat foto: {exc}", icon=":material/error:")


def _render_feature_overview():
    ui.empty_state("🖼️", "Belum ada foto",
                   "Unggah foto atau klik “Gambar contoh”. Pilih alat di sidebar — pratinjau diperbarui "
                   "otomatis, lalu tekan “Terapkan ke riwayat” untuk menyimpan langkah edit.")
    ui.section_title("Toolkit", f"{sum(len(c['tools']) for c in pipeline.CATEGORIES.values())} alat dalam 6 kategori")
    ui.metric_cards([{"label": c["label"], "value": f"{len(c['tools'])} alat", "icon": c["icon"],
                      "hint": " · ".join(c["tools"].values()), "delta": ", ".join(list(c["tools"].values())[:2]) + "…"}
                     for c in pipeline.CATEGORIES.values()])


def _render_messages(result):
    if result.error:
        st.error(f"**Filter tidak dapat diterapkan.** {result.error}", icon=":material/error:")
    if result.warning:
        st.warning(result.warning, icon=":material/face:")
    for msg in result.info:
        st.info(msg, icon=":material/info:")


def _render_metrics(before, after, result):
    sb = cached_stats(f"{state_manager.session_uid()}:{st.session_state['img_version']}", before)
    sa = cached_stats(st.session_state["_last_key"], after)

    res_delta = (f"dari {sb['width']}×{sb['height']}"
                 if (sa["width"], sa["height"]) != (sb["width"], sb["height"]) else f"{sa['megapixels']:.2f} MP")
    items = [{"label": "Resolusi", "value": f"{sa['width']}×{sa['height']}", "delta": res_delta, "icon": "📐",
              "hint": "Lebar × tinggi hasil (px)"}]
    for label, k, fmt, icon, hint, pct in [
        ("Kecerahan", "brightness", "{:.0f}", "☀️", "Rata-rata luminansi 0–255", False),
        ("Kontras", "contrast", "{:.1f}", "🌗", "Standar deviasi luminansi", False),
        ("Ketajaman", "sharpness", "{:.0f}", "🔍", "Varians Laplacian (makin tinggi makin tajam)", True),
        ("Entropi", "entropy", "{:.2f}", "🧮", "Kekayaan informasi tonal (bit, maks 8)", False),
        ("Colorfulness", "colorfulness", "{:.0f}", "🎨", "Metrik Hasler–Süsstrunk", False),
    ]:
        d, direction = ui.delta_text(sb[k], sa[k], fmt="{:+.2f}" if k == "entropy" else "{:+.1f}", pct=pct)
        items.append({"label": label, "value": fmt.format(sa[k]), "delta": d, "direction": direction,
                      "icon": icon, "hint": hint})
    items.append({"label": "Waktu proses", "value": f"{result.elapsed_ms:.0f} ms", "icon": "⚡",
                  "delta": "dari cache" if result.elapsed_ms and st.session_state.get("_from_cache") else "",
                  "hint": "Durasi komputasi filter pada resolusi kerja"})
    ui.metric_cards(items)
    return sb, sa


def _render_images(before, result, compare_mode):
    after = result.image
    same_ratio = abs(before.shape[1] / before.shape[0] - after.shape[1] / after.shape[0]) < 0.01
    if compare_mode == "slider" and same_ratio:
        ui.compare_slider(before, after)
    elif compare_mode == "result":
        ui.frame_title("Hasil", result.label, accent=True)
        st.image(helpers.bgr_to_rgb(after), width="stretch")
    else:
        if compare_mode == "slider":
            st.caption("Rasio gambar berubah (filter geometri) — ditampilkan berdampingan.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ops = state_manager.applied_operations()
            ui.frame_title("Saat ini", f"{len(ops)} langkah" if ops else "Asli")
            st.image(helpers.bgr_to_rgb(before), width="stretch")
        with c2:
            ui.frame_title("Pratinjau", result.label, accent=True)
            st.image(helpers.bgr_to_rgb(after), width="stretch")

    if result.aux:
        with st.expander(f"Visual pendukung ({len(result.aux)})", icon=":material/layers:", expanded=True):
            cols = st.columns(len(result.aux))
            for col, (name, img) in zip(cols, result.aux.items()):
                col.image(helpers.bgr_to_rgb(img), caption=name, width="stretch")


def _render_analysis(before, after, result, sb, sa, base_name):
    t_hist, t_stats, t_hist_log, t_export = st.tabs(
        [":material/bar_chart: Histogram", ":material/analytics: Statistik",
         ":material/history: Riwayat", ":material/download: Ekspor"])

    with t_hist:
        mode = st.segmented_control("Mode histogram", ["luma", "rgb"], default="luma", required=True,
                                    format_func={"luma": "Luminansi", "rgb": "Per channel RGB"}.get,
                                    key="hist_mode", label_visibility="collapsed")
        df = ui.histogram_chart(before, after, mode)
        wide = df.pivot_table(index="Intensitas", columns=["Versi", "Channel"], values="Jumlah piksel")
        wide.columns = [f"{v}_{c}" for v, c in wide.columns]
        st.download_button("Unduh data histogram (CSV)", wide.to_csv().encode("utf-8"),
                           file_name=f"{base_name}_histogram.csv", mime="text/csv",
                           icon=":material/table_view:")

    with t_stats:
        rows = [("Lebar (px)", "width"), ("Tinggi (px)", "height"), ("Megapiksel", "megapixels"),
                ("Kecerahan rata-rata", "brightness"), ("Kontras (std)", "contrast"),
                ("Ketajaman (var. Laplacian)", "sharpness"), ("Entropi (bit)", "entropy"),
                ("Colorfulness", "colorfulness"), ("Piksel terpotong 0/255 (%)", "clipped_pct")]
        table = pd.DataFrame([{"Metrik": n, "Sebelum": sb[k], "Sesudah": sa[k], "Perubahan": sa[k] - sb[k]}
                              for n, k in rows])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Sebelum": st.column_config.NumberColumn(format="%.2f"),
            "Sesudah": st.column_config.NumberColumn(format="%.2f"),
            "Perubahan": st.column_config.NumberColumn(format="%+.2f")})
        meta = st.session_state.get("image_meta", {})
        if meta:
            st.caption(f"Sumber: **{meta.get('name')}** · format {meta.get('format')} ({meta.get('mode')}) · "
                       f"{helpers.human_bytes(meta.get('file_size', 0))} · ukuran asli "
                       f"{meta['original_size'][0]}×{meta['original_size'][1]} · resolusi kerja "
                       f"{meta['working_size'][0]}×{meta['working_size'][1]}"
                       f"{' · EXIF dirotasi otomatis' if meta.get('exif') else ''}")

    with t_hist_log:
        ops = state_manager.applied_operations()
        if not ops:
            st.caption("Belum ada langkah yang diterapkan. Tekan **Terapkan ke riwayat** untuk menyimpan edit.")
        else:
            st.dataframe(pd.DataFrame({"Langkah": range(1, len(ops) + 1), "Operasi": ops}),
                         hide_index=True, width="stretch")
        log = {"aplikasi": "PicaLab", "diekspor": datetime.now().isoformat(timespec="seconds"),
               "sumber": st.session_state.get("image_meta", {}).get("name"), "langkah": ops,
               "pratinjau_aktif": result.label}
        st.download_button("Unduh log edit (JSON)", json.dumps(log, indent=2, ensure_ascii=False),
                           file_name=f"{base_name}_log.json", mime="application/json", icon=":material/description:")

    with t_export:
        c1, c2, c3 = st.columns([1, 1.4, 1.6])
        fmt = c1.selectbox("Format", list(helpers.EXPORT_FORMATS), key="exp_fmt")
        spec = helpers.EXPORT_FORMATS[fmt]
        quality = c2.slider("Kualitas", 50, 100, 92, key="exp_q", disabled=not spec["lossy"])
        name = c3.text_input("Nama file", f"{base_name}_picalab", key="exp_name").strip() or "picalab"
        key = st.session_state["_last_key"]
        data_prev = cached_encode(f"{key}:{fmt}:{quality}", after, spec["ext"], quality)
        data_cur = cached_encode(f"{state_manager.session_uid()}:{st.session_state['img_version']}:{fmt}:{quality}",
                                 before, spec["ext"], quality)
        d1, d2 = st.columns(2)
        d1.download_button(f"Unduh pratinjau · {helpers.human_bytes(len(data_prev))}", data_prev,
                           file_name=f"{name}{spec['ext']}", mime=spec["mime"], type="primary",
                           icon=":material/download:", width="stretch")
        d2.download_button(f"Unduh gambar saat ini · {helpers.human_bytes(len(data_cur))}", data_cur,
                           file_name=f"{name}_current{spec['ext']}", mime=spec["mime"],
                           icon=":material/save:", width="stretch")
        st.caption("“Pratinjau” = gambar saat ini + filter aktif. “Gambar saat ini” = hasil riwayat yang sudah diterapkan.")


def render_photo_workspace(params):
    # Catatan: _handle_photo_upload() dipanggil di main() SEBELUM sidebar dirender,
    # agar parameter yang bergantung ukuran gambar (Resize/Crop/Perspektif) memakai
    # dimensi foto yang baru diunggah, bukan foto sebelumnya.
    if not state_manager.has_image():
        _render_feature_overview()
        return

    current = st.session_state["current_image"]
    key = (f"{state_manager.session_uid()}:{st.session_state['img_version']}:"
           f"{helpers.params_fingerprint(params)}")
    st.session_state["_from_cache"] = st.session_state.get("_last_key") == key
    st.session_state["_last_key"] = key
    with st.spinner(f"Menerapkan {pipeline.tool_label(params)}…"):
        result = cached_process(key, current, params)
    changed = result.image.shape != current.shape or not np.array_equal(result.image, current)

    # Toolbar
    with st.container(border=True):
        t1, t2, t3 = st.columns([1.4, 2.2, 1.6], vertical_alignment="center")
        t1.button("Terapkan ke riwayat", icon=":material/check_circle:", type="primary", width="stretch",
                  disabled=not (result.ok and changed) or bool(result.warning), on_click=_apply_to_history,
                  args=(result.image, result.label),
                  help=("Simpan pratinjau sebagai langkah edit (bisa di-undo)" if changed
                        else "Filter belum mengubah gambar — atur parameter di sidebar"))
        compare_mode = t2.segmented_control(
            "Tampilan", ["side", "slider", "result"], default="side", required=True, key="compare_mode",
            format_func={"side": "Berdampingan", "slider": "Slider pembanding", "result": "Hasil saja"}.get,
            label_visibility="collapsed")
        t3.markdown(f'<div style="text-align:right"><span class="pl-tag accent">{result.label}</span> '
                    f'<span class="pl-tag">{result.elapsed_ms:.0f} ms</span></div>', unsafe_allow_html=True)

    _render_messages(result)
    sb, sa = _render_metrics(current, result.image, result)
    _render_images(current, result, compare_mode)

    base_name = Path(st.session_state.get("image_meta", {}).get("name") or "foto").stem
    ui.section_title("Analisis & Ekspor", "histogram interaktif, statistik, riwayat, unduhan")
    _render_analysis(current, result.image, result, sb, sa, base_name)


# ---------------------------------------------------------------------------
# Workspace: VIDEO
# ---------------------------------------------------------------------------
def _reset_video_state():
    for key in ("video_input_path", "video_result"):
        item = st.session_state.pop(key, None)
        path = item.get("output_path") if isinstance(item, dict) else item
        media_handler.safe_unlink(path)
    st.session_state.pop("video_info", None)


def _handle_video_upload() -> bool:
    """Uploader video. Return True bila ada video valid yang siap dirender."""
    with st.container(border=True):
        video_file = st.file_uploader("Unggah video", type=VIDEO_TYPES, key="video_upload",
                                      help="Disarankan klip pendek (< 1 menit). Audio tidak disertakan pada hasil.")
    if video_file is None:
        if st.session_state.get("video_input_path"):
            _reset_video_state()
            st.session_state.pop("video_sid", None)
        return False

    sid = helpers.source_id(video_file)
    if st.session_state.get("video_sid") != sid:
        _reset_video_state()
        st.session_state["video_sid"] = sid
        st.session_state["video_file_size"] = video_file.size
        path = media_handler.save_upload_to_temp(video_file)
        try:
            st.session_state["video_info"] = media_handler.probe_video(path)
            st.session_state["video_input_path"] = path
        except ValueError as exc:
            media_handler.safe_unlink(path)
            st.session_state.pop("video_sid", None)
            st.error(str(exc), icon=":material/error:")
            return False
    return "video_info" in st.session_state


def render_video_workspace(params, video_ready: bool):
    if not video_ready:
        ui.empty_state("🎬", "Belum ada video",
                       "Unggah klip video, pilih filter di sidebar, lalu render. Hasil ditampilkan berdampingan "
                       "dengan aslinya (opsional histogram live) dan siap diunduh sebagai MP4 H.264.")
        return

    info = st.session_state["video_info"]
    ui.metric_cards([
        {"label": "Resolusi", "value": f"{info.width}×{info.height}", "icon": "📐"},
        {"label": "Frame rate", "value": f"{info.fps:.2f} fps", "icon": "⏱️"},
        {"label": "Durasi", "value": f"{info.duration:.1f} s", "icon": "🕒"},
        {"label": "Total frame", "value": f"{info.frames:,}", "icon": "🎞️"},
        {"label": "Codec", "value": info.codec.upper(), "icon": "🧬"},
        {"label": "Ukuran file", "value": helpers.human_bytes(st.session_state.get("video_file_size", 0)),
         "icon": "💾"},
    ])

    with st.container(border=True):
        st.markdown('<div class="pl-side-label">Pengaturan render</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        layout = c1.selectbox("Tata letak", ["side", "result", "stack"], key="vid_layout",
                              format_func={"side": "Berdampingan", "result": "Hasil saja", "stack": "Atas-bawah"}.get)
        scale = c2.selectbox("Skala proses", [1.0, 0.75, 0.5], key="vid_scale",
                             format_func=lambda s: f"{int(s * 100)}%", help="Perkecil untuk render lebih cepat")
        max_dur = max(1, int(round(info.duration))) if info.duration else 60
        seconds = c3.slider("Durasi diproses (detik)", 1, min(max_dur, 300), min(max_dur, 15), key="vid_secs")
        show_hist = c4.toggle("Histogram live", False, key="vid_hist")
        st.caption(f"Filter aktif: **{pipeline.tool_label(params)}** · ±{int(seconds * info.fps)} frame akan diproses.")
        start = st.button("Render video", icon=":material/play_circle:", type="primary", key="vid_render")

    if start:
        old = st.session_state.pop("video_result", None)
        if old:
            media_handler.safe_unlink(old["output_path"])
        with st.status("Merender video frame-demi-frame…", expanded=True) as status:
            bar = st.progress(0.0)
            text = st.empty()

            def on_progress(done, total, elapsed):
                frac = min(done / total, 1.0) if total else 0.0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed and total else 0
                bar.progress(frac)
                text.caption(f"Frame {done:,}/{total:,} · {speed:.1f} fps · sisa ±{eta:.0f} s")

            try:
                result = media_handler.render_video(
                    st.session_state["video_input_path"],
                    lambda f: pipeline.apply_processing_frame(f, params),
                    layout=layout, show_hist=show_hist, max_seconds=seconds, process_scale=scale,
                    progress_cb=on_progress)
                result["label"] = pipeline.tool_label(params)
                st.session_state["video_result"] = result
                status.update(label=f"Render selesai dalam {result['seconds']:.1f} s", state="complete",
                              expanded=False)
                st.toast("Video selesai dirender", icon="🎬")
            except Exception as exc:
                status.update(label="Render gagal", state="error", expanded=True)
                st.error(f"Gagal merender video: {exc}", icon=":material/error:")

    res = st.session_state.get("video_result")
    if res and os.path.exists(res["output_path"]):
        ui.section_title("Hasil render", res["label"])
        ui.metric_cards([
            {"label": "Frame", "value": f"{res['frames']:,}", "icon": "🎞️"},
            {"label": "Waktu render", "value": f"{res['seconds']:.1f} s", "icon": "⏱️"},
            {"label": "Kecepatan", "value": f"{res['frames'] / max(res['seconds'], 1e-6):.1f} fps", "icon": "⚡"},
            {"label": "Ukuran", "value": helpers.human_bytes(res["size"]), "icon": "💾"},
            {"label": "Encoder", "value": res["codec"], "icon": "🧬"},
        ])
        if res["codec"] == "mpeg4":
            st.warning("Encoder H.264 tidak tersedia — video MPEG-4 mungkin tidak bisa diputar di browser, "
                       "tetapi tetap bisa diunduh.", icon=":material/warning:")
        st.video(res["output_path"])
        with open(res["output_path"], "rb") as fh:
            st.download_button("Unduh video hasil (MP4)", fh.read(), file_name="picalab_video.mp4",
                               mime="video/mp4", type="primary", icon=":material/download:")


# ---------------------------------------------------------------------------
# Workspace: LIVE CAMERA
# ---------------------------------------------------------------------------
def render_camera_workspace(params):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
        options = {
            "layout": c1.selectbox("Tata letak", ["side", "result", "stack"], key="cam_layout",
                                   format_func={"side": "Berdampingan", "result": "Hasil saja",
                                                "stack": "Atas-bawah"}.get),
            "mirror": c2.toggle("Mode cermin (selfie)", True, key="cam_mirror"),
            "show_hist": c3.toggle("Histogram live", False, key="cam_hist"),
            "show_fps": c4.toggle("Tampilkan FPS", True, key="cam_fps"),
        }

    ctx = media_handler.start_webrtc_camera(key="picalab-live")
    processor = ctx.video_processor
    if processor:
        processor.update(params, options)

    playing = bool(ctx.state.playing)
    st.markdown(f'<div class="pl-status pl-status-{"ok" if playing else "err"}" style="max-width:420px">'
                f'<span class="pl-dot"></span>{"Kamera aktif" if playing else "Kamera belum aktif"}'
                f'<small>Filter: {pipeline.tool_label(params)} — ubah di sidebar, langsung berlaku.</small></div>',
                unsafe_allow_html=True)

    if st.button("Ambil gambar", icon=":material/photo_camera:", type="primary", disabled=not playing):
        original, filtered = processor.snapshot() if processor else (None, None)
        if original is None:
            st.warning("Frame belum tersedia. Tunggu kamera tampil lalu coba lagi.", icon=":material/hourglass:")
        else:
            st.session_state["cam_capture"] = {"original": original, "filtered": filtered,
                                               "label": pipeline.tool_label(params),
                                               "time": datetime.now().strftime("%H%M%S")}
            st.toast("Gambar berhasil ditangkap", icon="📸")

    cap = st.session_state.get("cam_capture")
    if cap:
        ui.section_title("Hasil tangkapan", cap["label"])
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ui.frame_title("Asli")
            st.image(helpers.bgr_to_rgb(cap["original"]), width="stretch")
            st.download_button("Unduh asli (PNG)", helpers.encode_image_for_download(cap["original"]),
                               file_name=f"picalab_cam_{cap['time']}_asli.png", mime="image/png",
                               icon=":material/download:", width="stretch")
            st.button("Edit foto asli di Editor", icon=":material/edit:", width="stretch",
                      on_click=_send_capture_to_editor, args=("original",))
        with c2:
            ui.frame_title("Dengan filter", cap["label"], accent=True)
            st.image(helpers.bgr_to_rgb(cap["filtered"]), width="stretch")
            st.download_button("Unduh hasil filter (PNG)", helpers.encode_image_for_download(cap["filtered"]),
                               file_name=f"picalab_cam_{cap['time']}_filter.png", mime="image/png",
                               type="primary", icon=":material/download:", width="stretch")
            st.button("Lanjut edit hasil di Editor", icon=":material/tune:", width="stretch",
                      on_click=_send_capture_to_editor, args=("filtered",))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ui.inject_css()
    ui.hero("PicaLab Studio",
            "Mini Photoshop untuk pengolahan citra digital — geometri, warna, restorasi, segmentasi, "
            "efek kreatif, hingga beauty berbasis AI. Bekerja pada foto, video, dan kamera secara real-time.",
            chips=("35 alat", "Undo / Redo", "Histogram interaktif", "Video H.264", "Live WebRTC", "MediaPipe AI"))

    # Nilai awal lewat Session State (bukan `default=`) karena tombol "Kirim ke Editor"
    # di tab kamera mengubah nav_mode secara terprogram.
    st.session_state.setdefault("nav_mode", "photo")
    mode = st.segmented_control("Mode kerja", list(MODES), required=True,
                                format_func=MODES.get, key="nav_mode", label_visibility="collapsed")

    # Upload diproses SEBELUM sidebar agar parameter berbasis ukuran gambar selalu sinkron
    video_ready = False
    if mode == "photo":
        _handle_photo_upload()
    elif mode == "video":
        video_ready = _handle_video_upload()

    img_shape = None
    if mode == "photo" and state_manager.has_image():
        img_shape = st.session_state["current_image"].shape
    elif mode == "video" and st.session_state.get("video_info"):
        info = st.session_state["video_info"]
        img_shape = (info.height, info.width)
    elif mode == "camera":
        img_shape = (540, 960)

    params = sidebar.render_sidebar(img_shape, show_history=(mode == "photo"))

    if mode == "photo":
        render_photo_workspace(params)
    elif mode == "video":
        render_video_workspace(params, video_ready)
    else:
        render_camera_workspace(params)


if __name__ == "__main__":
    main()
