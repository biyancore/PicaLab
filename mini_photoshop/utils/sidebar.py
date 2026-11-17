"""
Sidebar PicaLab: kontrol riwayat + pemilihan kategori/alat + parameter filter.

Dipindah dari app.py (dulu ±180 baris di file utama). Perubahan utama:
- Setiap alat memakai KODE stabil (lihat utils.pipeline.CATEGORIES).
- Parameter geometri menyesuaikan ukuran gambar aktif (Resize default ke ukuran
  asli + kunci rasio; Crop pakai slider rentang; titik Perspektif default di
  sudut gambar). Versi lama: Resize selalu 800x600 dan titik Perspektif (0,0)
  sehingga fitur tersebut tidak berfungsi tanpa input manual.
- Undo/Redo/Reset memakai callback on_click (tanpa st.rerun ganda).
"""

import streamlit as st

from modules import beauty
from utils import helpers, state_manager
from utils.pipeline import CATEGORIES, NO_INTENSITY_CATEGORIES

LIPSTICK_PRESETS = {
    "Merah Merona": "#B40F1E",
    "Pink Soft": "#FF6993",
    "Peach": "#FF8264",
    "Plum / Gelap": "#64143C",
    "Nude": "#C47A6A",
}

# (h_min, h_max, s_min, s_max, v_min, v_max, wrap) — skala OpenCV
COLOR_PRESETS = {
    "Kustom": (0, 179, 0, 255, 0, 255, False),
    "Merah": (10, 170, 70, 255, 50, 255, True),
    "Oranye": (10, 25, 80, 255, 80, 255, False),
    "Kuning": (22, 35, 70, 255, 80, 255, False),
    "Hijau": (36, 85, 50, 255, 40, 255, False),
    "Biru": (90, 130, 60, 255, 40, 255, False),
    "Ungu": (130, 160, 50, 255, 40, 255, False),
    "Kulit": (0, 25, 30, 170, 60, 255, False),
}

POSITIONS = {"bottom-right": "Kanan Bawah", "bottom-left": "Kiri Bawah", "top-right": "Kanan Atas",
             "top-left": "Kiri Atas", "center": "Tengah"}


def _odd_slider(label, lo, hi, value, key, help=None):
    return st.sidebar.slider(label, lo, hi, value, step=2, key=key, help=help)


def _select(label, options: dict, key, help=None):
    """Selectbox berbasis kode -> label."""
    return st.sidebar.selectbox(label, list(options), format_func=options.get, key=key, help=help)


# ---------------------------------------------------------------------------
# Riwayat
# ---------------------------------------------------------------------------
def render_history_controls():
    st.sidebar.markdown('<div class="pl-side-label">Riwayat Edit</div>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    c1.button("Undo", icon=":material/undo:", on_click=state_manager.undo,
              disabled=not state_manager.can_undo(), width="stretch", key="btn_undo",
              help="Kembali satu langkah")
    c2.button("Redo", icon=":material/redo:", on_click=state_manager.redo,
              disabled=not state_manager.can_redo(), width="stretch", key="btn_redo",
              help="Maju satu langkah")
    st.sidebar.button("Reset ke foto asli", icon=":material/restart_alt:", on_click=state_manager.reset_to_original,
                      disabled=not state_manager.applied_operations(), width="stretch", key="btn_reset",
                      help="Hapus semua langkah edit")
    if state_manager.has_image():
        steps = len(state_manager.applied_operations())
        st.sidebar.caption(f"{steps} langkah diterapkan · riwayat ±{state_manager.history_memory_mb():.0f} MB "
                           f"(maks {state_manager.MAX_HISTORY} langkah undo)")


# ---------------------------------------------------------------------------
# Menu per kategori
# ---------------------------------------------------------------------------
def render_geometry_menu(tool, img_shape):
    h, w = (img_shape[:2] if img_shape is not None else (600, 800))
    dim = f"{w}x{h}"
    p = {}
    if tool == "resize":
        lock = st.sidebar.toggle("Kunci rasio aspek", value=True, key="p_resize_lock")
        p["width"] = st.sidebar.number_input("Lebar (px)", 1, 8000, min(w, 8000), key=f"p_resize_w_{dim}")
        if lock:
            p["height"] = max(1, round(p["width"] * h / w))
            st.sidebar.caption(f"Tinggi otomatis: **{p['height']} px**")
        else:
            p["height"] = st.sidebar.number_input("Tinggi (px)", 1, 8000, min(h, 8000), key=f"p_resize_h_{dim}")
        p["method"] = st.sidebar.selectbox("Interpolasi", ["Bilinear", "Bicubic", "Lanczos", "Area",
                                                           "Nearest Neighbor"], key="p_resize_m")
    elif tool == "resample":
        p["scale"] = st.sidebar.slider("Faktor skala", 0.1, 4.0, 1.0, 0.05, key="p_rs_scale",
                                       format="%.2fx")
        p["method"] = st.sidebar.selectbox("Interpolasi", ["Bicubic", "Bilinear", "Lanczos", "Area",
                                                           "Nearest Neighbor"], key="p_rs_m")
        st.sidebar.caption(f"Hasil: **{max(1, round(w * p['scale']))} × {max(1, round(h * p['scale']))} px**")
    elif tool == "crop":
        ratios = {"free": "Bebas", "1:1": "1:1 (Persegi)", "4:3": "4:3", "3:2": "3:2", "16:9": "16:9",
                  "9:16": "9:16 (Story)"}
        ratio = _select("Rasio", ratios, "p_crop_ratio")
        if ratio == "free":
            xs = st.sidebar.slider("Rentang X (px)", 0, w, (int(w * 0.1), int(w * 0.9)), key=f"p_crop_x_{dim}")
            ys = st.sidebar.slider("Rentang Y (px)", 0, h, (int(h * 0.1), int(h * 0.9)), key=f"p_crop_y_{dim}")
        else:
            rw, rh = (int(v) for v in ratio.split(":"))
            zoom = st.sidebar.slider("Ukuran area (%)", 20, 100, 90, key="p_crop_zoom") / 100
            cw = min(w, int(h * rw / rh)) * zoom
            ch = cw * rh / rw
            cx = st.sidebar.slider("Posisi horizontal (%)", 0, 100, 50, key="p_crop_cx") / 100
            cy = st.sidebar.slider("Posisi vertikal (%)", 0, 100, 50, key="p_crop_cy") / 100
            x0, y0 = (w - cw) * cx, (h - ch) * cy
            xs, ys = (int(x0), int(x0 + cw)), (int(y0), int(y0 + ch))
        p.update(x_start=xs[0], x_end=xs[1], y_start=ys[0], y_end=ys[1])
        st.sidebar.caption(f"Hasil crop: **{xs[1] - xs[0]} × {ys[1] - ys[0]} px**")
    elif tool == "flip":
        p["mode"] = _select("Mode cermin", {"Horizontal": "Horizontal ↔", "Vertical": "Vertikal ↕",
                                            "Both": "Keduanya"}, "p_flip")
    elif tool == "rotate":
        p["angle"] = st.sidebar.slider("Sudut (°)", -180, 180, 0, key="p_rot_angle",
                                       help="Positif = berlawanan arah jarum jam")
        p["keep_canvas"] = st.sidebar.toggle("Perluas kanvas (tidak terpotong)", True, key="p_rot_keep")
        p["border"] = helpers.hex_to_bgr(st.sidebar.color_picker("Warna latar sudut kosong", "#000000",
                                                                  key="p_rot_border"))
    elif tool == "perspective":
        st.sidebar.caption("Atur 4 titik sudut objek (urutan bebas — akan diurutkan otomatis).")
        mx, my = int(w * 0.05), int(h * 0.05)
        defaults = [("Kiri-Atas", mx, my), ("Kanan-Atas", w - mx, my),
                    ("Kanan-Bawah", w - mx, h - my), ("Kiri-Bawah", mx, h - my)]
        pts = []
        for i, (name, dx, dy) in enumerate(defaults):
            c1, c2 = st.sidebar.columns(2)
            x = c1.number_input(f"{name} X", 0, w, dx, key=f"p_pt{i}x_{dim}")
            y = c2.number_input(f"{name} Y", 0, h, dy, key=f"p_pt{i}y_{dim}")
            pts.append((x, y))
        p["pts"] = pts
    return p


def render_color_menu(tool):
    p = {}
    if tool == "saturation":
        p["factor"] = st.sidebar.slider("Intensitas saturasi", 0.0, 3.0, 1.4, 0.05, key="p_sat",
                                        help="0 = abu-abu, 1 = asli, >1 = lebih jenuh")
    elif tool == "hue":
        p["degrees"] = st.sidebar.slider("Pergeseran hue (°)", -180, 180, 60, key="p_hue",
                                         help="Rotasi pada roda warna 360°")
    elif tool == "channels":
        p["channel"] = _select("Channel", {"R": "🔴 Red", "G": "🟢 Green", "B": "🔵 Blue"}, "p_ch")
        p["tinted"] = st.sidebar.toggle("Tampilkan berwarna (tinted)", True, key="p_ch_tint")
    return p


def render_enhancement_menu(tool):
    p = {}
    if tool == "bc":
        p["brightness"] = st.sidebar.slider("Brightness", -100, 100, 15, key="p_bc_b")
        p["contrast"] = st.sidebar.slider("Contrast", 0.1, 3.0, 1.15, 0.05, key="p_bc_c")
    elif tool == "gamma":
        p["gamma"] = st.sidebar.slider("Gamma", 0.2, 3.0, 1.4, 0.05, key="p_gamma",
                                       help=">1 mencerahkan bayangan, <1 menggelapkan")
    elif tool == "histeq":
        p["method"] = _select("Metode", {"clahe": "CLAHE (adaptif, direkomendasikan)",
                                         "global": "Histogram Equalization global"}, "p_he_m")
        p["clip_limit"] = st.sidebar.slider("Clip limit", 0.5, 8.0, 2.0, 0.1, key="p_he_clip",
                                            disabled=p["method"] != "clahe")
        p["tile_size"] = st.sidebar.slider("Ukuran tile", 2, 16, 8, key="p_he_tile",
                                           disabled=p["method"] != "clahe")
    elif tool == "denoise":
        p["method"] = _select("Metode", {"nlm": "Non-Local Means (terbaik, lambat)", "gaussian": "Gaussian",
                                         "median": "Median (salt & pepper)", "bilateral": "Bilateral (jaga tepi)"},
                              "p_dn_m")
        p["strength"] = st.sidebar.slider("Kekuatan", 1, 30, 10, key="p_dn_s",
                                          disabled=p["method"] in ("gaussian", "median"))
        p["kernel"] = _odd_slider("Ukuran kernel", 3, 15, 5, "p_dn_k")
    elif tool == "blur":
        p["method"] = _select("Jenis blur", {"gaussian": "Gaussian", "averaging": "Averaging (Box)",
                                             "median": "Median", "bilateral": "Bilateral"}, "p_bl_m")
        hi = 25 if p["method"] == "bilateral" else 51
        p["kernel"] = _odd_slider("Ukuran kernel", 3, hi, 9, f"p_bl_k_{hi}")
        p["sigma_color"] = st.sidebar.slider("Sigma warna", 10, 200, 75, key="p_bl_sc",
                                             disabled=p["method"] != "bilateral")
    elif tool == "sharpen":
        p["method"] = _select("Metode", {"unsharp": "Unsharp Mask", "laplacian": "Kernel Laplacian",
                                         "high_boost": "Kernel High-boost"}, "p_sh_m")
        p["strength"] = st.sidebar.slider("Strength (kernel)", 0.0, 1.0, 0.7, 0.05, key="p_sh_s",
                                          disabled=p["method"] == "unsharp")
        p["amount"] = st.sidebar.slider("Amount (unsharp)", 0.0, 3.0, 1.2, 0.1, key="p_sh_a",
                                        disabled=p["method"] != "unsharp")
    elif tool == "deblur":
        p["method"] = _select("Metode", {"wiener_gaussian": "Wiener — blur umum",
                                         "wiener_motion": "Wiener — blur gerak",
                                         "unsharp": "Unsharp deblur", "inpaint": "Inpainting goresan"},
                              "p_db_m")
        m = p["method"]
        if m.startswith("wiener"):
            p["size"] = _odd_slider("Ukuran PSF", 3, 31, 7, "p_db_size")
            if m == "wiener_gaussian":
                p["sigma"] = st.sidebar.slider("Sigma PSF", 0.5, 6.0, 2.0, 0.1, key="p_db_sigma")
            else:
                p["angle"] = st.sidebar.slider("Arah gerak (°)", 0, 180, 0, key="p_db_angle")
            p["k"] = st.sidebar.select_slider("Regularisasi K", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1],
                                              0.01, key="p_db_k", help="Kecil = lebih tajam tapi noise/ringing naik")
        elif m == "unsharp":
            p["amount"] = st.sidebar.slider("Amount", 0.5, 4.0, 2.0, 0.1, key="p_db_amt")
            p["sigma"] = st.sidebar.slider("Sigma", 0.5, 5.0, 2.0, 0.1, key="p_db_us_sigma")
        else:
            p["damage"] = _select("Jenis kerusakan", {"bright": "Goresan terang", "dark": "Bercak gelap"},
                                  "p_db_dmg")
            p["threshold"] = st.sidebar.slider("Ambang deteksi", 150, 254, 240, key="p_db_thr")
            p["radius"] = st.sidebar.slider("Radius inpaint", 1, 10, 3, key="p_db_rad")
    elif tool == "auto":
        st.sidebar.caption("Pipeline: Denoise → CLAHE → Brightness/Contrast → Sharpen")
        p["denoise"] = st.sidebar.slider("Denoise (0 = mati)", 0, 15, 5, key="p_auto_dn")
        p["use_clahe"] = st.sidebar.toggle("CLAHE", True, key="p_auto_clahe")
        p["clip_limit"] = st.sidebar.slider("Clip limit", 0.5, 6.0, 2.0, 0.1, key="p_auto_clip",
                                            disabled=not p["use_clahe"])
        p["brightness"] = st.sidebar.slider("Brightness", -60, 60, 0, key="p_auto_b")
        p["contrast"] = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.05, key="p_auto_c")
        p["sharpen"] = st.sidebar.slider("Sharpen", 0.0, 2.0, 0.8, 0.1, key="p_auto_sh")
    return p


def render_segmentation_menu(tool):
    p = {}
    if tool == "threshold":
        p["method"] = _select("Metode", {"otsu": "Otsu (otomatis)", "global": "Global",
                                         "adaptive": "Adaptive (lokal)"}, "p_th_m")
        p["invert"] = st.sidebar.toggle("Balik hitam & putih", False, key="p_th_inv")
        if p["method"] == "global":
            p["value"] = st.sidebar.slider("Nilai ambang", 0, 255, 127, key="p_th_v",
                help="0 = hitam, 255 = putih. Tanpa pembalikan warna, ambang lebih tinggi menghasilkan lebih banyak hitam.")
        elif p["method"] == "adaptive":
            p["adaptive_method"] = _select("Metode adaptif", {"gaussian": "Gaussian", "mean": "Mean"}, "p_th_am",
                help="Mean memakai rata-rata area. Gaussian memberi bobot lebih besar pada piksel dekat pusat.")
            p["block_size"] = _odd_slider("Ukuran area lokal", 3, 51, 11, "p_th_bs",
                help="Ukuran lingkungan untuk menghitung ambang. Kecil lebih peka pada detail; besar mempertimbangkan area lebih luas. Harus ganjil.")
            p["c"] = st.sidebar.slider("Konstanta C", -10, 20, 2, key="p_th_c",
                help="Ambang lokal = rata-rata area − C. Tanpa pembalikan warna, C lebih besar menghasilkan lebih banyak putih.")
    elif tool == "colormask":
        preset = st.sidebar.selectbox("Preset warna", list(COLOR_PRESETS), index=1, key="p_cm_preset")
        h0, h1, s0, s1, v0, v1, wrap = COLOR_PRESETS[preset]
        hr = st.sidebar.slider("Rentang Hue (0–179)", 0, 179, (h0, h1), key=f"p_cm_h_{preset}")
        wrap = st.sidebar.toggle("Pilih di LUAR rentang hue (untuk merah)", wrap, key=f"p_cm_wrap_{preset}")
        sr = st.sidebar.slider("Rentang saturasi", 0, 255, (s0, s1), key=f"p_cm_s_{preset}")
        vr = st.sidebar.slider("Rentang kecerahan", 0, 255, (v0, v1), key=f"p_cm_v_{preset}")
        lo_h, hi_h = (hr[1], hr[0]) if wrap else hr
        p["lower"], p["upper"] = (lo_h, sr[0], vr[0]), (hi_h, sr[1], vr[1])
        p["output"] = _select("Tampilan hasil", {"highlight": "Sorot warna (latar abu-abu)",
                                                 "selected": "Warna terpilih saja", "mask": "Mask biner"},
                              "p_cm_out")
    elif tool == "morph":
        p["method"] = _select("Operasi", {"opening": "Opening", "closing": "Closing", "erosion": "Erosion",
                                          "dilation": "Dilation", "gradient": "Gradient (kontur)",
                                          "tophat": "Top-hat", "blackhat": "Black-hat"}, "p_mo_m")
        p["shape"] = _select("Bentuk kernel", {"rect": "Persegi", "ellipse": "Elips", "cross": "Silang"}, "p_mo_s",
            help="Bentuk area pemeriksaan: persegi mencakup sudut, elips mengikuti bentuk bulat, silang mengikuti arah mendatar dan tegak.")
        p["kernel"] = _odd_slider("Ukuran kernel", 1, 21, 5, "p_mo_k",
            help="Kernel adalah area kecil yang memeriksa objek putih. Ukuran lebih besar memengaruhi detail yang lebih besar.")
        p["iterations"] = st.sidebar.slider("Iterasi", 1, 10, 1, key="p_mo_it",
            help="Jumlah pengulangan erosion atau dilation dalam operasi. Nilai lebih besar memperkuat perubahan bentuk.")
    elif tool == "edges":
        p["method"] = _select("Operator", {"canny": "Canny", "sobel": "Sobel", "prewitt": "Prewitt",
                                           "laplacian": "Laplacian"}, "p_ed_m")
        p["kernel"] = st.sidebar.select_slider("Ukuran kernel", [3, 5, 7], 3, key="p_ed_k",
                                               disabled=p["method"] == "prewitt")
        if p["method"] == "canny":
            p["low"], p["high"] = st.sidebar.slider("Ambang Canny", 0, 255, (60, 160), key="p_ed_lh",
                help="Ambang atas menentukan tepi kuat. Tepi lemah di atas ambang bawah dipertahankan jika terhubung ke tepi kuat.")

    with st.sidebar.expander("Panduan singkat", expanded=False):
        if tool == "threshold":
            st.markdown({
                "global": "**Global** memakai satu batas kecerahan untuk seluruh gambar. Piksel di atas ambang menjadi putih, sisanya hitam.",
                "otsu": "**Otsu** memilih ambang otomatis dari histogram, lalu mengubah gambar menjadi hitam-putih. Cocok jika kecerahan objek dan latar cukup berbeda.",
                "adaptive": "**Adaptive** menghitung ambang untuk tiap area kecil. Berguna saat pencahayaan tidak merata atau ada bayangan.",
            }[p["method"]])
            st.caption("Balik hitam & putih menukar hasilnya. Histogram hasil biner hanya berisi nilai 0 dan 255; bagian tengah yang kosong itu normal.")
        elif tool == "colormask":
            st.markdown("**Color Masking** memilih bagian gambar berdasarkan warna. Mulai dari preset, lalu sesuaikan rentangnya.")
            st.markdown("**Hue** = jenis warna · **Saturasi** = kepekatan · **Kecerahan** = terang-gelap. Merah berada di dua ujung hue, sehingga memakai pilihan di luar rentang.")
            st.caption("Mask biner: putih = terpilih, hitam = tidak. Warna terpilih saja memberi latar hitam; Sorot warna memberi latar abu-abu.")
        elif tool == "morph":
            st.markdown({
                "erosion": "**Erosion** memperkecil objek putih dan mengikis bagian tepinya.",
                "dilation": "**Dilation** memperbesar objek putih dan dapat menyambung celah kecil.",
                "opening": "**Opening** mengikis lalu memperbesar kembali objek putih. Berguna untuk menghapus bintik putih kecil.",
                "closing": "**Closing** memperbesar lalu mengikis kembali objek putih. Berguna untuk menutup lubang dan celah hitam kecil.",
                "gradient": "**Gradient** mengambil selisih dilation dan erosion untuk menampilkan kontur objek.",
                "tophat": "**Top-hat** mengambil selisih gambar dengan hasil opening untuk menonjolkan detail putih kecil.",
                "blackhat": "**Black-hat** mengambil selisih hasil closing dengan gambar untuk menonjolkan detail hitam kecil.",
            }[p["method"]])
            st.caption("Foto diubah menjadi mask hitam-putih dengan Otsu terlebih dahulu. Operasi bekerja pada objek putih; arah efek bergantung pada warna objek di mask.")
        elif tool == "edges":
            st.markdown({
                "canny": "**Canny** menghasilkan garis tepi tipis dengan dua ambang. Naikkan ambang untuk menyaring tepi lemah; turunkan untuk menangkap lebih banyak detail.",
                "sobel": "**Sobel** mencari perubahan kecerahan arah mendatar dan tegak, lalu menggabungkannya menjadi peta tepi.",
                "prewitt": "**Prewitt** mencari perubahan kecerahan arah mendatar dan tegak dengan bobot seragam. Selalu memakai kernel 3×3.",
                "laplacian": "**Laplacian** memakai turunan kedua untuk menonjolkan perubahan kecerahan yang cepat. Noise juga bisa ikut terlihat.",
            }[p["method"]])
            st.caption("Garis terang menunjukkan batas objek atau perubahan kecerahan, termasuk tekstur dan noise.")
    return p


def render_effects_menu(tool):
    p = {}
    if tool == "sketch":
        p["blur_size"] = _odd_slider("Ukuran blur sketsa", 3, 51, 21, "p_sk_b")
    elif tool == "cartoon":
        p["line_size"] = _odd_slider("Ketebalan garis", 3, 21, 9, "p_ct_l")
        p["blur_value"] = _odd_slider("Penghalus garis", 3, 11, 5, "p_ct_b")
        p["passes"] = st.sidebar.slider("Kerataan warna (iterasi)", 1, 5, 3, key="p_ct_p")
    elif tool == "vignette":
        p["strength"] = st.sidebar.slider("Kekuatan vignette", 0.0, 2.0, 1.0, 0.05, key="p_vg")
    elif tool == "sepia":
        p["intensity"] = st.sidebar.slider("Intensitas sepia", 0.0, 1.0, 0.85, 0.05, key="p_sp")
    elif tool == "pixelate":
        p["block"] = st.sidebar.slider("Ukuran blok (px)", 2, 64, 12, key="p_px")
    elif tool == "emboss":
        p["strength"] = st.sidebar.slider("Kedalaman relief", 0.2, 3.0, 1.0, 0.1, key="p_em")
    elif tool == "watermark":
        p["wm_type"] = _select("Jenis", {"text": "Teks", "logo": "Logo (PNG)"}, "p_wm_t")
        p["pos"] = _select("Posisi", POSITIONS, "p_wm_pos")
        p["opacity"] = st.sidebar.slider("Opacity", 0.0, 1.0, 0.6, 0.05, key="p_wm_op")
        if p["wm_type"] == "text":
            p["text"] = st.sidebar.text_input("Teks watermark", "© PicaLab", max_chars=60, key="p_wm_txt")
            if not p["text"].strip():
                st.sidebar.warning("Teks watermark kosong — isi teks terlebih dahulu.")
                p["text"] = "PicaLab"
            p["color"] = helpers.hex_to_rgb(st.sidebar.color_picker("Warna teks", "#FFFFFF", key="p_wm_col"))
            p["font_pct"] = st.sidebar.slider("Ukuran font (% lebar gambar)", 1.0, 15.0, 4.0, 0.5, key="p_wm_fs")
        else:
            logo_file = st.sidebar.file_uploader("Upload logo (PNG transparan disarankan)",
                                                 type=["png", "jpg", "jpeg", "webp"], key="p_wm_logo")
            p["logo_pct"] = st.sidebar.slider("Lebar logo (% lebar gambar)", 5, 60, 20, key="p_wm_lp")
            p["logo"] = None
            if logo_file is not None:
                try:
                    p["logo"] = helpers.decode_logo(logo_file)
                except ValueError as exc:
                    st.sidebar.error(str(exc))
    return p


def render_beauty_menu(tool):
    p = {}
    status = beauty.backend_status()
    css = "ok" if status["available"] else "err"
    st.sidebar.markdown(
        f'<div class="pl-status pl-status-{css}"><span class="pl-dot"></span>{status["name"]}'
        f'<small>{status["detail"]}</small></div>', unsafe_allow_html=True)

    if tool == "teeth":
        p["intensity"] = st.sidebar.slider("Intensitas", 0.0, 1.0, 0.7, 0.05, key="p_bt_t")
    elif tool == "blush":
        p["color"] = helpers.hex_to_bgr(st.sidebar.color_picker("Warna blush", "#FF6993", key="p_bt_bc"))
        p["intensity"] = st.sidebar.slider("Ketebalan blush", 0.0, 1.0, 0.5, 0.05, key="p_bt_bi")
    elif tool == "lipstick":
        choice = st.sidebar.selectbox("Warna lipstik", list(LIPSTICK_PRESETS) + ["Kustom"], key="p_bt_lp")
        hex_color = (st.sidebar.color_picker("Warna kustom", "#C2185B", key="p_bt_lc")
                     if choice == "Kustom" else LIPSTICK_PRESETS[choice])
        p["color"] = helpers.hex_to_bgr(hex_color)
        p["intensity"] = st.sidebar.slider("Ketebalan lipstik", 0.0, 1.0, 0.55, 0.05, key="p_bt_li")
    elif tool == "bokeh":
        p["kernel"] = _odd_slider("Tingkat blur latar", 11, 151, 51, "p_bt_bk")
    elif tool == "mesh":
        st.sidebar.caption("Menampilkan 478 titik landmark wajah yang dipakai fitur Beauty.")
    return p


_MENUS = {
    "color": render_color_menu, "enhance": render_enhancement_menu,
    "segment": render_segmentation_menu, "effects": render_effects_menu, "beauty": render_beauty_menu,
}


def render_sidebar(img_shape=None, show_history: bool = True) -> dict:
    """Merender sidebar dan mengembalikan dict parameter filter."""
    st.sidebar.markdown(
        '<div class="pl-brand"><div class="pl-logo">P</div>'
        '<div><b>PicaLab</b><span>Studio Pengolahan Citra</span></div></div>',
        unsafe_allow_html=True)

    if show_history:
        render_history_controls()
        st.sidebar.divider()

    st.sidebar.markdown('<div class="pl-side-label">Alat</div>', unsafe_allow_html=True)
    category = st.sidebar.selectbox(
        "Kategori", list(CATEGORIES),
        format_func=lambda c: f"{CATEGORIES[c]['icon']}  {CATEGORIES[c]['label']}", key="p_category")
    tools = CATEGORIES[category]["tools"]
    tool = st.sidebar.selectbox("Alat", list(tools), format_func=tools.get, key=f"p_tool_{category}")

    params = {"category": category, "tool": tool}
    if category == "geometry":
        params.update(render_geometry_menu(tool, img_shape))
    else:
        params.update(_MENUS[category](tool))

    from utils.pipeline import NO_INTENSITY_TOOLS
    if category not in NO_INTENSITY_CATEGORIES and (category, tool) not in NO_INTENSITY_TOOLS:
        params["intensity_pct"] = st.sidebar.slider(
            "Kekuatan filter (%)", 0, 100, 100, step=5, key="p_intensity",
            help="Campuran hasil filter dengan gambar asli (alpha blending)")
    else:
        params["intensity_pct"] = 100

    st.sidebar.divider()
    st.sidebar.caption("PicaLab · Pengolahan Citra")
    return params
