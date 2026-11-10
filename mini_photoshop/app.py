"""Jalankan dari folder mini_photoshop: streamlit run app.py."""

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from modules import segmentation as seg
from modules import enhancement as enh


def png_bytes(image):
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def threshold_controls(image, prefix):
    method = st.selectbox("Metode threshold", ["Global", "Adaptive", "Otsu"], key=f"{prefix}_method")
    invert = st.checkbox("Balik hitam dan putih", key=f"{prefix}_invert")
    if method == "Global":
        value = st.slider("Nilai threshold", 0, 255, 127, key=f"{prefix}_value")
        return seg.global_threshold(image, value, invert)
    if method == "Adaptive":
        adaptive = st.selectbox("Metode adaptif", ["gaussian", "mean"], key=f"{prefix}_adaptive")
        size = st.slider("Ukuran area lokal", 3, 51, 11, step=2, key=f"{prefix}_size")
        c = st.slider("Konstanta C", -20, 20, 2, key=f"{prefix}_c")
        return seg.adaptive_threshold(image, adaptive, size, c, invert)
    return seg.otsu_threshold(image, invert)

ENHANCEMENT_FEATURES = ["Brightness dan Kontras", "Histogram Equalization", "Enhancement Komprehensif",
                        "Denoising", "Blur Suite", "Sharpening", "Deblurring dan Restorasi"]
HISTOGRAM_FEATURES = ("Brightness dan Kontras", "Histogram Equalization")

def show_histograms(before, after):
    """Grafik histogram sebelum dan sesudah (sumbu X intensitas, sumbu Y jumlah piksel)."""
    st.subheader("Histogram sebelum dan sesudah")
    mode = st.radio("Mode histogram", ["Intensitas (grayscale)", "Per kanal RGB"],
                    horizontal=True, key="enh_hist_mode")
    st.pyplot(enh.histogram_figure(before, after, "rgb" if mode.startswith("Per") else "gray"))

def enhancement_controls(image, feature):
    """Kontrol sidebar fitur Aliza. Semua widget diberi key berawalan 'enh_'."""
    if feature == "Brightness dan Kontras":
        brightness = st.slider("Brightness", -100, 100, 0, key="enh_bc_b")
        contrast = st.slider("Contrast", 0.1, 3.0, 1.0, step=0.1, key="enh_bc_c")
        gamma = st.slider("Gamma", 0.2, 3.0, 1.0, step=0.1, key="enh_bc_g")
        st.checkbox("Tampilkan histogram", True, key="enh_show_hist")      
        return enh.gamma_correction(enh.brightness_contrast(image, brightness, contrast), gamma)
    
    if feature == "Histogram Equalization":
        method = st.radio("Metode", ["Histogram Equalization", "CLAHE"], key="enh_he_method")
        if method == "Histogram Equalization":
            return enh.histogram_equalization(image)
        clip = st.slider("Clip limit", 1.0, 10.0, 2.0, step=0.5, key="enh_he_clip")
        tile = st.slider("Ukuran tile", 2, 16, 8, key="enh_he_tile")
        return enh.clahe(image, clip, tile)
    if feature == "Enhancement Komprehensif":
        st.caption("Urutan: denoise → CLAHE → brightness/contrast → sharpen. "
                   "Koreksi perspektif dikerjakan di menu Geometri.")
        den = st.slider("Kekuatan denoise (0 = mati)", 0, 20, 5, key="enh_c_den")
        use_clahe = st.checkbox("Perbaikan kontras (CLAHE)", True, key="enh_c_use")
        clip = st.slider("Clip limit CLAHE", 1.0, 8.0, 2.0, step=0.5, key="enh_c_clip")
        brightness = st.slider("Brightness", -100, 100, 0, key="enh_c_b")
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, step=0.1, key="enh_c_c")
        amount = st.slider("Kekuatan sharpen (0 = mati)", 0.0, 3.0, 0.8, step=0.1, key="enh_c_sh")
        return enh.comprehensive_enhancement(image, den, use_clahe, clip, 8, brightness, contrast, amount)
    if feature == "Denoising":
        labels = {"Non-Local Means": "nlm", "Gaussian": "gaussian",
                  "Median (salt & pepper)": "median", "Bilateral": "bilateral"}
        method = labels[st.selectbox("Metode", list(labels), key="enh_dn_method")]
        strength = st.slider("Kekuatan", 1, 30, 10, key="enh_dn_s")
        size = st.slider("Ukuran kernel", 3, 21, 5, step=2, key="enh_dn_k")
        return enh.denoise(image, method, strength, size)
    if feature == "Blur Suite":
        labels = {"Gaussian Blur": "gaussian", "Averaging Blur": "averaging",
                  "Median Blur": "median", "Bilateral Blur": "bilateral"}
        method = labels[st.selectbox("Jenis blur", list(labels), key="enh_bl_method")]
        size, sigma, diameter, sc, ss = 5, 0.0, 9, 75, 75
        if method == "bilateral":
            diameter = st.slider("Diameter", 3, 25, 9, key="enh_bl_d")
            sc = st.slider("Sigma color", 10, 200, 75, key="enh_bl_sc")
            ss = st.slider("Sigma space", 10, 200, 75, key="enh_bl_ss")
        else:
            size = st.slider("Ukuran kernel", 3 if method == "median" else 1, 51, 5, step=2, key="enh_bl_k")
            if method == "gaussian":
                sigma = st.slider("Sigma (0 = otomatis)", 0.0, 20.0, 0.0, step=0.5, key="enh_bl_sg")
        return enh.blur(image, method, size, sigma, diameter, sc, ss)
    if feature == "Sharpening":
        labels = {"Kernel Laplacian": "laplacian", "Kernel High-boost": "high_boost",
                  "Unsharp Mask": "unsharp"}
        method = labels[st.selectbox("Metode", list(labels), key="enh_sh_method")]
        if method == "unsharp":
            size = st.slider("Ukuran kernel", 3, 21, 5, step=2, key="enh_sh_k")
            sigma = st.slider("Sigma", 0.5, 10.0, 1.0, step=0.5, key="enh_sh_sg")
            amount = st.slider("Amount", 0.1, 5.0, 1.5, step=0.1, key="enh_sh_a")
            return enh.sharpen(image, method, kernel_size=size, sigma=sigma, amount=amount)
        strength = st.slider("Strength", 0.0, 1.0, 1.0, step=0.1, key="enh_sh_s")
        return enh.sharpen(image, method, strength)
    # Deblurring dan Restorasi
    method = st.selectbox("Metode", ["Wiener (blur umum)", "Wiener (blur gerak)",
                                     "Unsharp deblur", "Inpainting goresan"], key="enh_db_method")
    if method.startswith("Wiener"):
        ks = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
        k = st.select_slider("K (regularisasi)", options=ks, value=0.01, key="enh_db_k")
        if method == "Wiener (blur umum)":
            size = st.slider("Ukuran PSF", 3, 31, 7, step=2, key="enh_db_size")
            sigma = st.slider("Sigma PSF", 0.5, 10.0, 2.0, step=0.5, key="enh_db_sigma")
            return enh.wiener_deblur(image, "gaussian", size, sigma, 0, k)
        length = st.slider("Panjang blur gerak", 3, 51, 15, step=2, key="enh_db_len")
        angle = st.slider("Sudut (derajat)", 0, 180, 0, key="enh_db_angle")
        return enh.wiener_deblur(image, "motion", length, 0, angle, k)
    if method == "Unsharp deblur":
        amount = st.slider("Amount", 0.5, 5.0, 2.0, step=0.1, key="enh_db_amount")
        sigma = st.slider("Sigma", 0.5, 10.0, 2.0, step=0.5, key="enh_db_us")
        return enh.deblur_unsharp(image, amount, sigma)
    damage = "bright" if st.radio("Jenis kerusakan", ["Terang (goresan putih)", "Gelap (bercak hitam)"],
                                  key="enh_db_dmg").startswith("Terang") else "dark"
    threshold = st.slider("Threshold area rusak", 150, 255, 240, key="enh_db_th")
    radius = st.slider("Radius inpaint", 1, 10, 3, key="enh_db_rad")
    return enh.inpaint_damage(image, damage, threshold, radius)


def process_image(image, feature):
    auxiliary = None
    with st.sidebar:
        st.subheader("Pengaturan")
        if feature in ENHANCEMENT_FEATURES:
            result = enhancement_controls(image, feature)
        elif feature == "Thresholding":
            result = threshold_controls(image, "threshold")
        elif feature == "Color masking":
            st.caption("Hue: 0–179. Hue bawah > atas memilih warna melewati nol, misalnya merah 170–10.")
            h_min = st.slider("Hue bawah", 0, 179, 0)
            h_max = st.slider("Hue atas", 0, 179, 179)
            s_min, s_max = st.slider("Rentang saturasi", 0, 255, (0, 255))
            v_min, v_max = st.slider("Rentang kecerahan", 0, 255, (0, 255))
            mask, selected = seg.color_masking(image, (h_min, s_min, v_min), (h_max, s_max, v_max))
            output = st.radio("Hasil yang disimpan", ["Objek berwarna", "Mask biner"])
            result, auxiliary = (selected, mask) if output == "Objek berwarna" else (mask, selected)
        elif feature == "Morfologi":
            st.caption("Buat mask terlebih dahulu. Objek putih akan diproses pada latar hitam.")
            binary = threshold_controls(image, "morph")
            operation = st.selectbox("Operasi morfologi", ["erosion", "dilation", "opening", "closing"])
            size = st.slider("Ukuran kernel", 1, 31, 3, step=2)
            shape = st.selectbox("Bentuk kernel", ["rect", "ellipse", "cross"])
            iterations = st.slider("Jumlah iterasi", 1, 10, 1)
            result = seg.morphology(binary, operation, size, iterations, shape)
            auxiliary = binary
        elif feature == "Deteksi tepi":
            method = st.selectbox("Operator tepi", ["canny", "sobel", "prewitt", "laplacian"])
            size = 3
            low, high = 100, 200
            if method != "prewitt":
                size = st.select_slider("Ukuran kernel tepi", options=[3, 5, 7], value=3)
            if method == "canny":
                low, high = st.slider("Ambang Canny (bawah, atas)", 0, 255, (100, 200))
            result = seg.edge_detection(image, method, size, low, high)
        elif feature == "Sketch":
            size = st.slider("Ukuran blur sketsa", 3, 51, 21, step=2)
            result = seg.sketch(image, size)
        else:
            kind = st.radio("Jenis watermark", ["Teks", "Logo"])
            position_labels = {"Kanan bawah": "bottom-right", "Kiri bawah": "bottom-left",
                               "Kanan atas": "top-right", "Kiri atas": "top-left", "Tengah": "center"}
            position = position_labels[st.selectbox("Posisi", list(position_labels))]
            opacity = st.slider("Opacity", 0.0, 1.0, 0.5, step=0.05)
            margin = int(st.number_input("Margin (piksel)", min_value=0, value=10, step=1))
            if kind == "Teks":
                text = st.text_input("Teks watermark", "PicaLab")
                font_size = st.slider("Ukuran teks", 8, 150, 24)
                hex_color = st.color_picker("Warna teks", "#FFFFFF")
                color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
                result = seg.text_watermark(image, text, opacity, position, font_size, color, margin)
            else:
                logo_file = st.file_uploader("Upload logo PNG/JPG", type=["png", "jpg", "jpeg"], key="logo")
                if logo_file is None:
                    st.info("Upload logo untuk menambahkan watermark. PNG mendukung transparansi.")
                    return None, None
                with Image.open(logo_file) as source:
                    logo = ImageOps.exif_transpose(source).convert("RGBA")
                width_percent = st.slider("Lebar logo (% gambar)", 1, 100, 20)
                width = max(1, round(image.shape[1] * width_percent / 100))
                height = max(1, round(logo.height * width / logo.width))
                logo = np.array(logo.resize((width, height), Image.Resampling.LANCZOS))
                result = seg.logo_watermark(image, logo, opacity, position, margin)
    return result, auxiliary


def main():
    st.set_page_config(page_title="PicaLab — Mini Photoshop", layout="wide")
    st.title("PicaLab")
    st.caption("Pengolahan citra • Fitur Andini")
    with st.sidebar:
        uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], key="source")
        feature = st.selectbox("Pilih fitur", [
            "Thresholding", "Color masking", "Morfologi", "Deteksi tepi", "Sketch", "Watermark",
        ] + ENHANCEMENT_FEATURES)
        st.caption("Setiap operasi memakai gambar asli. Hasil berubah saat parameter diatur.")
    if uploaded is None:
        st.info("Upload gambar JPG atau PNG melalui sidebar untuk mulai.")
        return
    try:
        with Image.open(uploaded) as source:
            image = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        st.error(f"Gambar tidak dapat dibaca: {exc}")
        return

    original_column, result_column = st.columns(2)
    with original_column:
        st.subheader("Gambar asli")
        st.image(image, width="stretch")
        st.caption(f"{image.shape[1]} × {image.shape[0]} piksel")
    try:
        result, auxiliary = process_image(image, feature)
    except (ValueError, OSError, cv2.error, Image.DecompressionBombError) as exc:
        with result_column:
            st.error(f"Gambar belum dapat diproses: {exc}")
        return
    if result is None:
        with result_column:
            st.info("Lengkapi pengaturan di sidebar untuk melihat hasil.")
        return
    with result_column:
        st.subheader("Hasil")
        st.image(result, width="stretch")
        st.download_button("Download hasil PNG", data=png_bytes(result),
                           file_name=f"picalab_{feature.lower().replace(' ', '_')}.png", mime="image/png")
    if feature in HISTOGRAM_FEATURES and (
            feature == "Histogram Equalization" or st.session_state.get("enh_show_hist", True)):
        show_histograms(image, result)
    
    if auxiliary is not None:
        with st.expander("Lihat mask sebelum morfologi" if feature == "Morfologi" else "Lihat keluaran pendamping"):
            st.image(auxiliary, width="stretch")


if __name__ == "__main__":
    main()
