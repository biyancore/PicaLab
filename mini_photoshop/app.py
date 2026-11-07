"""Jalankan dari folder mini_photoshop: streamlit run app.py."""

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from modules import segmentation as seg


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


def process_image(image, feature):
    auxiliary = None
    with st.sidebar:
        st.subheader("Pengaturan")
        if feature == "Thresholding":
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
        feature = st.selectbox("Pilih fitur", ["Thresholding", "Color masking", "Morfologi",
                                               "Deteksi tepi", "Sketch", "Watermark"])
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
    if auxiliary is not None:
        with st.expander("Lihat mask sebelum morfologi" if feature == "Morfologi" else "Lihat keluaran pendamping"):
            st.image(auxiliary, width="stretch")


if __name__ == "__main__":
    main()
