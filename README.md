# PicaLab

PicaLab adalah aplikasi web pengolahan citra digital interaktif berbasis Python yang dikembangkan menggunakan Streamlit dan OpenCV. Dirancang sebagai proyek mata kuliah Pengolahan Citra, aplikasi ini mengombinasikan antarmuka yang ramah pengguna dengan implementasi algoritma pengolahan citra tingkat lanjut secara real-time.

## Fitur

| Mode | Kemampuan |
|---|---|
| **Editor Foto** | 35 alat, pratinjau langsung, Undo/Redo/Reset (riwayat 10 langkah), slider pembanding before/after, metric cards (kecerahan, kontras, ketajaman, entropi, colorfulness), histogram interaktif (luminansi & per channel), ekspor PNG/JPEG/WEBP + CSV histogram + log edit JSON |
| **Video** | Render frame-demi-frame ke MP4 H.264 (bisa diputar di browser), tata letak berdampingan/hasil saja/atas-bawah, histogram live, batas durasi & skala proses, progress + ETA |
| **Live Camera** | WebRTC real-time dengan FPS & latensi, mode cermin, histogram live, tangkap gambar, kirim hasil tangkapan ke Editor Foto |

Kategori alat: Geometri & Ukuran · Warna & Dekomposisi · Peningkatan & Restorasi · Segmentasi & Morfologi · Efek Kreatif · Beauty (AI, MediaPipe).

## Menjalankan

```bash
cd mini_photoshop
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> **Beauty (AI):** dengan MediaPipe ≥ 0.10.30 / 1.x (`mp.solutions` sudah dihapus), model
> `face_landmarker.task` & `selfie_segmenter.tflite` (~4 MB) diunduh otomatis ke `mini_photoshop/models/`
> saat fitur pertama kali dipakai. Tanpa internet, unduh manual dari URL di pesan error lalu taruh di folder itu.
> Dengan MediaPipe lama (≤ 0.10.21, Python ≤ 3.12) model bawaan paket dipakai langsung.

## Struktur

```
mini_photoshop/
├── app.py                 # Orkestrasi halaman (Foto / Video / Kamera)
├── .streamlit/config.toml # Tema
├── modules/               # Algoritma (tanpa Streamlit)
│   ├── geometry.py        # resize, resample, crop, flip, rotasi, perspektif
│   ├── color_space.py     # grayscale, saturasi, hue, dekomposisi RGB, negatif
│   ├── enhancement.py     # brightness/kontras, gamma, HE/CLAHE, blur, sharpen, denoise, deblur, auto
│   ├── segmentation.py    # threshold, color masking, morfologi, tepi, sketch, watermark
│   ├── effects.py         # cartoon, vignette, sepia, pixelate, emboss
│   └── beauty.py          # teeth whitening, blush, lipstick, bokeh, face mesh (MediaPipe)
└── utils/
    ├── pipeline.py        # Router filter + statistik citra (dipakai foto, video, kamera)
    ├── sidebar.py         # Menu & parameter alat
    ├── ui.py              # Tema CSS & komponen (metric card, slider pembanding, histogram)
    ├── state_manager.py   # Undo / Redo / Reset
    ├── helpers.py         # I/O gambar, konversi, ekspor
    └── media_handler.py   # Render video (PyAV) & WebRTC
```
