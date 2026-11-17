# PicaLab — Mini Photoshop

PicaLab adalah aplikasi web pengolahan citra digital berbasis **Python, Streamlit, dan OpenCV**, yang dikembangkan sebagai proyek mata kuliah **Pengolahan Citra Digital**. Pengguna dapat mencoba algoritma, membandingkan hasil, serta menganalisis histogram dan statistik citra.

Aplikasi menyediakan **35 alat dalam 6 kategori**, dengan tiga mode kerja: **Editor Foto**, **Video**, dan **Live Camera**.

## Fitur pengolahan citra

| Kategori | Alat dan metode |
| --- | --- |
| **Geometri & Ukuran** | Resize, resample berdasarkan skala, crop, flip, rotasi, dan koreksi perspektif. Interpolasi: Nearest Neighbor, Bilinear, Bicubic, Lanczos, dan Area. |
| **Warna & Dekomposisi** | Grayscale, saturasi, pergeseran hue, pemisahan kanal RGB dalam grayscale atau warna kanal, serta negatif. |
| **Peningkatan & Restorasi** | Brightness/contrast, gamma, histogram equalization global dan CLAHE, denoising, blur Gaussian/averaging/median/bilateral, sharpening, deblurring Wiener, inpainting, dan Auto Enhance. |
| **Segmentasi & Morfologi** | Threshold global, adaptive Mean/Gaussian, dan Otsu; masking HSV; erosion, dilation, opening, closing, gradient, top-hat, dan black-hat; deteksi tepi Sobel, Prewitt, Laplacian, dan Canny. |
| **Efek Kreatif** | Sketch pensil, cartoon, vignette, sepia, pixelate, emboss, serta watermark teks atau logo transparan. |
| **Beauty (AI)** | Teeth whitening, blush, lipstick, background blur/bokeh, dan visualisasi landmark Face Mesh menggunakan MediaPipe. |

### Mode kerja

| Mode | Kemampuan |
| --- | --- |
| **Editor Foto** | Upload atau gambar contoh, pilihan resolusi kerja, pratinjau filter, tampilan berdampingan/slider pembanding/hasil saja, Undo/Redo hingga 10 langkah, dan reset ke foto asli. |
| **Video** | Pemrosesan setiap frame, tata letak berdampingan/hasil saja/atas-bawah, pengaturan skala dan durasi proses, histogram, progres render, dan unduhan MP4. |
| **Live Camera** | Filter langsung melalui WebRTC, mode cermin, FPS/latensi, histogram, pengambilan gambar, serta pengiriman foto asli atau hasil filter ke Editor Foto. |

Editor Foto juga menyediakan histogram luminansi dan RGB, statistik kecerahan/kontras/ketajaman/entropi/colorfulness, unduhan histogram CSV, log edit JSON, dan ekspor **PNG, JPEG, atau WEBP**. Kualitas JPEG/WEBP dapat diatur untuk menyesuaikan ukuran berkas.

## Teknologi

| Library | Kegunaan |
| --- | --- |
| Streamlit | Antarmuka web, kontrol parameter, dan pengelolaan sesi. |
| OpenCV dan NumPy | Transformasi, filter, dan perhitungan matriks citra. |
| Pillow | Pembacaan gambar, orientasi EXIF, transparansi, dan watermark. |
| Pandas dan Altair | Tabel statistik serta histogram interaktif. |
| PyAV (av) | Pembacaan dan encoding video. |
| streamlit-webrtc | Streaming kamera dari browser. |
| MediaPipe | Landmark wajah dan segmentasi orang untuk Beauty. |
| Matplotlib | Fungsi visualisasi histogram tambahan pada modul enhancement. |

Daftar dependensi terdapat di [requirements.txt](mini_photoshop/requirements.txt).

## Instalasi dan menjalankan aplikasi

Siapkan **Git** dan **Python**. Python **3.11 atau 3.12** disarankan untuk memudahkan kompatibilitas dependensi, terutama MediaPipe.

### Windows (PowerShell)

Jalankan perintah berikut satu per satu. Setelah perintah kedua, terminal berada di folder repository hasil clone.

```powershell
git clone https://github.com/biyancore/PicaLab.git
cd PicaLab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r mini_photoshop/requirements.txt
cd mini_photoshop
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Perintah tersebut menggunakan Python dari virtual environment secara langsung sehingga tidak membutuhkan aktivasi PowerShell.

### macOS / Linux

```bash
git clone https://github.com/biyancore/PicaLab.git
cd PicaLab
python3 -m venv .venv
.venv/bin/python -m pip install -r mini_photoshop/requirements.txt
cd mini_photoshop
../.venv/bin/python -m streamlit run app.py
```

Buka alamat yang muncul di terminal, biasanya **[http://localhost:8501](http://localhost:8501)**. Biarkan terminal tetap terbuka selama aplikasi digunakan; tekan **Ctrl+C** untuk menghentikannya. Jalankan Streamlit dari folder **mini_photoshop** agar konfigurasi tema dan server di [.streamlit/config.toml](mini_photoshop/.streamlit/config.toml) terbaca.

Jika repository sudah tersedia secara lokal, masuk ke folder repository tersebut dan jalankan **git pull --ff-only** untuk mengambil pembaruan. Instal ulang dependensi bila requirements.txt berubah.

## Cara penggunaan

### Editor Foto

1. Pilih **Editor Foto**, kemudian unggah gambar atau klik **Gambar contoh**.
2. Pilih resolusi kerja: sisi terpanjang 1280, 2048, 4096 piksel, atau resolusi asli. Foto besar dapat diperkecil agar pemrosesan lebih cepat.
3. Pilih kategori dan alat di sidebar, lalu atur parameternya. Pratinjau diperbarui otomatis.
4. Klik **Terapkan ke riwayat** untuk menjadikan pratinjau sebagai gambar aktif sebelum melanjutkan dengan filter lain. Gunakan **Undo**, **Redo**, atau **Reset ke foto asli** untuk mengelola perubahan.
5. Periksa tab **Histogram**, **Statistik**, dan **Riwayat** untuk menganalisis hasil.
6. Pada tab **Ekspor**, pilih format, kualitas, dan nama berkas. **Unduh pratinjau** menyimpan gambar aktif beserta filter yang sedang dipilih; **Unduh gambar saat ini** menyimpan hasil edit yang sudah diterapkan ke riwayat.

### Video

1. Pilih **Video** dan unggah klip. Klip pendek disarankan untuk percobaan awal.
2. Pilih filter di sidebar, lalu atur tata letak, skala proses, durasi, dan histogram.
3. Klik **Render video**, tunggu proses selesai, lalu unduh hasil MP4. Mengubah parameter memerlukan render ulang.

Hasil video **tidak menyertakan audio**. Encoder H.264 digunakan bila tersedia; aplikasi dapat beralih ke MPEG-4 yang mungkin perlu diputar di aplikasi pemutar video.

### Live Camera

1. Pilih **Live Camera**, mulai kamera melalui kontrol WebRTC, dan izinkan akses kamera pada browser.
2. Pilih filter serta atur mode cermin, tata letak, histogram, dan tampilan FPS.
3. Klik **Ambil gambar** untuk menyimpan frame. Unduh foto atau gunakan tombol untuk melanjutkan pengeditan di Editor Foto.

Akses kamera membutuhkan browser yang mendukung WebRTC. Gunakan **localhost** untuk menjalankan secara lokal; jika aplikasi diakses melalui server, gunakan **HTTPS**. Koneksi WebRTC juga dapat dipengaruhi izin browser, firewall, dan jaringan.

## Model untuk Beauty

Modul Beauty mendukung backend MediaPipe legacy (mp.solutions) dan Tasks API, sesuai kemampuan paket yang terpasang. Pada backend Tasks, model yang diperlukan diunduh otomatis saat fitur pertama kali digunakan ke folder **mini_photoshop/models/**:

| Model | Digunakan untuk |
| --- | --- |
| [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | Landmark wajah, teeth whitening, blush, lipstick, dan Face Mesh. |
| [selfie_segmenter.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite) | Pemisahan orang dari latar untuk bokeh. |

Unduhan awal membutuhkan internet. Jika gagal, simpan model dari tautan di atas ke folder tersebut dengan nama berkas yang sama. Lokasi model dapat diubah melalui variabel lingkungan **PICALAB_MODEL_DIR**. Berkas model tidak disertakan dalam Git.

Jika MediaPipe gagal dimuat, aplikasi tetap dapat digunakan untuk fitur pengolahan citra lainnya. Efek wajah memerlukan wajah yang berhasil terdeteksi; bokeh memerlukan segmentasi subjek.

## Struktur proyek

```text
PicaLab/
├── README.md
├── .gitignore
└── mini_photoshop/
    ├── app.py                  # Halaman Editor Foto, Video, dan Live Camera
    ├── requirements.txt
    ├── .streamlit/
    │   └── config.toml         # Tema, batas upload, dan konfigurasi server
    ├── assets/                # Aset pendukung
    ├── models/                # Model MediaPipe yang diunduh saat diperlukan
    ├── modules/
    │   ├── __init__.py
    │   ├── geometry.py        # Transformasi geometri dan interpolasi
    │   ├── color_space.py     # Warna dan dekomposisi kanal
    │   ├── enhancement.py     # Peningkatan kualitas dan restorasi
    │   ├── segmentation.py    # Threshold, masking, morfologi, tepi, sketch, watermark
    │   ├── effects.py         # Efek kreatif tambahan
    │   └── beauty.py          # Beauty berbasis MediaPipe
    └── utils/
        ├── __init__.py
        ├── pipeline.py        # Pemilihan fungsi filter dan statistik citra
        ├── sidebar.py         # Menu alat dan pengaturan parameter
        ├── ui.py              # Tampilan, perbandingan, dan histogram
        ├── state_manager.py   # Riwayat edit, Undo, Redo, dan Reset
        ├── helpers.py         # I/O gambar, konversi warna, dan ekspor
        └── media_handler.py   # Pemrosesan video dan kamera WebRTC
```

## Pembagian modul utama

| Anggota | Modul utama | Tanggung jawab |
| --- | --- | --- |
| Biyan | geometry.py | Transformasi geometri, ukuran, perspektif, dan interpolasi. |
| Putri | color_space.py | Grayscale, saturasi, hue, dekomposisi RGB, dan negatif. |
| Aliza | enhancement.py | Brightness/contrast, histogram equalization, blur, denoise, sharpening, dan restorasi. |
| Andini | segmentation.py | Thresholding, masking HSV, morfologi, deteksi tepi, sketch, dan watermark. |

Pembagian ini mengikuti tanggung jawab modul utama tim. Pada menu aplikasi, sketch dan watermark berada di kategori **Efek Kreatif**, sementara implementasinya tetap di segmentation.py.

## Catatan penggunaan dan pengembangan

- Format foto yang diterima: JPG/JPEG, PNG, WEBP, BMP, dan TIFF. Format video: MP4, MOV, AVI, MKV, WEBM, dan M4V, sesuai dukungan decoder. Batas upload pada konfigurasi proyek adalah **300 MB per berkas**.
- Transparansi foto masukan dikomposisikan di atas latar putih; gambar kerja tidak mempertahankan kanal alpha. Logo watermark tetap mendukung transparansi.
- Riwayat foto disimpan dalam sesi aplikasi. Unduh hasil sebelum menutup atau memulai ulang sesi.
- Pemrosesan foto besar, video, dan Beauty dipengaruhi kemampuan perangkat. Turunkan resolusi kerja atau skala proses jika terasa lambat. Restorasi blur tidak menjamin detail yang hilang dapat dipulihkan sepenuhnya.
- Pipeline menggunakan array NumPy uint8 dengan urutan kanal **BGR**. Modul segmentation.py menggunakan **RGB**; konversinya ditangani pada batas pemanggilan di utils/pipeline.py.
- Logika pemrosesan berada di modules/, kontrol alat di utils/sidebar.py, dan penghubung filter di utils/pipeline.py. Gunakan pola ini ketika menambahkan alat agar bisa dipakai pada foto, video, dan kamera.
