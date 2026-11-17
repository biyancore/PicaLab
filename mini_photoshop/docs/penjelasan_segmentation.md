# Penjelasan Fitur Segmentation — PicaLab

Modul `modules/segmentation.py` berisi bagian Andini. Isinya enam kelompok fitur:
thresholding, color masking, morfologi, deteksi tepi, sketch, dan watermark.
Semua fungsi menerima citra NumPy `uint8` berformat RGB dan mengembalikan citra baru.
Hasil biner memakai nilai 0 (latar) dan 255 (objek).

## Alur singkat saat tombol diterapkan

1. `utils/sidebar.py` mengumpulkan pilihan pengguna.
2. `utils/pipeline.py` mengubah citra BGR dari OpenCV menjadi RGB, lalu memanggil fungsi di `segmentation.py`.
3. Hasil dikembalikan ke pipeline dan ditampilkan oleh `app.py`.

Contoh untuk thresholding:

```python
# utils/pipeline.py
def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _segment(img, p, ctx):
    rgb = _rgb(img)
    if p["tool"] == "threshold":
        if p["method"] == "otsu":
            return segmentation.otsu_threshold(rgb, invert=p["invert"])
```

## 1. Thresholding

Thresholding mengubah citra abu-abu menjadi citra biner. Setiap piksel dibandingkan
dengan sebuah ambang: piksel di atas ambang menjadi putih, piksel di bawahnya menjadi hitam.

| Metode | Cara kerja | Kapan dipakai |
|---|---|---|
| Global | Satu ambang untuk seluruh gambar | Latar terang merata, objek jelas |
| Adaptive | Ambang dihitung per area kecil | Latar tidak merata atau bayangan |
| Otsu | Ambang dipilih otomatis dari histogram | Objek dan latar cukup terpisah |

```python
# modules/segmentation.py — global_threshold, baris 41
mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
return cv2.threshold(_gray(image), threshold, 255, mode)[1]

# adaptive_threshold, baris 49
return cv2.adaptiveThreshold(
    _gray(image), 255, methods[method], mode, _odd_size(block_size), c
)

# otsu_threshold, baris 61
return cv2.threshold(_gray(image), 0, 255, mode | cv2.THRESH_OTSU)[1]
```

Parameter yang perlu disebut:

- **Nilai ambang**: 0–255. Nilai 127 adalah titik tengah yang umum.
- **Block size**: ukuran area lokal untuk adaptive threshold; harus ganjil.
- **C**: konstanta yang dikurangkan dari ambang lokal. Nilai lebih besar membuat lebih sedikit piksel menjadi putih.
- **Invert**: menukar hitam dan putih.

Hasil yang diharapkan: objek berwarna atau gelap menjadi putih, latar menjadi hitam.
Pada contoh `threshold.png`, Otsu memilih ambang 161. Adaptive threshold
masih menangkap gradasi latar, sehingga tepi objek ikut terlihat sebagai garis.

Kalimat presentasi: “Global memakai satu nilai ambang. Adaptive menghitung ambang
secara lokal, jadi lebih tahan terhadap bayangan. Otsu mencari ambang terbaik dari
histogram, sehingga pengguna tidak perlu menebak angka.”

## 2. Color Masking

Color masking memilih piksel berdasarkan rentang warna. Citra RGB diubah dulu ke HSV,
karena HSV memisahkan jenis warna (H), kepekatan warna (S), dan kecerahan (V).

```python
# modules/segmentation.py, baris 72–102
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
if lower[0] < upper[0]:
    mask = cv2.inRange(hsv, lower, upper)
else:
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower, np.array([179, *upper[1:]], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([0, *lower[1:]], dtype=np.uint8), upper),
    )
return mask, cv2.bitwise_and(image, image, mask=mask)
```

Di OpenCV, Hue bernilai 0–179, sedangkan Saturasi dan Value bernilai 0–255.
Merah berada di dua ujung hue, sekitar 170–179 dan 0–10. Karena itu mask merah
dihitung dari dua rentang lalu digabung dengan `bitwise_or`.

Hasil yang diharapkan: tiga pilihan tampilan di aplikasi.

- **Mask biner**: area warna terpilih putih, sisanya hitam.
- **Warna terpilih saja**: warna asli dipertahankan, latar menjadi hitam.
- **Sorot warna**: objek berwarna tetap, latar diubah menjadi abu-abu.

Pada contoh `color_masking.png`, mask merah mencakup 25,8% piksel.
Jika preset tidak menemukan apa pun, naikkan rentang Saturasi atau Value.

Kalimat presentasi: “Saya memakai HSV karena Hue memisahkan warna dengan rapi.
Merah melintasi nol, jadi saya gabungkan dua rentang hue.”

## 3. Operasi Morfologi

Morfologi mengubah bentuk objek pada mask biner menggunakan kernel. Kernel adalah
kotak kecil yang bergerak di seluruh mask. Ukuran dan bentuk kernel memengaruhi hasil.

| Operasi | Rumus | Kegunaan |
|---|---|---|
| Erosion | — | Memperkecil objek, menghapus bintik kecil |
| Dilation | — | Memperbesar objek, menutup lubang kecil |
| Opening | Erosion → Dilation | Menghapus noise tanpa banyak mengubah ukuran objek |
| Closing | Dilation → Erosion | Menutup celah dan lubang di dalam objek |
| Gradient | Dilation − Erosion | Menampilkan kontur objek |
| Top-hat | Citra − Opening | Menonjolkan detail kecil yang terang |
| Black-hat | Closing − Citra | Menonjolkan detail kecil yang gelap |

```python
# modules/segmentation.py, baris 112–139
if auto_binarize:
    image = to_binary_mask(image)
size = _odd_size(kernel_size, minimum=1)
kernel = cv2.getStructuringElement(shapes[shape], (size, size))
return cv2.morphologyEx(image, operations[operation], kernel, iterations=int(iterations))
```

Parameter yang perlu disebut:

- **Kernel size**: ukuran jendela, misalnya 5×5 atau 7×7; harus ganjil.
- **Bentuk kernel**: persegi, elips, atau silang.
- **Iterations**: berapa kali operasi diulang; makin banyak, efeknya makin kuat.
- **Auto binarize**: pada aplikasi, foto berwarna diubah dulu menjadi mask dengan Otsu.

Hasil yang diharapkan: bintik kecil hilang setelah opening, sedangkan objek utama
tetap. Pada `morfologi.png`, opening menghapus titik-titik di bawah objek.

Kalimat presentasi: “Opening cocok untuk membersihkan noise, closing cocok untuk
menutup lubang. Saya memakai kernel elips agar bentuk objek bulat tidak terlalu berubah.”

## 4. Deteksi Tepi

Deteksi tepi mencari piksel yang intensitasnya berubah tajam. Hasilnya citra
abu-abu dengan garis putih pada batas objek.

```python
# modules/segmentation.py, baris 141–168
if method == "canny":
    return cv2.Canny(gray, low_threshold, high_threshold, apertureSize=size)
if method == "laplacian":
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=size))
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=size)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=size)
return np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)
```

| Operator | Ide dasar |
|---|---|
| Sobel | Turunan pertama; gradien horizontal dan vertikal digabung |
| Prewitt | Mirip Sobel, memakai kernel 3×3 dengan bobot seragam |
| Laplacian | Turunan kedua; menonjolkan perubahan intensitas yang cepat |
| Canny | Gradien, dua ambang, lalu penipisan dan penyambungan tepi |

Parameter yang perlu disebut:

- **Kernel size**: 3, 5, atau 7 untuk Canny; ukuran lain untuk Sobel dan Laplacian.
- **Ambang bawah dan atas**: khusus Canny. Ambang bawah menangkap tepi lemah,
  ambang atas memastikan tepi kuat.

Hasil yang diharapkan: Canny menghasilkan garis tipis dan bersih, Sobel dan Prewitt
menghasilkan tepi yang lebih tebal, Laplacian menghasilkan tepi paling tipis.

Kalimat presentasi: “Sobel dan Prewitt mencari perubahan intensitas di arah X dan Y.
Laplacian memakai turunan kedua. Canny menambahkan penyaringan dua ambang agar
tepi lebih bersih.”

## 5. Sketch Pensil

Sketch mengubah foto menjadi gambar seperti pensil. Langkahnya: grayscale, balik
intensitas, blur, lalu color dodge.

```python
# modules/segmentation.py, baris 171–179
gray = _gray(image)
blurred = cv2.GaussianBlur(255 - gray, (size, size), 0)
denominator = 255 - blurred.astype(np.float32)
result = np.full(gray.shape, 255, dtype=np.float32)
np.divide(gray.astype(np.float32) * 255, denominator, out=result, where=denominator > 0)
return np.clip(result, 0, 255).astype(np.uint8)
```

Color dodge membagi nilai asli dengan kebalikan citra blur. Bagian yang berubah
tajam menjadi gelap, sedangkan area rata menjadi terang. Hasilnya garis tepi
seperti goresan pensil. Nilai `where=denominator > 0` mencegah pembagian
dengan nol, dan `np.clip` menjaga hasil tetap 0–255.

Parameter: **blur size**, harus ganjil. Blur lebih besar menghasilkan garis yang
lebih tebal dan lembut.

Kalimat presentasi: “Sketch adalah hasil color dodge antara foto abu-abu dan
kebalikan foto yang diblur. Area dengan perubahan kuat tampil sebagai garis.”

## 6. Watermark

Watermark menambahkan teks atau logo dengan transparansi. Teks dibuat menjadi
lapisan RGBA, alpha-nya dikalikan dengan opacity, lalu ditempel ke citra.

```python
# modules/segmentation.py, baris 202–225
base = Image.fromarray(image).convert("RGBA")
mark.putalpha(mark.getchannel("A").point(lambda alpha: round(alpha * opacity)))
base.alpha_composite(mark, (x, y))
return np.array(base.convert("RGB"))
```

`text_watermark` membuat teks dan bisa menambahkan outline. `logo_watermark`
menerima PNG dengan kanal alpha, sehingga bagian transparan tetap transparan.

Parameter yang perlu disebut:

- **Opacity**: 0–1; makin besar, watermark makin pekat.
- **Position**: kiri atas, kanan atas, kiri bawah, kanan bawah, atau tengah.
- **Margin**: jarak dari tepi citra.
- **Scale**: lebar logo relatif terhadap lebar citra.
- **Fit**: mengecilkan watermark agar tetap muat di dalam gambar.

Kalimat presentasi: “Watermark ditempel sebagai lapisan RGBA. Opacity mengatur
seberapa kuat watermark terlihat, sedangkan alpha pada logo tetap dipertahankan.”

## Urutan demo yang disarankan

1. Buka **Segmentasi & Morfologi → Thresholding**, pilih Otsu, lalu balik hitam-putih.
2. Ganti ke Adaptive dan tunjukkan perbedaan pada latar bergradasi.
3. Buka **Color Masking**, pilih preset Merah, lalu bandingkan mask, warna terpilih, dan sorot.
4. Buka **Morfologi**, pilih Opening dengan kernel 7, lalu bandingkan dengan Closing.
5. Buka **Deteksi Tepi**, tampilkan Canny, lalu ganti ke Sobel.
6. Buka **Efek Kreatif → Sketch** dan **Watermark**.

## Hal yang paling mungkin ditanya penguji

- **Kenapa HSV, bukan RGB?** Di HSV, jenis warna dipisahkan dari kecerahan,
  sehingga ambang warna lebih mudah ditentukan.
- **Kenapa block size harus ganjil?** Adaptive threshold membutuhkan piksel pusat
  dan area simetris di sekitarnya. Kernel morfologi juga memakai aturan yang sama.
- **Apa bedanya Otsu dan global?** Keduanya memakai satu ambang, tetapi Otsu
  memilih nilainya dari histogram secara otomatis.
- **Kenapa opening menghapus noise?** Erosion menghapus objek yang lebih kecil
  dari kernel, lalu dilation mengembalikan ukuran objek yang tersisa.
- **Kenapa hasil biner berwarna hitam-putih?** Nilainya 0 dan 255, sehingga
  langsung cocok untuk masking dan operasi morfologi.

## Gambar pendukung

Semua gambar ada di `mini_photoshop/assets/segmentation/`:

- `ringkasan_segmentation.png` — sembilan hasil dalam satu gambar.
- `threshold.png` — global, Otsu, dan adaptive.
- `color_masking.png` — mask, warna terpilih, dan sorot.
- `morfologi.png` — erosion, dilation, opening, closing.
- `morfologi_tambahan.png` — gradient, top-hat, black-hat.
- `deteksi_tepi.png` — Canny, Sobel, Prewitt, Laplacian.
- `sketch.png` dan `watermark.png`.
