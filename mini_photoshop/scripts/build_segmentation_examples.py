
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

root = Path.cwd()
sys.path.insert(0, str(root / "mini_photoshop"))
from modules import segmentation as seg

out_dir = root / "mini_photoshop" / "assets" / "segmentation"
out_dir.mkdir(parents=True, exist_ok=True)

# Gambar sintetis: latar bergradasi, objek berwarna, dan bintik untuk morfologi.
h, w = 360, 560
yy, xx = np.mgrid[0:h, 0:w]
image = np.zeros((h, w, 3), dtype=np.uint8)
image[..., 0] = np.clip(185 + xx * 35 / w, 0, 255)
image[..., 1] = np.clip(205 + yy * 20 / h, 0, 255)
image[..., 2] = np.clip(220 + (xx + yy) * 20 / (w + h), 0, 255)
cv2.circle(image, (150, 155), 78, (220, 48, 55), -1)
cv2.rectangle(image, (305, 95), (465, 225), (220, 145, 46), -1)
cv2.ellipse(image, (390, 280), (92, 46), 0, 0, 360, (85, 170, 65), -1)
cv2.line(image, (320, 280), (460, 280), (255, 255, 255), 9)
cv2.circle(image, (390, 280), 20, (255, 255, 255), -1)
cv2.putText(image, "PicaLab", (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (35, 45, 55), 2, cv2.LINE_AA)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
Image.fromarray(image).save(out_dir / "contoh_asli.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
global_result = seg.global_threshold(image, 127)
otsu_result = seg.otsu_threshold(image)
adaptive_result = seg.adaptive_threshold(image, "gaussian", 31, 5)
# Preset Merah di sidebar memakai rentang 10-170 dengan mode "di luar rentang"
# untuk merah yang berada di kedua ujung hue.
mask, selected = seg.color_masking(image, (10, 70, 50), (170, 255, 255))
highlight = np.where(mask[..., None] > 0, image, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

binary = np.zeros((h, w), dtype=np.uint8)
cv2.circle(binary, (150, 155), 78, 255, -1)
cv2.rectangle(binary, (305, 95), (465, 225), 255, -1)
cv2.ellipse(binary, (390, 280), (92, 46), 0, 0, 360, 255, -1)
cv2.line(binary, (320, 280), (460, 280), 0, 9)
cv2.circle(binary, (390, 280), 20, 0, -1)
for x, y in [(250, 310), (268, 318), (292, 306), (310, 330), (338, 312)]:
    cv2.circle(binary, (x, y), 3, 255, -1)
cv2.line(binary, (285, 300), (285, 335), 255, 2)
morph_results = {name: seg.morphology(binary, name, 7, 1, "ellipse") for name in
                 ("erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat")}

edges = {
    "Canny": seg.edge_detection(image, "canny", 3, 60, 160),
    "Sobel": seg.edge_detection(image, "sobel", 3),
    "Prewitt": seg.edge_detection(image, "prewitt", 3),
    "Laplacian": seg.edge_detection(image, "laplacian", 3),
}
sketch = seg.sketch(image, 21)
text_wm = seg.text_watermark(image, "PicaLab", opacity=0.8, position="bottom-right",
                             font_size=32, color=(255, 255, 255), stroke_width=2, fit=True)
logo = Image.new("RGBA", (220, 76), (0, 0, 0, 0))
draw = ImageDraw.Draw(logo)
draw.rounded_rectangle((0, 0, 219, 75), radius=18, fill=(12, 18, 30, 180))
draw.text((24, 18), "PicaLab", fill=(255, 255, 255, 255))
logo_result = seg.logo_watermark(image, np.array(logo), opacity=0.9, position="top-left",
                                 margin=12, scale=0.3, fit=True)

def save_steps(name, steps):
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 3.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, img) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

save_steps("threshold.png", [("Gambar asli", image), ("Global, ambang 127", global_result),
                             ("Otsu otomatis", otsu_result), ("Adaptive Gaussian", adaptive_result)])
save_steps("color_masking.png", [("Gambar asli", image), ("Mask HSV merah", mask),
                                 ("Warna terpilih", selected), ("Sorot warna", highlight)])
save_steps("morfologi.png", [("Mask biner 7x7", binary), ("Erosion", morph_results["erosion"]),
                             ("Dilation", morph_results["dilation"]), ("Opening", morph_results["opening"]),
                             ("Closing", morph_results["closing"])])
save_steps("morfologi_tambahan.png", [("Mask biner", binary), ("Gradient", morph_results["gradient"]),
                                      ("Top-hat", morph_results["tophat"]), ("Black-hat", morph_results["blackhat"])])
save_steps("deteksi_tepi.png", [("Gambar asli", image), ("Canny", edges["Canny"]), ("Sobel", edges["Sobel"]),
                                ("Prewitt", edges["Prewitt"]), ("Laplacian", edges["Laplacian"])])
save_steps("sketch.png", [("Gambar asli", image), ("Sketch", sketch)])
save_steps("watermark.png", [("Watermark teks", text_wm), ("Watermark logo", logo_result)])

summary = [("Threshold global", global_result), ("Otsu", otsu_result), ("Adaptive", adaptive_result),
           ("Mask warna", mask), ("Opening", morph_results["opening"]), ("Closing", morph_results["closing"]),
           ("Canny", edges["Canny"]), ("Sketch", sketch), ("Watermark", text_wm)]
fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), constrained_layout=True)
for ax, (title, img) in zip(axes.flat, summary):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
fig.savefig(out_dir / "ringkasan_segmentation.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Otsu threshold:", round(seg.otsu_value(image), 1))
print("Mask coverage:", round(100 * np.count_nonzero(mask) / mask.size, 1), "%")
print("Files:", sorted(p.name for p in out_dir.glob("*.png")))

