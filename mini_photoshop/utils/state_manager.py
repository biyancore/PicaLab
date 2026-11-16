"""
Manajemen state (riwayat Undo/Redo/Reset) untuk tab Foto.

Revisi:
- BUG FIX: init_session_state() versi lama hanya mengisi key yang BELUM ada,
  sehingga saat pengguna mengunggah foto kedua, gambar lama tetap tampil.
  Sekarang ada load_new_image() yang selalu mengganti seluruh state.
- Setiap perubahan gambar menaikkan `img_version`. Angka ini dipakai sebagai
  kunci cache pemrosesan sehingga filter tidak dihitung ulang tanpa perlu.
- Riwayat menyimpan label operasi (timeline) + estimasi pemakaian memori.
"""

import uuid

import numpy as np
import streamlit as st

# Batasi jumlah riwayat agar RAM komputer/server tidak jebol
# Mengingat matriks citra (terutama foto resolusi tinggi) memakan banyak memori
MAX_HISTORY = 10

_KEYS = ["original_image", "current_image", "history", "redo_stack",
         "history_labels", "redo_labels", "img_version", "image_source", "image_meta"]


def _bump_version():
    st.session_state["img_version"] = st.session_state.get("img_version", 0) + 1


def session_uid() -> str:
    """ID unik per sesi browser (dipakai sebagai bagian kunci cache)."""
    if "session_uid" not in st.session_state:
        st.session_state["session_uid"] = uuid.uuid4().hex
    return st.session_state["session_uid"]


def has_image() -> bool:
    return st.session_state.get("current_image") is not None


def init_session_state(original_image: np.ndarray):
    """
    Inisialisasi variabel state bila belum ada (kompatibel dengan versi lama).
    Untuk mengganti gambar yang sudah ada, gunakan load_new_image().
    """
    if "current_image" not in st.session_state:
        load_new_image(original_image)


def load_new_image(original_image: np.ndarray, source_id: str = None, meta: dict = None):
    """
    Muat gambar baru: hapus riwayat lama & jadikan gambar ini titik awal.
    """
    clear_all_states()
    st.session_state["original_image"] = original_image.copy()
    st.session_state["current_image"] = original_image.copy()
    st.session_state["history"] = []
    st.session_state["redo_stack"] = []
    st.session_state["history_labels"] = []
    st.session_state["redo_labels"] = []
    st.session_state["image_source"] = source_id
    st.session_state["image_meta"] = dict(meta or {})
    _bump_version()


def is_new_source(source_id: str) -> bool:
    """True jika file upload berbeda dari gambar yang sedang diedit."""
    return st.session_state.get("image_source") != source_id


def update_image(new_image: np.ndarray, label: str = "Edit"):
    """
    Dipanggil SETIAP KALI pengguna menekan tombol "Terapkan Filter".
    Gambar lama dimasukkan ke riwayat, gambar baru ditayangkan.
    """
    if not has_image():
        return
    history = st.session_state["history"]
    history.append(st.session_state["current_image"].copy())
    st.session_state["history_labels"].append(label)

    # Batasi ukuran history agar RAM tidak penuh (label tetap disimpan utuh)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    st.session_state["current_image"] = new_image.copy()

    # Timeline berubah -> redo tidak lagi valid
    st.session_state["redo_stack"].clear()
    st.session_state["redo_labels"].clear()
    _bump_version()


def can_undo() -> bool:
    return bool(st.session_state.get("history"))


def can_redo() -> bool:
    return bool(st.session_state.get("redo_stack"))


def undo():
    """Mengembalikan gambar ke satu langkah sebelumnya."""
    if not can_undo():
        return
    st.session_state["redo_stack"].append(st.session_state["current_image"].copy())
    st.session_state["current_image"] = st.session_state["history"].pop()
    if st.session_state["history_labels"]:
        st.session_state["redo_labels"].append(st.session_state["history_labels"].pop())
    _bump_version()


def redo():
    """Maju satu langkah ke gambar yang baru saja di-undo."""
    if not can_redo():
        return
    st.session_state["history"].append(st.session_state["current_image"].copy())
    st.session_state["current_image"] = st.session_state["redo_stack"].pop()
    if st.session_state["redo_labels"]:
        st.session_state["history_labels"].append(st.session_state["redo_labels"].pop())
    _bump_version()


def reset_to_original():
    """Menghapus semua editan dan kembali ke foto awal yang diunggah."""
    if st.session_state.get("original_image") is None:
        return
    st.session_state["history"].clear()
    st.session_state["redo_stack"].clear()
    st.session_state["history_labels"].clear()
    st.session_state["redo_labels"].clear()
    st.session_state["current_image"] = st.session_state["original_image"].copy()
    _bump_version()


def applied_operations() -> list:
    """Daftar label operasi yang membentuk gambar saat ini (urut lama -> baru)."""
    return list(st.session_state.get("history_labels", []))


def history_memory_mb() -> float:
    """Estimasi RAM yang dipakai riwayat (MB) untuk ditampilkan di UI."""
    total = 0
    for key in ("original_image", "current_image"):
        img = st.session_state.get(key)
        if img is not None:
            total += img.nbytes
    for key in ("history", "redo_stack"):
        total += sum(img.nbytes for img in st.session_state.get(key, []))
    return total / (1024 * 1024)


def clear_all_states():
    """
    Digunakan jika pengguna menutup gambar atau mengunggah file yang sama sekali baru.
    img_version TIDAK dihapus agar kunci cache lama tidak terpakai ulang.
    """
    for key in _KEYS:
        if key != "img_version" and key in st.session_state:
            del st.session_state[key]
    _bump_version()
