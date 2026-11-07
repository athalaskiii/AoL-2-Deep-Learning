"""Lokasi berkas, skema label, dan pemuat data HoASA."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "hoasa"
SOURCE_DIR = ROOT / "data" / "raw" / "hoasa_source"
RESOURCE_DIR = ROOT / "data" / "raw" / "resources"
PROC_DIR = ROOT / "data" / "processed"
SPLITS = ["train", "valid", "test"]

# Urutan aspek dan kelas mengikuti kode resmi IndoNLU (utils/data_utils.py).
ASPECTS = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service", "sunrise_meal", "tv", "wifi"]
CLASSES = ["neg", "neut", "pos", "neg_pos"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# Nama aspek dalam bahasa alami (dipakai untuk input pasangan kalimat dan prompt).
ASPECT_TEXT = {
    "ac": "ac", "air_panas": "air panas", "bau": "bau", "general": "umum", "kebersihan": "kebersihan",
    "linen": "linen", "service": "pelayanan", "sunrise_meal": "sarapan", "tv": "tv", "wifi": "wifi",
}


def load_raw(split: str) -> pd.DataFrame:
    """Berkas resmi IndoNLU: kolom `review` + 10 kolom aspek berisi neg/neut/pos/neg_pos."""
    return pd.read_csv(RAW_DIR / f"{split}_preprocess.csv")


def load_processed(split: str) -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / f"{split}.csv", keep_default_na=False)


def to_ids(df: pd.DataFrame) -> np.ndarray:
    """Label wide -> matriks (n_ulasan, 10) berisi indeks kelas."""
    return df[ASPECTS].replace(LABEL2ID).to_numpy(dtype=np.int64)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Format panjang: satu baris per (ulasan, aspek) -- bentuk tugas ACSA."""
    out = df.reset_index(names="review_id").melt(
        id_vars=["review_id", "review"], value_vars=ASPECTS, var_name="aspect", value_name="sentiment")
    out["aspect"] = pd.Categorical(out["aspect"], categories=ASPECTS, ordered=True)
    return out.sort_values(["review_id", "aspect"], ignore_index=True)


def to_tuples(row: pd.Series) -> list:
    """Himpunan pasangan (aspek, polaritas) yang dibahas dalam satu ulasan; `neut` = tidak dibahas."""
    return [(a, row[a]) for a in ASPECTS if row[a] != "neut"]
