"""Pipeline preprocessing bertingkat untuk ulasan HoASA.

Tingkat (masing-masing dibangun dari tingkat sebelumnya):
  raw    teks asli
  clean  NFKC, emoji -> kata, huruf kecil, reduplikasi angka, elongasi, tanda baca berulang
  norm   clean + normalisasi kata informal (kamus terkurasi) + penyeragaman negasi
  stem   norm + stemming Sastrawi
  lemma  norm + lemmatisasi Stanza
  stop   norm + penghapusan stopword (negasi, intensitas, dan kata kontras dipertahankan)
"""
import re
import unicodedata
from functools import lru_cache

import emoji
import pandas as pd

from .data import RESOURCE_DIR

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[!?]")

# Kata yang tidak boleh diubah kamus: nama/penanda aspek dan kata Inggris yang lazim di ulasan campur kode.
PROTECTED = {
    "ac", "tv", "wifi", "wc", "snack", "staff", "lobby", "linen", "bed", "room", "shower", "toilet",
    "check", "in", "out", "so", "it", "on", "at", "no", "ok", "oke", "service", "hotel",
}
NEGATION = {"ga", "gak", "gk", "g", "nggak", "ngga", "nggk", "enggak", "engga", "ngak", "tdk", "tak", "ndak",
            "nda", "kagak", "gx", "tida", "tdak"}
# Dipertahankan saat stopword dihapus: penentu polaritas dan intensitas opini.
KEEP_WORDS = {
    "tidak", "bukan", "belum", "jangan", "tanpa", "kurang", "ada", "sangat", "sekali", "banget", "terlalu", "cukup",
    "agak", "lumayan", "lebih", "paling", "tapi", "tetapi", "namun", "padahal", "walaupun", "meskipun", "masih",
    "sudah", "harus", "bisa", "!", "?",
    # kata opini/aspek yang ternyata ada di daftar stopword Sastrawi
    "baik", "enak", "masalah", "lama", "baru", "besar", "kecil", "dekat", "jauh", "keluar",
}
CLITICS = {"nya", "ku", "mu", "lah", "kah", "pun"}


def clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("’", "'")
    t = emoji.replace_emoji(t, replace=lambda e, d: " " + _emoji_id(e) + " ")
    t = t.lower()
    t = re.sub(r"\b([a-z]{2,})2(?=\b|nya\b)", r"\1-\1", t)   # mati2 -> mati-mati
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)                   # bangettt -> banget
    t = re.sub(r"([!?.,])\1+", r"\1", t)                     # !!! -> !
    return re.sub(r"\s+", " ", t).strip()


def _emoji_id(e: str) -> str:
    name = emoji.demojize(e, language="id", delimiters=("", ""))
    return "" if name == e else name.replace("_", " ")


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text)


@lru_cache(maxsize=1)
def lexicon() -> dict:
    """Kamus kata informal -> baku: kamus alay (Ibrohim & Budi, 2019) diprioritaskan,
    dilengkapi Colloquial Indonesian Lexicon (Salsabila et al., 2018), lalu dikurasi."""
    sal = pd.read_csv(RESOURCE_DIR / "colloquial-indonesian-lexicon.csv")
    sal = sal.groupby("slang")["formal"].agg(lambda s: s.mode().iloc[0])
    ala = pd.read_csv(RESOURCE_DIR / "new_kamusalay.csv", encoding="latin-1", header=None, names=["slang", "formal"])
    lex = {**dict(zip(sal.index.str.lower(), sal.str.lower())),
           **dict(zip(ala.slang.str.lower(), ala.formal.str.lower()))}
    lex = {k: v.strip() for k, v in lex.items()
           if k not in PROTECTED and k.removesuffix("nya") not in PROTECTED and k != v.strip()}
    lex.update({w: "tidak" for w in NEGATION})
    lex.update({"k": "ke", "dmn": "di mana"})
    lex.update({"gada": "tidak ada", "gaada": "tidak ada", "tdkada": "tidak ada"})
    return lex


def normalize(text: str) -> str:
    lex = lexicon()
    return " ".join(lex.get(w, w) for w in tokenize(clean(text)))


@lru_cache(maxsize=1)
def _stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=None)
def stem_word(w: str) -> str:
    return w if not w.isalpha() else (_stemmer().stem(w) or w)


def stem(norm_text: str) -> str:
    return " ".join(stem_word(w) for w in norm_text.split())


@lru_cache(maxsize=1)
def _stanza():
    import stanza
    import torch
    stanza.download("id", processors="tokenize,mwt,pos,lemma", verbose=False)
    return stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma", verbose=False, use_gpu=torch.cuda.is_available())


def lemmatize_all(norm_texts: list) -> list:
    """Lemmatisasi Stanza (model UD Indonesian-GSD) secara batch; klitik hasil pemisahan MWT dibuang."""
    import stanza
    docs = _stanza().bulk_process([stanza.Document([], text=t) for t in norm_texts])
    out = []
    for d in docs:
        words = [(w.lemma or w.text).lower() for s in d.sentences for w in s.words if w.text.lower() not in CLITICS]
        out.append(" ".join(w for w in words if TOKEN_RE.fullmatch(w)))
    return out


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    return frozenset(StopWordRemoverFactory().get_stop_words()) - KEEP_WORDS


def remove_stop(norm_text: str) -> str:
    sw = stopwords()
    return " ".join(w for w in norm_text.split() if w not in sw)


def build_levels(reviews: pd.Series) -> pd.DataFrame:
    """Semua tingkat preprocessing untuk satu kolom ulasan."""
    out = pd.DataFrame({"raw": reviews.values})
    out["clean"] = out.raw.map(clean)
    out["norm"] = out.raw.map(normalize)
    out["stem"] = out.norm.map(stem)
    out["lemma"] = lemmatize_all(out.norm.tolist())
    out["stop"] = out.norm.map(remove_stop)
    return out
