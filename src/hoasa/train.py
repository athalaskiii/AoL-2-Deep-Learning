"""Pelatihan pada split resmi IndoNLU: latih di train (bersih), pilih epoch terbaik dengan F1 valid, laporkan test.

Setiap run menambahkan satu baris ke outputs/results.jsonl (metrik valid/test pada epoch terbaik + riwayat per epoch)
dan menyimpan probabilitas valid/test ke outputs/preds/<nama>_s<seed>.npz untuk evaluasi dan ensemble.
Test dievaluasi setiap epoch hanya untuk disimpan; pemilihan epoch memakai valid saja.

Label latih memakai 4 kelas (kolom <aspek>_orig, neg_pos sebagai kelas sendiri). Saat prediksi, probabilitas neg_pos
digabung ke neg (`fold`), sehingga keluaran selalu 3 kelas dan jebakan metrik neg_pos (notebook 1) tidak terjadi.
Diagnostik: memetakan neg_pos -> neg sebelum pelatihan menurunkan F1 IndoBERT-base (outputs/results_3kelas.jsonl).
"""
import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ASPECT_TEXT, ASPECTS, ROOT, load_processed, to_ids
from .metrics import metrics
from .models import N_CLS, WORD_MODELS, TransformerAspect, TransformerCLS
from .preprocess import TOKEN_RE

SEEDS = [42, 1337, 2024]
OUT_DIR = ROOT / "outputs"
RESULTS = OUT_DIR / "results.jsonl"
PRED_DIR = OUT_DIR / "preds"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SPLITS = ["valid", "test"]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def done():
    if not RESULTS.exists():
        return set()
    return {(r["name"], r["seed"]) for r in map(json.loads, RESULTS.read_text().splitlines())}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    data = {s: load_processed(s) for s in ["train", *EVAL_SPLITS]}
    Y = {s: to_ids(d) for s, d in data.items()}
    Y["train"] = to_ids(data["train"][[f"{a}_orig" for a in ASPECTS]].set_axis(ASPECTS, axis=1))   # 4 kelas
    return data, Y


def fold(p):
    """Probabilitas 4 kelas -> 3 kelas: neg_pos (aspek dibahas negatif dan positif) dihitung sebagai neg."""
    p = np.array(p, dtype=np.float32)
    p[..., 0] += p[..., 3]
    return p[..., :3]


def save_run(name, family, seed, cfg, best, history, secs, n_params):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{slug(name)}_s{seed}.npz", **best["probs"])
    row = {"name": name, "family": family, "seed": seed, "cfg": cfg, "best_epoch": best["epoch"],
           "epochs_run": len(history), **{s: best[s] for s in EVAL_SPLITS}, "params": n_params,
           "minutes": round(secs / 60, 2), "history": history}
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    v, t = best["valid"], best["test"]
    print(f"DONE {name} s{seed}: best ep {best['epoch']} | valid f1 {v['f1_indonlu']:.4f} | "
          f"test f1 {t['f1_indonlu']:.4f} pair {t['f1_pair']:.4f} exact {t['exact']:.4f} | {secs / 60:.1f} min",
          flush=True)


# --------------------------------------------------------------------------- loop bersama untuk model neural
def fit(name, seed, model, opt, sched, train_batches, predict, Y, epochs, patience, smooth=0.0, on_best=None):
    """train_batches(rng) -> iterasi (input, y 4 kelas); predict(split) -> probabilitas 3 kelas (n, 10, 3).

    on_best(): dipanggil setiap F1 valid membaik, mis. untuk menyalin bobot epoch terbaik.
    """
    best, history, bad, t0 = {"epoch": 0, "valid": {"f1_indonlu": -1}}, [], 0, time.time()
    rng = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for inp, y in train_batches(rng):
            loss = F.cross_entropy(model(*inp).float().reshape(-1, N_CLS), y.reshape(-1), label_smoothing=smooth)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            probs = {s: predict(s) for s in EVAL_SPLITS}
        m = {s: metrics(Y[s], probs[s].argmax(-1)) for s in EVAL_SPLITS}
        history.append({"epoch": ep, "loss": round(float(np.mean(losses)), 5),
                        "valid_f1": round(m["valid"]["f1_indonlu"], 5), "test_f1": round(m["test"]["f1_indonlu"], 5)})
        improved = m["valid"]["f1_indonlu"] > best["valid"]["f1_indonlu"]
        if improved:
            best, bad = {"epoch": ep, **m, "probs": probs}, 0
            if on_best:
                on_best()
        else:
            bad += 1
        print(f"[{name} s{seed}] ep {ep:2d} loss {np.mean(losses):.4f} valid f1 {m['valid']['f1_indonlu']:.4f}"
              f"{' *' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            break
    return best, history, time.time() - t0


# --------------------------------------------------------------------------- baseline klasik
def run_tfidf(name="TF-IDF + LR", text="norm", C=10.0):
    """TF-IDF n-gram kata (1-2) + karakter (2-5) dan Logistic Regression per aspek (konfigurasi notebook 1)."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data, Y = load_data()
    t0 = time.time()
    txt = {s: d[f"text_{text}"].str.lower() for s, d in data.items()}
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    vw.fit(txt["train"])
    vc.fit(txt["train"])
    X = {s: hstack([vw.transform(t), vc.transform(t)]).tocsr() for s, t in txt.items()}
    probs = {s: np.zeros((X[s].shape[0], Y["train"].shape[1], N_CLS), dtype=np.float32) for s in EVAL_SPLITS}
    for j in range(Y["train"].shape[1]):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X["train"], Y["train"][:, j])
        for s in EVAL_SPLITS:
            probs[s][:, j, clf.classes_] = clf.predict_proba(X[s])
    probs = {s: fold(p) for s, p in probs.items()}
    best = {"epoch": 0, "probs": probs, **{s: metrics(Y[s], probs[s].argmax(-1)) for s in EVAL_SPLITS}}
    save_run(name, "klasik", 0, {"text": text, "C": C}, best, [], time.time() - t0, int(X["train"].shape[1] * 30))


# --------------------------------------------------------------------------- model berbasis kata
_FT = None


def fasttext():
    global _FT
    if _FT is None:
        from gensim.models.fasttext import load_facebook_vectors
        _FT = load_facebook_vectors(ROOT / "data" / "embeddings" / "cc.id.300.bin")
    return _FT


def run_word(name, seed, arch, text="norm", epochs=100, patience=10, bs=32, lr=1e-3, emb_lr=1e-4, max_len=128):
    """LSTM / BiLSTM / CNN dengan embedding FastText cc.id.300 yang di-fine-tune (lr embedding lebih kecil)."""
    data, Y = load_data()
    set_seed(seed)
    toks = {s: d[f"text_{text}"].str.lower().map(TOKEN_RE.findall).tolist() for s, d in data.items()}
    vocab = {}
    for ws in toks["train"]:
        for w in ws:
            vocab.setdefault(w, len(vocab) + 2)          # 0 = padding, 1 = OOV
    ft = fasttext()
    emb = np.zeros((len(vocab) + 2, ft.vector_size), dtype=np.float32)
    for w, k in vocab.items():
        emb[k] = ft[w]
    ids = {s: [[vocab.get(w, 1) for w in ws][:max_len] or [1] for ws in t] for s, t in toks.items()}

    def pad(seqs):
        x = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
        for i, q in enumerate(seqs):
            x[i, :len(q)] = torch.tensor(q)
        return x.to(DEV)

    Yt = {s: torch.tensor(y).to(DEV) for s, y in Y.items()}
    model = WORD_MODELS[arch](emb).to(DEV)
    opt = torch.optim.Adam([{"params": model.emb.parameters(), "lr": emb_lr},
                            {"params": [p for n, p in model.named_parameters() if not n.startswith("emb")], "lr": lr}])

    def train_batches(rng):
        for b in torch.randperm(len(ids["train"]), generator=rng).split(bs):
            yield (pad([ids["train"][i] for i in b]),), Yt["train"][b.to(DEV)]

    def predict(split):
        seqs = ids[split]
        return fold(torch.cat([model(pad(seqs[i:i + 256])).softmax(-1).cpu() for i in range(0, len(seqs), 256)]))

    best, history, secs = fit(name, seed, model, opt, None, train_batches, predict, Y, epochs, patience)
    cfg = {"arch": arch, "text": text, "lr": lr, "emb_lr": emb_lr, "bs": bs, "epochs": epochs, "patience": patience}
    save_run(name, "kata", seed, cfg, best, history, secs, sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- transformer
def aspect_descriptions(train, k=8):
    """Deskripsi tiap aspek = nama aspek + k kata yang paling berasosiasi (chi2) dengan aspek itu dibahas di train.

    Hanya memakai data train, sehingga konvensi anotator (mis. snack/air minum masuk linen) dipelajari tanpa
    melihat valid/test. Kata yang sangat umum (muncul di >25% ulasan) dibuang.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import chi2

    cv = CountVectorizer(binary=True, min_df=5, max_df=0.25, token_pattern=r"(?u)\b[a-z]{3,}\b")
    X = cv.fit_transform(train["text_clean"].str.lower())
    vocab = cv.get_feature_names_out()
    out = {}
    for a in ASPECTS:
        y = (train[a] != "neut").to_numpy()
        score = np.where(np.asarray(X[y].mean(0) > X[~y].mean(0)).ravel(), chi2(X, y)[0], 0)
        top = [w for w in vocab[np.argsort(-score)] if w not in ASPECT_TEXT[a].split()][:k]
        out[a] = " ".join([ASPECT_TEXT[a]] + top)
    return out


@torch.no_grad()
def encode_descriptions(backbone, tok, desc):
    """Rata-rata state terakhir encoder atas token deskripsi (tanpa token spesial) -> (10, hidden)."""
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(backbone).eval()
    t = tok([desc[a] for a in ASPECTS], padding=True, return_tensors="pt", return_special_tokens_mask=True)
    keep = (t.pop("special_tokens_mask") == 0) & (t["attention_mask"] == 1)
    h = enc(**t).last_hidden_state
    return (h * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)


def run_transformer(name, seed, backbone, text="clean", lr=3e-5, epochs=20, patience=5, bs=32, max_len=128,
                    head="cls", head_lr=None, smooth=0.0, save_dir=None):
    """Fine-tuning penuh: AdamW (wd 0,01), warmup 10% + decay linear, bf16, clip 1,0, early stopping F1 valid.

    head: "cls" (linear di atas token pertama), "aspect" (attention per aspek, query acak) atau "aspect_desc"
    (attention per aspek, query diinisialisasi dari deskripsi aspek hasil `aspect_descriptions`).
    head_lr: learning rate parameter head (default sama dengan lr). smooth: label smoothing cross-entropy.
    save_dir: jika diisi, bobot epoch terbaik disimpan ke save_dir/model.pt beserta konfigurasinya (config.json).
    """
    from transformers import AutoTokenizer

    data, Y = load_data()
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = {s: tok(d[f"text_{text}"].tolist(), max_length=max_len, truncation=True, padding=True, return_tensors="pt")
           for s, d in data.items()}
    Yt = {s: torch.tensor(y) for s, y in Y.items()}
    desc = aspect_descriptions(data["train"]) if head == "aspect_desc" else None
    if head == "cls":
        model = TransformerCLS(backbone)
    else:
        model = TransformerAspect(backbone, encode_descriptions(backbone, tok, desc) if desc else None)
    model.to(DEV)
    enc_params = list(model.enc.parameters())
    ids = {id(p) for p in enc_params}
    opt = torch.optim.AdamW([{"params": enc_params},
                             {"params": [p for p in model.parameters() if id(p) not in ids], "lr": head_lr or lr}],
                            lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(Y["train"]) / bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))

    def batch(split, idx):
        m = enc[split]["attention_mask"][idx]
        n = int(m.sum(1).max())                          # potong padding yang tidak terpakai di batch ini
        return enc[split]["input_ids"][idx, :n].to(DEV), m[:, :n].to(DEV)

    class BF16(nn.Module):                               # bungkus forward dengan autocast bf16
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids, mask):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                return self.m(ids, mask)

    wrapped = BF16(model)

    def train_batches(rng):
        for b in torch.randperm(len(Y["train"]), generator=rng).split(bs):
            yield batch("train", b), Yt["train"][b].to(DEV)

    def predict(split):
        idx = torch.arange(len(Y[split])).split(128)
        return fold(torch.cat([wrapped(*batch(split, b)).float().softmax(-1).cpu() for b in idx]))

    state = {}

    def keep_best():
        state.update({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})

    best, history, secs = fit(name, seed, wrapped, opt, sched, train_batches, predict, Y, epochs, patience, smooth,
                              keep_best if save_dir else None)
    cfg = {"backbone": backbone, "text": text, "lr": lr, "bs": bs, "epochs": epochs, "patience": patience,
           "max_len": max_len, "head": head, "head_lr": head_lr, "smooth": smooth, **({"desc": desc} if desc else {})}
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, save_dir / "model.pt")
        (save_dir / "config.json").write_text(json.dumps({"name": name, "seed": seed, "best_epoch": best["epoch"],
                                                          **cfg}, indent=1))
    save_run(name, "transformer", seed, cfg, best, history, secs, sum(p.numel() for p in model.parameters()))
    del model, wrapped, opt
    torch.cuda.empty_cache()
