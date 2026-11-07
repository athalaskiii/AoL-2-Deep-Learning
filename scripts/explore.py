"""Eksplorasi metode tambahan, terpisah dari pipeline utama (notebook 2/03).

Hasil -> outputs/explore/results.jsonl dan outputs/explore/preds/; run yang sudah ada dilewati.
Pembanding: baseline di outputs/results.jsonl dengan protokol yang sama (3 seed, epoch terbaik menurut F1 valid).

uv run python scripts/explore.py mdeberta indolem xlmr_base_norm rdrop concat tapt cl indobert_large_p1 pair
"""
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from hoasa import train as T  # noqa: E402
from hoasa.data import ASPECT_TEXT, ASPECTS  # noqa: E402
from hoasa.metrics import metrics  # noqa: E402
from hoasa.models import N_CLS, TransformerAspect, TransformerCLS  # noqa: E402

hf_logging.disable_progress_bar()
EXP = T.ROOT / "outputs" / "explore"
EXP.mkdir(parents=True, exist_ok=True)
T.RESULTS, T.PRED_DIR = EXP / "results.jsonl", EXP / "preds"     # simpan terpisah dari hasil utama
DEV = T.DEV
NEUT, IGNORE = 1, -100
BASE = "indobenchmark/indobert-base-p1"
COOLDOWN = int(os.environ.get("HOASA_COOLDOWN", "0"))


class PairModel(nn.Module):
    """Formulasi auxiliary sentence (Sun et al., 2019): input (ulasan, pertanyaan aspek) -> 4 kelas untuk satu aspek."""

    def __init__(self, name):
        from transformers import AutoModel
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.enc.config.hidden_size, N_CLS)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.out(self.drop(h)).view(-1, 1, N_CLS)


class BF16(nn.Module):
    def __init__(self, m, enabled=True):
        super().__init__()
        self.m, self.enabled = m, enabled

    def forward(self, ids, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.enabled and DEV == "cuda"):
            return self.m(ids, mask)


def question(a):
    return f"bagaimana {ASPECT_TEXT[a]}?"


def concat_pairs(Y, lengths, max_len):
    """Pasangan ulasan train (i, j) yang aspek dibahasnya tidak beririsan dan gabungannya muat dalam max_len token."""
    mentioned = Y != NEUT
    disjoint = (mentioned.astype(int) @ mentioned.T.astype(int)) == 0
    fits = (lengths[:, None] + lengths[None, :]) <= max_len - 3
    ok = disjoint & fits
    np.fill_diagonal(ok, False)
    return [np.flatnonzero(r) for r in ok]


# --------------------------------------------------------------------------- satu run fine-tuning
def run(name, seed, backbone, text="clean", lr=3e-5, epochs=20, patience=5, bs=32, max_len=128, head="cls",
        head_lr=None, rdrop=0.0, concat=0.0, clean=None, pair=False, train_idx=None, heldout_idx=None, save=True,
        bf16=True):
    """Fine-tuning dengan opsi eksplorasi.

    rdrop: bobot R-Drop (KL simetris antar dua forward dengan dropout berbeda).
    concat: rasio contoh tambahan per epoch dari penggabungan dua ulasan dengan aspek yang tidak beririsan.
    clean: nama berkas .npy (n_train, 10) bool; sel bertanda diabaikan di loss (hasil `cl`).
    pair: formulasi pasangan kalimat (10 contoh per ulasan).
    train_idx / heldout_idx: subset train untuk pelatihan dan bagian train yang diprediksi (dipakai `cl`).
    bf16: autocast bf16 (default, sama dengan pipeline utama).
    """
    from transformers import AutoTokenizer

    data, Y = T.load_data()
    T.set_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    texts = {s: d[f"text_{text}"].tolist() for s, d in data.items()}
    y_train = Y["train"].copy()
    if clean:
        y_train[np.load(EXP / clean)] = IGNORE
    idx_train = np.arange(len(y_train)) if train_idx is None else np.asarray(train_idx)

    if pair:
        model = PairModel(backbone)
    elif head == "cls":
        model = TransformerCLS(backbone)
    else:
        model = TransformerAspect(backbone, T.encode_descriptions(backbone, tok, T.aspect_descriptions(data["train"])))
    model.float().to(DEV)   # beberapa checkpoint (mis. mDeBERTa) tersimpan fp16; bobot latih selalu fp32
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([{"params": list(model.enc.parameters())},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": head_lr or lr}],
                            lr=lr, weight_decay=0.01)
    n_per_epoch = len(idx_train) * (10 if pair else 1) * (1 + concat)
    steps = epochs * math.ceil(n_per_epoch / bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    wrapped = BF16(model, bf16)

    def encode(a, b=None):
        t = tok(a, b, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
        return t["input_ids"].to(DEV), t["attention_mask"].to(DEV)

    if concat:
        lengths = np.array([len(tok.tokenize(t)) for t in texts["train"]])
        partners = concat_pairs(Y["train"], lengths, max_len)
        has_partner = np.array([i for i in idx_train if len(np.intersect1d(partners[i], idx_train)) > 0])

    def train_batches(rng):
        xs = [texts["train"][i] for i in idx_train]
        ys = [y_train[i] for i in idx_train]
        if concat:
            for i in rng.choice(has_partner, int(concat * len(idx_train))):
                cand = np.intersect1d(partners[i], idx_train)
                j = rng.choice(cand)
                xs.append(texts["train"][i] + " " + texts["train"][j])
                ys.append(np.where(Y["train"][i] != NEUT, y_train[i], y_train[j]))
        if pair:
            xs, qs, ys = ([x for x in xs for _ in ASPECTS], [question(a) for _ in xs for a in ASPECTS],
                          [v for y in ys for v in y])
        order = rng.permutation(len(xs))
        for k in range(0, len(order), bs):
            b = order[k:k + bs]
            x = encode([xs[i] for i in b], [qs[i] for i in b] if pair else None)
            y = torch.tensor(np.array([ys[i] for i in b])).reshape(len(b), -1).to(DEV)
            yield x, y

    split_text = {**texts, "heldout": [texts["train"][i] for i in (heldout_idx if heldout_idx is not None else [])]}

    def predict(split):
        xs, out = split_text[split], []
        for k in range(0, len(xs), 64):
            chunk = xs[k:k + 64]
            if pair:
                logits = wrapped(*encode([x for x in chunk for _ in ASPECTS], [question(a) for _ in chunk for a in ASPECTS]))
                logits = logits.view(len(chunk), len(ASPECTS), N_CLS)
            else:
                logits = wrapped(*encode(chunk))
            out.append(logits.float().softmax(-1).cpu())
        return T.fold(torch.cat(out))

    splits = T.EVAL_SPLITS + (["heldout"] if heldout_idx is not None else [])
    best, history, bad, t0 = {"epoch": 0, "valid": {"f1_indonlu": -1}}, [], 0, time.time()
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        wrapped.train()
        losses = []
        for x, y in train_batches(rng):
            out = wrapped(*x).float()
            loss = F.cross_entropy(out.reshape(-1, N_CLS), y.reshape(-1), ignore_index=IGNORE)
            if rdrop:
                out2 = wrapped(*x).float()
                loss = (loss + F.cross_entropy(out2.reshape(-1, N_CLS), y.reshape(-1), ignore_index=IGNORE)) / 2
                lp, lq = out.log_softmax(-1), out2.log_softmax(-1)
                keep = (y != IGNORE).unsqueeze(-1).float()
                kl = ((lp.exp() * (lp - lq) + lq.exp() * (lq - lp)) * keep).sum() / keep.sum().clamp(min=1) / 2
                loss = loss + rdrop * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        wrapped.eval()
        with torch.no_grad():
            probs = {s: predict(s) for s in splits}
        m = {s: metrics(Y[s], probs[s].argmax(-1)) for s in T.EVAL_SPLITS}
        history.append({"epoch": ep, "loss": round(float(np.mean(losses)), 5),
                        "valid_f1": round(m["valid"]["f1_indonlu"], 5), "test_f1": round(m["test"]["f1_indonlu"], 5)})
        improved = m["valid"]["f1_indonlu"] > best["valid"]["f1_indonlu"]
        if improved:
            best, bad = {"epoch": ep, **m, "probs": probs}, 0
        else:
            bad += 1
        print(f"[{name} s{seed}] ep {ep:2d} loss {np.mean(losses):.4f} valid f1 {m['valid']['f1_indonlu']:.4f}"
              f"{' *' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            break
    secs = time.time() - t0
    if save:
        cfg = {"backbone": str(backbone), "text": text, "lr": lr, "bs": bs, "epochs": epochs, "patience": patience,
               "max_len": max_len, "head": head, "head_lr": head_lr, "rdrop": rdrop, "concat": concat, "clean": clean,
               "pair": pair, "bf16": bf16}
        best["probs"] = {s: best["probs"][s] for s in T.EVAL_SPLITS}
        T.save_run(name, "eksplorasi", seed, cfg, best, history, secs, sum(p.numel() for p in model.parameters()))
    del model, wrapped, opt
    torch.cuda.empty_cache()
    return best


# --------------------------------------------------------------------------- TAPT
def tapt(backbone=BASE, out="tapt_indobert_base", epochs=40, lr=5e-5, bs=32, max_len=128, seed=42):
    """Task-adaptive pretraining (Gururangan et al., 2020): lanjutkan MLM pada teks ulasan train saja (tanpa label)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

    path = EXP / out
    if (path / "config.json").exists():
        return path
    data, _ = T.load_data()
    T.set_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForMaskedLM.from_pretrained(backbone).to(DEV)
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    enc = tok(data["train"]["text_clean"].tolist(), max_length=max_len, truncation=True)["input_ids"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(enc) / bs)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    rng, t0 = np.random.default_rng(seed), time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for k in range(0, len(enc), bs):
            b = coll([{"input_ids": enc[i]} for i in rng.permutation(len(enc))[k:k + bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = model(**{n: v.to(DEV) for n, v in b.items()}).loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if ep % 5 == 0 or ep == 1:
            print(f"[TAPT] ep {ep:2d} mlm loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    del model
    torch.cuda.empty_cache()
    return path


# --------------------------------------------------------------------------- confident learning
def confident_learning(backbone=BASE, k=5, seed=42, out="cl_mask.npy"):
    """Prediksi out-of-fold (k-fold di dalam train) lalu tandai sel yang labelnya kemungkinan salah (Northcutt et al., 2021).

    Ambang per kelas t_c = rata-rata p(c) pada sel berlabel c. Sel ditandai bila kelas lain k memenuhi p(k) >= t_k
    dan kelas itu (argmax di antara yang memenuhi) berbeda dari label. Sel neg_pos tidak pernah ditandai.
    """
    if (EXP / out).exists():
        return out
    data, Y = T.load_data()
    n = len(Y["train"])
    folds = np.random.default_rng(seed).permutation(n) % k
    oof = np.zeros((n, len(ASPECTS), 3), dtype=np.float32)
    for f in range(k):
        held = np.flatnonzero(folds == f)
        best = run(f"CL-fold{f}", seed, backbone, train_idx=np.flatnonzero(folds != f), heldout_idx=held, save=False)
        oof[held] = best["probs"]["heldout"]
        time.sleep(COOLDOWN)
    y3 = np.where(Y["train"] == 3, 0, Y["train"])
    thr = np.array([oof[..., c][y3 == c].mean() for c in range(3)])
    above = oof >= thr
    cand = np.where(above, oof, -1).argmax(-1)
    flag = above.any(-1) & (cand != y3) & (Y["train"] != 3)
    np.save(EXP / out, flag)
    np.save(EXP / "cl_oof.npy", oof)
    print(f"[CL] ambang {np.round(thr, 3).tolist()} | sel ditandai {flag.sum()} dari {flag.size} "
          f"({100 * flag.mean():.2f}%), di {flag.any(1).sum()} ulasan", flush=True)
    return out


# --------------------------------------------------------------------------- daftar eksperimen
ASPDESC = {"head": "aspect_desc", "head_lr": 1e-3}
EXPERIMENTS = {
    "mdeberta": lambda s: run("mDeBERTa-v3-base", s, "microsoft/mdeberta-v3-base"),
    "indolem": lambda s: run("IndoLEM-IndoBERT-base", s, "indolem/indobert-base-uncased"),
    "indobert_large_p1": lambda s: run("IndoBERT-large-p1", s, "indobenchmark/indobert-large-p1"),
    "xlmr_base_norm": lambda s: run("XLM-R-base (text_norm)", s, "FacebookAI/xlm-roberta-base", lr=2e-5, text="norm"),
    "rdrop": lambda s: run("IndoBERT-base + R-Drop", s, BASE, rdrop=1.0),
    "concat": lambda s: run("IndoBERT-base + ConcatAug", s, BASE, concat=0.5),
    "tapt": lambda s: run("IndoBERT-base + TAPT", s, tapt()),
    "cl": lambda s: run("IndoBERT-base + CL-clean", s, BASE, clean=confident_learning()),
    "pair": lambda s: run("IndoBERT-base PAIR", s, BASE, pair=True, epochs=10, patience=3),
}
NAMES = {"mdeberta": "mDeBERTa-v3-base", "indolem": "IndoLEM-IndoBERT-base", "indobert_large_p1": "IndoBERT-large-p1",
         "xlmr_base_norm": "XLM-R-base (text_norm)", "rdrop": "IndoBERT-base + R-Drop", "concat": "IndoBERT-base + ConcatAug",
         "tapt": "IndoBERT-base + TAPT", "cl": "IndoBERT-base + CL-clean", "pair": "IndoBERT-base PAIR"}

if __name__ == "__main__":
    for key in sys.argv[1:]:
        for seed in T.SEEDS:
            if (NAMES[key], seed) not in T.done():
                EXPERIMENTS[key](seed)
                time.sleep(COOLDOWN)
