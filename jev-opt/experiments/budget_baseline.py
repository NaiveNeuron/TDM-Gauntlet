"""Same-label-budget trained baselines on the 13 RQ3 sets (pre-registered).

Each baseline is trained on exactly the labels the decision-model pipeline used for that set: the 500-example
train split of the optimizer runs (all 500 for training) and the same validation split for every choice. The test
split is read once per family and set, after the family's configuration is fixed on validation.

Subcommands (run from the repository root under the jev-opt Python 3.11 environment plus SetFit/sentencepiece:
torch 2.14, transformers 5.17, sentence-transformers 5.x, scikit-learn 1.9; the `--with` overlay below adds SetFit
and DeBERTa's tokenizer dependencies without changing uv.lock):
  jev-preds              re-execute the tested Jev representatives from their Jev caches (posting disabled; a cache
                         miss is an error) -> results/budget_baseline/jev_test_preds.json
  emb    [--sets ...]    B1: frozen sentence encoder + logistic regression (or ridge for QWK sets)
  setfit [--sets ...]    B2: SetFit (contrastive fine-tuning of the encoder + logistic-regression head), 3 seeds
  ft     [--sets ...]    B3: full fine-tuning of RoBERTa-large and DeBERTa-v3-large, lr grid x 3 seeds
  timing --sets ...      B3 timing probe on train/validation only (never reads test)

The analysis (paired comparisons, paper macros and tables) is budget_report.py.

  uv run --project jev-opt/python --with setfit==1.2.0 --with sentencepiece --with protobuf \
      python jev-opt/experiments/budget_baseline.py <subcommand>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENROUTER_API_KEY", "offline-cache-only")  # never used: no Jev request is made
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from jev_opt import metrics  # noqa: E402
from jev_opt.hygiene import allow_test_split, load_split  # noqa: E402
from jev_opt.runs import _load_run_config  # noqa: E402
from resample import cluster_index, jackknife_t  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "budget_baseline"
OUT_MD = ROOT / "results" / "BUDGET_BASELINE.md"
OUT_JSON = ROOT / "results" / "BUDGET_BASELINE.json"
TEX = ROOT / "paper" / "generated"
B = 10_000
CHUNK = 500
SEEDS = (0, 1, 2)

# slug, short name, macro tag, group ("opt": 8 configurations; "frozen": 5 rule-selected sets; 3 optimizer seeds each)
SETS = [
    ("contempo-tbdense", "TB-Dense", "Tbd", "opt"),
    ("ezstance-subtaskA", "EZ-STANCE A", "Ez", "opt"),
    ("smarter-implicit-hate", "SMARTER implicit", "Sm", "opt"),
    ("edit-intent", "Edit intent", "Ed", "opt"),
    ("inli", "INLI", "Inli", "opt"),
    ("dress-new-content", "DREsS content", "DrC", "opt"),
    ("dress-new-organization", "DREsS organization", "DrO", "opt"),
    ("dress-new-language", "DREsS language", "DrL", "opt"),
    ("contempo-matres", "MATRES", "Mat", "frozen"),
    ("lifetox", "LifeTox", "Lt", "frozen"),
    ("sciprompt-s2orc", "SciPrompt S2ORC", "Sci", "frozen"),
    ("smarter-hatexplain", "HateXplain", "Hx", "frozen"),
    ("smarter-latent-hate", "Latent Hatred", "Lh", "frozen"),
]
SLUGS = [s for s, *_ in SETS]

# B1
EMB_ENCODERS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5")
EMB_C = (0.01, 0.1, 1.0, 10.0, 100.0)
EMB_CW = (None, "balanced")
EMB_RIDGE = (0.1, 1.0, 10.0, 100.0, 1000.0)
# B2
SETFIT_BODY = "BAAI/bge-base-en-v1.5"
# B3
FT_MODELS = ("FacebookAI/roberta-large", "microsoft/deberta-v3-large")
FT_LRS = (1e-5, 2e-5, 3e-5)
FT_MAX_EPOCHS = 20
FT_PATIENCE = 5
FT_BS = 16
FT_WARMUP = 0.1
MAX_LEN = 512
TRAIN_TOKENS = 4096  # micro-batch token budget (gradient accumulation to batch 16)
EVAL_TOKENS = 16384


# ---- data ------------------------------------------------------------------------------------------------------
def jev_runs(slug: str, group: str) -> list[str]:
    if group == "opt":
        seeds = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"][slug]["seeds"]
        return [seeds[str(s)]["run"] for s in SEEDS]
    return [f"fz-{slug}-P", f"fz-{slug}-P-s1", f"fz-{slug}-P-s2"]  # seed 0 first


def group_of(slug: str) -> str:
    return next(g for s, _, _, g in SETS if s == slug)


def text_of(fields: dict[str, str], inputs: list[str]) -> str:
    """The template: one `name: value` line per input field of the task config, in config order."""
    return "\n".join(f"{k}: {fields[k]}" for k in inputs)


class Data:
    def __init__(self, slug: str):
        runs = jev_runs(slug, group_of(slug))
        cfgs = [_load_run_config(RUNS / r) for r in runs]
        c = cfgs[0]
        for other in cfgs[1:]:  # every optimizer seed used the same files
            for split in ("train", "val", "test"):
                assert other.split_path(split).read_bytes() == c.split_path(split).read_bytes(), (slug, split)
            assert list(other.labels) == list(c.labels) and other.primary_metric == c.primary_metric
        self.slug, self.config = slug, c
        self.labels = list(c.labels)
        self.metric, self.neg = c.primary_metric, c.negative_label
        self.inputs = list(c.inputs)
        self.train = load_split(c, "train")
        self.val = load_split(c, "val")
        assert len(self.train) == 500, (slug, len(self.train))
        self.lid = {lab: i for i, lab in enumerate(self.labels)}

    def texts(self, rows) -> list[str]:
        return [text_of(e.fields, self.inputs) for e in rows]

    def y(self, rows) -> np.ndarray:
        return np.array([self.lid[e.label] for e in rows])

    def score(self, rows, pred_idx) -> float:
        return metrics.primary(self.metric, [e.label for e in rows], [self.labels[i] for i in pred_idx], self.labels, self.neg)

    def test(self):
        with allow_test_split():
            return load_split(self.config, "test")


def out_path(family: str, slug: str) -> Path:
    return OUT / family / f"{slug}.json"


def write_result(family: str, slug: str, rec: dict) -> None:
    p = out_path(family, slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1) + "\n")


def guard(family: str, slug: str) -> bool:
    """One test evaluation per family and set: an existing result file is never overwritten."""
    if out_path(family, slug).exists():
        print(f"{family}/{slug}: result exists, skipped (test already evaluated)", flush=True)
        return False
    return True


def ordinal(d: Data) -> bool:
    return d.metric == "qwk"


def round_to_grid(v: np.ndarray, k: int) -> np.ndarray:
    return np.clip(np.rint(v), 0, k - 1).astype(int)


# ---- Jev predictions -------------------------------------------------------------------------------------------
def cmd_jev_preds(_args) -> None:
    from analyze_test_v3 import DISCREPANCIES, evaluate  # import disables Jev posting (cache-only)

    out: dict[str, Any] = {}
    for slug, _, _, group in SETS:
        d = Data(slug)
        test = d.test()
        out[slug] = {"runs": {}}
        for run in jev_runs(slug, group):
            r = evaluate(RUNS / run)
            assert r["ids"] == [e.id for e in test] and r["gold"] == [e.label for e in test], (slug, run)
            assert abs(r["primary"] - r["recorded"]) < 1e-9, (run, r["primary"], r["recorded"])
            out[slug]["runs"][run] = {"pred": r["pred"], "primary": r["primary"]}
            print(slug, run, round(100 * r["primary"], 2), flush=True)
        out[slug]["ids"] = [e.id for e in test]
        out[slug]["gold"] = [e.label for e in test]
    if DISCREPANCIES:
        raise SystemExit(f"re-executed scores differ from recorded ones: {DISCREPANCIES}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jev_test_preds.json").write_text(json.dumps(out) + "\n")


# ---- B1: embeddings + linear model -----------------------------------------------------------------------------
def encode(model_name: str, texts: list[str], device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = MAX_LEN
    e = m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    del m
    gc.collect()
    return e


def emb_grid(d: Data):
    from sklearn.linear_model import LogisticRegression, Ridge

    for C in EMB_C:
        for cw in EMB_CW:
            yield {"model": "logreg", "C": C, "class_weight": cw}, lambda C=C, cw=cw: LogisticRegression(C=C, class_weight=cw, max_iter=5000)
    if ordinal(d):
        for a in EMB_RIDGE:
            yield {"model": "ridge", "alpha": a}, lambda a=a: Ridge(alpha=a)


def emb_predict(cfg: dict, m, X: np.ndarray, k: int) -> np.ndarray:
    return round_to_grid(m.predict(X), k) if cfg["model"] == "ridge" else m.predict(X)


def cmd_emb(args) -> None:
    for slug in args.sets:
        if not guard("emb", slug):
            continue
        t0 = time.time()
        d = Data(slug)
        ytr = d.y(d.train)
        grid, best = [], None
        for enc in EMB_ENCODERS:
            Xtr, Xva = encode(enc, d.texts(d.train), args.device), encode(enc, d.texts(d.val), args.device)
            for cfg, make in emb_grid(d):
                m = make().fit(Xtr, ytr)
                v = d.score(d.val, emb_predict(cfg, m, Xva, len(d.labels)))
                grid.append({"encoder": enc, **cfg, "val": v})
                if best is None or v > best[0]:
                    best = (v, enc, cfg, m)
        v, enc, cfg, m = best
        test = d.test()  # the one test read of this family and set
        pred = emb_predict(cfg, m, encode(enc, d.texts(test), args.device), len(d.labels))
        rec = {"family": "emb", "slug": slug, "chosen": {"encoder": enc, **cfg}, "val": [v],
               "test": [d.score(test, pred)], "test_pred": [[d.labels[i] for i in pred]], "grid": grid,
               "seconds": time.time() - t0}
        write_result("emb", slug, rec)
        print(f"emb {slug}: {enc} {cfg} val {100 * v:.1f} test {100 * rec['test'][0]:.1f}", flush=True)


# ---- B2: SetFit ------------------------------------------------------------------------------------------------
def cmd_setfit(args) -> None:
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    for slug in args.sets:
        if not guard("setfit", slug):
            continue
        t0 = time.time()
        d = Data(slug)
        train_ds = Dataset.from_dict({"text": d.texts(d.train), "label": d.y(d.train).tolist()})
        models, vals = [], []
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = SetFitModel.from_pretrained(SETFIT_BODY, head_params={"max_iter": 5000}, device=args.device)
            targs = TrainingArguments(batch_size=FT_BS, num_epochs=1, num_iterations=20, body_learning_rate=2e-5,
                                      max_length=MAX_LEN, seed=seed, report_to="none", show_progress_bar=False)
            Trainer(model=model, args=targs, train_dataset=train_ds).train()
            v = d.score(d.val, np.asarray(model.predict(d.texts(d.val), batch_size=64, as_numpy=True)))
            vals.append(v)
            models.append(model)
            print(f"setfit {slug} seed {seed}: val {100 * v:.1f} ({time.time() - t0:.0f}s)", flush=True)
        test = d.test()
        preds = [np.asarray(m.predict(d.texts(test), batch_size=64, as_numpy=True)) for m in models]
        rec = {"family": "setfit", "slug": slug, "chosen": {"body": SETFIT_BODY, "num_iterations": 20, "num_epochs": 1,
                                                             "body_learning_rate": 2e-5, "batch_size": FT_BS},
               "val": vals, "test": [d.score(test, p) for p in preds],
               "test_pred": [[d.labels[i] for i in p] for p in preds], "seconds": time.time() - t0}
        write_result("setfit", slug, rec)
        print(f"setfit {slug}: val {pm(vals)} test {pm(rec['test'])}", flush=True)
        del models
        gc.collect()
        torch.cuda.empty_cache()


# ---- B3: full fine-tuning --------------------------------------------------------------------------------------
class FT:
    def __init__(self, d: Data, model_name: str, device: str):
        from transformers import AutoTokenizer

        self.d, self.model_name, self.device = d, model_name, device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.reg = ordinal(d)
        ytr = d.y(d.train).astype(float)
        self.mu, self.sd = float(ytr.mean()), float(ytr.std()) or 1.0
        self.enc_train = self.tok(d.texts(d.train), truncation=True, max_length=MAX_LEN)["input_ids"]
        self.enc_val = self.tok(d.texts(d.val), truncation=True, max_length=MAX_LEN)["input_ids"]

    def batch(self, ids_list: list[list[int]]):
        import torch

        n = max(len(x) for x in ids_list)
        pad = self.tok.pad_token_id
        ids = torch.full((len(ids_list), n), pad, dtype=torch.long)
        att = torch.zeros((len(ids_list), n), dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        return ids.to(self.device), att.to(self.device)

    @staticmethod
    def chunks(idx: list[int], tokens: int, cap: int, enc: list[list[int]]) -> list[list[int]]:
        """Consecutive pieces of `idx` with (size x longest sequence) <= tokens and size <= cap."""
        out, cur, longest = [], [], 0
        for j in idx:
            n = max(longest, len(enc[j]))
            if cur and (len(cur) + 1 > cap or (len(cur) + 1) * n > tokens):
                out.append(cur)
                cur, n = [], len(enc[j])
            cur.append(j)
            longest = n
        return out + [cur] if cur else out

    def predict(self, model, enc: list[list[int]]) -> np.ndarray:
        import torch

        model.eval()
        order = np.argsort([len(x) for x in enc])
        out = np.empty(len(enc), dtype=int)
        with torch.no_grad():
            for idx in self.chunks([int(j) for j in order], EVAL_TOKENS, 64, enc):
                idx = np.array(idx)
                ids, att = self.batch([enc[j] for j in idx])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(input_ids=ids, attention_mask=att).logits.float()
                if self.reg:
                    out[idx] = round_to_grid(lo[:, 0].cpu().numpy() * self.sd + self.mu, len(self.d.labels))
                else:
                    out[idx] = lo.argmax(-1).cpu().numpy()
        return out

    def train(self, lr: float, seed: int, max_epochs: int = FT_MAX_EPOCHS) -> dict:
        """One run; returns the best-validation epoch's weights (CPU) and the validation curve."""
        import torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

        d = self.d
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1 if self.reg else len(d.labels), dtype=torch.float32).to(self.device)
        ytr = d.y(d.train)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        per_epoch = (len(ytr) + FT_BS - 1) // FT_BS
        total = FT_MAX_EPOCHS * per_epoch
        sch = get_linear_schedule_with_warmup(opt, int(FT_WARMUP * total), total)
        rng = random.Random(seed)
        curve, best, since = [], (-float("inf"), None, -1), 0
        for ep in range(max_epochs):
            model.train()
            order = list(range(len(ytr)))
            rng.shuffle(order)
            for i in range(0, len(order), FT_BS):
                batch = order[i:i + FT_BS]
                for idx in self.chunks(batch, TRAIN_TOKENS, FT_BS, self.enc_train):  # gradient accumulation
                    ids, att = self.batch([self.enc_train[j] for j in idx])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lo = model(input_ids=ids, attention_mask=att).logits.float()
                    if self.reg:
                        y = torch.tensor((ytr[idx] - self.mu) / self.sd, dtype=torch.float, device=self.device)
                        loss = torch.nn.functional.mse_loss(lo[:, 0], y, reduction="sum")
                    else:
                        loss = torch.nn.functional.cross_entropy(lo, torch.tensor(ytr[idx], device=self.device), reduction="sum")
                    (loss / len(batch)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
            v = d.score(d.val, self.predict(model, self.enc_val))
            curve.append(v)
            if v > best[0]:
                best, since = (v, {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}, ep), 0
            else:
                since += 1
                if since >= FT_PATIENCE:
                    break
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        return {"val": best[0], "state": best[1], "best_epoch": best[2] + 1, "epochs_run": len(curve), "curve": curve}

    def load(self, state):
        import torch
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1 if self.reg else len(self.d.labels), dtype=torch.float32)
        model.load_state_dict(state)
        return model.to(self.device)


def cmd_ft(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    for slug in args.sets:
        if not guard("ft", slug):
            continue
        t0 = time.time()
        d = Data(slug)
        grid: list[dict] = []
        best = None  # (mean val, model, lr, [run dicts with state])
        for model_name in FT_MODELS:
            ft = FT(d, model_name, args.device)
            per_lr: dict[float, list[dict]] = {}
            for lr in FT_LRS:
                per_lr[lr] = []
                for seed in SEEDS:
                    r = ft.train(lr, seed)
                    per_lr[lr].append(r)
                    grid.append({"model": model_name, "lr": lr, "seed": seed, "val": r["val"], "best_epoch": r["best_epoch"],
                                 "epochs_run": r["epochs_run"], "curve": r["curve"]})
                    print(f"ft {slug} {model_name} lr {lr:g} seed {seed}: val {100 * r['val']:.1f} "
                          f"(epoch {r['best_epoch']}/{r['epochs_run']}, {time.time() - t0:.0f}s)", flush=True)
            lr_best = max(FT_LRS, key=lambda lr: statistics.mean(r["val"] for r in per_lr[lr]))  # first max wins ties
            mv = statistics.mean(r["val"] for r in per_lr[lr_best])
            if best is None or mv > best[0]:
                best = (mv, model_name, lr_best, per_lr[lr_best])
            del per_lr
            gc.collect()
        mv, model_name, lr, runs = best
        ft = FT(d, model_name, args.device)
        test = d.test()  # the one test read of this family and set
        enc_test = ft.tok(d.texts(test), truncation=True, max_length=MAX_LEN)["input_ids"]
        preds, vals_check = [], []
        for r in runs:
            model = ft.load(r["state"])
            vals_check.append(d.score(d.val, ft.predict(model, ft.enc_val)))
            preds.append(ft.predict(model, enc_test))
            del model
            torch.cuda.empty_cache()
        rec = {"family": "ft", "slug": slug, "chosen": {"model": model_name, "lr": lr, "batch_size": FT_BS,
                                                         "max_epochs": FT_MAX_EPOCHS, "patience": FT_PATIENCE,
                                                         "head": "regression+rounding" if ft.reg else "classification"},
               "val": [r["val"] for r in runs], "val_reloaded": vals_check,
               "test": [d.score(test, p) for p in preds], "test_pred": [[d.labels[i] for i in p] for p in preds],
               "grid": grid, "seconds": time.time() - t0}
        write_result("ft", slug, rec)
        print(f"ft {slug}: {model_name} lr {lr:g} val {pm(rec['val'])} test {pm(rec['test'])} ({rec['seconds']:.0f}s)", flush=True)
        del best, runs
        gc.collect()


def cmd_timing(args) -> None:
    for slug in args.sets:
        d = Data(slug)
        for model_name in FT_MODELS:
            ft = FT(d, model_name, args.device)
            t0 = time.time()
            r = ft.train(2e-5, 0, max_epochs=2)
            print(f"timing {slug} {model_name}: {(time.time() - t0) / 2:.1f}s/epoch (incl. val), val {r['curve']}", flush=True)


# ---- formatting ------------------------------------------------------------------------------------------------
def pm(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{100 * xs[0]:.1f}"
    return f"{100 * statistics.mean(xs):.1f} ± {100 * statistics.stdev(xs):.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jev-preds", "emb", "setfit", "ft", "timing"])
    ap.add_argument("--sets", default=",".join(SLUGS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.sets = a.sets.split(",")
    assert set(a.sets) <= set(SLUGS), set(a.sets) - set(SLUGS)
    {"jev-preds": cmd_jev_preds, "emb": cmd_emb, "setfit": cmd_setfit, "ft": cmd_ft, "timing": cmd_timing}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
