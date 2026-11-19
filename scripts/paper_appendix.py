# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Appendix generator: renders paper/generated/app_*.tex from repo data (called by paper_tables.py).

Per-paper prose lives in scripts/appendix/<prefix>.tex: LaTeX with data placeholders, so every
count, score, hash and cost is read from the repo, never typed by hand.

Placeholders (single line)::

    <<KIND:ID:PATH|FMT>>

    KIND  m   data/manifests/ID.json         s   runs/ID/scores.json
          r   runs/ID/manifest.json          pn  results/paper_numbers.json, entry ID
          q   prompts/ID.json                j   ID is a repo-relative JSON file path
          jl  ID is a repo-relative JSONL file; PATH starts with the 0-based line index
    PATH  dot-separated keys (greedy: keys that themselves contain dots, slashes or spaces
          match whole) and list indices; empty PATH = the whole document.
    FMT   (optional) pct pct2 f0..f4 int usd tt url sha commit list items len keys counts
          upper; default: strings escaped, ints with thousands separators, floats 3 decimals.

Blocks::

    <<papernums:ID=Label;ID2=Label2|FIELD>>   the paper's reported numbers (x100) as a table;
                                             FIELD defaults to "paper" (e.g. paper_accuracy)
    <<metrics
    cols: ID=Label; ID2=Label2
    path.in.scores = Row label | fmt
    ?optional.path = Row label | fmt           ("?" renders -- when absent in a column)
    >>
    <<block:NAME>>                            registered Python block (see BLOCKS)

A missing path is a hard error, so a typo can never silently print a wrong number.

Check one template in isolation (renders, then compiles a two-column stub with tectonic):

    uv run scripts/paper_appendix.py --check lifetox
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
TEMPLATES = ROOT / "scripts" / "appendix"

# --------------------------------------------------------------------------------------------
# LaTeX escaping
# --------------------------------------------------------------------------------------------
SPECIALS = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
            "_": r"\_\allowbreak{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
            ">": r"\textgreater{}", "|": r"\textbar{}", "/": r"/\allowbreak{}"}
UNICODE = {"→": r"$\rightarrow$", "←": r"$\leftarrow$", "’": "'", "‘": "`", "“": "``", "”": "''", "–": "--", "—": "---",
           "†": r"$\dagger$", "δ": r"$\delta$", "Δ": r"$\Delta$", "§": r"\S{}", "×": r"$\times$", "±": r"$\pm$",
           "é": r"\'e", "è": r"\`e", "ä": r'\"a', "ö": r'\"o', "Ö": r'\"O', "ü": r'\"u', "ñ": r"\~n", "ç": r"\c{c}",
           "ş": r"\c{s}", "ı": r"\i{}", "°": r"$^\circ$", "…": r"\ldots{}", "∪": r"$\cup$", "∩": r"$\cap$",
           "€": "EUR", "£": r"\pounds{}", "μ": r"$\mu$", "−": r"$-$", "β": r"$\beta$", "α": r"$\alpha$", "ρ": r"$\rho$",
           "τ": r"$\tau$", "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "\xa0": "~", "´": "'", "©": r"\textcopyright{}",
           "κ": r"$\kappa$", "≠": r"$\neq$", "·": r"$\cdot$", "✓": r"$\checkmark$", "Σ": r"$\Sigma$", "θ": r"$\theta$",
           # glyphs quoted in language-screen notes that the paper fonts cannot render
           "ඞ": "[Sinhala glyph]", "\u0361": "", "\u035c": "", "ʖ": "?", "关": "[CJK]", "键": "", "词": "", "汉": "[CJK]", "字": ""}
ASCII = {"→": "->", "←": "<-", "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "--", "…": "...", "×": "x",
         "≥": ">=", "≤": "<=", "−": "-", "\xa0": " ", "é": "e", "è": "e", "δ": "delta", "§": "S", "°": "deg"}
UNKNOWN: set[str] = set()


def esc(s: Any) -> str:
    out = []
    prev = " "
    for ch in str(s).replace("\u0338=", "≠"):  # PDF-extracted "j \u0338= k" is j != k
        opening = prev in " ([{/\n\t"
        if ch == "'" and opening:
            out.append("`")
        elif ch == '"':
            out.append("``" if opening else "''")
        elif ch in SPECIALS:
            out.append(SPECIALS[ch])
        elif ord(ch) < 128:
            out.append(ch)
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        else:
            UNKNOWN.add(ch)
            out.append("?")
        prev = ch
    return "".join(out)


def ascii_only(s: str) -> str:
    """For verbatim listings: transliterate non-ASCII (verbatim cannot hold LaTeX escapes)."""
    return "".join(ch if ord(ch) < 128 else ASCII.get(ch, "?") for ch in s)


# --------------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------------
@cache
def load_json(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@cache
def load_jsonl(rel: str) -> list[Any]:
    p = ROOT / rel
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


KIND_PATH = {"m": "data/manifests/{}.json", "s": "runs/{}/scores.json", "r": "runs/{}/manifest.json",
             "q": "prompts/{}.json", "j": "{}"}


@cache
def load_csv(rel: str) -> list[dict[str, str]]:
    import csv

    p = ROOT / rel
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t" if p.suffix == ".tsv" else ","))


@cache
def load_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def document(kind: str, ident: str) -> Any:
    if kind == "pn":
        return load_json("results/paper_numbers.json")[ident]
    if kind == "jl":
        return load_jsonl(ident)
    if kind == "csv":
        return load_csv(ident)
    if kind not in KIND_PATH:
        raise KeyError(f"unknown placeholder kind {kind!r}")
    return load_json(KIND_PATH[kind].format(ident))


def regex_quote(ident: str, pattern: str, where: str) -> str:
    """<<rx:FILE:REGEX|fmt>>: verbatim text of the first match (group 1 if the regex has one) in a text file."""
    m = re.search(pattern, load_text(ident), re.S)
    if m is None:
        raise KeyError(f"{where}: regex {pattern!r} not found in {ident}")
    return m.group(1) if m.re.groups else m.group(0)


def resolve(node: Any, path: str, where: str = "") -> Any:
    rest = path
    while rest:
        if isinstance(node, dict):
            match = next((k for k in sorted(node, key=len, reverse=True) if rest == k or rest.startswith(k + ".")), None)
            if match is None:
                raise KeyError(f"{where}: no key for {rest!r} (have: {sorted(node)[:30]})")
            node, rest = node[match], rest[len(match) + 1:]
        elif isinstance(node, list):
            head, _, rest = rest.partition(".")
            if not head.lstrip("-").isdigit() or not -len(node) <= int(head) < len(node):
                raise KeyError(f"{where}: bad list index {head!r} (len {len(node)})")
            node = node[int(head)]
        else:
            raise KeyError(f"{where}: cannot descend into {type(node).__name__} with {rest!r}")
    return node


def fmt(v: Any, spec: str = "") -> str:
    if re.fullmatch(r"c\d+", spec):  # clip text to N chars at a word boundary, whitespace collapsed
        text, limit = " ".join(str(v).split()), int(spec[1:])
        if len(text) <= limit:
            return esc(text)
        return esc(text[:limit].rsplit(" ", 1)[0]) + "~\\ldots{}"
    if spec == "len":
        return f"{len(v):,}"
    if spec == "keys":
        return ", ".join(esc(k) for k in v)
    if spec == "counts":
        return ", ".join(f"{esc(k)} {fmt(x)}" for k, x in v.items())
    if spec == "list":
        return ", ".join(fmt(x) for x in v)
    if spec == "items":
        return "\\begin{itemize}\n" + "".join(f"\\item {fmt(x)}\n" for x in v) + "\\end{itemize}"
    if v is None or (isinstance(v, float) and v != v):  # None and NaN (e.g. undefined coverage) render as --
        return "--"
    if spec == "pct":
        return f"{100 * v:.1f}"
    if spec == "pct2":
        return f"{100 * v:.2f}"
    if re.fullmatch(r"f\d", spec):
        return f"{v:,.{spec[1]}f}"
    if spec == "int":
        return f"{int(round(v)):,}"
    if spec == "usd":
        return f"\\${v:,.2f}"
    if spec == "tt":
        return f"\\texttt{{{esc(v)}}}"
    if spec == "url":
        return "\\url{" + str(v).replace("%", "\\%").replace("#", "\\#") + "}"
    if spec == "sha":
        return f"\\texttt{{{esc(str(v)[:12])}}}"
    if spec == "commit":
        return f"\\texttt{{{esc(str(v)[:7])}}}"
    if spec == "upper":
        return esc(str(v).upper())
    if spec:
        raise ValueError(f"unknown format {spec!r}")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.3f}"
    if isinstance(v, (list, dict)):
        return esc(json.dumps(v, ensure_ascii=False))
    return esc(v)


def lookup(spec: str, where: str) -> Any:
    kind, _, rest = spec.partition(":")
    ident, _, path = rest.partition(":")
    if kind == "rx":
        return regex_quote(ident, path, where)
    return resolve(document(kind, ident), path, f"{where} <<{spec}>>")


# --------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------
METRIC_TEX = {"macro_f1": "macro-F1", "micro_f1": "micro-F1", "auc": "AUC", "accuracy": "accuracy", "ndcg@10": "nDCG@10",
              "kendall_tau_b": "Kendall $\\tau_b$", "balanced_accuracy": "balanced accuracy", "spearman_rho": "Spearman $\\rho$",
              "pearson_mean_groups": "mean Pearson $r$ over source groups", "all_macro_f1": "macro-F1 (ID+OOD)",
              "qwk_total": "QWK (total)", "p@1": "P@1", "f1": "micro-F1", "single": "Single", "p_at_10": "P@10",
              "weighted_f1": "weighted F1", "ndcg@5": "nDCG@5"}


def row_label(text: str, ncols: int, limit: int = 34) -> str:
    """Long first-column labels wrap in a parbox sized so the table fits one column."""
    plain = re.sub(r"\\[a-zA-Z]+|[{}$]", "", text)
    if len(plain) <= limit:
        return text
    return f"\\parbox[t]{{\\dimexpr\\columnwidth-1.3cm*{ncols}-2\\tabcolsep-0.7cm\\relax}}{{\\raggedright {text}\\strut}}"


def collapse_sources(sources: list[str]) -> str:
    """Join sources as sentences; when short ones share a long common prefix (same table, different columns), print
    it once and list the differing tails."""
    if len(sources) < 2:
        return ". ".join(esc(s) for s in sources)
    prefix = sources[0]
    for s in sources[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix[:prefix.rfind(" ") + 1] if " " in prefix else ""
    if len(prefix) < 25 or any(". " in s for s in sources):
        return ". ".join(esc(s) for s in sources)
    return esc(prefix.rstrip()) + " " + "; ".join(esc(s[len(prefix):]) for s in sources)


def papernums(arg: str, where: str) -> str:
    body, _, opts = arg.partition("|")
    field, _, flag = opts.partition("|")
    field = field or "paper"
    if flag not in ("", "src1"):
        raise ValueError(f"{where}: unknown papernums flag {flag!r}")
    cols = [c.split("=", 1) for c in body.split(";")]
    paper = load_json("results/paper_numbers.json")
    methods: list[str] = []
    for ident, _ in cols:
        tab = resolve(paper[ident], field, f"{where} papernums {ident}")
        if not isinstance(tab, dict):
            raise ValueError(f"{where}: papernums {ident}.{field} is not a dict")
        methods += [m for m in tab if m not in methods]
    lines = ["\\begin{center}\\scriptsize", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Reported in the paper & " + " & ".join(label for _, label in cols) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for ident, _ in cols:
            v = resolve(paper[ident], field, where).get(m)
            cells.append("--" if v is None else fmt(v, "pct") if isinstance(v, (int, float)) else esc(v))
        name = esc(m)
        if field == "paper" and any(paper[i].get("method") == m for i, _ in cols):
            name = f"\\textbf{{{name}}}"
        lines.append(f"{{}}{row_label(name, len(cols))} & " + " & ".join(cells) + " \\\\")  # {} stops \\[ as \\[<len>]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    metric = {paper[i]["metric"] for i, _ in cols}
    sources = []
    for ident, _ in cols[:1] if flag == "src1" else cols:
        # The subsection header already cites the paper: drop a leading "Author et al. YEAR, ".
        s = re.sub(r"^[^,;()]*?\bet al\.?,?\s*\d{4}[a-z]?,\s*", "", paper[ident]["source"]).rstrip(". ")
        if s not in sources:
            sources.append(s)
    what = (f"Metric: {', '.join(METRIC_TEX.get(x, esc(x)) for x in sorted(metric))}" if field == "paper" else
            f"Values: {esc(field.split('.')[0].removeprefix('paper_').replace('_', ' '))}"
            + (f" ({esc(field.split('.', 1)[1])})" if "." in field else "") + " as reported")
    note = (what + " ($\\times$100). "
            + ("Bold: the paper's proposed method. " if field == "paper" else "")
            + "Source: " + collapse_sources(sources) + ".")
    return "\n".join(lines) + f"\n{{\\scriptsize {note}\\par}}\n"


def metrics(arg: str, where: str) -> str:
    rows = [ln.strip() for ln in arg.strip().splitlines() if ln.strip()]
    if not rows or not rows[0].startswith("cols:"):
        raise ValueError(f"{where}: metrics block must start with 'cols:'")
    cols = [c.strip().split("=", 1) for c in rows[0][5:].split(";")]
    lines = ["\\begin{center}\\scriptsize", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Jev & " + " & ".join(esc(label) for _, label in cols) + " \\\\", "\\midrule"]
    for row in rows[1:]:
        if row == "---":
            lines.append("\\midrule")
            continue
        path, _, rest = row.partition(" = ")
        label, _, spec = rest.rpartition(" | ") if " | " in rest else (rest, "", "")
        optional = path.startswith("?")
        path = path.lstrip("?").strip()
        cells = []
        for ident, _ in cols:
            try:
                v = resolve(load_json(f"runs/{ident}/scores.json"), path, f"{where} metrics {ident}")
            except KeyError:
                if not optional:
                    raise
                v = None
            cells.append(fmt(v, spec.strip()))
        lines.append(f"{{}}{row_label(label.strip(), len(cols))} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(lines) + "\n"


BLOCKS: dict[str, Callable[[], str]] = {}


def register(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        BLOCKS[name] = fn
        return fn
    return deco


def _load_block_modules() -> None:
    for mod in ("paper_appendix_opt",):
        if (Path(__file__).parent / f"{mod}.py").exists():
            __import__(mod)


# --------------------------------------------------------------------------------------------
# Template rendering
# --------------------------------------------------------------------------------------------
BLOCK_RE = re.compile(r"<<metrics\n(.*?)\n\s*>>", re.S)
INLINE_RE = re.compile(r"<<([a-z]+):([^<>\n]*?)>>")


def render(text: str, where: str) -> str:
    text = BLOCK_RE.sub(lambda m: metrics(m.group(1), where), text)

    def inline(m: re.Match) -> str:
        kind, arg = m.group(1), m.group(2)
        if kind == "papernums":
            return papernums(arg, where)
        if kind == "block":
            _load_block_modules()
            if arg not in BLOCKS:
                raise KeyError(f"{where}: unknown block {arg!r} (have {sorted(BLOCKS)})")
            return BLOCKS[arg]()
        spec, _, f = arg.rpartition("|") if "|" in arg else (arg, "", "")
        return fmt(lookup(f"{kind}:{spec}", where), f)

    out = INLINE_RE.sub(inline, text)
    if "<<" in out or ">>" in out:
        bad = re.findall(r"<<.{0,60}", out)
        raise ValueError(f"{where}: unrendered placeholder(s): {bad[:3]}")
    return out


def render_template(name: str) -> str:
    path = TEMPLATES / f"{name}.tex"
    return render(path.read_text(encoding="utf-8"), path.name)


# --------------------------------------------------------------------------------------------
# Papers, in selection order (scripts/appendix/selection.json): (prefix, short title, bib key, scored dataset ids, tier)
# Tiers set the implementation order only.
# --------------------------------------------------------------------------------------------
APP_PAPERS: list[tuple[str, str, str, list[str], str]] = [
    ("lifetox", "LifeTox", "kim-etal-2024-lifetox", ["lifetox"], "A"),
    ("pinose", "PiNose", "zhang-etal-2024-transferable", ["pinose/true_false", "pinose/nq", "pinose/triviaqa", "pinose/webq"], "A"),
    ("fenice", "FENICE", "scire-etal-2024-fenice", ["fenice"], "B"),
    ("factpico", "FactPICO", "joseph-etal-2024-factpico", ["factpico"], "B"),
    ("cicc", "Conformal Intent Classification and Clarification (CICC)", "hengst-etal-2024-conformal",
     ["cicc/mtod", "cicc/hwu64", "cicc/c150", "cicc/acid", "cicc/atis", "cicc/b77", "cicc/c150_oos", "cicc/b77_oos"], "C"),
    ("edit_intent", "Edit intent classification (Re3-Sci)", "ruan-etal-2024-large", ["edit_intent"], "A"),
    ("ezstance", "EZ-STANCE", "zhao-caragea-2024-ez", ["ezstance/subtaskA", "ezstance/subtaskB"], "A"),
    ("contempo", "ConTempo (MATRES, TB-Dense)", "niu-etal-2024-contempo", ["contempo/matres", "contempo/tbdense"], "A"),
    ("mismatched", "MISMATCHED", "shaik-etal-2025-mismatched", ["mismatched"], "B"),
    ("inli", "INLI", "havaldar-etal-2025-entailed", ["inli"], "B"),
    ("relemb", "Entity or Relation Embeddings (Wiki-WD, NYT-FB)", "mtumbuka-schockaert-2024-entity", ["relemb/wikiwd", "relemb/nytfb"], "B"),
    ("prp_graph", "PRP-Graph", "luo-etal-2024-prp", ["prp_graph/trec_covid"], "A"),
    ("scalar_constructs", "Measuring Scalar Constructs", "licht-etal-2025-measuring",
     [f"scalar_constructs/{c}/{m}" for c in ("immigration_fear", "ad_negativity", "grandstanding") for m in ("pairwise", "pointwise")], "B"),
    ("dress_new", "DREsS", "yoo-etal-2025-dress", ["dress_new"], "A"),
    ("medreadme", "MedReadMe", "jiang-xu-2024-medreadme", ["medreadme"], "B"),
    ("expl_rank", "Explanation quality ranking", "bailleux-etal-2026-explanation", ["expl_rank"], "B"),
    ("made", "MADE", "agarwal-etal-2026-made", ["made"], "C"),
    ("ttp", "Security attack pattern (TTP) recognition", "nguyen-etal-2024-noise",
     ["ttp/tram", "ttp/procedures_plus", "ttp/expert", "ttp/expert_full"], "C"),
    ("legalseg", "LegalSeg", "nigam-etal-2025-legalseg", ["legalseg"], "C"),
    ("sciprompt", "SciPrompt", "you-etal-2024-sciprompt", ["sciprompt/arxiv", "sciprompt/s2orc", "sciprompt/sdpra", "sciprompt/emerging_nlp"], "C"),
    ("nav_unknown", "Navigating the Unknown (CLINC150, BANKING77)", "sali-toraman-2025-navigating", ["nav_unknown/clinc150", "nav_unknown/banking77"], "C"),
    ("smarter", "SMARTER", "nghiem-etal-2026-smarter", ["smarter/hatexplain", "smarter/latent_hate", "smarter/implicit_hate"], "B"),
    ("biorc800", "BIORC800", "lan-etal-2024-multi", ["biorc800"], "B"),
    ("xstest", "XSTest (response judge)", "rottger-etal-2024-xstest", ["xstest"], "B"),
]
TIER_NAME = {"A": "A (harness pilot)", "B": "B (core study)", "C": "C (adversarial or engineering-heavy)"}


def md(s: str) -> str:
    """Markdown cell -> LaTeX: **bold**, `code`, [text](url) (text only); everything else escaped."""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)", s):
        out.append(esc(s[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f"\\textbf{{{md(m.group(1))}}}")
        elif m.group(2) is not None:
            out.append(f"\\texttt{{{esc(m.group(2))}}}")
        else:
            out.append(esc(m.group(3)))
        pos = m.end()
    out.append(esc(s[pos:]))
    return "".join(out)


# Run directories of a paper dropped from the study after selection; its runs are not reported.
DROPPED_RUNS = ("delta_stance",)
SELECTION = TEMPLATES / "selection.json"


@cache
def selection() -> dict[str, Any]:
    """Selection facts (scripts/appendix/selection.json): corpus counts, criteria, rubric, core table, near-misses,
    language exclusions."""
    return json.loads(SELECTION.read_text(encoding="utf-8"))


TABLE_NAME = {"cicc": "CICC", "edit_intent": "Edit intent", "contempo": "ConTempo", "relemb": "Entity or Relation Emb.",
              "scalar_constructs": "Scalar Constructs", "expl_rank": "Explanation ranking", "ttp": "Security TTP matching",
              "nav_unknown": "Navigating the Unknown", "xstest": "XSTest"}


def first_sentence(text: str, limit: int = 200) -> str:
    """First sentence of a ledger entry (markdown kept), clipped at a clause boundary if still long."""
    m = re.search(r"^(.+?[.;])(?:\s+[A-Z(*]|\s*$)", text)
    out = m.group(1) if m else text
    if len(out) > limit:
        cut = max(out.rfind(",", 0, limit), out.rfind(";", 0, limit), out.rfind(":", 0, limit))
        out = (out[:cut] if cut > 60 else out[:limit].rsplit(" ", 1)[0]) + " \u2026"
    return out

def selection_tex() -> str:
    s = selection()
    if len(s["core"]) != len(APP_PAPERS):
        raise ValueError(f"selection core has {len(s['core'])} rows, APP_PAPERS {len(APP_PAPERS)}")
    crit = "\\begin{description}\\setlength{\\itemsep}{0pt}\n" + "".join(
        f"\\item[{code} {esc(name)}.] {md(desc)}\n" for code, name, desc in s["criteria"]) + "\\end{description}\n"
    rub = ", ".join(f"\\textbf{{{k}}}~{esc(v)}" for k, v in s["rubric"])
    years = " & ".join(f"{y}" for y, _ in s["years"]) + " & Total \\\\\n\\midrule\n" + " & ".join(
        f"{c:,}" for _, c in s["years"]) + f" & {sum(c for _, c in s['years']):,} \\\\\n"
    corpus = ("\\begin{center}\\footnotesize\n\\begin{tabular}{" + "r" * (len(s["years"]) + 1) + "}\n\\toprule\n" + years
              + "\\bottomrule\n\\end{tabular}\n\\end{center}\n")
    rows = []
    for c, (prefix, short, key, _, tier) in zip(s["core"], APP_PAPERS):
        dagger = "$^\\dagger$" if c["stress"] else ""
        name = TABLE_NAME.get(prefix, short)
        rows.append(f"{c['no']}{dagger} & {name}~\\citep{{{key}}} & {esc(c['venue'])} & {tier} & "
                    + "/".join(str(x) for x in c["scores"]) + f" & {c['base']} & \\S\\ref{{app:{prefix}}} \\\\")
    core = ("\\begin{table*}[!tp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
            "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"
            "\\begin{tabular}{rllccrl}\n\\toprule\n"
            "\\# & Paper & Venue & Tier & A/B/C/D/E/F & Base & Details \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}}\n"
            f"\\caption{{The {len(s['core'])} preregistered papers in the order of the selection list. "
            f"$^\\dagger$~prespecified hard case ({sum(c['stress'] for c in s['core'])} of {len(s['core'])}). "
            "Screening scores: A centrality, B reproducibility, C fit to a bounded decision, D stress-test value, "
            "E task-family distinctiveness (shown, not summed), F practical feasibility. Base = A+B+C+D+F. "
            "Tiers set implementation order only.}\n\\label{tab:selection}\n\\end{table*}\n")
    near_rows = [f"{esc(n['name'])} & {esc(n['venue'])} & {md(first_sentence(n['status']))} & "
                 f"{n['base'] if n['base'] is not None else '--'} \\\\" for n in s["near"]]
    near = ("\\begin{table*}[!p]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{p{3.4cm}p{2.6cm}p{8.4cm}r}\n\\toprule\n"
            "Candidate & Venue & Status and main screen-out reason & Base \\\\\n\\midrule\n" + "\n".join(near_rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{Additional English candidates and near-misses considered before the core set was fixed "
            f"({len(s['near'])} records, first sentence of each screening-ledger entry). "
            "``Eligible'' = could meet E1--E5 on the stated English slice. Losses are on redundancy, scope, release "
            "or task fit, never on predicted model performance.}\n\\label{tab:nearmiss}\n\\end{table*}\n")
    langex = "\\begin{description}\\setlength{\\itemsep}{0pt}\n" + "".join(
        f"\\item[{esc(k)}.] {md(v)}\n" for k, v in s["langex"]) + "\\end{description}\n"
    return ("% Generated by scripts/paper_appendix.py from scripts/appendix/selection.json. Do not edit.\n"
            f"\\newcommand{{\\appCorpusYears}}{{{corpus}}}\n\\newcommand{{\\appCriteria}}{{{crit}}}\n"
            f"\\newcommand{{\\appRubric}}{{{rub}}}\n\\newcommand{{\\appLangEx}}{{{langex}}}\n" + core + near)


def spec_macros() -> list[str]:
    s = selection()
    return [f"\\newcommand{{\\specRecords}}{{{s['records']:,}}}", f"\\newcommand{{\\specInScope}}{{{s['in_scope']:,}}}",
            f"\\newcommand{{\\specTaskHits}}{{{s['task_hits']:,}}}", f"\\newcommand{{\\specResourceHits}}{{{s['resource_hits']:,}}}",
            f"\\newcommand{{\\specExportSha}}{{\\texttt{{{s['export_sha'][:16]}}}}}",
            f"\\newcommand{{\\specFrom}}{{{s['dates'][0]}}}", f"\\newcommand{{\\specCutoff}}{{{s['dates'][1]}}}",
            f"\\newcommand{{\\specNearMiss}}{{{len(s['near'])}}}",
            f"\\newcommand{{\\specStress}}{{{sum(c['stress'] for c in s['core'])}}}",
            f"\\newcommand{{\\specCore}}{{{len(s['core'])}}}"]


# --------------------------------------------------------------------------------------------
# Task catalogue: scripts/appendix/catalogue/<prefix>.tex
#
#   @task            plain-language description (LaTeX + placeholders)
#   @output          output space / labels (placeholders into prompts/*.json preferred)
#   @example <provenance>   rows "Field & value" copied verbatim from data files via placeholders
#   @sets            lines "dataset/id = Label" or "dataset/id = Label | unit of N"
#
# Each paper is one row of a longtable, titled with its name in Table 1. Its test sets are named as the results tables
# name them, without the paper's name in front (paper_tables.PAPER_LABEL, SET_LABEL); a set named only by the paper's
# name gets no name.
# Sets outside those tables (supplementary, dev or rebuilt sets) are left out unless the paper has no other set.
# --------------------------------------------------------------------------------------------
CATALOGUE = TEMPLATES / "catalogue"
CAT_METRIC = {**METRIC_TEX, "accuracy": "accuracy"}


def catalogue_entry(prefix: str) -> str:
    from paper_tables import PAPER_LABEL, SET_LABEL  # paper_tables imports this module

    key = next(k for p, _, k, _, _ in APP_PAPERS if p == prefix)
    path = CATALOGUE / f"{prefix}.tex"
    where = f"catalogue/{path.name}"
    blocks: dict[str, list[str]] = {}
    provenance = ""
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@"):
            current, _, arg = line[1:].partition(" ")
            if current not in ("task", "output", "example", "sets"):
                raise ValueError(f"{where}: unknown block @{current}")
            if current == "example":
                provenance = arg.strip()
            blocks[current] = []
        elif current is not None and not line.lstrip().startswith("%"):
            blocks[current].append(line)
    missing = {"task", "output", "example", "sets"} - set(blocks)
    if missing or not provenance:
        raise ValueError(f"{where}: missing blocks {sorted(missing)} or @example provenance")
    text = {k: render("\n".join(v).strip(), where) for k, v in blocks.items() if k != "sets"}
    provenance = render(provenance, where)
    paper = load_json("results/paper_numbers.json")
    sets = []
    lines = [ln.strip() for ln in blocks["sets"] if ln.strip()]
    compared = [ln for ln in lines if ln.partition(" = ")[0].strip() in SET_LABEL]
    for line in compared or lines:
        ident, _, rest = line.partition(" = ")
        # "Label", "Label | unit of N" or "Label | unit | metric label override" (the unit may be empty)
        label, unit, metric_label = ([part.strip() for part in rest.split("|")] + ["", ""])[:3]
        title = PAPER_LABEL[key]
        label = SET_LABEL.get(ident.strip(), label)
        label = label[len(title):].lstrip(" :") if label.startswith(title) else label
        label = "" if label.startswith("(") and label.endswith(")") else label  # e.g. "XSTest (judge)"
        n = load_json(f"runs/{ident.strip()}/scores.json")["n"]
        metric = metric_label.strip() or CAT_METRIC.get(paper[ident.strip()]["metric"], esc(paper[ident.strip()]["metric"]))
        unit = unit.strip()
        sets.append(f"{label + ': ' if label else ''}$N={n:,}$".replace(",", "{,}") + f"{(' ' + unit) if unit else ''}, {metric}")
    rows = [r.strip() for r in text["example"].split("\\\\") if r.strip()] if "\\\\" in text["example"] else \
        [r.strip() for r in text["example"].splitlines() if r.strip()]
    example = "\n".join(f"{r.split(' & ', 1)[0]} & {r.split(' & ', 1)[1]} \\\\" for r in rows)  # the label column is italic
    source = provenance[0].upper() + provenance[1:] + ("" if provenance.endswith(".") else ".")
    return (f"\\textbf{{{PAPER_LABEL[key]}}} \\citep{{{key}}}\\par\n"
            f"\\textbf{{Task.}} {text['task']}\\par\n\\textbf{{Output.}} {text['output']}\\par\n"
            f"\\textbf{{Test {'set' if len(sets) == 1 else 'sets'}.}} {'; '.join(sets)}.\n"
            f"& \\textbf{{Source.}} {source}\\par\n"
            "\\begin{tabular}[t]{@{}>{\\itshape}p{1.45cm}>{\\raggedright\\arraybackslash}p{\\dimexpr\\linewidth-1.45cm-2\\tabcolsep\\relax}@{}}\n"
            + example + "\n\\end{tabular} \\\\")


def catalogue_tex() -> str:
    """One longtable (one-column pages): a row per paper in alphabetical order, so rows run on across pages without
    gaps and a name from the paper is quick to find."""
    from paper_tables import PAPER_LABEL

    missing = [p for p, *_ in APP_PAPERS if not (CATALOGUE / f"{p}.tex").exists()]
    if missing:
        print("WARNING: missing catalogue entries:", ", ".join(missing))
    head = "\\textbf{Paper and data} & \\multicolumn{1}{c}{\\textbf{Example}} \\\\\n\\midrule\n"
    order = sorted((p for p, *_ in APP_PAPERS if p not in missing),
                   key=lambda p: PAPER_LABEL[next(k for q, _, k, _, _ in APP_PAPERS if q == p)].lower())
    rows = "\n\\midrule\n".join(catalogue_entry(p) for p in order)
    return ("% Generated by scripts/paper_appendix.py from scripts/appendix/catalogue/*.tex. Do not edit.\n"
            "{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\setlength{\\aboverulesep}{0.2ex}\\setlength{\\belowrulesep}{0.3ex}\n"
            "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{7.6cm}"
            ">{\\raggedright\\arraybackslash}p{\\dimexpr\\textwidth-7.6cm-2\\tabcolsep\\relax}@{}}\n"
            "\\caption{The data of each paper, in alphabetical order of the names in Table~\\ref{tab:tasks}.}\\label{tab:catalogue}\\\\\n"
            "\\toprule\n" + head + "\\endfirsthead\n"
            "\\multicolumn{2}{@{}l}{\\textit{Table~\\ref{tab:catalogue}, continued}} \\\\\n\\toprule\n" + head
            + "\\endhead\n\\bottomrule\n\\endlastfoot\n" + rows + "\n\\end{longtable}}\n")



# --------------------------------------------------------------------------------------------
# B. Per-paper sections
# --------------------------------------------------------------------------------------------
def tasks_tex() -> str:
    s = {c["no"]: c for c in selection()["core"]}
    parts = ["% Generated by scripts/paper_appendix.py from scripts/appendix/<prefix>.tex templates. Do not edit.\n"
             "\\begingroup\\small"]
    missing = []
    for i, (prefix, short, key, datasets, tier) in enumerate(APP_PAPERS, start=1):
        c, paper = s[i], load_json(f"data/manifests/{prefix}.json")["paper"]
        head = (f"\\subsection{{{short}}}\\label{{app:{prefix}}}\n"
                f"{{\\footnotesize\\noindent\\textit{{{esc(paper['title'])}}}~\\citep{{{key}}}, {esc(c['venue'])}. "
                f"Selection list \\#{c['no']}, tier {TIER_NAME[tier]}"
                + (", prespecified hard case" if c["stress"] else "")
                + f". Screening score {'/'.join(map(str, c['scores']))} (base {c['base']}).\\par}}\n")
        tpl = TEMPLATES / f"{prefix}.tex"
        if tpl.exists():
            parts.append(head + render_template(prefix))
        else:
            missing.append(prefix)
            parts.append(head)
    if missing:
        print("WARNING: missing appendix templates:", ", ".join(missing))
    return "\n".join(parts) + "\n\\endgroup\n"


# --------------------------------------------------------------------------------------------
# C. Secondary metrics and supplementary test sets
# --------------------------------------------------------------------------------------------
_ACC = [("accuracy", "acc.", "pct")]
SECONDARY: dict[str, list[tuple[str, str, str]]] = {
    **{f"pinose/{d}": [("accuracy", "acc.\\ at 0.5", "pct")] for d in ("true_false", "nq", "triviaqa", "webq")},
    "prp_graph/trec_covid": [("ndcg@10", "nDCG@10 without BM25 interpolation", "pct"), ("bm25_ndcg@10", "BM25 nDCG@10", "pct")],
    "xstest": [("macro_f1", "macro-F1", "pct"), ("cohen_kappa", "Cohen $\\kappa$", "f3")],
    "factpico": [("spearman_rho", "Spearman $\\rho$", "pct"), ("acc_eq", "exact agreement", "pct")],
    "fenice": [("roc_auc", "ROC-AUC", "pct"), ("balanced_accuracy_at_0.5", "bal.\\ acc.\\ at 0.5", "pct")],
    "biorc800": [("macro_f1", "macro-F1", "pct"), ("exact_match", "exact match", "pct")],
    **{f"sciprompt/{d}": [("macro_f1", "macro-F1", "pct")] for d in ("arxiv", "s2orc", "sdpra", "emerging_nlp")},
    **{f"scalar_constructs/{c}/{m}": [("accuracy", "pairwise acc.", "pct"), ("rmse", "RMSE", "f3")]
       for c in ("immigration_fear", "ad_negativity", "grandstanding") for m in ("pairwise", "pointwise")},
    "lifetox": _ACC + [("roc_auc_unsafe", "ROC-AUC", "pct")],
    "edit_intent": _ACC + [("by_action.modify.macro_f1", "modify macro-F1", "pct"),
                           ("by_action.add_delete.macro_f1", "add/delete macro-F1", "pct")],
    "ezstance/subtaskA": [("N.macro_f1", "noun-phrase targets", "pct"), ("C.macro_f1", "claim targets", "pct")],
    "ezstance/subtaskB": [("macro_f1_mean_over_domains_N", "noun-phrase targets", "pct"),
                          ("macro_f1_mean_over_domains_C", "claim targets", "pct")],
    "contempo/matres": _ACC + [("macro_f1_all_labels", "macro-F1 (all labels)", "pct")],
    "contempo/tbdense": _ACC + [("macro_f1_all_labels", "macro-F1 (all labels)", "pct")],
    "mismatched": _ACC,
    "inli": [("implied_entailment_accuracy", "implied-entailment acc.", "pct"), ("macro_f1", "macro-F1", "pct"),
             ("accuracy_3way", "3-way acc.", "pct")],
    "medreadme": [("spearman_mean_groups", "mean Spearman", "pct"), ("pearson_pooled", "pooled Pearson", "pct")],
    **{f"nav_unknown/{d}": [("id_macro_f1", "ID macro-F1", "pct"), ("ood_f1", "OOD F1", "pct"), ("ood_recall", "OOD recall", "pct")]
       for d in ("clinc150", "banking77")},
    "made": [("jaccard", "Jaccard", "pct"), ("prr", "PRR", "pct"), ("ece_plus", "ECE+", "f3")],
    **{f"smarter/{d}": _ACC for d in ("hatexplain", "latent_hate", "implicit_hate")},
    "dress_new": [("qwk_content", "QWK content", "pct"), ("qwk_organization", "QWK organization", "pct"),
                  ("qwk_language", "QWK language", "pct")],
    **{f"ttp/{d}": [("r@1", "R@1", "pct"), ("f1@3", "F1@3", "pct"), ("mrr@5", "MRR@5", "pct")]
       for d in ("tram", "procedures_plus", "expert", "expert_full")},
    "legalseg": _ACC + [("weighted_f1", "weighted F1", "pct"), ("mcc", "MCC", "f3")],
    "relemb/wikiwd": [("precision", "P", "pct"), ("recall", "R", "pct"), ("macro_f1", "macro-F1", "pct")],
    "relemb/nytfb": [("p_at_30", "P@30", "pct")],
    **{f"cicc/{d}": [("accuracy", "top-1 acc.", "pct"), ("marginal_coverage", "coverage", "pct"),
                     ("mean_set_size", "mean set size", "f2")] for d in ("mtod", "hwu64", "c150", "acid", "atis", "b77")},
}


def secondary_tex(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        s = load_json(f"runs/{r['task']}/scores.json")
        sec = "; ".join(f"{label} {fmt(resolve(s, path, r['task']), f)}" for path, label, f in SECONDARY[r["task"]])
        trunc = s.get("n_truncated")
        lines.append(f"{r['name']} & {r['metric']} {fmt(r['jev'], 'pct')} & {sec} & {fmt(trunc) if trunc is not None else '--'} \\\\")
    return ("% Generated by scripts/paper_appendix.py. Do not edit.\n"
            "\\begin{table*}[!tp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{llp{8.4cm}r}\n\\toprule\nDataset & Primary (Jev) & Secondary metrics (Jev, $\\times$100 unless "
            "marked) & Trunc. \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Secondary metrics for every scored test set, in the order of Table~\\ref{tab:full}. "
            "Primary as in Table~\\ref{tab:full} (paper protocol where marked there). "
            "Trunc.: inputs shortened by the frozen truncation rule (unit as defined per task in Appendix~\\ref{app:tasks}).}\n"
            "\\label{tab:secondary}\n\\end{table*}\n")


SUPPLEMENTARY = [
    # (dataset, display, metric path, metric label, why it is not in Table full)
    ("expl_rank", "Explanation ranking", "ndcg@5", "nDCG@5", None),
    ("cicc/b77", "CICC B77", "single", "Single", None),
    ("cicc/c150_oos", "CICC C150 OOS", "f1", "OOS F1", "CICC-OOS variant, secondary in the paper"),
    ("cicc/b77_oos", "CICC B77 OOS", "f1", "OOS F1", "CICC-OOS variant, secondary in the paper"),
    ("ttp/expert_full", "TTP Expert (full)", "p@1", "P@1", "the paper's full Expert set overlaps the Expert test split"),
    ("biorc800/dev", "BIORC800 dev", "micro_f1", "micro-F1 at 0.5", "development split used only to fit thresholds"),
]


def supplementary_tex(not_comparable: dict[str, str]) -> str:
    paper = load_json("results/paper_numbers.json")
    lines = []
    for task, name, path, label, why in SUPPLEMENTARY:
        s = load_json(f"runs/{task}/scores.json")
        ref = paper.get(task, {})
        pv = ref.get("paper", {}).get(ref.get("method")) if ref else None
        reason = esc(not_comparable[task]) if task in not_comparable else esc(why)
        lines.append(f"{name} & {fmt(s['n'])} & {label} & {fmt(resolve(s, path, task), 'pct')} & {fmt(pv, 'pct')} & {reason} \\\\")
    return ("% Generated by scripts/paper_appendix.py. Do not edit.\n"
            "\\begin{table*}[!tp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lrlrrp{7.4cm}}\n\\toprule\nTest set & $N$ & Metric & Jev & Paper & Why not in Table~\\ref{tab:full} \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Supplementary runs and test sets excluded from the comparison. Paper: the paper's proposed method on "
            "the corresponding set where one is reported (--: none or not comparable).}\n\\label{tab:supp}\n\\end{table*}\n")


# --------------------------------------------------------------------------------------------
# D. Prompts
# --------------------------------------------------------------------------------------------
def trim(obj: Any, max_str: int = 360, max_items: int = 6) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + f" [... {len(obj) - max_str:,} more chars]"
    if isinstance(obj, dict):
        keys = list(obj)
        out = {k: trim(obj[k], max_str, max_items) for k in keys[:max_items]}
        if len(keys) > max_items:
            out["..."] = f"{len(keys) - max_items:,} more entries"
        return out
    if isinstance(obj, list):
        out = [trim(x, max_str, max_items) for x in obj[:max_items]]
        if len(obj) > max_items:
            out.append(f"... {len(obj) - max_items:,} more items")
        return out
    return obj


def wrap_json(obj: Any, width: int = 104) -> str:
    lines = []
    for line in ascii_only(json.dumps(obj, indent=1, ensure_ascii=False)).splitlines():
        indent = len(line) - len(line.lstrip(" ")) + 2
        while len(line) > width:
            cut = line.rfind(" ", indent + 20, width)
            cut = cut if cut > 0 else width
            lines.append(line[:cut])
            line = " " * indent + line[cut:].lstrip(" ")
        lines.append(line)
    return "\n".join(lines)


# (prompt file, keys to show (None = all non-underscore keys), description). Descriptions state structure only.
PROMPT_SHOWCASE = [
    ("lifetox", "LifeTox", None, "a single \\textsc{choice} question over the paper's labels. The instruction is the "
     "paper's own zero-shot LLM prompt and the criteria quote its label definitions."),
    ("edit_intent", "Edit intent", ["modify", "add_delete"], "\\textsc{choice} questions routed by the edit action "
     "(fixed before any output): Modify edits see all intents, Add/Delete edits only those that can apply."),
    ("dress_new", "DREsS", ["levels", "questions"], "one \\textsc{score} question per rubric (content, organization, "
     "language) over the rubric's half-point levels, sharing one essay state."),
    ("prp_graph_trec_covid", "PRP-Graph", None, "the paper's pairwise prompt as a two-option \\textsc{choice}. Each "
     "Swiss-system pairing is asked in both orders."),
    ("ttp", "Security TTP", None, "one \\textsc{noul} question per ATT\\&CK candidate profile. Each text's ranking "
     "sorts all candidates by $P(\\text{yes})$. Only the template is shown. The candidate profiles are frozen separately."),
    ("relemb_wikiwd_v2", "Wiki-WD (v2)", ["question", "none_option", "halves", "labels"], "relation names only (the paper's classifier sees no label text). The "
     "inventory exceeds the option cap, so it is split into fixed halves, each with a ``none'' option, and a final "
     "round between the half-winners. Label lists truncated."),
]


def prompts_tex() -> str:
    """Prompt files as inline column-width listings (no floats, so they flow without float pages)."""
    out = ["% Generated by scripts/paper_appendix.py from prompts/*.json. Do not edit.\n"]
    for name, title, keys, description in PROMPT_SHOWCASE:
        p = load_json(f"prompts/{name}.json")
        shown = {k: p[k] for k in (keys or [k for k in p if not k.startswith("_")])}
        meta = {k: p[k] for k in p if k.startswith("_") and k != "_source"}
        body = wrap_json(trim({**meta, **shown}, max_str=200, max_items=4), width=56)
        src = p.get("_source", "")
        src_short = esc(src) if len(src) <= 260 else esc(src[:260].rsplit(" ", 1)[0]) + " \\ldots"
        out.append(f"\\paragraph{{{title}}}\\label{{app:prompt-{name.replace('_', '-')}}} "
                   f"(\\texttt{{prompts/{esc(name)}.json}}): {description} "
                   f"{{\\footnotesize Source note (abridged): {src_short}}}\n"
                   "{\\scriptsize\n\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n}\n")
    return "\n".join(out)


def example_request_tex() -> str:
    """A verbatim LifeTox request/response pair, reconstructed from the frozen item and the prompt file and
    checked byte-for-byte against the request hash stored with the saved raw response."""
    item_id = "lifetox-test-0001"
    row = next(r for r in load_jsonl("data/frozen/lifetox/test.jsonl") if r["id"] == item_id)
    resp = next(r for r in load_jsonl("runs/lifetox/responses.jsonl.gz") if r["item_id"] == item_id)
    model = re.search(r'^MODEL = "([^"]+)"', (ROOT / "src/gauntlet/jev.py").read_text(), re.M).group(1)
    body = {"model": model, "state": {"question": row["query"], "answer": row["response"]},
            "questions": {resp["key"]: load_json("prompts/lifetox.json")["question"]}}
    import hashlib
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != resp["request_hash"]:
        raise ValueError("example request does not reproduce the stored request hash")
    shown = {"model": model, "state": body["state"], "questions": {resp["key"]: {"type": "choice", "...": "as in Figure~X"}}}
    req = wrap_json(trim(shown, max_str=300), width=50).replace('"as in Figure~X"', '"(prompts/lifetox.json question)"')
    ans = wrap_json({k: resp["response"][k] for k in ("model", "answers", "usage", "provider")}, width=50)
    return ("% Generated by scripts/paper_appendix.py. Do not edit.\n"
            "\\begin{figure*}[!tp]\n\\scriptsize\n\\begin{minipage}[t]{0.49\\textwidth}\n\\begin{verbatim}\n" + req
            + "\n\\end{verbatim}\n\\end{minipage}\\hfill\n\\begin{minipage}[t]{0.49\\textwidth}\n\\begin{verbatim}\n" + ans
            + "\n\\end{verbatim}\n\\end{minipage}\n"
            f"\\caption{{A complete LifeTox request (left, question body abbreviated, see "
            f"the LifeTox prompt in Appendix~\\ref{{app:prompts}}) and the stored raw response (right) for item \\texttt{{{esc(item_id)}}}. "
            f"The request was rebuilt from the frozen test file and the prompt file and reproduces the SHA-256 request hash "
            f"stored with the response (\\texttt{{{digest[:16]}}}), which is also the key of the on-disk cache.}}\n"
            "\\label{fig:example-request}\n\\end{figure*}\n")


# --------------------------------------------------------------------------------------------
# E. Cost and volume; G. Reproducibility
# --------------------------------------------------------------------------------------------
def is_audit_run(rel: str) -> bool:
    """True for a zero-shot Jev audit run directory (runs/<rel>). The optimization study (jev-opt), smoke tests,
    its development runs (dev_v3), its large-label-space test runs (<task>_v3) and the fitted readouts on the
    audit's answers (<task>+head, which reuse the audit run's responses) are reported separately; runs of
    papers dropped from the study (DROPPED_RUNS) are not reported. Clef and Clef-flash runs (runs/clef*/,
    local compute, no API cost) are reported by scripts/paper_clef.py, GPT-6 Luna runs (runs/luna/) by
    scripts/analyze_luna.py."""
    top = rel.split("/")[0]
    return (top not in ("jev-opt", "_smoke", "dev_v3", "clef", "clef-flash", "luna", "msd1", *DROPPED_RUNS) and not top.endswith("_v3")
            and not rel.endswith("+head"))


def run_manifests() -> list[tuple[str, dict]]:
    """(run path relative to runs/, manifest) for every zero-shot audit run."""
    out = []
    for p in sorted((ROOT / "runs").glob("**/manifest.json")):
        rel = p.parent.relative_to(ROOT / "runs").as_posix()
        if is_audit_run(rel):
            out.append((rel, json.loads(p.read_text())))
    return out


def cost_tex() -> str:
    ms = run_manifests()
    rows = [f"\\texttt{{{esc(t)}}} & {m['n_items']:,} & {m['n_requests']:,} & {m['input_tokens'] / 1e6:,.1f} & {m['cost_usd']:,.2f} \\\\"
            for t, m in ms]
    total = (f"\\midrule\nTotal & {sum(m['n_items'] for _, m in ms):,} & {sum(m['n_requests'] for _, m in ms):,} & "
             f"{sum(m['input_tokens'] for _, m in ms) / 1e6:,.1f} & {sum(m['cost_usd'] for _, m in ms):,.2f} \\\\")
    half = (len(rows) + 1) // 2

    def tab(chunk: list[str], last: bool) -> str:
        return ("\\begin{tabular}[t]{lrrrr}\n\\toprule\nRun & Items & Requests & M tok. & \\$ \\\\\n\\midrule\n"
                + "\n".join(chunk) + ("\n" + total if last else "") + "\n\\bottomrule\n\\end{tabular}")
    return ("% Generated by scripts/paper_appendix.py from runs/*/manifest.json. Do not edit.\n"
            "\\begin{table*}[!tp]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
            + tab(rows[:half], False) + "\\hfill\n" + tab(rows[half:], True) + "\n"
            "\\caption{Volume and cost of every gauntlet run (test sets, the BIORC800 dev threshold fit and the "
            "supplementary runs of Table~\\ref{tab:supp}). Items: scored units as defined by the task (e.g.\\ calibration "
            "plus test utterances for CICC). Requests: Jev calls. M tok.: input tokens in millions. \\$: OpenRouter "
            "charge. Output tokens are not billed.}\n\\label{tab:cost}\n\\end{table*}\n")


def client_constants() -> dict[str, str]:
    src = (ROOT / "src" / "gauntlet" / "jev.py").read_text()

    def const(name: str) -> str:
        return re.search(rf"^\s+{name}: \w+ = ([\d_.]+)", src, re.M).group(1).replace("_", "")
    cap = re.search(r"caps [\w/.\-]+ at ([\d,]+) requests/minute", src).group(1)
    return {"rpm": const("requests_per_minute"), "concurrency": const("concurrency"), "retries": const("max_retries"),
            "timeout": const("timeout_s").rstrip("0").rstrip("."), "cap": cap,
            "model": re.search(r'^MODEL = "([^"]+)"', src, re.M).group(1)}


def repro_macros() -> list[str]:
    ms = run_manifests()
    snaps: dict[str, int] = {}
    for _, m in ms:
        for k, v in m["model_snapshots"].items():
            snaps[k] = snaps.get(k, 0) + v
    c = client_constants()
    tokens = sum(m["input_tokens"] for _, m in ms)
    cost = sum(m["cost_usd"] for _, m in ms)
    return [f"\\newcommand{{\\jevModel}}{{\\texttt{{{esc(c['model'])}}}}}",
            "\\newcommand{\\jevSnapshots}{" + ", ".join(f"\\texttt{{{esc(k)}}}" for k in snaps) + "}",
            f"\\newcommand{{\\nSnapshots}}{{{len(snaps)}}}",
            f"\\newcommand{{\\nRuns}}{{{len(ms)}}}",
            f"\\newcommand{{\\jevPricePerM}}{{\\${cost / tokens * 1e6:.3f}}}",
            f"\\newcommand{{\\jevRPM}}{{{int(c['rpm']):,}}}", f"\\newcommand{{\\jevRPMCap}}{{{c['cap']}}}",
            f"\\newcommand{{\\jevConcurrency}}{{{c['concurrency']}}}", f"\\newcommand{{\\jevMaxRetries}}{{{c['retries']}}}",
            f"\\newcommand{{\\jevTimeout}}{{{c['timeout']}}}"]


def row_has_text(path: Path, n_rows: int = 20) -> bool:
    """True if the first rows of a frozen JSONL file carry input text (not only ids, labels and hashes)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for _, line in zip(range(n_rows), f) if line.strip()]

    def texts(node: Any) -> list[str]:
        if isinstance(node, dict):
            return [t for k, v in node.items() if not re.search(r"sha|hash|id$|_id|label|gold|file", k) for t in texts(v)]
        if isinstance(node, list):
            return [t for v in node for t in texts(v)]
        return [node] if isinstance(node, str) else []
    return any(len(t) >= 12 and " " in t for t in texts(rows))


def freeze_tex() -> str:
    rows = []
    for prefix, short, _, _, _ in APP_PAPERS:
        d = ROOT / "data" / "frozen" / prefix
        files = sorted(p for p in d.rglob("*") if p.is_file())
        data = [p for p in files if p.name.endswith((".jsonl", ".jsonl.gz"))
                and not re.search(r"non_english|excluded|candidates|expert_train|expert_dev|/dev/", p.as_posix())]
        kinds = {row_has_text(p) for p in data}
        form = "inputs (JSONL)" if kinds == {True} else "IDs + row hashes" if kinds == {False} else "mixed"
        shards = len([p for p in (ROOT / "runs").glob(f"{prefix}/**/responses*.jsonl.gz") if is_audit_run(
            p.parent.relative_to(ROOT / "runs").as_posix())])
        rows.append(f"{TABLE_NAME.get(prefix, short.split(' (')[0])} & {form} & {len(files)} & "
                    f"\\texttt{{prepare\\_{esc(prefix)}.py}} & {shards} \\\\")
    return ("% Generated by scripts/paper_appendix.py from data/frozen/ and runs/. Do not edit.\n"
            "\\begin{table*}[!tp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{5pt}\n"
            "\\begin{tabular}{llrlr}\n\\toprule\nPaper & Test inputs & Files & Rebuild script & Resp.\\ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{What the frozen files hold per paper. Inputs: the frozen files hold the test inputs "
            "themselves. IDs + row hashes: they hold only identifiers and per-row SHA-256 hashes, "
            "and the rebuild script regenerates the inputs from the licensed or gated source and checks every hash. "
            "Mixed: some test files of the paper carry inputs, others only hashes. Files: all files under "
            "\\texttt{data/frozen/<paper>/} (tests, ID manifests, language screens, splits). "
            "Resp.: raw-response shards stored with the run.}\n"
            "\\label{tab:freeze}\n\\end{table*}\n")


def macros_tex() -> list[str]:
    """Appendix macros appended to generated/numbers.tex."""
    lines = spec_macros() + repro_macros()
    _load_block_modules()
    if "opt_macros" in BLOCKS:
        lines += [ln for ln in BLOCKS["opt_macros"]().splitlines() if ln.strip()]
    return lines


def write_all(rows: list[dict], not_comparable: dict[str, str]) -> None:
    outputs = {
        "app_catalogue.tex": catalogue_tex(),
        "app_selection.tex": selection_tex(),
        "app_tasks.tex": tasks_tex(),
        "app_secondary.tex": secondary_tex(rows),
        "app_supp.tex": supplementary_tex(not_comparable),
        "app_prompts.tex": prompts_tex(),
        "app_example.tex": example_request_tex(),
        "app_cost.tex": cost_tex(),
        "app_freeze.tex": freeze_tex(),
    }
    _load_block_modules()
    if "opt_settings" in BLOCKS:  # the settings table alone, for the compact appendix of the submission PDF
        (OUT / "app_opt_settings.tex").write_text("% Generated by scripts/paper_appendix.py (block opt_settings). Do not edit.\n"
                                                  + BLOCKS["opt_settings"](), encoding="utf-8")
    if (TEMPLATES / "opt.tex").exists():
        outputs["app_opt.tex"] = "% Generated by scripts/paper_appendix.py from scripts/appendix/opt.tex. Do not edit.\n" + render_template("opt")
    else:
        print("WARNING: missing scripts/appendix/opt.tex")
        outputs["app_opt.tex"] = "% scripts/appendix/opt.tex missing\n"
    for name, text in outputs.items():
        (OUT / name).write_text(text, encoding="utf-8")
    if UNKNOWN:
        print("WARNING: non-ASCII characters rendered as '?':", sorted(UNKNOWN))


# --------------------------------------------------------------------------------------------
# Standalone check for template authors
# --------------------------------------------------------------------------------------------
STUB = r"""\documentclass[11pt,twocolumn]{article}
\usepackage[a4paper,margin=2.5cm,columnsep=0.6cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{times}\usepackage{inconsolata}\usepackage{booktabs}
\usepackage{amsmath,amssymb}\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}\usepackage{graphicx}
\usepackage{hyperref}\usepackage{url}
\newcommand{\citep}[1]{[#1]}\newcommand{\citet}[1]{#1}\newcommand{\jev}{Jev}\newcommand{\jevopt}{\texttt{jev-opt}}
\newcommand{\ai}[1]{\textcolor{orange}{[#1]}}
\begin{document}
\section{Check}
{\small
%s
}
\end{document}
"""


def check(name: str) -> int:
    body = catalogue_entry(name[4:]) if name.startswith("cat:") else render_template(name)
    name = name.replace(":", "_")
    if UNKNOWN:
        print("WARNING: non-ASCII characters rendered as '?':", sorted(UNKNOWN))
    d = Path(tempfile.gettempdir()) / "appcheck" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "doc.tex").write_text(STUB % body, encoding="utf-8")
    res = subprocess.run(["tectonic", "-X", "compile", "doc.tex"], cwd=d, capture_output=True, text=True)
    log = res.stdout + res.stderr
    problems = [ln for ln in log.splitlines() if re.search(r"error|Overfull|Undefined", ln, re.I)]
    print("\n".join(problems) if problems else "compiled cleanly")
    if res.returncode == 0:
        pages = subprocess.run(["pdfinfo", str(d / "doc.pdf")], capture_output=True, text=True).stdout
        print(next((ln for ln in pages.splitlines() if ln.startswith("Pages")), ""), f"-> {d / 'doc.pdf'}")
    return res.returncode


if __name__ == "__main__":
    import paper_appendix  # one module instance, so blocks registered by companion modules are visible

    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(paper_appendix.check(sys.argv[2]))
    print(__doc__)
