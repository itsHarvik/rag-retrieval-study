"""Part 2, Day 3: evaluate the detectors in verification/scores.csv.
  Per condition: NLI 3-way accuracy, false-alarm rate, miss rate, and AUROC for NLI (nli_entail) and overlap.
  Splits: (a) evidence found vs missed, (b) contradicted vs unsupported, (c) human vs LLM supported answers.
  Every number gets a 95% bootstrap CI (10,000 resamples of questions; all items of a question stay together).
Saves results/verification/summary.csv and results/verification/false_alarms.png, and prints 10 error cases.
Run from the project folder after detect.py: python verification/run_verification.py"""
import csv
import json
import os
import random
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verification"))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from pipeline import bootstrap_ci  # Part 1: CI for a mean of per-question values
from detect import windows, pair_key, load_json, NLI_CACHE_PATH, CTX_PATH, SET_PATH, CONDITIONS

SCORES_PATH = Path("verification/scores.csv")
OUT_DIR = Path("results/verification")
N_RESAMPLES = 10000
SEED = 0

# ---------- load ----------

with open(SET_PATH, encoding="utf-8") as f:
    drafted_by = {i["id"]: i["drafted_by"] for i in json.load(f)}
with open(SCORES_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["overlap"], r["nli_entail"], r["nli_contra"] = float(r["overlap"]), float(r["nli_entail"]), float(r["nli_contra"])
    r["evidence_hit"] = {"True": True, "False": False, "": None}[r["evidence_hit"]]
    r["drafted_by"] = drafted_by[r["id"]]
    r["bad"] = r["label"] != "supported"            # contradicted or unsupported
    r["accepted"] = r["nli_pred"] == "supported"    # NLI lets the answer through


def select(condition=None, **filters):
    """Rows of one condition (or all), filtered by exact field values, e.g. label="supported"."""
    return [r for r in rows if (condition is None or r["condition"] == condition)
            and all(r[k] == v for k, v in filters.items())]


# ---------- measures ----------
# Each rate is "share of items where something is true". Every question has the same number of items in each
# subset (one answer per label), so the mean over items equals the mean of the per-question values.

def correct(r):
    return r["label"] == r["nli_pred"]


def false_alarm(r):  # used on supported answers: a good answer that NLI does not accept
    return not r["accepted"]


def missed(r):  # used on bad answers: a bad answer that NLI accepts
    return r["accepted"]


def per_question(subset, fn):
    """{question_id: share of that question's items where fn is true}."""
    groups = defaultdict(list)
    for r in subset:
        groups[r["question_id"]].append(fn(r))
    return {q: sum(v) / len(v) for q, v in groups.items()}


def auroc(pos, neg):
    """Chance that a random supported answer scores higher than a random not-supported one (ties count half).
    1.0 = the score separates them perfectly, 0.5 = no better than a coin flip."""
    # Same as comparing every (pos, neg) pair, but faster: sort neg once, then binary-search each pos score
    neg = sorted(neg)
    wins = sum(bisect_left(neg, p) + 0.5 * (bisect_right(neg, p) - bisect_left(neg, p)) for p in pos)
    return wins / (len(pos) * len(neg))


def auroc_groups(condition, score):
    """{question_id: (scores of supported answers, scores of bad answers)} for one condition and score column."""
    groups = defaultdict(lambda: ([], []))
    for r in select(condition):
        groups[r["question_id"]][1 if r["bad"] else 0].append(r[score])
    return dict(groups)


def pooled_auroc(groups):
    return auroc([s for g in groups for s in g[0]], [s for g in groups for s in g[1]])


# ---------- bootstrap ----------
# Same resampling as Part 1 (random.Random(0), 10,000 resamples of questions, 2.5% / 97.5% points),
# written for any statistic, because AUROC and unpaired differences are not a plain mean.

def bootstrap_stat(groups, stat_fn):
    """groups: {question_id: data}. Resample questions with replacement, compute stat_fn on the list of data."""
    rng = random.Random(SEED)
    ids = list(groups)
    stats = sorted(stat_fn([groups[q] for q in rng.choices(ids, k=len(ids))]) for _ in range(N_RESAMPLES))
    return stats[int(0.025 * N_RESAMPLES)], stats[int(0.975 * N_RESAMPLES)]


def bootstrap_diff_unpaired(a, b):
    """CI for mean(a) - mean(b) when a and b come from different questions: resample each group separately."""
    rng = random.Random(SEED)
    diffs = sorted(sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b)
                   for _ in range(N_RESAMPLES))
    return diffs[int(0.025 * N_RESAMPLES)], diffs[int(0.975 * N_RESAMPLES)]


def mean(values):
    return sum(values) / len(values)


# ---------- collect results ----------

summary = []


def add(section, metric, condition, group, value, ci, n_q, n_items, detector="nli", is_diff=False):
    lo, hi = ci
    summary.append({
        "section": section, "detector": detector, "metric": metric, "condition": condition, "group": group,
        "n_questions": n_q, "n_items": n_items, "value": round(value, 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        # a difference is significant when its whole 95% CI is on one side of zero (same rule as Part 1)
        "significant": ("yes" if lo > 0 or hi < 0 else "no") if is_diff else "",
    })


def rate(section, metric, condition, group, subset, fn):
    pq = per_question(subset, fn)
    add(section, metric, condition, group, mean(list(pq.values())), bootstrap_ci(list(pq.values())),
        len(pq), len(subset))
    return pq


# 1. Main measures per condition
for cond in CONDITIONS:
    rate("main", "accuracy", cond, "all", select(cond), correct)
    rate("main", "false_alarm", cond, "all", select(cond, label="supported"), false_alarm)
    rate("main", "miss", cond, "all", select(cond, bad=True), missed)
    for detector, score in [("nli", "nli_entail"), ("overlap", "overlap")]:
        g = auroc_groups(cond, score)
        add("main", "auroc", cond, "all", pooled_auroc(g.values()), bootstrap_stat(g, pooled_auroc),
            len(g), len(select(cond)), detector=detector)

# 2. Paired differences between conditions (same questions, so compare question by question)
for a, b in [("hybrid", "oracle"), ("weak", "oracle"), ("weak", "hybrid")]:
    for metric, subset_filter, fn in [("accuracy", {}, correct), ("false_alarm", {"label": "supported"}, false_alarm),
                                      ("miss", {"bad": True}, missed)]:
        pa, pb = per_question(select(a, **subset_filter), fn), per_question(select(b, **subset_filter), fn)
        diffs = [pa[q] - pb[q] for q in pa]
        add("condition_diff", metric, f"{a} - {b}", "all", mean(diffs), bootstrap_ci(diffs), len(diffs),
            len(select(a, **subset_filter)), is_diff=True)
    ga, gb = auroc_groups(a, "nli_entail"), auroc_groups(b, "nli_entail")
    paired = {q: (ga[q], gb[q]) for q in ga}
    stat = lambda gs: pooled_auroc([x for x, _ in gs]) - pooled_auroc([y for _, y in gs])
    add("condition_diff", "auroc", f"{a} - {b}", "all", stat(list(paired.values())), bootstrap_stat(paired, stat),
        len(paired), len(select(a)), is_diff=True)

# 3. NLI vs overlap: is the model better than counting shared words? (AUROC, paired by question)
for cond in CONDITIONS:
    gn, go = auroc_groups(cond, "nli_entail"), auroc_groups(cond, "overlap")
    paired = {q: (gn[q], go[q]) for q in gn}
    stat = lambda gs: pooled_auroc([x for x, _ in gs]) - pooled_auroc([y for _, y in gs])
    add("detector_diff", "auroc", cond, "nli - overlap", stat(list(paired.values())), bootstrap_stat(paired, stat),
        len(paired), len(select(cond)), detector="nli vs overlap", is_diff=True)

# (a) Evidence found vs missed: false alarms on supported answers
for cond in ["hybrid", "weak"]:
    pq = {}
    for hit, name in [(True, "found"), (False, "missed")]:
        pq[name] = rate("a_evidence", "false_alarm", cond, name, select(cond, label="supported", evidence_hit=hit),
                        false_alarm)
    a, b = list(pq["missed"].values()), list(pq["found"].values())
    add("a_evidence", "false_alarm", cond, "missed - found", mean(a) - mean(b), bootstrap_diff_unpaired(a, b),
        len(a) + len(b), len(a) + len(b), is_diff=True)

# (b) Contradicted vs unsupported: which kind of bad answer slips through more?
for cond in CONDITIONS:
    pq = {label: rate("b_bad_type", "miss", cond, label, select(cond, label=label), missed)
          for label in ["contradicted", "unsupported"]}
    diffs = [pq["unsupported"][q] - pq["contradicted"][q] for q in pq["unsupported"]]  # same questions: paired
    add("b_bad_type", "miss", cond, "unsupported - contradicted", mean(diffs), bootstrap_ci(diffs),
        len(diffs), 2 * len(diffs), is_diff=True)

# (c) Supported answers written by the author (human) vs by the LLM
for cond in CONDITIONS:
    pq = {who: rate("c_author", "false_alarm", cond, who, select(cond, label="supported", drafted_by=who),
                    false_alarm)
          for who in ["human", "llm"]}
    a, b = list(pq["human"].values()), list(pq["llm"].values())
    add("c_author", "false_alarm", cond, "human - llm", mean(a) - mean(b), bootstrap_diff_unpaired(a, b),
        len(a) + len(b), len(a) + len(b), is_diff=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "summary.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

# ---------- print ----------

SECTION_TITLES = {
    "main": "Main measures (NLI unless the detector says overlap)",
    "condition_diff": "Paired differences between conditions",
    "detector_diff": "NLI vs overlap (AUROC difference)",
    "a_evidence": "(a) False alarms: evidence found vs missed",
    "b_bad_type": "(b) Miss rate: contradicted vs unsupported",
    "c_author": "(c) False alarms on supported answers: human vs LLM",
}
for section, title in SECTION_TITLES.items():
    print(f"\n=== {title} ===")
    print(f"  {'metric':12}{'detector':16}{'condition':16}{'group':28}{'n_q':>4}{'value':>9}   95% CI")
    for s in (s for s in summary if s["section"] == section):
        sig = f"  -> {'significant' if s['significant'] == 'yes' else 'not significant'}" if s["significant"] else ""
        print(f"  {s['metric']:12}{s['detector']:16}{s['condition']:16}{s['group']:28}{s['n_questions']:>4}"
              f"{s['value']:>9.3f}   [{s['ci_low']:.3f}, {s['ci_high']:.3f}]{sig}")
print(f"\nSaved {OUT_DIR / 'summary.csv'} ({len(summary)} rows)")

# ---------- chart: false-alarm rate by condition, evidence found vs missed ----------

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"all": "#52514e", "found": "#2a78d6", "missed": "#eb6834"}
LABELS = {"all": "Oracle (evidence always given)", "found": "Evidence found", "missed": "Evidence missed"}
lookup = {(s["condition"], s["group"]): s for s in summary if s["metric"] == "false_alarm"
          and s["section"] in ("main", "a_evidence")}
bars = [(0, "oracle", "all"), (0.8, "hybrid", "found"), (1.2, "hybrid", "missed"),
        (2.0, "weak", "found"), (2.4, "weak", "missed")]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
used = set()
for x, cond, group in bars:
    s = lookup[(cond, group)]
    ax.bar(x, s["value"], width=0.36, color=COLORS[group], zorder=2,
           label=LABELS[group] if group not in used else None)
    used.add(group)
    ax.errorbar(x, s["value"], yerr=[[s["value"] - s["ci_low"]], [s["ci_high"] - s["value"]]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    ax.text(x, s["ci_high"] + 0.02, f"{s['value']:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
cond_names = {"oracle": "Oracle", "hybrid": "Hybrid", "weak": "Weak"}
ax.set_xticks([x for x, _, _ in bars])
ax.set_xticklabels([f"{cond_names[c]}" + ("" if g == "all" else f", {g}") + f"\n{lookup[(c, g)]['n_questions']} questions"
                    for _, c, g in bars], color=INK, fontsize=8)
ax.set_ylabel("False-alarm rate (lower is better)", color=INK)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("NLI false alarms on supported answers, by retrieval condition (95% bootstrap CI)",
             color=INK, fontsize=10, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.tick_params(colors=MUTED, length=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK, bbox_to_anchor=(0, 0.97))
fig.tight_layout()
fig.savefig(OUT_DIR / "false_alarms.png", facecolor=SURFACE)
print(f"Saved {OUT_DIR / 'false_alarms.png'}")

# ---------- 10 error cases ----------

with open(CTX_PATH, encoding="utf-8") as f:
    contexts = {r["question_id"]: r for r in json.load(f)}
with open("eval/questions.json", encoding="utf-8") as f:
    evidence = {q["id"]: " | ".join(ev["text"] for ev in q["evidence"]) for q in json.load(f)}
nli_cache = load_json(NLI_CACHE_PATH, {})


def pick(candidates, n=5):
    """Up to n cases, taking conditions in turn (oracle, hybrid, weak, ...) and never the same item twice."""
    by_cond = {c: [r for r in candidates if r["condition"] == c] for c in CONDITIONS}
    chosen, seen = [], set()
    while len(chosen) < n and any(by_cond.values()):
        for c in CONDITIONS:
            while by_cond[c] and by_cond[c][0]["id"] in seen:
                by_cond[c].pop(0)
            if by_cond[c] and len(chosen) < n:
                r = by_cond[c].pop(0)
                chosen.append(r)
                seen.add(r["id"])
    return chosen


def top_chunk(r):
    """The chunk (and window inside it) where NLI found the highest entailment for this answer."""
    ctx = contexts[r["question_id"]]
    best = max(((nli_cache[pair_key(w, answers[r["id"]])]["entailment"], rank, chunk, w)
                for rank, chunk in enumerate(ctx["contexts"][r["condition"]]["chunks"], 1)
                for w in windows(chunk["text"])), key=lambda t: t[0])
    return best, ctx["evidence_chunk_ids"]


with open(SET_PATH, encoding="utf-8") as f:
    answers = {i["id"]: i["answer"] for i in json.load(f)}

cases = [("FALSE ALARM (supported answer, NLI did not accept it)",
          pick(select(label="supported", accepted=False))),
         ("ACCEPTED BAD ANSWER (contradicted/unsupported answer, NLI called it supported)",
          pick(select(bad=True, accepted=True)))]
for title, chosen in cases:
    print(f"\n=== {title} ===")
    for r in chosen:
        (entail, rank, chunk, window), evidence_ids = top_chunk(r)
        hit = "" if r["evidence_hit"] is None else f", evidence_hit={r['evidence_hit']}"
        print(f"\n- {r['id']} [{r['condition']}{hit}]  label={r['label']}  nli_pred={r['nli_pred']}  "
              f"entail={r['nli_entail']:.2f}  contra={r['nli_contra']:.2f}  overlap={r['overlap']:.2f}")
        print(f"  answer:   {answers[r['id']]}")
        print(f"  evidence: {evidence[r['question_id']]}")
        print(f"  top chunk: #{rank} in context (chunk {chunk['id']}, page {chunk['page']}, "
              f"{'an evidence chunk' if chunk['id'] in evidence_ids else 'NOT an evidence chunk'}); "
              f"best window (entail={entail:.2f}):")
        print(f"    {window}")
