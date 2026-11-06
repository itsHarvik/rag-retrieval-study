"""Part 2, Day 2: score every answer in faithfulness_set.json against each context condition.
  overlap - share of the answer's content words found in the best-matching chunk (simple baseline)
  NLI     - cross-encoder/nli-deberta-v3-base; premise = a ~3-sentence window of a chunk, hypothesis = the answer.
            We keep the max entailment and max contradiction over all windows of all chunks.
  judge   - optional LLM judge, only if ANTHROPIC_API_KEY is set
Output: verification/scores.csv, plus a sanity check on the oracle condition.
NLI scores are cached in verification/nli_cache.json, so a re-run takes seconds.
Run from the project folder: python verification/detect.py"""
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

SET_PATH = Path("verification/faithfulness_set.json")
CTX_PATH = Path("verification/contexts.json")
OUT_PATH = Path("verification/scores.csv")
NLI_CACHE_PATH = Path("verification/nli_cache.json")
JUDGE_CACHE_PATH = Path("verification/judge_cache.json")

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
CONDITIONS = ["oracle", "hybrid", "weak"]
LABELS = ["supported", "contradicted", "unsupported"]
THRESHOLD = 0.5
SENTS_PER_WINDOW = 3     # "about 3 sentences" per window
WINDOW_STRIDE = 2        # windows overlap by 1 sentence, so a claim split over a boundary is still seen whole
MAX_WINDOW_WORDS = 200   # ~300 DeBERTa tokens, leaves room for the answer inside the 512-token limit
MAX_TOKENS = 512
BATCH_SIZE = 16

# Same stopword list and word rule as check_set.py, so "content word" means the same thing everywhere
STOPWORDS = set("""a an the and or but of to in on at by for with from as is are was were be been being it its
this that these those than then so such which who what when where why how do does did not no can could would
should will they their them we our us he she his her you your i about into over only also very more most""".split())


def content_words(text):
    """Lowercase words without punctuation, minus stopwords."""
    text = unicodedata.normalize("NFKC", text)
    words = (re.sub(r"\W+", "", w.lower()) for w in text.split())
    return {w for w in words if w and w not in STOPWORDS}


def overlap_score(answer, chunks):
    """Max over chunks of: (answer content words that appear in the chunk) / (all answer content words)."""
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    return max(len(answer_words & content_words(c["text"])) / len(answer_words) for c in chunks)


# ---------- splitting chunks into windows ----------

def split_sentences(text):
    """Split after . ! ? when the next word starts like a new sentence (capital, digit, bracket, quote).
    PDF text is messy, so this is approximate; very long 'sentences' (tables, lists) are cut further below."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])", text.strip())
    pieces = []
    for p in parts:
        words = p.split()
        for start in range(0, len(words), MAX_WINDOW_WORDS):
            pieces.append(" ".join(words[start:start + MAX_WINDOW_WORDS]))
    return [p for p in pieces if p]


def windows(text):
    """Sliding windows of up to 3 sentences (and at most MAX_WINDOW_WORDS words), stepping 2 sentences."""
    sents = split_sentences(text)
    result = []
    for start in range(0, len(sents), WINDOW_STRIDE):
        window, n_words = [], 0
        for s in sents[start:start + SENTS_PER_WINDOW]:
            if window and n_words + len(s.split()) > MAX_WINDOW_WORDS:
                break
            window.append(s)
            n_words += len(s.split())
        result.append(" ".join(window))
        if start + SENTS_PER_WINDOW >= len(sents):  # this window already reaches the last sentence
            break
    return result


def pair_key(premise, hypothesis):
    return hashlib.sha1(f"{NLI_MODEL}\n{premise}\n{hypothesis}".encode("utf-8")).hexdigest()


def load_json(path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------- NLI scoring with a cache ----------

def score_missing_pairs(pairs, cache):
    """Run the NLI model on (premise, hypothesis) pairs that aren't cached yet.
    Each cache entry: probabilities by label name, plus the token count (to check truncation)."""
    missing = list({pair_key(p, h): (p, h) for p, h in pairs if pair_key(p, h) not in cache}.items())
    print(f"NLI pairs: {len({pair_key(p, h) for p, h in pairs})} unique, {len(missing)} not cached yet")
    if not missing:
        return
    from sentence_transformers import CrossEncoder  # imported here so a fully cached run doesn't load torch
    model = CrossEncoder(NLI_MODEL, max_length=MAX_TOKENS)
    id2label = {int(i): name.lower() for i, name in model.config.id2label.items()}
    print(f"Label order from model.config.id2label: {id2label}")

    for start in tqdm(range(0, len(missing), BATCH_SIZE), desc="NLI", unit="batch"):
        batch = missing[start:start + BATCH_SIZE]
        probs = model.predict([pair for _, pair in batch], apply_softmax=True, show_progress_bar=False)
        for (key, (premise, hypothesis)), row in zip(batch, probs):
            n_tokens = len(model.tokenizer(premise, hypothesis)["input_ids"])
            cache[key] = {**{id2label[i]: float(p) for i, p in enumerate(row)}, "tokens": n_tokens}
        if (start // BATCH_SIZE) % 50 == 49:  # save now and then, so an interrupted run keeps its progress
            save_json(NLI_CACHE_PATH, cache)
    save_json(NLI_CACHE_PATH, cache)


def nli_scores(answer, chunks, cache):
    """Max entailment and max contradiction over all windows of all chunks, and the windows they came from."""
    best_e, best_c = (-1.0, ""), (-1.0, "")
    for chunk in chunks:
        for w in windows(chunk["text"]):
            s = cache[pair_key(w, answer)]
            best_e = max(best_e, (s["entailment"], w))
            best_c = max(best_c, (s["contradiction"], w))
    return best_e, best_c


def nli_predict(entail, contra):
    if entail >= THRESHOLD:
        return "supported"
    if contra >= THRESHOLD:
        return "contradicted"
    return "unsupported"


# ---------- optional LLM judge ----------

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_PROMPT = """You check whether an answer is faithful to the given context from a research paper.
Reply with exactly one word:
supported - the context states or directly implies the answer
contradicted - the context says something that conflicts with the answer
unsupported - the context neither supports nor contradicts the answer

<context>
{context}
</context>

<answer>
{answer}
</answer>"""


def judge_all(jobs):
    """jobs: list of (key, answer, chunks). Returns {key: label}. Cached, so each pair is paid for only once."""
    import anthropic
    client = anthropic.Anthropic()
    cache = load_json(JUDGE_CACHE_PATH, {})
    for key, answer, chunks in tqdm(jobs, desc="Judge", unit="call"):
        if key in cache:
            continue
        context = "\n\n".join(f"[chunk {c['id']}, page {c['page']}]\n{c['text']}" for c in chunks)
        response = client.beta.messages.create(
            model=JUDGE_MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines, the API retries on a fallback model
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
        )
        text = " ".join(b.text for b in response.content if b.type == "text").strip().lower()
        cache[key] = next((label for label in LABELS if label in text), "unparsed")
        save_json(JUDGE_CACHE_PATH, cache)
    return {key: cache[key] for key, _, _ in jobs}


# ---------- main ----------

with open(SET_PATH, encoding="utf-8") as f:
    items = json.load(f)
with open(CTX_PATH, encoding="utf-8") as f:
    contexts = {r["question_id"]: r["contexts"] for r in json.load(f)}

# 1. Collect every (window, answer) pair we need, then score the ones that aren't cached
pairs = [(w, item["answer"])
         for item in items
         for cond in CONDITIONS
         for chunk in contexts[item["question_id"]][cond]["chunks"]
         for w in windows(chunk["text"])]
nli_cache = load_json(NLI_CACHE_PATH, {})
score_missing_pairs(pairs, nli_cache)

too_long = sum(nli_cache[pair_key(p, h)]["tokens"] > MAX_TOKENS for p, h in set(pairs))
print(f"Pairs longer than {MAX_TOKENS} tokens (would be truncated): {too_long}")

# 2. One row per item per condition
rows = []
for item in items:
    for cond in CONDITIONS:
        ctx = contexts[item["question_id"]][cond]
        (entail, entail_win), (contra, contra_win) = nli_scores(item["answer"], ctx["chunks"], nli_cache)
        rows.append({
            "id": item["id"],
            "question_id": item["question_id"],
            "label": item["label"],
            "condition": cond,
            "evidence_hit": "" if cond == "oracle" else ctx["evidence_hit"],
            "overlap": round(overlap_score(item["answer"], ctx["chunks"]), 4),
            "nli_entail": round(entail, 4),
            "nli_contra": round(contra, 4),
            "nli_pred": nli_predict(entail, contra),
            # kept only for printing examples below, not saved
            "_answer": item["answer"], "_entail_win": entail_win, "_contra_win": contra_win,
        })

# 3. Optional LLM judge
columns = ["id", "question_id", "label", "condition", "evidence_hit", "overlap", "nli_entail", "nli_contra", "nli_pred"]
if os.environ.get("ANTHROPIC_API_KEY"):
    jobs = [(f"{r['id']}|{r['condition']}", r["_answer"], contexts[r["question_id"]][r["condition"]]["chunks"])
            for r in rows]
    verdicts = judge_all(jobs)
    for r, (key, _, _) in zip(rows, jobs):
        r["judge_pred"] = verdicts[key]
    columns.append("judge_pred")
else:
    print("LLM judge skipped: ANTHROPIC_API_KEY is not set.")

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {OUT_PATH} ({len(rows)} rows = {len(items)} items x {len(CONDITIONS)} conditions)")

# 4. Sanity check on oracle: confusion table (rows = true label, columns = NLI prediction)
oracle = [r for r in rows if r["condition"] == "oracle"]
counts = Counter((r["label"], r["nli_pred"]) for r in oracle)
print("\nOracle condition: true label (rows) vs nli_pred (columns)")
print(f"{'':14}" + "".join(f"{p:>14}" for p in LABELS) + f"{'total':>8}")
for true in LABELS:
    print(f"{true:14}" + "".join(f"{counts[(true, p)]:>14}" for p in LABELS)
          + f"{sum(counts[(true, p)] for p in LABELS):>8}")
correct = sum(counts[(l, l)] for l in LABELS)
print(f"Accuracy: {correct}/{len(oracle)} = {correct / len(oracle):.1%}")

# 3 wrong examples, from a different true label each where possible
wrong = [r for r in oracle if r["label"] != r["nli_pred"]]
examples = []
for true in LABELS:
    examples += [r for r in wrong if r["label"] == true][:1]
examples += [r for r in wrong if r not in examples][:3 - len(examples)]
print(f"\n{len(wrong)} oracle items wrong. Examples:")
for r in examples[:3]:
    window = r["_entail_win"] if r["nli_pred"] == "supported" or r["label"] == "supported" else r["_contra_win"]
    print(f"\n- {r['id']}: true={r['label']}, nli_pred={r['nli_pred']} "
          f"(entail={r['nli_entail']:.2f}, contra={r['nli_contra']:.2f}, overlap={r['overlap']:.2f})")
    print(f"  answer: {r['_answer']}")
    print(f"  most relevant window: {window[:400]}{'...' if len(window) > 400 else ''}")
