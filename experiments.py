"""Stage 5: BM25 vs semantic vs hybrid, at 3 chunk sizes, plus windowed embedding
(semantic-win / hybrid-win: chunks embedded as windows of <= 180 words). Run: python experiments.py
Prints the tables and saves results/per_question.csv and results/summary.csv."""
import csv
import os

from pipeline import (extract_pages, chunk_pages, load_eval_set, load_model, build_index, build_bm25,
                      semantic_search, bm25_search, hybrid_search, evaluate_search, summarize,
                      random_ranking, bootstrap_ci, split_windows)

CHUNK_SIZES = [100, 200, 400]
METHODS = ["bm25", "semantic", "hybrid", "semantic-win", "hybrid-win"]
WINDOW_WORDS = 180
METRICS = ["p@1", "p@3", "p@5", "r@1", "r@3", "r@5", "rr"]
RANDOM_RUNS = 200

with open("test.pdf", "rb") as f:
    pages = extract_pages(f.read())
questions = load_eval_set()
model = load_model()

all_rows = []  # one row per (chunk size, method, question)
summary = []   # one row per (chunk size, method)

for size in CHUNK_SIZES:
    chunks = chunk_pages(pages, size)
    index, owners = build_index(chunks, model)
    win_index, win_owners = build_index(chunks, model, window_words=WINDOW_WORDS)
    bm25 = build_bm25(chunks)
    n = len(chunks)

    def count_too_long(texts):
        return sum(len(model.tokenizer(t)["input_ids"]) > model.max_seq_length for t in texts)
    too_long = count_too_long([c["text"] for c in chunks])
    windows, _ = split_windows(chunks, WINDOW_WORDS)
    windows_too_long = count_too_long(windows)

    searches = {
        "bm25": lambda q: [r["id"] for r in bm25_search(q, bm25, chunks, k=n)],
        "semantic": lambda q: [r["id"] for r in semantic_search(q, model, index, owners, chunks, k=n)],
        "hybrid": lambda q: [r["id"] for r in hybrid_search(q, model, index, owners, bm25, chunks, k=n)],
        "semantic-win": lambda q: [r["id"] for r in semantic_search(q, model, win_index, win_owners, chunks, k=n)],
        "hybrid-win": lambda q: [r["id"] for r in hybrid_search(q, model, win_index, win_owners, bm25, chunks, k=n)],
    }
    random_rows = []
    for seed in range(RANDOM_RUNS):
        random_rows += evaluate_search(lambda q: random_ranking(chunks, seed), questions, pages, chunks)
    rand = summarize(random_rows, METRICS)

    print(f"\n=== chunk size {size} words: {n} chunks, {too_long} longer than the model's "
          f"{model.max_seq_length}-token limit; windowed: {len(windows)} windows, {windows_too_long} over the limit ===")
    print(f"  {'method':12}" + "".join(f"{('MRR' if m == 'rr' else m.upper()):>8}" for m in METRICS)
          + "   MRR 95% CI     MRR lex  MRR para")
    print(f"  {'random':12}" + "".join(f"{rand[m]:>8.3f}" for m in METRICS))

    for method in METHODS:
        rows = evaluate_search(searches[method], questions, pages, chunks)
        for r in rows:
            r.update(chunk_size=size, method=method)
        all_rows += rows

        avg = summarize(rows, METRICS)
        lo, hi = bootstrap_ci([r["rr"] for r in rows])
        by_type = {t: summarize([r for r in rows if r["type"] == t], ["rr"])["rr"] for t in ["lexical", "paraphrase"]}
        by_source = {s: summarize([r for r in rows if r["source"] == s], ["rr"])["rr"] for s in ["human", "llm"]}
        summary.append({"chunk_size": size, "method": method, "n_chunks": n,
                        "n_vectors": len(windows) if method.endswith("-win") else n,
                        **{m: round(v, 4) for m, v in avg.items()},
                        "rr_ci_low": round(lo, 4), "rr_ci_high": round(hi, 4),
                        "rr_lexical": round(by_type["lexical"], 4), "rr_paraphrase": round(by_type["paraphrase"], 4),
                        "rr_human": round(by_source["human"], 4), "rr_llm": round(by_source["llm"], 4),
                        "random_rr": round(rand["rr"], 4), "random_r@5": round(rand["r@5"], 4)})
        print(f"  {method:12}" + "".join(f"{avg[m]:>8.3f}" for m in METRICS)
              + f"   [{lo:.3f}, {hi:.3f}]   {by_type['lexical']:.3f}    {by_type['paraphrase']:.3f}")


def rr_of(size, method):
    return {r["id"]: r["rr"] for r in all_rows if r["chunk_size"] == size and r["method"] == method}


# Paired comparison: same questions, two methods. Is the difference bigger than the noise?
print("\n=== Paired comparisons (difference in reciprocal rank, per question) ===")
for size in CHUNK_SIZES:
    for a, b in [("hybrid", "semantic"), ("hybrid", "bm25"), ("semantic", "bm25"),
                 ("semantic-win", "semantic"), ("hybrid-win", "hybrid"), ("hybrid-win", "bm25")]:
        ra, rb = rr_of(size, a), rr_of(size, b)
        diffs = [ra[q] - rb[q] for q in ra]
        lo, hi = bootstrap_ci(diffs)
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        verdict = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"  size {size:3}  {a:12} - {b:8}: {sum(diffs) / len(diffs):+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
              f"wins {wins} / ties {len(diffs) - wins - losses} / losses {losses}  -> {verdict}")

os.makedirs("results", exist_ok=True)
with open("results/per_question.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["chunk_size", "method", "id", "type", "source"] + METRICS)
    writer.writeheader()
    writer.writerows(all_rows)
with open("results/summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print("\nSaved results/per_question.csv and results/summary.csv")
