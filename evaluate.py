"""Stage 4: score semantic search on the eval set with P@k, R@k and MRR. Run: python evaluate.py"""
from pipeline import (extract_pages, chunk_pages, load_eval_set, load_model, build_index,
                      semantic_search, evaluate_search, summarize, random_ranking)

METRICS = ["p@1", "p@3", "p@5", "r@1", "r@3", "r@5", "rr"]
RANDOM_RUNS = 200  # average the random baseline over many shuffles so it isn't luck


def print_table(title, groups):
    print(f"\n{title}")
    print(f"  {'group':24}{'n':>4}" + "".join(f"{('MRR' if m == 'rr' else m.upper()):>8}" for m in METRICS))
    for name, rows in groups:
        avg = summarize(rows, METRICS)
        print(f"  {name:24}{len(rows):>4}" + "".join(f"{avg[m]:>8.3f}" for m in METRICS))


with open("test.pdf", "rb") as f:
    pages = extract_pages(f.read())
questions = load_eval_set()
chunks = chunk_pages(pages, chunk_size=200)
model = load_model()
index, owners = build_index(chunks, model)


def search(question):
    # Rank ALL chunks, so MRR sees the true position of the first relevant chunk
    return [r["id"] for r in semantic_search(question, model, index, owners, chunks, k=len(chunks))]


rows = evaluate_search(search, questions, pages, chunks)

# Random baseline: one row per question per shuffle, averaged over all of them
random_rows = []
for seed in range(RANDOM_RUNS):
    random_rows += evaluate_search(lambda q: random_ranking(chunks, seed), questions, pages, chunks)

print(f"{len(questions)} questions, {len(chunks)} chunks of 200 words")
print_table("Semantic search vs random baseline", [("semantic", rows), (f"random (avg of {RANDOM_RUNS})", random_rows)])
print_table("Semantic search by question type and author", [
    (f"{field}={value}", [r for r in rows if r[field] == value])
    for field in ["type", "source"] for value in sorted({r[field] for r in rows})
])

misses = [r for r in rows if r["r@5"] == 0]
print(f"\nQuestions with no relevant chunk in the top 5 ({len(misses)}):")
for r in misses:
    q = next(q for q in questions if q["id"] == r["id"])
    print(f"  {r['id']} [{r['type']}] RR={r['rr']:.2f}  {q['question']}")
