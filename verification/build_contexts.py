"""Part 2: for every evaluation question, save 3 context conditions to verification/contexts.json.
  oracle - only the chunk(s) that contain the evidence quote (perfect retrieval)
  hybrid - our best Part 1 setup: windowed hybrid search, 400-word chunks, top 5
  weak   - plain semantic search (no windows), 400-word chunks, top 5. The model reads only the first
           256 tokens of each chunk, so the end of every long chunk is invisible to it (Part 1's truncation problem)
Run from the project folder: python verification/build_contexts.py"""
import json
import os
import sys
from pathlib import Path

# Make "import pipeline" and the relative paths (test.pdf, eval/questions.json) work
# no matter which folder the script is started from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")  # the Windows console can't print characters like "−" otherwise

from pipeline import (extract_pages, chunk_pages, load_eval_set, load_model, build_index, build_bm25,
                      hybrid_search, semantic_search, relevant_chunk_ids)

CHUNK_SIZE = 400    # best chunk size in Part 1
WINDOW_WORDS = 180  # same window size as "hybrid-win" in experiments.py
TOP_K = 5
OUT_PATH = Path("verification/contexts.json")

with open("test.pdf", "rb") as f:
    pages = extract_pages(f.read())
chunks = chunk_pages(pages, CHUNK_SIZE)
questions = load_eval_set()

model = load_model()
win_index, win_owners = build_index(chunks, model, window_words=WINDOW_WORDS)  # for "hybrid"
plain_index, plain_owners = build_index(chunks, model)                          # for "weak"
bm25 = build_bm25(chunks)


def condition(chunk_ids, evidence_ids=None):
    """One context condition: the chunk ids and texts, plus evidence_hit for retrieved conditions."""
    cond = {
        "chunk_ids": chunk_ids,
        "chunks": [{"id": i, "page": chunks[i]["page"], "text": chunks[i]["text"]} for i in chunk_ids],
    }
    if evidence_ids is not None:
        # Same matching as Part 1: a chunk "contains the evidence" if it overlaps the word span of an evidence quote
        cond["evidence_hit"] = any(i in evidence_ids for i in chunk_ids)
    return cond


records = []
for q in questions:
    evidence_ids = relevant_chunk_ids(q, pages, chunks)
    hybrid_ids = [r["id"] for r in hybrid_search(q["question"], model, win_index, win_owners, bm25, chunks, k=TOP_K)]
    weak_ids = [r["id"] for r in semantic_search(q["question"], model, plain_index, plain_owners, chunks, k=TOP_K)]
    records.append({
        "question_id": q["id"],
        "question": q["question"],
        "evidence_chunk_ids": sorted(evidence_ids),
        "contexts": {
            "oracle": condition(sorted(evidence_ids)),
            "hybrid": condition(hybrid_ids, evidence_ids),
            "weak": condition(weak_ids, evidence_ids),
        },
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"{len(chunks)} chunks of {CHUNK_SIZE} words, {len(records)} questions")
oracle_sizes = [len(r["contexts"]["oracle"]["chunk_ids"]) for r in records]
print(f"oracle: {min(oracle_sizes)}-{max(oracle_sizes)} chunks per question "
      f"(avg {sum(oracle_sizes) / len(oracle_sizes):.2f})")
for name in ["hybrid", "weak"]:
    hits = sum(r["contexts"][name]["evidence_hit"] for r in records)
    misses = [r["question_id"] for r in records if not r["contexts"][name]["evidence_hit"]]
    print(f"{name:6}: evidence_hit = true for {hits} / {len(records)} questions"
          + (f"  (misses: {', '.join(misses)})" if misses else ""))
print(f"Saved {OUT_PATH}")
