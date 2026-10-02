"""Check the labeled evaluation set against test.pdf. Run: python evaluate.py"""
from collections import Counter

from pipeline import extract_pages, chunk_pages, load_eval_set, relevant_chunk_ids

with open("test.pdf", "rb") as f:
    pages = extract_pages(f.read())
questions = load_eval_set()

print(len(questions), "questions:", dict(Counter(q["type"] for q in questions)))

# Every quote must be found, for every chunk size we plan to test in Stage 5
for chunk_size in [100, 200, 400]:
    chunks = chunk_pages(pages, chunk_size)
    counts = [len(relevant_chunk_ids(q, pages, chunks)) for q in questions]
    print(f"chunk_size={chunk_size}: {len(chunks)} chunks, "
          f"relevant chunks per question: avg {sum(counts) / len(counts):.2f}, min {min(counts)}, max {max(counts)}")

print("\nRelevant chunks per question (chunk_size=200):")
chunks = chunk_pages(pages, 200)
for q in questions:
    ids = sorted(relevant_chunk_ids(q, pages, chunks))
    print(f"  {q['id']} [{q['type']:10}] chunks {ids}  {q['question']}")
