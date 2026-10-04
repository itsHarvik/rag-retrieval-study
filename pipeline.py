import json
import math
import random
import re
import unicodedata
from collections import Counter

import faiss
import pymupdf
from sentence_transformers import SentenceTransformer

# A word broken across two lines, e.g. "com-\nputational"
LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")


def extract_pages(pdf_bytes):
    """Read a PDF and return a list of pages: [{"page": 1, "text": "..."}, ...]"""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    # NFKC turns ligature characters like "ﬁ" (one symbol) into normal "fi"
    texts = [unicodedata.normalize("NFKC", page.get_text()) for page in doc]

    # Words that really contain a hyphen somewhere in the document (e.g. "cosine-similarity").
    # If a line-break hyphen produces one of these, we keep the hyphen; otherwise we join the halves.
    real_hyphen_words = set(re.findall(r"\w+-\w+", " ".join(texts).lower()))

    def fix(match):
        left, right = match.group(1), match.group(2)
        if f"{left}-{right}".lower() in real_hyphen_words:
            return f"{left}-{right}"
        return left + right

    return [{"page": i + 1, "text": LINE_BREAK_HYPHEN.sub(fix, t)} for i, t in enumerate(texts)]


def chunk_pages(pages, chunk_size=200):
    chunks = []

    for page in pages:
        page_text = page["text"]
        page_number = page["page"]

        words = page_text.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            chunks.append({
                "id": len(chunks),  # position in the chunk list, used by search results and metrics
                "text": chunk,
                "page": page_number,
                # Word positions this chunk covers on its page: [start, end)
                "start": i,
                "end": min(i + chunk_size, len(words)),
            })

    return chunks


def keyword_search(chunks, query):
    """Return the chunks whose text contains the query (case-insensitive)."""
    return [c for c in chunks if query.lower() in c["text"].lower()]


def load_model():
    """Load the sentence embedding model (downloads it the first time)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks, model):
    """Embed every chunk and store the vectors in a FAISS index."""
    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True makes every vector length 1,
    # so inner product (IP) = cosine similarity
    embeddings = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # 384 dimensions
    index.add(embeddings)
    return index


def semantic_search(query, model, index, chunks, k=5):
    """Return the k chunks most similar in meaning to the query, with scores."""
    query_vector = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(query_vector, k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:  # FAISS uses -1 when there are fewer than k chunks
            continue
        results.append({**chunks[i], "score": float(score)})
    return results


def normalize_words(text):
    """Split text into lowercase words without punctuation, for matching.
    Splits exactly like chunk_pages, so word positions line up."""
    text = unicodedata.normalize("NFKC", text)  # same normalization as extract_pages
    return [re.sub(r"\W+", "", w.lower()) for w in text.split()]


def find_evidence(pages, page_number, evidence_text):
    """Find where a quote appears on a page. Returns the word span (start, end), or None."""
    page_words = normalize_words(pages[page_number - 1]["text"])
    quote = normalize_words(evidence_text)
    for start in range(len(page_words) - len(quote) + 1):
        if page_words[start:start + len(quote)] == quote:
            return start, start + len(quote)
    return None


def load_eval_set(path="eval/questions.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def relevant_chunk_ids(question, pages, chunks):
    """Indices of the chunks that contain (part of) any evidence quote for this question.
    Works for any chunk size, because labels are quotes, not chunk numbers."""
    relevant = set()
    for ev in question["evidence"]:
        span = find_evidence(pages, ev["page"], ev["text"])
        if span is None:
            raise ValueError(f"{question['id']}: quote not found on page {ev['page']}: {ev['text']!r}")
        for idx, c in enumerate(chunks):
            # Two ranges overlap if each one starts before the other ends
            if c["page"] == ev["page"] and c["start"] < span[1] and span[0] < c["end"]:
                relevant.add(idx)
    return relevant


# ---------- Stage 5: BM25 and hybrid search ----------

def tokenize(text):
    """Lowercase words without punctuation (same rules as evidence matching)."""
    return [w for w in normalize_words(text) if w]


def build_bm25(chunks):
    """Precompute what BM25 needs: word counts per chunk, chunk lengths, and IDF per word."""
    counts = [Counter(tokenize(c["text"])) for c in chunks]
    lengths = [sum(c.values()) for c in counts]
    n = len(chunks)
    # Document frequency: in how many chunks does each word appear?
    df = Counter(word for c in counts for word in c)
    # IDF: rare words get a high weight, common words ("the") a weight close to 0
    idf = {word: math.log((n - d + 0.5) / (d + 0.5) + 1) for word, d in df.items()}
    return {"counts": counts, "lengths": lengths, "avg_length": sum(lengths) / n, "idf": idf}


def bm25_search(query, bm25, chunks, k=5, k1=1.5, b=0.75):
    """Rank chunks by BM25 score: shared words count, weighted by rarity (IDF).
    k1 limits how much repeating a word helps; b penalizes long chunks."""
    query_words = set(tokenize(query))
    scored = []
    for i, counts in enumerate(bm25["counts"]):
        length_norm = 1 - b + b * bm25["lengths"][i] / bm25["avg_length"]
        score = 0.0
        for w in query_words:
            tf = counts.get(w, 0)
            if tf:
                score += bm25["idf"][w] * tf * (k1 + 1) / (tf + k1 * length_norm)
        scored.append((score, i))
    scored.sort(key=lambda x: -x[0])
    return [{**chunks[i], "score": score} for score, i in scored[:k]]


def hybrid_search(query, model, index, bm25, chunks, k=5, rrf_k=60):
    """Reciprocal Rank Fusion: each method votes 1 / (rrf_k + rank) for every chunk.
    Uses ranks, not raw scores, because BM25 scores (0-20+) and cosine scores (0-1) aren't comparable."""
    fused = Counter()
    for results in (semantic_search(query, model, index, chunks, k=len(chunks)),
                    bm25_search(query, bm25, chunks, k=len(chunks))):
        for rank, r in enumerate(results, start=1):
            fused[r["id"]] += 1 / (rrf_k + rank)
    return [{**chunks[i], "score": score} for i, score in fused.most_common(k)]


# ---------- Stage 4: retrieval metrics ----------

def precision_at_k(ranked_ids, relevant, k):
    """Of the top k results, what fraction are relevant?"""
    return len(set(ranked_ids[:k]) & relevant) / k


def recall_at_k(ranked_ids, relevant, k):
    """Of all relevant chunks, what fraction appear in the top k?"""
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids, relevant):
    """1 / (position of the first relevant result). 1st -> 1.0, 2nd -> 0.5, never found -> 0."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def evaluate_search(search, questions, pages, chunks, ks=(1, 3, 5)):
    """Run search(question_text) -> ranked list of chunk ids for every question.
    Returns one row per question with its scores, so we can average over any group later."""
    rows = []
    for q in questions:
        relevant = relevant_chunk_ids(q, pages, chunks)
        ranked = search(q["question"])
        row = {"id": q["id"], "type": q["type"], "source": q["source"], "rr": reciprocal_rank(ranked, relevant)}
        for k in ks:
            row[f"p@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"r@{k}"] = recall_at_k(ranked, relevant, k)
        rows.append(row)
    return rows


def summarize(rows, metrics):
    """Average each metric over the rows. The average of "rr" is MRR."""
    return {m: sum(r[m] for r in rows) / len(rows) for m in metrics}


def random_ranking(chunks, seed):
    """Baseline: a random order of all chunks. Any real method must beat this."""
    ids = [c["id"] for c in chunks]
    random.Random(seed).shuffle(ids)
    return ids


def bootstrap_ci(values, n_resamples=10000, seed=0):
    """95% confidence interval for the mean, by resampling the questions with replacement.
    Shows how much the average could change with a different set of 43 questions."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


if __name__ == "__main__":
    # Quick test without Streamlit: run "python pipeline.py"
    with open("test.pdf", "rb") as f:
        pages = extract_pages(f.read())
    chunks = chunk_pages(pages)
    print(len(pages), "pages,", len(chunks), "chunks")

    model = load_model()
    index = build_index(chunks, model)

    query = "How long does BERT take to find the most similar sentence pair?"
    print("\nQuery:", query)
    print("Keyword matches:", len(keyword_search(chunks, query)))
    print("Semantic top 3:")
    for r in semantic_search(query, model, index, chunks, k=3):
        print(f"  page {r['page']}  score {r['score']:.3f}  {r['text'][:100]}...")