import json
import re
import unicodedata

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


def load_eval_set(path="eval_set.json"):
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