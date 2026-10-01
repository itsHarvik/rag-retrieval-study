import faiss
import pymupdf
from sentence_transformers import SentenceTransformer

def extract_pages(pdf_bytes):
    """Read a PDF and return a list of pages: [{"page": 1, "text": "..."}, ...]"""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        pages.append({"page": i + 1, "text": page.get_text()})
    return pages


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
                "page": page_number
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