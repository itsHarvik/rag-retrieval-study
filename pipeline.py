import pymupdf

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


if __name__ == "__main__":
    # Quick test without Streamlit: run "python pipeline.py"
    with open("test.pdf", "rb") as f:
        pages = extract_pages(f.read())
    chunks = chunk_pages(pages)
    print(len(pages), "pages,", len(chunks), "chunks")
    print(chunks[0])