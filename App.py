import streamlit as st
from pipeline import (extract_pages, chunk_pages, keyword_search, load_model, build_index, semantic_search,
                      build_bm25, bm25_search, hybrid_search)


@st.cache_resource
def get_model():
    # Load the model once and reuse it for every search
    return load_model()


@st.cache_resource
def process_pdf(pdf_bytes):
    # Chunk and index each PDF only once (cached by the file's bytes)
    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages)
    index, owners = build_index(chunks, get_model(), window_words=180)
    return chunks, index, owners, build_bm25(chunks)


st.title("Document Search")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    chunks, index, owners, bm25 = process_pdf(uploaded_file.read())

    mode = st.radio("Search mode", ["Keyword", "BM25", "Semantic", "Hybrid"], horizontal=True)
    query = st.text_input("Search the document")
    if query:
        if mode == "Keyword":
            results = keyword_search(chunks, query)
            st.write(f"Found {len(results)} matching chunks")
        elif mode == "BM25":
            results = bm25_search(query, bm25, chunks, k=5)
            st.write(f"Top {len(results)} chunks by BM25 word matching")
        elif mode == "Semantic":
            results = semantic_search(query, get_model(), index, owners, chunks, k=5)
            st.write(f"Top {len(results)} chunks by meaning")
        else:
            results = hybrid_search(query, get_model(), index, owners, bm25, chunks, k=5)
            st.write(f"Top {len(results)} chunks by BM25 + semantic (rank fusion)")

        for r in results:
            if "score" in r:
                st.markdown(f"**Page {r['page']}** · score {r['score']:.3f}")
            else:
                st.markdown(f"**Page {r['page']}**")
            st.write(r["text"])
