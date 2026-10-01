import streamlit as st
from pipeline import extract_pages, chunk_pages, keyword_search

st.title("Document Search")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pages = extract_pages(uploaded_file.read())
    chunks = chunk_pages(pages)

    query = st.text_input("Search the document")
    if query:
        results = keyword_search(chunks, query)
        st.write(f"Found {len(results)} matching chunks")
        for r in results:
            st.markdown(f"**Page {r['page']}**")
            st.write(r["text"])