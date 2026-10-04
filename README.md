# PDF Question Answering: BM25 vs Semantic vs Hybrid Retrieval

A small research project on the **retrieval** step of PDF question answering (RAG). Given a question about a PDF, which method finds the passage that answers it best? I compare keyword search (BM25), semantic search (sentence embeddings + FAISS) and a hybrid of both, at three chunk sizes, on a hand-labeled evaluation set.

**Full write-up: [REPORT.md](REPORT.md)**

![MRR by method and chunk size](results/chart.png)

## Key results (43 questions about the Sentence-BERT paper)

- **BM25 is a strong baseline.** It beat plain semantic search at every chunk size (significantly at 200 and 400 words).
- **The embedding model's 256-token limit silently cut off most chunks.** As a result, semantic search got *worse* with bigger chunks.
- **Embedding chunks as windows of ≤ 180 words fixes most of this.** The best configuration, windowed hybrid with 400-word chunks, reaches **MRR 0.70 and Recall@5 0.90**.
- Evaluation set: **28 questions drafted by an LLM, 15 written by me**. Results are reported for both groups. See the report for limitations.

## How it works

| File | What it does |
|---|---|
| `pipeline.py` | All logic: PDF extraction and cleaning, chunking, BM25 (hand-written), semantic search (all-MiniLM-L6-v2 + FAISS, optional windowing), hybrid search (Reciprocal Rank Fusion), metrics (P@k, R@k, MRR), bootstrap confidence intervals |
| `App.py` | Streamlit app: upload a PDF and search it with Keyword / BM25 / Semantic / Hybrid |
| `eval/questions.json` | 43 labeled questions. Each has evidence quotes (page + exact text), a type (lexical / paraphrase) and an author (`llm` / `human`) |
| `check_eval_set.py` | Checks that every evidence quote is found in the PDF |
| `evaluate.py` | Stage 4 report: semantic search vs a random baseline |
| `experiments.py` | All experiments (5 methods × 3 chunk sizes) with paired bootstrap tests. Writes `results/*.csv` |
| `make_chart.py` | Draws `results/chart.png` from the results |
| `LEARNING.md` | Stage-by-stage notes: what was built, why, and what we found |

## Run it

```bash
pip install -r requirements.txt

# The test paper is not included in the repo. Download Sentence-BERT (arXiv:1908.10084):
curl -L -o test.pdf https://arxiv.org/pdf/1908.10084

python check_eval_set.py   # verify the labels match the PDF
python experiments.py      # run all experiments (writes results/)
python make_chart.py       # redraw the chart
streamlit run App.py       # interactive search app
```

The labels are exact quotes, so the PDF text extraction must match. If you get a "quote not found" error, check that you downloaded the original arXiv v1 PDF.

## Built with

Python, PyMuPDF, sentence-transformers, FAISS, Streamlit, matplotlib. The code was written with an AI coding assistant (Claude Code) under my direction. LEARNING.md documents the reasoning behind every choice.
