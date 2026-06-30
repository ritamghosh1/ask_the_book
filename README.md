# 📚 Ask the Book

> Upload any PDF. Ask anything. Get grounded, cited answers in under 3 seconds.

Ask the Book is a Retrieval-Augmented Generation (RAG) system built from scratch. It goes well beyond basic vector search — implementing a full **Hybrid Retrieval Pipeline**, **Reciprocal Rank Fusion**, and a **Cross-Encoder Reranker** to deliver precise, hallucination-free answers with automatic page citations.

---

##  Demo

> Upload a textbook → Ask a question → Get a cited answer in ~2s

```
User:  What are the two ways to handle overflow in a hash table?

Bot:   To handle overflows, two primary methods are used:
       1. Chaining — maintain a linked list per bucket for synonyms.
       2. Open Addressing — probe alternative cells sequentially
          until an empty slot is found.

       Sources: pages 32, 33
```

---

##  Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────┐
│  unstructured.io Parser     │  ← Handles digital PDFs + auto OCR fallback
│  + Text Cleaning            │    for scanned/image-based documents
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Semantic Chunker           │  ← Splits on idea boundaries, not character
│  (LlamaIndex + BGE-small)   │    limits. breakpoint_percentile_threshold=85
└──────┬──────────────────────┘
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
┌─────────────────┐                   ┌──────────────────────┐
│  ChromaDB       │                   │  BM25 Keyword Index  │
│  (Vector Store) │                   │  (rank_bm25)         │
└────────┬────────┘                   └──────────┬───────────┘
         │                                       │
         ▼                                       ▼
    BGE-Small                            Exact keyword
    Semantic Search                      matching
    (top-15 candidates)                  (top-15 candidates)
         │                                       │
         └──────────────┬────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  Reciprocal Rank Fusion      │  ← Merges both lists. Chunks
         │  score = 1 / (k + rank)      │    appearing in both ranked higher
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  Cross-Encoder Reranker      │  ← ms-marco-MiniLM-L-6-v2
         │  (ms-marco-MiniLM-L-6-v2)   │    strictly grades each chunk
         └──────────────┬───────────────┘    against the query
                        ▼
                   Top-K Chunks
                   + Page Metadata
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Groq Llama 3.1 8B           │  ← Grounded system prompt.
         │  (Streaming)                 │    "Answer ONLY from context.
         └──────────────┬───────────────┘    If unknown, say so."
                        ▼
            Streaming Answer + Citations
```

---

## Latency Benchmarks

Tested on a 35-page academic PDF (862 extracted elements, 62 semantic chunks).

| Operation | Latency |
|---|---|
| PDF Ingestion + Semantic Chunking | ~18.5s (one-time per document) |
| Hybrid Retrieval + RRF + Reranking | ~0.4s |
| LLM Generation (Llama 3.1 8B via Groq) | ~1.5 – 2.5s |
| **Total Query Latency** | **< 3.0 seconds** |

Ingestion runs once per document. All subsequent questions on the same PDF answer in under 3 seconds.

---

## Evaluation Results

The retrieval pipeline is evaluated using a **tiered test suite** across 4 difficulty levels with substring matching against ground-truth phrases from the source document.

```
EASY       | ██  | 2/2  (100%)   Direct definition lookup
MEDIUM     | ██  | 2/2  (100%)   Paraphrased questions (wording ≠ book)
HARD       | ██░ | 2/3   (67%)   Cross-chunk reasoning across pages
NEGATIVE   | ███ | 3/3  (100%)   Out-of-scope hallucination guard
─────────────────────────────────
OVERALL    |     | 9/10  (90%)
```

**On evaluation methodology:** String matching was chosen deliberately for **determinism** — LLM-as-judge introduces non-determinism where identical correct answers can fail across runs depending on the judge's phrasing. For a closed-domain document with known ground truth (lecture notes with exact definitions), substring matching against source text produces more consistent, reproducible results.

The single hard-tier miss (Q5) is a known retrieval depth issue: the best-case complexity answer lives on page 10 while worst-case is on page 9, and both chunks don't always surface together in top-3. Raising `top_k` to 5 resolves it at the cost of slightly more LLM context. For open-domain or abstractive QA, LLM-as-judge would be the more appropriate evaluation method.

---

## Features

**Hybrid Search Engine**
Combines semantic understanding (BAAI/bge-small-en-v1.5 embeddings) with exact keyword matching (BM25). Neither alone is as strong as both together — vector search handles paraphrased questions, BM25 handles specific terminology like algorithm names and function signatures.

**Reciprocal Rank Fusion**
Merges the two ranked lists using the formula `score = 1 / (k + rank)`. A chunk appearing high in both lists scores significantly higher than one appearing in only one. Deduplication runs before the reranker to avoid scoring the same chunk twice.

**Cross-Encoder Reranking**
After RRF fusion, every candidate is re-scored by a cross-encoder model that reads the query and chunk together (not independently). This catches cases where embedding similarity is high but actual relevance is low.

**Semantic Chunking**
Splits text on natural idea boundaries using embedding similarity between sentences, not arbitrary character limits. A `breakpoint_percentile_threshold` of 75 produces ~60 focused chunks from a 35-page document.

**Grounded System Prompt + Citations**
The LLM is strictly instructed to answer only from provided context. Page numbers are preserved through the entire pipeline as chunk metadata and injected into the prompt, so every answer ends with `Sources: pages X, Y`.

**Streaming Responses**
Groq's LPU inference engine streams tokens in real-time so answers appear immediately rather than after a full generation delay.

**Hallucination Guard**
Out-of-scope questions (neural networks, graph algorithms, geography) return `"No context found from the book"` — not a hallucinated answer. Validated across 3 negative test cases.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Parsing | `unstructured` | Handles tables, mixed layouts, auto OCR |
| Embeddings | `BAAI/bge-small-en-v1.5` | Outperforms MiniLM on retrieval benchmarks, runs locally |
| Vector DB | `ChromaDB` (in-memory) | Zero cost, no server needed |
| Keyword Search | `rank_bm25` | Exact match for terminology, names, code |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local, significant precision improvement |
| LLM | `Llama 3.1 8B` via Groq | Free tier, fastest inference available |
| UI | `Streamlit` | Rapid prototyping, session state, streaming support |
| Orchestration | `LlamaIndex` | Semantic chunking, vector index management |

**Total infrastructure cost: $0**

---

##  Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ask-the-book.git
cd ask-the-book
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**4. Run the app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload any PDF, and start asking questions.

---

## 📁 Project Structure

```
ask-the-book/
│
├── app.py          # Streamlit UI — file upload, chat interface, session state
├── ingestion.py    # PDF parsing via unstructured, text cleaning, page grouping
├── embed.py        # BGE-small embeddings, ChromaDB setup, semantic chunking
├── retriever.py    # BM25 index, RRF fusion, cross-encoder reranking
├── llm.py          # Groq client, grounded system prompt, citation injection
├── e2e.py          # End-to-end latency benchmarks + tiered evaluation suite
│
├── .env            # GROQ_API_KEY (not committed)
├── .gitignore
└── requirements.txt
```

---

## Running the Evaluation Suite

```bash
python e2e.py
```

This runs:
1. **Phase 1** — Ingestion latency benchmark (how fast is the pipeline?)
2. **Phase 2** — Query latency benchmark on 3 test questions
3. **Tiered Evaluation** — 10 Q&A pairs across Easy / Medium / Hard / Negative tiers

Note: The evaluation includes 5-second delays between LLM calls to respect Groq's free-tier rate limit (6,000 TPM). Full eval takes approximately 3 minutes.

---

## ⚠️ Known Limitations

- **Single document** — One PDF per session. Multi-document support (ask across multiple books simultaneously) is planned for v2.
- **PDF only** — Currently accepts PDF files only. DOCX, EPUB, and plain text support planned for v2.
- **In-memory storage** — ChromaDB is ephemeral. Re-uploading the same PDF re-embeds from scratch (~18s). Persistent storage planned for v2.
- **English only** — Optimized for English text. Multilingual support planned for v2.
- **Groq free tier rate limits** — 6,000 tokens/minute. The evaluation suite includes automatic delays to handle this. Heavy usage may require Groq Dev tier.
- **Benchmarked on small corpus** — All benchmarks run on a 35-page document. Performance on 200+ page books with cross-chapter questions is the next test target.
- **String matching evaluation** — See Evaluation section above for reasoning.
