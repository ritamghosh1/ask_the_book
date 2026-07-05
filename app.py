import os
import tempfile
import uuid
import streamlit as st
from embed import VectorDBManager
from retriever import HybridRetriever
from llm import RAGGenerator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask the Book",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
# Polishes the source chunk expander, relevance badges, and suggested question buttons
st.markdown("""
<style>
    /* Suggested question buttons — make them feel like chips, not full-width buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
        text-align: left;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        color: #ccc;
        font-size: 13px;
        padding: 10px 14px;
        line-height: 1.4;
        white-space: normal;
        height: auto;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.25);
        color: #fff;
    }

    /* Relevance score badge inside expanders */
    .relevance-badge {
        display: inline-block;
        background: rgba(74,222,128,0.12);
        border: 1px solid rgba(74,222,128,0.3);
        color: #4ade80;
        font-size: 11px;
        font-family: monospace;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Summary box */
    .summary-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #4ade80;
        border-radius: 6px;
        padding: 14px 16px;
        font-size: 14px;
        line-height: 1.7;
        color: #ccc;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "retriever": None,
    "generator": None,
    "current_file": None,
    "session_id": str(uuid.uuid4()),   # unique collection per session (concurrency fix)
    "book_summary": None,
    "suggested_questions": [],
    "nodes": None,                     # keep nodes for score display
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helper: generate document summary ─────────────────────────────────────────
def generate_summary(generator, nodes):
    """
    Grab the first 4 chunks as a document preview and ask the LLM to summarise.
    Capped at 4 chunks so this stays fast and doesn't eat Groq's rate limit.
    """
    preview_nodes = nodes[:4]
    return generator.generate(
        "In exactly 3 sentences, summarise what this document is about. "
        "Be specific — mention the main topic, who it is for, and what it covers.",
        preview_nodes
    )


# ── Helper: generate suggested questions ──────────────────────────────────────
def generate_suggestions(generator, nodes):
    """
    Ask the LLM to produce 4 questions a reader might ask about this document.
    Returns a list of strings.
    """
    preview_nodes = nodes[:6]
    raw = generator.generate(
        "Generate exactly 4 interesting questions that a student or reader might ask "
        "about this document. Return ONLY a numbered list like:\n"
        "1. Question one\n2. Question two\n3. Question three\n4. Question four\n"
        "No preamble, no explanation, just the 4 questions.",
        preview_nodes
    )
    lines = [
        line.strip().lstrip("1234567890.)-").strip()
        for line in raw.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    return lines[:4]


# ── Helper: smart truncation at sentence boundary ─────────────────────────────
def smart_truncate(text, max_chars=500):
    """
    Truncate at the last sentence boundary before max_chars,
    rather than cutting mid-word.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Walk back to the last full stop, question mark, or exclamation
    for punct in (".", "?", "!"):
        pos = truncated.rfind(punct)
        if pos > max_chars // 2:
            return truncated[:pos + 1] + "  ···"
    # No sentence boundary found — fall back to word boundary
    pos = truncated.rfind(" ")
    return (truncated[:pos] if pos > 0 else truncated) + "···"


# ── Helper: normalise reranker score to 0–100% ────────────────────────────────
def score_to_pct(score):
    """
    Cross-encoder scores are unbounded logits, typically in range ~-15 to +10.
    Map to 0–100% with a sigmoid-like clamp so they display sensibly.
    """
    import math
    clamped = max(-10.0, min(10.0, float(score)))
    pct = int((clamped + 10) / 20 * 100)
    return max(1, min(99, pct))


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Ask the Book")
    st.caption("Upload any PDF and ask questions about it instantly.")
    st.divider()

    # ── File upload ──
    st.markdown("**1. Upload your document**")
    uploaded_file = st.file_uploader(
        "Supports PDF files",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        # New file — reset everything
        st.session_state.current_file   = uploaded_file.name
        st.session_state.messages       = []
        st.session_state.book_summary   = None
        st.session_state.suggested_questions = []
        st.session_state.session_id     = str(uuid.uuid4())  # fresh collection

        progress_bar = st.progress(10, text="Saving file temporarily...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            progress_bar.progress(25, text="Extracting and chunking text…")
            manager = VectorDBManager(
                collection_name=f"atb_{st.session_state.session_id}"
            )
            index, nodes = manager.process_and_store(tmp_path)
            st.session_state.nodes = nodes

            progress_bar.progress(60, text="Building Hybrid Search & BM25 indexes…")
            st.session_state.retriever = HybridRetriever(index, nodes)

            progress_bar.progress(75, text="Initialising Llama 3.1 LLM…")
            st.session_state.generator = RAGGenerator()

            progress_bar.progress(88, text="Generating document summary…")
            st.session_state.book_summary = generate_summary(
                st.session_state.generator, nodes
            )

            progress_bar.progress(95, text="Generating suggested questions…")
            st.session_state.suggested_questions = generate_suggestions(
                st.session_state.generator, nodes
            )

            progress_bar.progress(100, text="Ready!")
            st.success(f"✓ {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.session_state.retriever = None

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Document info panel (shown after upload) ──
    if st.session_state.current_file:
        st.divider()
        st.markdown("**Document**")
        st.caption(st.session_state.current_file)

        if st.session_state.nodes:
            st.caption(f"{len(st.session_state.nodes)} semantic chunks · "
                       f"{len(st.session_state.messages) // 2} questions asked")

    # ── Clear chat button ──
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Settings ──
    st.divider()
    st.markdown("**Settings**")
    top_k = st.slider(
        "Source chunks per answer",
        min_value=1, max_value=6, value=3,
        help="How many retrieved passages the LLM sees. More = richer answers, slower retrieval."
    )
    show_scores = st.toggle(
        "Show relevance scores",
        value=True,
        help="Display a relevance % on each source chunk."
    )


# ── Main area ──────────────────────────────────────────────────────────────────
if not st.session_state.retriever:
    # ── Welcome screen ──
    st.markdown("## Welcome to Ask the Book")
    st.markdown(
        "Upload any PDF in the sidebar — textbook, research paper, manual, "
        "lecture notes — and chat with it in plain English."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔍 Hybrid Search")
        st.caption("Combines semantic understanding with keyword matching for precise retrieval.")
    with col2:
        st.markdown("#### 🎯 Reranking")
        st.caption("A cross-encoder grades each result against your question before answering.")
    with col3:
        st.markdown("#### 📄 Citations")
        st.caption("Every answer includes the exact page numbers it was drawn from.")

    st.info("👈 Upload a PDF in the sidebar to begin.", icon="📂")

else:
    # ── Book summary (shown once, above the chat) ──
    if st.session_state.book_summary and not st.session_state.messages:
        st.markdown("#### 📖 Document Summary")
        st.markdown(
            f'<div class="summary-box">{st.session_state.book_summary}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

    # ── Suggested starter questions ──
    if st.session_state.suggested_questions and not st.session_state.messages:
        st.markdown("#### 💡 Suggested questions")
        st.caption("Click any question to ask it, or type your own below.")

        # Render in a 2-column grid
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}"):
                    st.session_state["pending_question"] = question
                    st.rerun()

        st.markdown("---")

    # ── Handle a suggestion click (set as pending, rerun picks it up) ──
    pending = st.session_state.pop("pending_question", None)

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Source chunks for assistant messages
            if msg["role"] == "assistant" and msg.get("chunks"):
                chunks  = msg["chunks"]
                scores  = msg.get("scores", [None] * len(chunks))

                with st.expander(f"View {len(chunks)} source chunks"):
                    for i, (node, score) in enumerate(zip(chunks, scores)):
                        page = node.metadata.get("page", "Unknown")

                        # Header row: chunk number + page + optional relevance badge
                        header = f"**Chunk {i+1}** · Page {page}"
                        if show_scores and score is not None:
                            pct = score_to_pct(score)
                            header += (
                                f' <span class="relevance-badge">'
                                f'relevance {pct}%</span>'
                            )
                        st.markdown(header, unsafe_allow_html=True)

                        # Smart truncation at sentence boundary
                        st.markdown(smart_truncate(node.text, max_chars=500))

                        if i < len(chunks) - 1:
                            st.divider()

    # ── Accept user input (typed or from suggestion click) ──
    prompt = st.chat_input("Ask a question about the uploaded book…") or pending

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching book…"):
                best_chunk_tuples = st.session_state.retriever.retrieve(
                    prompt, top_k=top_k
                )
                just_nodes  = [node  for node, score in best_chunk_tuples]
                just_scores = [score for node, score in best_chunk_tuples]

            # Streaming response
            stream = st.session_state.generator.generate_stream(prompt, just_nodes)
            answer = st.write_stream(stream)

            # Source chunks with relevance scores
            with st.expander(f"View {len(just_nodes)} source chunks"):
                for i, (node, score) in enumerate(zip(just_nodes, just_scores)):
                    page = node.metadata.get("page", "Unknown")

                    header = f"**Chunk {i+1}** · Page {page}"
                    if show_scores:
                        pct = score_to_pct(score)
                        header += (
                            f' <span class="relevance-badge">'
                            f'relevance {pct}%</span>'
                        )
                    st.markdown(header, unsafe_allow_html=True)
                    st.markdown(smart_truncate(node.text, max_chars=500))
                    if i < len(just_nodes) - 1:
                        st.divider()

        # Persist to session state
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "chunks":  just_nodes,
            "scores":  just_scores,
        })