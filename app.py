"""CS4106 Session 9: run with `streamlit run app.py`."""
import hashlib
import os

import streamlit as st
from groq import Groq, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
from sentence_transformers import SentenceTransformer

from rag_core import (DEFAULT_MODEL, EMBEDDING_MODEL, VectorStore,
                      read_document, make_chunks, answer_from_hits)

st.set_page_config(page_title="CS4106 | Document Q&A", page_icon="📚", layout="wide")
st.title("📚 Document Q&A")
st.caption("CS4106 · Session 9 · Groq + MiniLM + FAISS + Streamlit")
st.write("Upload your documents, build a knowledge base, and ask a question with traceable sources.")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(os.environ.get("EMBEDDING_MODEL_PATH", EMBEDDING_MODEL), device="cpu")


def clear_knowledge():
    for key in ("store", "signature", "result", "built_files"):
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("Settings")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if api_key:
        st.success("Server API key configured")
    else:
        api_key = st.text_input("Groq API key", type="password").strip()
    model = st.text_input("Groq model ID", value=os.environ.get("GROQ_MODEL", DEFAULT_MODEL)).strip()
    st.caption("Model unavailable? Copy a chat model ID from console.groq.com/docs/models.")
    top_k = st.slider("Chunks to retrieve (k)", 1, 5, 3)
    size = st.slider("Chunk size (words)", 50, 200, 150, 10)
    overlap = st.slider("Overlap (words)", 0, 40, 30, 5)
    st.button("Clear knowledge base", on_click=clear_knowledge, use_container_width=True)
    st.caption("Changing files or chunk settings requires rebuilding. Uploads stay in session memory; retrieved text and your question are sent to Groq when you request an answer.")

files = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"],
                         accept_multiple_files=True)
payloads = [(f.name, f.getvalue()) for f in files]
signature = (tuple((name, hashlib.sha256(data).hexdigest()) for name, data in payloads), size, overlap)
current = st.session_state.get("signature") == signature

if st.button("Build knowledge base", type="primary", disabled=not files):
    # Clear stale answers and indexes before attempting a replacement.
    clear_knowledge()
    passages, valid_names = [], []
    for name, data in payloads:
        if len(data) > 20 * 1024 * 1024:
            st.warning(f"Skipped {name}: maximum file size is 20 MB.")
            continue
        try:
            passages.extend(read_document(name, data))
            valid_names.append(name)
        except Exception:
            st.warning(f"Could not read {name}. Use an unlocked, text-based PDF, a valid DOCX, or a UTF-8 TXT file.")
    if passages:
        try:
            with st.spinner("Embedding passages and building the index..."):
                chunks = make_chunks(passages, size, overlap)
                if len(chunks) > 5000:
                    raise ValueError("Too many chunks")
                store = VectorStore(load_embedder(), chunks)
            st.session_state.update(store=store, signature=signature, built_files=valid_names)
            current = True
        except Exception:
            st.error("Could not build the index. Check the model download connection, or use smaller documents (maximum 5,000 chunks).")
    else:
        st.error("No readable text was found. Scanned PDFs require OCR before uploading.")

store = st.session_state.get("store")
if store and current:
    st.success(f"Ready: {len(store.chunks)} chunks from {len(st.session_state.built_files)} file(s).")
elif store:
    st.warning("Your files or chunk settings changed. Rebuild the knowledge base before asking.")

with st.form("question_form"):
    question = st.text_input("Ask a question about your documents", max_chars=2000,
                             placeholder="What does RAG stand for?")
    retrieve = st.form_submit_button("Preview sources")
    answer = st.form_submit_button("Answer", type="primary")

if retrieve or answer:
    st.session_state.pop("result", None)
    if not store or not current:
        st.error("Upload documents and build the knowledge base first.")
    elif not question.strip():
        st.error("Enter a question.")
    elif answer and not api_key:
        st.error("Enter your Groq API key in the sidebar, or configure GROQ_API_KEY on the server.")
    elif answer and not model:
        st.error("Enter a Groq model ID.")
    else:
        try:
            with st.spinner("Finding relevant sources..."):
                hits = store.search(question.strip(), top_k)
            text = None
            if answer:
                with st.spinner("Generating an answer from your sources..."):
                    with Groq(api_key=api_key, timeout=45.0, max_retries=1) as client:
                        text = answer_from_hits(client, question.strip(), hits, model)
            st.session_state.result = {"question": question.strip(), "text": text, "hits": hits}
        except AuthenticationError:
            st.error("Groq rejected the API key. Check it privately and try again.")
        except RateLimitError:
            st.error("Groq rate limit reached. Wait briefly, reduce k, or check your account quota.")
        except APIConnectionError:
            st.error("Could not connect to Groq. Check your connection and retry.")
        except APIStatusError as exc:
            st.error(f"Groq returned HTTP {exc.status_code}. Check your model ID, permissions and quota.")
        except Exception:
            st.error("Could not complete the request. Rebuild the index, or try a shorter question or another model.")

result = st.session_state.get("result")
if result and current:
    st.subheader("Answer" if result["text"] else "Source preview")
    st.caption(result["question"])
    if result["text"]:
        st.write(result["text"])
    else:
        st.info("Retrieval only. No model answer has been generated.")
    with st.expander("Sources (retrieved chunks)", expanded=True):
        for number, (chunk, score) in enumerate(result["hits"], 1):
            st.text(f"[{number}] {chunk.filename} · {chunk.location} · chunk {chunk.number}")
            st.caption(f"Cosine similarity: {score:.3f} (not an answer-confidence score)")
            st.write(chunk.text)
            st.divider()
