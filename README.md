# CS4106 Document Q&A

A Streamlit Retrieval-Augmented Generation app using Groq, MiniLM embeddings,
and FAISS. Upload PDF, DOCX or TXT documents, build a knowledge base, and ask
questions with visible source passages and page references.

## Run locally

Use Python 3.12:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your Groq API key in the password field. It is kept in session memory and
is not committed to the repository. Retrieved passages and the question are sent
to Groq when you request an answer.

## Deploy on Streamlit Community Cloud

Select this repository, branch `main`, entrypoint `app.py`, and Python 3.12.
No server API key is required to deploy: visitors can enter their own Groq key.
For an owner-managed key, use a root-level `GROQ_API_KEY` in the Cloud secrets
settings; never commit secrets to GitHub.

The first build of a knowledge base downloads `sentence-transformers/all-MiniLM-L6-v2`.
The default generation model is `openai/gpt-oss-20b`, served by Groq.

## Demo

Upload `Demo_Library_Handbook.pdf` (fictional demonstration data), then build.
Use chunk size 150, overlap 30, and k=2.

- What are the library opening hours on weekdays? (8 AM to 8 PM.)
- How many books can I borrow, and for how long? (Four; fourteen days.)
- How long can I borrow a book and reserve a quiet study room? (Fourteen days; two hours.)
- How much is the late-return fine per day? (Not provided: the app should abstain.)

Inspect the Sources panel to verify the answer. Similarity is not confidence;
a context-only prompt does not guarantee that all generated claims are supported.
Scanned PDFs require OCR before upload.
