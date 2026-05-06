# LedgerLens

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-red?style=for-the-badge)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ledgerlens--vectorless--rag-181717?style=for-the-badge&logo=github)](https://github.com/jayaprakash2207/ledgerlens-vectorless-rag)

> **Ask anything about any document. No embeddings. No vector database. Fully explainable.**

LedgerLens is a vectorless RAG system that lets you upload any PDF or HTML document and ask it any question. It retrieves relevant context using transparent rule-based scoring — no black-box embedding similarity — and sends that context to Gemini to produce a clear, sourced answer.

---

## What It Does

- Upload a **PDF or HTML** document
- Ask **any question** — summaries, facts, figures, comparisons, risks, definitions, anything
- Get a **Gemini-powered answer** backed by retrieved document chunks
- See **exactly what was retrieved** and why — no hidden vector magic
- Optionally get **risk analysis** when your question involves risks or red flags
- **Compare two documents** to detect what changed between versions

---

## Why Vectorless RAG?

Most RAG systems bury retrieval inside embedding models and cosine similarity. You get an answer but you can't see the path. LedgerLens takes the opposite approach.

| Typical RAG | LedgerLens |
|---|---|
| Embedding model required | No embeddings |
| Vector database required | Plain Parquet files |
| Retrieval is a black box | Every score is visible |
| Hard to debug | Full fetch trace in the UI |
| Complex setup | `pip install -r requirements.txt` |

Instead of storing vectors, LedgerLens stores:

1. Cleaned plain text
2. Named document sections
3. Parquet-backed chunk store
4. Retrieval scores you can inspect directly in the UI

---

## How It Works

```
Document (PDF or HTML)
        ↓
   Parser cleans text
        ↓
 Section Splitter detects named buckets
 (risk_factors · mda · notes · legal · auditor_notes)
        ↓
  Parquet Store saves documents + chunks
        ↓
  Chunk Retriever scores chunks
  (keyword overlap + section priority)
        ↓
   Top chunks sent to Gemini
        ↓
  Answer + Sources returned to you
```

---

## Features

### Ask Any Question
Upload a document and type any question. LedgerLens retrieves the most relevant chunks from the document and lets Gemini answer directly from that context.

```
"What were the total revenues last year?"
"Who are the key executives mentioned?"
"What legal proceedings are described?"
"Summarize the main findings of this report."
"What risks does the company highlight?"
```

### Automatic Risk Analysis
When your question contains risk-related terms, the app automatically runs a deeper risk analysis pass and returns:

- Top risks identified
- Risk categories breakdown
- Red flags detected
- Highlighted risky sentences
- Confidence score

### Document Comparison
Upload two versions of a document (old and new) and detect:

- New risks that appeared
- Risks that were removed
- Tone shifts
- Risk intensity changes
- New red flags

### Vectorless RAG Visualization
The UI includes a full retrieval walkthrough panel showing:

- **Retrieval Flow Diagram** — the 5-step pipeline from query to answer
- **Fetch Trace** — query tokens, prioritized sections, and scoring breakdown
- **Ranked Section Cards** — every section scored with progress bars
- **Worked Example** — step-by-step trace of exactly how your answer was found
- **Parquet Memory Shape** — the live data structure behind the answer

### Parquet Data Store
Every uploaded document is persisted to local Parquet files. The Data Store tab lets you browse stored documents, inspect chunks, and view raw text.

---

## Pipeline in Detail

### Step 1 — Parse
The parser reads your PDF or HTML file and extracts clean plain text, stripping markup and formatting noise.

### Step 2 — Split into Sections
The section splitter detects named buckets in the text:

| Section | What It Captures |
|---|---|
| `risk_factors` | Disclosed risks and uncertainties |
| `mda` | Management discussion and analysis |
| `notes` | Financial notes and disclosures |
| `legal` | Legal proceedings |
| `auditor_notes` | Auditor commentary |

### Step 3 — Store in Parquet
Documents and chunks are saved to local Parquet files via PyArrow and Pandas. No database required.

### Step 4 — Retrieve
The chunk retriever tokenizes your question and scores every chunk using:

- **Section priority score** — sections more likely to contain the answer rank higher
- **Keyword overlap score** — chunks where more question words appear rank higher
- **Red-flag bonus** — chunks containing risk signal words get a boost on risk queries

Final score = base score + keyword score + red-flag bonus

### Step 5 — Answer with Gemini
The top-scoring chunks are sent to Gemini as context. Gemini returns a structured JSON response with an answer and supporting source quotes.

---

## Project Structure

```
.
├── analysis/
│   ├── comparison_engine.py   # Compare two document versions
│   ├── qa_engine.py           # General Q&A over retrieved chunks
│   └── risk_analyzer.py       # Risk-specific deep analysis
├── data/
│   └── apple_2023.html        # Sample SEC filing for testing
├── llm/
│   └── gemini_interface.py    # Gemini API client
├── parser/
│   ├── html_parser.py         # HTML → plain text
│   └── pdf_parser.py          # PDF → plain text
├── retrieval/
│   ├── chunk_retriever.py     # Parquet chunk scoring and ranking
│   └── rule_engine.py         # Rule-based section retrieval + explain
├── segmentation/
│   └── section_splitter.py    # Named section detection
├── storage/
│   └── parquet_store.py       # Parquet persistence layer
├── tests/
│   └── test_section_splitter.py
├── ui/
│   └── app.py                 # Streamlit app (3 tabs + RAG visualization)
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## Architecture

| Layer | File(s) | What It Does |
|---|---|---|
| Parsing | `parser/html_parser.py`, `parser/pdf_parser.py` | Extract clean text from HTML and PDF |
| Segmentation | `segmentation/section_splitter.py` | Detect and name document sections |
| Storage | `storage/parquet_store.py` | Persist documents and chunks to Parquet |
| Retrieval | `retrieval/chunk_retriever.py`, `retrieval/rule_engine.py` | Score and rank chunks with rule-based logic |
| LLM | `llm/gemini_interface.py` | Send retrieved context to Gemini and parse response |
| Analysis | `analysis/qa_engine.py`, `analysis/risk_analyzer.py` | Q&A and optional risk deep-dive |
| Comparison | `analysis/comparison_engine.py` | Diff two document versions |
| UI | `ui/app.py` | Streamlit interface with 3 tabs + vectorless RAG panel |

---

## Tech Stack

- **Python 3.x**
- **Streamlit** — UI with tabs and expanders
- **Gemini 2.5 Flash** — LLM for answering questions
- **BeautifulSoup4** — HTML parsing
- **pdfplumber + PyMuPDF** — PDF parsing
- **Pandas + PyArrow** — Parquet storage
- **Requests** — Gemini API calls

---

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/jayaprakash2207/ledgerlens-vectorless-rag.git
cd ledgerlens-vectorless-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Gemini API Key

You need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

**Option 1 — Paste it in the app**

The Streamlit UI has a secure password field at the top. Paste your key there before running analysis.

**Option 2 — Set an environment variable**

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

---

## Run

### Streamlit app (recommended)

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

Then:

1. Paste your Gemini API key
2. Upload a PDF or HTML document
3. Type any question
4. Click **Analyze Report**
5. Open the **How Vectorless RAG Stores This Report** panel to inspect every retrieval step

### CLI mode

```powershell
.\.venv\Scripts\python.exe main.py --file data\apple_2023.html --query "What are the main risks?" --model gemini-2.5-flash
```

Available arguments:

| Flag | Default | Description |
|---|---|---|
| `--file` | `data/apple_2023.html` | Path to a PDF or HTML document |
| `--query` | `What are the main risks?` | Your question |
| `--model` | `gemini-2.5-flash` | Gemini model name |

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Example Questions You Can Ask

```
What is the total revenue reported?
Who signed off on the audit?
What are the main business risks?
How many employees does the company have?
What legal cases are currently pending?
Summarize the management discussion section.
What changed in risk disclosures between the two reports?
Are there any red flags in the financial notes?
```

---

## Limitations

- Retrieval is rule-based, not semantic — exact keyword overlap matters
- Section detection is tuned for structured report-style documents
- Answer quality depends on the parsed text quality and Gemini response
- Very short or scanned PDFs may parse poorly

---

## Security

Do not commit API keys to source control. Use environment variables or the in-app key field. If a key has been exposed, rotate it immediately at [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

All rights reserved. Anyone who wants to use, modify, redistribute, or deploy this project must obtain prior written permission from the owner. See [LICENSE](LICENSE).
