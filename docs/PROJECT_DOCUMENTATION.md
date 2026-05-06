# LedgerLens — Project Documentation

**Document Version:** 1.0  
**Repository:** [ledgerlens-vectorless-rag](https://github.com/jayaprakash2207/ledgerlens-vectorless-rag)  
**License:** All Rights Reserved  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Module Reference](#5-module-reference)
   - 5.1 [Document Parsing](#51-document-parsing)
   - 5.2 [Section Segmentation](#52-section-segmentation)
   - 5.3 [Parquet Storage Layer](#53-parquet-storage-layer)
   - 5.4 [Retrieval Engine](#54-retrieval-engine)
   - 5.5 [LLM Interface (Gemini)](#55-llm-interface-gemini)
   - 5.6 [Analysis Layer](#56-analysis-layer)
   - 5.7 [Web Interface (Streamlit)](#57-web-interface-streamlit)
   - 5.8 [CLI Entry Point](#58-cli-entry-point)
6. [Data Flow — End to End](#6-data-flow--end-to-end)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Technology Stack](#8-technology-stack)
9. [Project Structure](#9-project-structure)
10. [Setup and Installation](#10-setup-and-installation)
11. [Running the Application](#11-running-the-application)
12. [Testing](#12-testing)
13. [Retrieval Scoring Formula](#13-retrieval-scoring-formula)
14. [Feature Summary](#14-feature-summary)
15. [Limitations and Known Constraints](#15-limitations-and-known-constraints)
16. [Security Considerations](#16-security-considerations)

---

## 1. Executive Summary

**LedgerLens** is a financial document intelligence platform built on a *vectorless Retrieval-Augmented Generation (RAG)* architecture. It enables users to upload any PDF or HTML financial filing—such as SEC 10-K annual reports—and ask free-form questions about the document. The system retrieves the most relevant sections of the document using transparent, rule-based scoring and passes that retrieved context to Google's Gemini large language model (LLM) to produce accurate, sourced answers.

Unlike conventional RAG systems that rely on embedding models and vector databases, LedgerLens performs all retrieval using keyword scoring and rule-based prioritization. Every retrieval decision is visible and explainable, making the system auditable and debuggable without any black-box components.

The platform also performs automated risk analysis on financial documents, identifying risk statements, categorizing them, detecting red flags (e.g., "going concern", "material weakness"), and comparing two document versions to surface changes in risk posture.

---

## 2. Problem Statement

Financial analysts and compliance teams regularly review large, complex documents such as SEC annual reports (10-K filings), prospectuses, audit reports, and earnings disclosures. These documents can be hundreds of pages long. Key challenges include:

- **Information overload**: Finding specific facts, risk disclosures, or compliance language inside lengthy documents is time-consuming.
- **Opaque AI tooling**: Most AI-powered document QA systems use embedding-based retrieval that cannot be audited—users cannot verify why a particular passage was retrieved.
- **Missing risk visibility**: Identifying and tracking risk language changes across document versions requires significant manual effort.
- **Infrastructure complexity**: Vector databases, embedding servers, and similarity indexes are complex to deploy and maintain.

LedgerLens addresses all of these problems with a simple, explainable, and deployable solution.

---

## 3. Solution Overview

LedgerLens processes a document through a five-stage pipeline:

| Stage | What Happens |
|---|---|
| **Parse** | PDF or HTML is converted to clean plain text |
| **Segment** | Text is split into named sections (risk factors, MD&A, notes, legal, auditor notes) |
| **Store** | Cleaned text and chunked sections are persisted to local Parquet files |
| **Retrieve** | User query is matched to the most relevant sections using rule-based scoring |
| **Answer** | Top retrieved sections are sent to Gemini, which returns a structured answer |

The system exposes three interfaces:
- A **Streamlit web application** with tabs for single-document analysis, document comparison, and a data store browser.
- A **command-line interface (CLI)** for scripted or automated use.
- A **Python API** through which each module can be used programmatically.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│              Streamlit App (ui/app.py)  /  CLI (main.py)             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │        Document Parsing        │
                │  html_parser / pdf_parser      │
                └───────────────┬───────────────┘
                                │  cleaned plain text
                ┌───────────────▼───────────────┐
                │      Section Segmentation      │
                │    section_splitter.py         │
                └───────────────┬───────────────┘
                                │  named section dict
                ┌───────────────▼───────────────┐
                │       Parquet Storage          │
                │    parquet_store.py            │
                │  documents.parquet             │
                │  chunks.parquet                │
                └───────┬───────────────┬────────┘
                        │               │
           ┌────────────▼──┐     ┌──────▼─────────────┐
           │ Rule-Based     │     │  Chunk Retriever    │
           │ Retriever      │     │  chunk_retriever.py │
           │ rule_engine.py │     └──────┬──────────────┘
           └────────┬───────┘            │
                    │                    │
                    └────────┬───────────┘
                             │  top scored chunks / sections
                ┌────────────▼────────────┐
                │     Gemini LLM Client   │
                │   gemini_interface.py   │
                └────────────┬────────────┘
                             │  structured JSON response
          ┌──────────────────┼──────────────────────┐
          │                  │                       │
   ┌──────▼──────┐  ┌────────▼───────┐  ┌───────────▼──────┐
   │  QA Engine  │  │ Risk Analyzer  │  │ Comparison Engine│
   │ qa_engine.py│  │risk_analyzer.py│  │comparison_engine │
   └─────────────┘  └────────────────┘  └──────────────────┘
```

---

## 5. Module Reference

### 5.1 Document Parsing

**Files:** `parser/html_parser.py`, `parser/pdf_parser.py`

#### HTML Parser — `SEC10KHtmlParser`

Designed specifically for SEC 10-K HTML filings. The parser:

1. Reads the HTML file from disk with UTF-8 encoding (ignoring decode errors).
2. Removes noise tags: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<noscript>`, `<svg>`, elements with `aria-hidden="true"`, and elements with `display:none`.
3. Extracts text from heading tags (`h1`–`h6`) and paragraph tags (`p`), classifying each block as either `"heading"` or `"paragraph"`.
4. Falls back to full-page `get_text()` line splitting when no structured tags are found.
5. Returns a single normalized plain-text string.

**Key class / function:**
```
SEC10KHtmlParser.parse(file_path, return_blocks=False) → str | List[HtmlBlock]
parse_html(file_path) → str   # convenience wrapper
```

#### PDF Parser — `PDFParser`

Uses `pdfplumber` as the primary PDF reader, with `PyMuPDF` (fitz) as an automatic fallback.

1. Iterates over every page, extracting page text.
2. Optionally extracts tables (`extract_tables=True`).
3. Normalizes whitespace: collapses runs of spaces/tabs and triples of newlines.
4. Returns a `ParsedDocument` dataclass: `{ text, pages, tables }`.

**Key class:**
```
PDFParser(extract_tables=False).parse(file_path) → ParsedDocument
```

---

### 5.2 Section Segmentation

**File:** `segmentation/section_splitter.py`

`SectionSplitter` scans the cleaned plain text line by line, matching each line against a set of regular-expression patterns that correspond to well-known SEC filing section headers. The following named sections are recognized:

| Section Key | Patterns Matched |
|---|---|
| `risk_factors` | "Item 1A", "Item 1A. Risk Factors", "Risk Factors" |
| `mda` | "Item 7", "Item 7. Management's Discussion and Analysis" |
| `notes` | "Notes to Financial Statements", "Item 8", "Item 8. Financial Statements" |
| `legal` | "Item 3", "Item 3. Legal Proceedings", "Legal Proceedings" |
| `auditor_notes` | "Report of Independent Registered Public Accounting Firm", "Independent Auditor's Report" |

The splitter builds a dictionary `{ section_name: section_text }`. If no named sections are detected (e.g., for non-standard documents), the first 5,000 characters are stored under `"other"`.

**Key class / function:**
```
SectionSplitter.split(text) → Dict[str, str]
split_sections(text) → Dict[str, str]   # convenience wrapper
```

---

### 5.3 Parquet Storage Layer

**File:** `storage/parquet_store.py`

`ParquetStore` persists documents and their chunks to two local Parquet files under `data/store/`:

| File | Contents |
|---|---|
| `documents.parquet` | One row per document: `doc_id`, `source_name`, `file_type`, `created_at`, `text_len`, `raw_text` |
| `chunks.parquet` | One row per chunk: `doc_id`, `chunk_id`, `section`, `chunk_index`, `text` |

**Document ID** is a 16-character SHA-256 hash of `(source_name + raw_text[:5000])`, ensuring that re-uploading the same file returns the same ID without duplicating records.

**Chunking strategy:**  
Text is split at sentence boundaries (`.` delimiter). Each chunk is capped at **1,200 characters** with a **200-character overlap** from the previous chunk's tail, preserving context across boundaries.

**Key methods:**
```
save_document(source_name, file_type, raw_text, sections) → StoredDocument
load_chunks(doc_id) → List[Dict]
list_documents() → List[Dict]
load_document(doc_id) → Optional[Dict]
```

---

### 5.4 Retrieval Engine

Two complementary retrievers are used for different purposes.

#### Rule-Based Retriever — `RuleBasedRetriever`

**File:** `retrieval/rule_engine.py`

Used for **section-level retrieval** (retrieving entire named document sections). Scoring formula:

```
total_score = base_score + keyword_score + red_flag_score
```

| Component | Value |
|---|---|
| `base_score` | 1.0 if section is in a query-prioritized list; 0.5 otherwise |
| `keyword_score` | (matched query tokens in section) / (total query tokens) |
| `red_flag_score` | 0.2 bonus if query contains red-flag keywords (restatement, material weakness, going concern, default) |

**Section prioritization rules:**
- Risk-related queries → prioritize `risk_factors`, `mda`
- Queries containing "legal" → prioritize `legal`
- Queries containing "auditor" → prioritize `auditor_notes`
- Queries containing "note/notes" → prioritize `notes`
- Default → all five named sections prioritized equally

The `explain_retrieval()` method returns full scoring detail for every section, enabling the UI to render the "Fetch Trace" visualization.

**Key methods:**
```
retrieve(query, sections, max_sections=2) → List[RetrievalResult]
explain_retrieval(query, sections) → RetrievalExplanation
```

#### Chunk Retriever — `ChunkRetriever`

**File:** `retrieval/chunk_retriever.py`

Used for **chunk-level retrieval** (scoring individual text chunks stored in Parquet). Scoring formula:

```
total_score = base_score + keyword_score + length_score * 0.1
```

| Component | Value |
|---|---|
| `base_score` | 1.0 if chunk's section is a priority section; 0.6 otherwise |
| `keyword_score` | (matched query tokens in chunk text) / (total query tokens) |
| `length_score` | min(chunk_length / 1200, 1.0) — slight bonus for longer chunks |

Returns the top-k scored chunks as `ChunkResult` objects (containing `chunk_id`, `score`, `section`, `text`).

**Key methods:**
```
retrieve(query, chunks, top_k=5) → List[ChunkResult]
explain(query, chunks, top_k=5) → ChunkExplanation
```

---

### 5.5 LLM Interface (Gemini)

**File:** `llm/gemini_interface.py`

`GeminiClient` wraps the Google Generative Language REST API (`/v1beta/models/{model}:generateContent`).

- API key is read from the `GEMINI_API_KEY` environment variable or passed directly to the constructor.
- All requests use a `temperature=0.2` default for consistent, factual responses.
- On HTTP errors or empty responses, `last_error` is set and `None` is returned so callers can surface the error message.
- Default model: `gemini-2.5-flash`.

**Key method:**
```
GeminiClient(model, api_key).generate(prompt, temperature=0.2) → Optional[str]
```

---

### 5.6 Analysis Layer

#### QA Engine — `QAEngine`

**File:** `analysis/qa_engine.py`

Answers a free-form question using retrieved document chunks as context.

- Sends up to 6 chunks to Gemini with a financial analyst system prompt.
- Instructs Gemini to return a JSON object with keys `answer` (string) and `sources` (array of short quotes).
- Extracts the first `{…}` JSON block from the raw LLM response with a regex-based fallback.
- Falls back to returning the raw response text if JSON parsing fails.

**Key method:**
```
QAEngine(llm_client).answer(question, chunks) → QAResult
```

`QAResult` fields: `answer: str`, `sources: List[str]`

#### Risk Analyzer — `RiskAnalyzer`

**File:** `analysis/risk_analyzer.py`

Performs deep risk extraction on a block of document text.

**LLM mode** (default when `require_llm=True`):  
Sends up to 12,000 characters of text to Gemini with a risk-analyst prompt. Expects JSON with keys: `top_risks`, `risk_categories` (Financial / Operational / Market / Regulatory), `red_flags`, `confidence_score`, `summary`.

**Heuristic fallback** (used when no LLM is configured):  
- Splits text into sentences; flags sentences containing risk signal words (`risk`, `uncertain`, `volatility`, `may`, `could`, `potential`, `threat`).
- Categorizes risks by keyword matching (e.g., "liquidity" → Financial; "cyber" → Operational).
- Detects red flags via regex for: `material weakness`, `going concern`, `restatement`, `significant doubt`.
- Scores confidence: `min(100, 40 + risk_sentence_count + red_flag_count * 10)`.

**Key method:**
```
RiskAnalyzer(llm_client, require_llm).analyze(text) → RiskAnalysisResult
```

`RiskAnalysisResult` fields: `top_risks`, `risk_categories`, `red_flags`, `confidence_score`, `summary`, `risky_sentences`

#### Risk Comparison Engine — `RiskComparisonEngine`

**File:** `analysis/comparison_engine.py`

Compares two document versions to surface risk changes.

| Analysis | Method |
|---|---|
| New risks | Risk sentences in the new document not matched (by token overlap ≥ 55% or Jaccard similarity ≥ 0.8) in the old document |
| Removed risks | Risk sentences in the old document with no match in the new document |
| Tone change | Counts cautious vs. positive words in the MD&A sections; returns "more optimistic", "more cautious", or "neutral" |
| Risk intensity change | Counts risk keyword hits and warning phrase occurrences; returns "increased", "decreased", or "stable" |
| New red flags | Warning phrases present in the new document but absent in the old document |

**Key method:**
```
RiskComparisonEngine().compare_reports(report_old, report_new) → ComparisonResult
compare_reports(report_old, report_new) → ComparisonResult  # convenience wrapper
```

---

### 5.7 Web Interface (Streamlit)

**File:** `ui/app.py`

A wide-layout Streamlit application with three tabs:

#### Tab 1 — Single Report Analysis

1. User uploads a PDF or HTML file.
2. User types any question.
3. On "Analyze Report":
   - Document is parsed and split into sections.
   - Sections and chunks are saved to Parquet.
   - `ChunkRetriever` scores all chunks; top 5 are passed to `QAEngine`.
   - `RuleBasedRetriever` retrieves the top 2 sections.
   - If the query is risk-related, `RiskAnalyzer` is run on the retrieved text.
4. Results displayed: answer, sources, risk breakdown (top risks, categories, red flags, risky sentences), confidence score.
5. A collapsible **"How Vectorless RAG Stores This Report"** panel renders the full retrieval pipeline as a visual flow diagram, fetch trace, ranked section cards, worked example, section breakdown, and Parquet memory shape.

#### Tab 2 — Compare Reports

1. User uploads two documents (old and new version).
2. On "Compare Reports":
   - Both documents are parsed and segmented.
   - `RiskComparisonEngine` compares the `risk_factors` and `mda` sections.
3. Results displayed: new risks, removed risks, tone change, risk intensity change, new red flags, highlighted sentences, confidence score.
4. JSON export button available.

#### Tab 3 — Data Store

- Lists all previously analyzed documents stored in Parquet.
- Shows document ID, source name, file type, creation timestamp, and text length.
- Allows inspection of individual document chunks by selecting from a dropdown.

**Global UI controls** (above the tabs):
- Gemini model name input (default: `gemini-2.5-flash`)
- Gemini API key password field (or read from `GEMINI_API_KEY` env var)

---

### 5.8 CLI Entry Point

**File:** `main.py`

Provides a command-line interface that runs the same pipeline as the web app.

```
python main.py [--file PATH] [--query "QUESTION"] [--model MODEL_NAME]
```

| Flag | Default | Description |
|---|---|---|
| `--file` | `data/apple_2023.html` | Path to a PDF or HTML document |
| `--query` | `What are the main risks?` | Free-form question |
| `--model` | `gemini-2.5-flash` | Gemini model name |

Outputs: section preview, chunk retrieval results, QA answer, and (when the query contains "risk") full risk analysis as JSON printed to stdout.

---

## 6. Data Flow — End to End

```
User uploads a document + types a question
          │
          ▼
┌──────────────────────────┐
│  1. PARSE                │
│  html_parser / pdf_parser│
│  → cleaned plain text    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  2. SEGMENT              │
│  section_splitter        │
│  → { risk_factors,       │
│      mda, notes,         │
│      legal,              │
│      auditor_notes,      │
│      other }             │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  3. STORE                │
│  parquet_store           │
│  → documents.parquet     │
│  → chunks.parquet        │
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐  ┌───────────────────┐
│  Chunk   │  │  Rule-Based       │
│ Retriever│  │  Retriever        │
│ (Parquet │  │  (Section-level   │
│  chunks) │  │   scoring)        │
└────┬─────┘  └────────┬──────────┘
     │                 │
     └────────┬────────┘
              │ top scored context
              ▼
┌──────────────────────────┐
│  4. LLM (Gemini)         │
│  gemini_interface        │
│  → structured JSON       │
└────────────┬─────────────┘
             │
      ┌──────┴──────────┐
      │                 │
      ▼                 ▼
┌──────────┐   ┌──────────────────┐
│ QAEngine │   │  RiskAnalyzer    │
│ answer + │   │  top_risks,      │
│ sources  │   │  categories,     │
└──────────┘   │  red_flags,      │
               │  confidence,     │
               │  summary         │
               └──────────────────┘
                        │
                        ▼
              Rendered in Streamlit UI
              or printed as JSON (CLI)
```

---

## 7. Key Design Decisions

### Vectorless RAG

Standard RAG systems embed all document chunks as dense vectors and retrieve by cosine similarity. LedgerLens deliberately avoids this approach:

| Dimension | Standard RAG | LedgerLens |
|---|---|---|
| Retrieval method | Embedding + vector similarity | Keyword overlap + rule scoring |
| Infrastructure | Vector database (Chroma, Pinecone, Weaviate) | Local Parquet files |
| Explainability | Black box | Every score component is visible |
| Setup complexity | High | `pip install -r requirements.txt` |
| Offline capability | Requires embedding model | Fully offline except for Gemini calls |

The trade-off is that LedgerLens is less effective on paraphrased queries where the exact vocabulary doesn't overlap with the document text. However, for structured financial documents—where section headers and risk terminology are predictable—rule-based retrieval is highly reliable.

### Parquet as Storage

Parquet was chosen over SQLite or a document database because:
- Zero server setup required.
- Columnar format is efficient for appending and deduplication.
- Directly readable by Pandas for browsing in the UI.
- Portable — the `data/store/` directory can be copied to another machine.

### LLM as Answering Layer Only

Gemini is used exclusively for natural language generation, not for retrieval. This means:
- Retrieval is fully deterministic and reproducible.
- LLM costs are bounded to the generation step only.
- The system degrades gracefully — if the API key is missing, parsing, retrieval, and storage still function.

---

## 8. Technology Stack

| Component | Library / Service | Purpose |
|---|---|---|
| Web UI | Streamlit | Interactive browser-based interface |
| HTML Parsing | BeautifulSoup4 | Extract text from HTML filings |
| PDF Parsing (primary) | pdfplumber | Page-by-page text + table extraction |
| PDF Parsing (fallback) | PyMuPDF (fitz) | Fallback for PDFs pdfplumber cannot open |
| Storage | Pandas + PyArrow | Parquet read/write |
| LLM | Google Gemini API (via `requests`) | Answer generation and risk extraction |
| Environment | python-dotenv | Load `.env` files for API keys |
| Language | Python 3.x | — |

---

## 9. Project Structure

```
ledgerlens-vectorless-rag/
│
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── .gitignore
├── LICENSE                        # All rights reserved
├── CONTRIBUTING.md
├── README.md
│
├── analysis/
│   ├── __init__.py
│   ├── comparison_engine.py       # Document-to-document risk comparison
│   ├── qa_engine.py               # Free-form Q&A using retrieved chunks
│   └── risk_analyzer.py           # Risk extraction and categorization
│
├── data/
│   ├── apple_2023.html            # Sample SEC 10-K filing (Apple Inc., FY2023)
│   └── store/                     # Auto-created Parquet storage directory
│       ├── documents.parquet      # Document metadata + raw text
│       └── chunks.parquet         # Chunked text indexed by doc_id
│
├── docs/
│   └── screenshots/               # UI screenshots
│
├── llm/
│   ├── __init__.py
│   ├── gemini_interface.py        # Google Gemini REST API client
│   └── ollama_interface.py        # Unused local LLM interface (legacy)
│
├── parser/
│   ├── __init__.py
│   ├── html_parser.py             # SEC 10-K HTML → plain text
│   └── pdf_parser.py              # PDF → plain text (pdfplumber + PyMuPDF)
│
├── retrieval/
│   ├── __init__.py
│   ├── chunk_retriever.py         # Parquet-chunk scoring and ranking
│   └── rule_engine.py             # Section-level rule-based retrieval + explain
│
├── segmentation/
│   ├── __init__.py
│   └── section_splitter.py        # Named section detection from plain text
│
├── storage/
│   ├── __init__.py
│   └── parquet_store.py           # Parquet persistence, chunking, deduplication
│
├── tests/
│   └── test_section_splitter.py   # Unit tests for section splitter
│
└── ui/
    ├── __init__.py
    └── app.py                     # Streamlit app (3 tabs + vectorless RAG panel)
```

---

## 10. Setup and Installation

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key (free tier available at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/jayaprakash2207/ledgerlens-vectorless-rag.git
cd ledgerlens-vectorless-rag
```

**2. Create and activate a virtual environment**

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure the Gemini API key**

Option A — Environment variable (recommended for CLI use):
```bash
# Windows
$env:GEMINI_API_KEY = "your_key_here"

# macOS / Linux
export GEMINI_API_KEY="your_key_here"
```

Option B — `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

Option C — Paste directly into the Streamlit UI at runtime (no file needed).

---

## 11. Running the Application

### Streamlit Web Application (recommended)

```bash
# Windows
.\.venv\Scripts\python.exe -m streamlit run ui\app.py

# macOS / Linux
python -m streamlit run ui/app.py
```

The app opens in your browser at `http://localhost:8501`.

**Workflow:**
1. Paste your Gemini API key in the key field (or pre-set it via env var).
2. Select the "Single Report Analysis" tab.
3. Upload a PDF or HTML financial document.
4. Type your question in the query box.
5. Click **Analyze Report**.
6. Expand the "How Vectorless RAG Stores This Report" panel to inspect the retrieval trace.

### CLI Mode

```bash
python main.py --file data/apple_2023.html --query "What are the main risks?" --model gemini-2.5-flash
```

Output is printed as JSON to stdout.

---

## 12. Testing

The test suite uses `pytest`.

```bash
# Run all tests
python -m pytest -q

# Run with verbose output
python -m pytest -v
```

Current test coverage:
- `tests/test_section_splitter.py` — validates that `SectionSplitter` correctly identifies `risk_factors`, `mda`, and `legal` sections from sample text containing SEC-style Item headers.

---

## 13. Retrieval Scoring Formula

### Section-Level Score (RuleBasedRetriever)

```
total_score = base_score + keyword_score + red_flag_score

base_score      = 1.0  if section in prioritized_sections
                  0.5  otherwise

keyword_score   = (count of query tokens found in section text)
                  ─────────────────────────────────────────────
                  (total query tokens)

red_flag_score  = 0.2  if query contains any red-flag keyword
                  0.0  otherwise
```

Red-flag keywords: `restatement`, `material weakness`, `going concern`, `default`

### Chunk-Level Score (ChunkRetriever)

```
total_score = base_score + keyword_score + length_score × 0.1

base_score    = 1.0  if chunk.section in priority_sections
                0.6  otherwise

keyword_score = (matched query tokens in chunk) / (total query tokens)

length_score  = min(len(chunk_text) / 1200, 1.0)
```

Priority sections: `risk_factors`, `mda`, `notes`, `legal`, `auditor_notes`

### Risk Comparison Matching

Two risk sentences are considered the same if:
- Token overlap ≥ 55%: `|A ∩ B| / max(|A|, 1) ≥ 0.55`
- OR Jaccard similarity ≥ 0.8: `|A ∩ B| / |A ∪ B| ≥ 0.8`

---

## 14. Feature Summary

| Feature | Description |
|---|---|
| **Document Ingestion** | Upload PDF or HTML documents via UI or CLI |
| **Free-Form Q&A** | Ask any question; answer sourced from retrieved document chunks |
| **Automatic Risk Detection** | Triggered when query contains risk-related terms |
| **Risk Categorization** | Financial, Operational, Market, Regulatory |
| **Red Flag Detection** | Material weakness, going concern, restatement, significant doubt |
| **Document Comparison** | New / removed risks, tone change, risk intensity, new red flags |
| **Retrieval Visualization** | 5-step pipeline diagram, fetch trace, ranked section cards, worked example |
| **Parquet Data Store** | Browse all stored documents and inspect chunks in the browser |
| **Confidence Scoring** | 0–100 score reflecting risk density in retrieved text |
| **JSON Export** | Download analysis results as structured JSON |
| **CLI Support** | Full pipeline accessible from the command line |
| **No Vector Database** | Fully explainable retrieval with no embedding infrastructure |

---

## 15. Limitations and Known Constraints

| Limitation | Detail |
|---|---|
| **Keyword-only retrieval** | Synonyms or paraphrased queries may miss relevant content if exact words don't overlap |
| **Section detection scope** | Section patterns are tuned for SEC 10-K style filings; arbitrary documents may fall through to the "other" bucket |
| **Scanned PDFs** | Image-based PDFs without embedded text layer will parse as empty text |
| **LLM dependency** | All natural language answers require a valid Gemini API key and network access to Google's API |
| **Context window** | Risk analyzer sends at most 12,000 characters to Gemini; very long risk sections may be truncated |
| **Table extraction** | PDF table extraction is available but not used in the default pipeline |
| **Single language** | All retrieval patterns and heuristics assume English-language documents |
| **Local storage only** | Parquet files are stored on the local filesystem; no cloud persistence is included |

---

## 16. Security Considerations

- **API keys must never be committed to source control.** Use environment variables or the in-app key field.
- If a key has been accidentally exposed, rotate it immediately at [Google AI Studio](https://aistudio.google.com/app/apikey).
- Uploaded documents are written to a system temporary directory (`tempfile.NamedTemporaryFile`) and are not permanently stored on the server filesystem (only the extracted text and chunks are saved to Parquet).
- The application is designed for internal or single-user local deployment. If deployed to a shared server, access controls and API key isolation should be added.
- Do not commit `.env` files, virtual environments, or local cache files to source control (all excluded by `.gitignore`).

---

*End of Document*
