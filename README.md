# LedgerLens

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Repo](https://img.shields.io/badge/GitHub-LedgerLens-181717?style=for-the-badge&logo=github)](https://github.com/jayaprakash2207/ledgerlens)

An explainable **vectorless RAG** system for SEC filings Q&A and insights.

LedgerLens reads SEC-style HTML and PDF reports, extracts structured sections, retrieves relevant chunks with a rule-based pipeline instead of embeddings, and uses Gemini to answer any user question. The result is a simpler, more transparent RAG workflow that is easier to inspect, debug, and demo.

## Highlights

- No vector database
- No embeddings pipeline
- Explainable rule-based retrieval
- Financial-report-focused section splitting
- Gemini-powered Q&A over filings
- Parquet-backed persistence of documents and chunks
- Streamlit UI with a built-in vectorless RAG visualization panel
- Report comparison workflow for detecting changes in risk framing

## Why Vectorless RAG?

Many RAG systems hide retrieval logic behind embeddings and similarity search. This project keeps retrieval visible.

Instead of storing vectors, LedgerLens stores:

1. cleaned plain text
2. named report sections
3. Parquet-backed chunk store
4. retrieval scores and fetched context sent to Gemini

That means you can actually see:

- what was parsed
- how it was grouped
- what was retrieved
- why the model got that context

## How It Works

```text
Report File
   ->
HTML/PDF Parser
   ->
Cleaned Plain Text
   ->
Section Splitter
   ->
Named Section Buckets
   ->
Parquet Store (documents + chunks)
   ->
Chunk Retriever
   ->
Gemini
   ->
Answer + Sources (and risk analysis if requested)
```

## Screenshots

### App Preview

![LedgerLens app preview](docs/screenshots/app-ui-preview.svg)

### Vectorless RAG Flow

![Vectorless RAG flow](docs/screenshots/vectorless-rag-flow.svg)

## How Retrieval Works

This project does not query a vector database to fetch context.

Instead, it performs vectorless retrieval over chunked report text in Parquet:

1. the parser reads a PDF or HTML filing into plain text
2. the section splitter groups that text into named buckets like `risk_factors`, `mda`, `notes`, and `legal`
3. the Parquet store persists the raw text and chunked sections
4. the chunk retriever tokenizes the user question
5. it scores chunks using section priority + keyword overlap
6. it ranks the chunks
7. it fetches the top-scoring chunks
8. it sends that fetched context to Gemini for an answer

In short:

`question -> query tokens -> chunk ranking -> top chunk fetch -> Gemini output`

## Worked Example

For a question like:

`How many shares were repurchased?`

the retriever will usually:

1. tokenize the query into words like `how`, `many`, `shares`, `repurchased`
2. recognize that those terms map strongly to `notes` and `legal` sections
3. score chunks higher where those terms appear
4. fetch the top matched chunks
5. pass those chunks to Gemini
6. produce an answer with short supporting quotes

## In-App Retrieval Visualization

The Streamlit UI now includes a detailed retrieval walkthrough under:

`How Vectorless RAG Stores This Report`

Inside that panel, the app shows:

- `Retrieval Flow Diagram`
- `Fetch Trace`
- `Ranked Section Cards`
- `Worked Example: How Data Was Found`

That means you can inspect:

- how the query was understood
- how the matching section was found
- what text was fetched
- what context was sent to Gemini
- how the final structured output was formed

## Features

### Single Report Analysis

Upload one filing and get:

- answer to any question
- top chunks and source quotes
- optional risk summary when a risk question is asked

### Report Comparison

Upload an older and newer filing to detect:

- new risks
- removed risks
- tone changes
- risk intensity changes
- new red flags

### Vectorless RAG Visualization

The UI includes a dedicated panel showing:

- parsed text size
- stored section buckets
- retrieval scores
- selected context passed to Gemini
- in-memory data shape

This makes the retrieval path easy to explain in demos, interviews, and project reviews.

## Example Questions

You can ask prompts like:

- `How many shares were repurchased?`
- `What were net sales in 2023?`
- `Which legal proceedings are described?`
- `What are the main risks?`

## Project Structure

```text
.
|-- analysis/
|   |-- comparison_engine.py
|   |-- qa_engine.py
|   `-- risk_analyzer.py
|-- data/
|   |-- apple_2023.html
|   `-- README.txt
|-- llm/
|   |-- gemini_interface.py
|   `-- ollama_interface.py
|-- parser/
|   |-- html_parser.py
|   `-- pdf_parser.py
|-- retrieval/
|   |-- chunk_retriever.py
|   `-- rule_engine.py
|-- segmentation/
|   `-- section_splitter.py
|-- storage/
|   `-- parquet_store.py
|-- tests/
|   `-- test_section_splitter.py
|-- ui/
|   `-- app.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Architecture Snapshot

| Layer | File(s) | Responsibility |
|---|---|---|
| Parsing | `parser/html_parser.py`, `parser/pdf_parser.py` | Extract clean text from HTML and PDF |
| Segmentation | `segmentation/section_splitter.py` | Detect financial-report sections like `risk_factors` and `mda` |
| Storage | `storage/parquet_store.py` | Persist documents and chunks to Parquet |
| Retrieval | `retrieval/chunk_retriever.py` | Rank chunks using rule-based keyword and section-priority logic |
| LLM | `llm/gemini_interface.py` | Send retrieved context to Gemini |
| Analysis | `analysis/risk_analyzer.py`, `analysis/qa_engine.py` | Produce answers and optional risk analysis |
| Comparison | `analysis/comparison_engine.py` | Compare old vs new reports |
| UI | `ui/app.py` | Streamlit interface and vectorless-RAG visualization |

## Tech Stack

- Python
- Streamlit
- BeautifulSoup4
- pdfplumber
- PyMuPDF
- Requests
- Pandas + PyArrow (Parquet)
- Gemini API

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Gemini API Key

You can provide the Gemini API key in either of these ways:

### Option 1. Enter it in the Streamlit UI

The app includes a secure password-style field for the key.

### Option 2. Set an environment variable

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

## Run the Project

### CLI mode

```powershell
.\.venv\Scripts\python.exe main.py --model gemini-2.5-flash
```

Example:

```powershell
.\.venv\Scripts\python.exe main.py --file data\apple_2023.html --query "What are the main risks?" --model gemini-2.5-flash
```

Available CLI arguments:

- `--file` path to a PDF or HTML report
- `--query` analysis question
- `--model` Gemini model name

### Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

Then:

1. paste the Gemini API key
2. upload a PDF or HTML report
3. run analysis
4. open `How Vectorless RAG Stores This Report`

## Testing

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Sample Data

The repo currently includes:

- `data/apple_2023.html`

You can add more sample reports to the `data/` folder for local testing.

## What Makes This Useful

- Great for explainable RAG demos
- Useful for SEC filing analysis prototypes
- Easier to debug than embedding-heavy pipelines
- Good learning project for document parsing and retrieval
- Strong base for building deeper financial AI workflows

## Current Limitations

- Retrieval is heuristic rather than embedding-based
- Section detection depends on common SEC-style headings
- Output quality depends on parsed text quality and Gemini responses
- Test coverage is still minimal

## Roadmap Ideas

- richer parser and retriever tests
- visual retrieval trace exports
- more filing formats
- parsed-document caching
- chart-based retrieval analytics
- stronger comparison reporting

## Security Note

Do not commit API keys to source control. Use environment variables or the Streamlit key field for local testing. If a key has been exposed publicly, rotate it immediately.

## Repository Status

This repository is actively set up for:

- local development
- GitHub publishing
- interactive Streamlit demos
- explainable vectorless RAG experiments

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is distributed under a **permission required / all rights reserved** license.

See [LICENSE](LICENSE). Anyone who wants to use, modify, redistribute, or deploy this project must obtain prior permission from the owner.
