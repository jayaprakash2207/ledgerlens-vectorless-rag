import argparse
import json
import os

from analysis.qa_engine import QAEngine
from analysis.risk_analyzer import RiskAnalyzer
from dotenv import load_dotenv
from llm.gemini_interface import GeminiClient
from parser.html_parser import SEC10KHtmlParser
from parser.pdf_parser import PDFParser
from retrieval.chunk_retriever import ChunkRetriever
from retrieval.rule_engine import RuleBasedRetriever
from segmentation.section_splitter import SectionSplitter
from storage.parquet_store import ParquetStore


def _load_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith((".html", ".htm")):
        html_parser = SEC10KHtmlParser()
        return html_parser.parse(file_path)

    if file_path.lower().endswith(".pdf"):
        parser = PDFParser(extract_tables=False)
        return parser.parse(file_path).text

    raise ValueError("Unsupported file type. Use .pdf, .html, or .htm")


def run_pipeline(file_path: str, query: str, use_ollama: bool, model: str) -> None:
    splitter = SectionSplitter()
    llm_client = GeminiClient(model=model)
    qa_engine = QAEngine(llm_client=llm_client)
    risk_analyzer = RiskAnalyzer(llm_client=llm_client, require_llm=True)
    store = ParquetStore()

    raw_text = _load_text(file_path)
    print(f"Parsed text length: {len(raw_text)}")
    if not raw_text.strip():
        print("Parsed text is empty; aborting pipeline.")
        return

    if len(raw_text) < 500:
        print("Warning: parsed text is very short; the HTML/PDF may be incomplete.")

    print("=== TEXT PREVIEW (first 500 chars) ===")
    print(raw_text[:500])
    print("=== END PREVIEW ===")

    sections = splitter.split(raw_text)
    print(f"Detected sections: {list(sections.keys())}")
    print(f"Risk factors size: {len(sections.get('risk_factors', ''))}")
    print("=== SECTION PREVIEW ===")
    sections_preview = {key: value[:200] for key, value in sections.items() if value}
    print(json.dumps(sections_preview, indent=2))
    print("=== END SECTION PREVIEW ===")

    stored = store.save_document(
        source_name=os.path.basename(file_path),
        file_type=os.path.splitext(file_path)[1].lstrip(".").lower(),
        raw_text=raw_text,
        sections=sections,
    )
    chunks = store.load_chunks(stored.doc_id)
    chunk_retriever = ChunkRetriever()
    top_chunks = chunk_retriever.retrieve(query, chunks, top_k=5)
    qa_result = qa_engine.answer(query, [item.text for item in top_chunks])

    print("=== GENERAL ANSWER ===")
    print("Answer:", qa_result.answer)
    print("Sources:", qa_result.sources)

    output = {
        "doc_id": stored.doc_id,
        "sections": sections,
        "answer": qa_result.answer,
        "sources": qa_result.sources,
    }

    if "risk" in query.lower():
        retriever = RuleBasedRetriever()
        retrieved = retriever.retrieve(query, sections, max_sections=2)
        combined_text = "\n\n".join(result.text for result in retrieved)
        analysis = risk_analyzer.analyze(combined_text)
        output.update(
            {
                "top_risks": analysis.top_risks,
                "risk_categories": analysis.risk_categories,
                "red_flags": analysis.red_flags,
                "confidence_score": analysis.confidence_score,
                "summary": analysis.summary,
                "risky_sentences": analysis.risky_sentences,
            }
        )
    print(json.dumps(output, indent=2))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Financial Risk Intelligence Assistant")
    parser.add_argument(
        "--file",
        default=os.path.join("data", "apple_2023.html"),
        help="Path to a PDF or HTML report",
    )
    parser.add_argument("--query", default="What are the main risks?", help="User query")
    parser.add_argument("--use-ollama", action="store_true", help="Ignored (Gemini required)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    print(f"Using file: {args.file}")
    print(f"File exists: {os.path.exists(args.file)}")

    try:
        run_pipeline(args.file, args.query, args.use_ollama, args.model)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
