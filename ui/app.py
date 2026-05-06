import json
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from analysis.comparison_engine import compare_reports
from analysis.qa_engine import QAEngine
from analysis.risk_analyzer import RiskAnalyzer
from llm.gemini_interface import GeminiClient
from parser.html_parser import SEC10KHtmlParser
from parser.pdf_parser import PDFParser
from retrieval.chunk_retriever import ChunkRetriever
from retrieval.rule_engine import RuleBasedRetriever
from segmentation.section_splitter import SectionSplitter
from storage.parquet_store import ParquetStore


st.set_page_config(page_title="LedgerLens", layout="wide")

st.title("LedgerLens")
st.caption("Vectorless filings intelligence for SEC reports, Q&A, and comparisons.")

use_ollama = st.checkbox("Use LLM (Gemini)", value=True, disabled=True)
model_name = st.text_input("Gemini model", value="gemini-2.5-flash")
api_key_value = st.text_input(
    "Gemini API key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Paste your Gemini API key here or set GEMINI_API_KEY before launching Streamlit.",
)
if not api_key_value:
    st.warning("Gemini API key is not set. Paste it above or set GEMINI_API_KEY before running analysis.")

def _parse_upload(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    parser = PDFParser(extract_tables=False)
    html_parser = SEC10KHtmlParser()
    splitter = SectionSplitter()

    file_name = uploaded_file.name.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_name) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if file_name.endswith(".pdf"):
            parsed = parser.parse(tmp_path)
            cleaned_text = parsed.text
        elif file_name.endswith(".html") or file_name.endswith(".htm"):
            cleaned_text = html_parser.parse(tmp_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None, None

        sections = splitter.split(cleaned_text)
        return sections, cleaned_text
    except Exception as exc:
        st.error(f"Failed to parse {uploaded_file.name}: {exc}")
        return None, None


def _extract_year(name: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", name)
    if not match:
        return None
    return int(match.group(0))


def _build_report_payload(name: str, sections: Dict[str, str]) -> Dict[str, str]:
    return {
        "company": name,
        "year": _extract_year(name),
        "risk_factors": sections.get("risk_factors", ""),
        "mda": sections.get("mda", ""),
        "notes": sections.get("notes", ""),
        "legal": sections.get("legal", ""),
        "auditor_notes": sections.get("auditor_notes", ""),
    }


def _missing_section_warnings(sections: Dict[str, str]) -> None:
    if not sections.get("risk_factors"):
        st.warning("Risk factors section missing or empty.")
    if not sections.get("mda"):
        st.warning("MD&A section missing or empty.")


def _json_download_button(data: Dict[str, object], label: str) -> None:
    st.download_button(
        label=label,
        data=json.dumps(data, indent=2),
        file_name="risk_analysis.json",
        mime="application/json",
    )


def _section_stats(sections: Dict[str, str]) -> List[Dict[str, object]]:
    stats: List[Dict[str, object]] = []
    total_chars = sum(len(text) for text in sections.values() if text)
    for name, text in sections.items():
        if not text:
            continue
        char_count = len(text)
        share = round((char_count / total_chars) * 100, 1) if total_chars else 0.0
        stats.append(
            {
                "section": name,
                "chars": char_count,
                "share": share,
                "preview": text[:220],
            }
        )
    return stats


def _highlight_terms(text: str, terms: List[str], limit: int = 900) -> str:
    snippet = text[:limit]
    if not terms:
        return snippet

    highlighted = snippet
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        highlighted = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r"[[\1]]",
            highlighted,
        )
    return highlighted


def _render_vectorless_rag_view(
    cleaned_text: str,
    sections: Dict[str, str],
    retrieved: List[object],
    query: str,
    retrieval_explanation: object,
    top_chunks: List[object],
) -> None:
    st.markdown("### Vectorless RAG View")
    st.info(
        "This project does not store embeddings or vectors. It stores cleaned text and chunked sections in "
        "Parquet, then uses keyword and rule scoring to fetch relevant context for Gemini."
    )

    total_chars = len(cleaned_text)
    section_count = len([name for name, text in sections.items() if text])
    retrieved_count = len(retrieved)
    top_context_chars = sum(len(item.text) for item in retrieved)

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    with metric_a:
        st.metric("Raw Text Chars", total_chars)
    with metric_b:
        st.metric("Stored Sections", section_count)
    with metric_c:
        st.metric("Retrieved Sections", retrieved_count)
    with metric_d:
        st.metric("Context Chars Sent", top_context_chars)

    st.markdown("#### Pipeline")
    pipe_a, pipe_b, pipe_c, pipe_d = st.columns(4)
    with pipe_a:
        st.markdown("**1. Parse**")
        st.caption("PDF/HTML to cleaned plain text")
        st.code(cleaned_text[:350] or "No text", language="text")
    with pipe_b:
        st.markdown("**2. Store**")
        st.caption("Named in-memory section buckets")
        st.json({name: len(text) for name, text in sections.items() if text})
    with pipe_c:
        st.markdown("**3. Retrieve**")
        st.caption(f"Rule match for query: {query}")
        st.json(
            [
                {"section": item.section, "score": round(item.score, 3), "chars": len(item.text)}
                for item in retrieved
            ]
        )
    with pipe_d:
        st.markdown("**4. Analyze**")
        st.caption("Fetched context goes to Gemini")
        st.code("\n\n".join(item.text[:180] for item in retrieved) or "No retrieved text", language="text")

    st.markdown("#### Retrieval Flow Diagram")
    top_ranked = retrieval_explanation.ranked_sections[:3]
    top_ranked_labels = "<br/>".join(
        f"{index + 1}. {item['section']} ({item['score']})"
        for index, item in enumerate(top_ranked)
    ) or "No ranked sections"
    fetched_labels = "<br/>".join(
        f"{index + 1}. {item.section} ({len(item.text)} chars)"
        for index, item in enumerate(retrieved)
    ) or "No fetched sections"
    query_labels = ", ".join(retrieval_explanation.query_terms) or "No query tokens"
    prioritized_labels = ", ".join(retrieval_explanation.prioritized_sections) or "No priority sections"

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 70px 1fr 70px 1fr 70px 1fr;gap:8px;align-items:center;margin:8px 0 20px 0;">
          <div style="background:#10243e;color:#ffffff;border-radius:16px;padding:16px;min-height:150px;">
            <div style="font-weight:700;font-size:18px;margin-bottom:8px;">1. Query</div>
            <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">User question</div>
            <div style="font-size:16px;">{query}</div>
            <div style="font-size:13px;opacity:0.85;margin-top:10px;">Tokens: {query_labels}</div>
          </div>
          <div style="text-align:center;font-size:34px;color:#54789c;">→</div>
          <div style="background:#ffffff;border:1px solid #d9e3f0;border-radius:16px;padding:16px;min-height:150px;">
            <div style="font-weight:700;font-size:18px;margin-bottom:8px;color:#10243e;">2. Prioritize</div>
            <div style="font-size:14px;color:#49627f;margin-bottom:10px;">Retriever chooses likely sections</div>
            <div style="font-size:15px;color:#10243e;">{prioritized_labels}</div>
          </div>
          <div style="text-align:center;font-size:34px;color:#54789c;">→</div>
          <div style="background:#ffffff;border:1px solid #d9e3f0;border-radius:16px;padding:16px;min-height:150px;">
            <div style="font-weight:700;font-size:18px;margin-bottom:8px;color:#10243e;">3. Rank</div>
            <div style="font-size:14px;color:#49627f;margin-bottom:10px;">Sections scored by rules</div>
            <div style="font-size:15px;color:#10243e;line-height:1.5;">{top_ranked_labels}</div>
          </div>
          <div style="text-align:center;font-size:34px;color:#54789c;">→</div>
          <div style="background:#ffffff;border:1px solid #d9e3f0;border-radius:16px;padding:16px;min-height:150px;">
            <div style="font-weight:700;font-size:18px;margin-bottom:8px;color:#10243e;">4. Fetch</div>
            <div style="font-size:14px;color:#49627f;margin-bottom:10px;">Top sections selected</div>
            <div style="font-size:15px;color:#10243e;line-height:1.5;">{fetched_labels}</div>
          </div>
          <div style="text-align:center;font-size:34px;color:#54789c;">→</div>
          <div style="background:#1d4d7a;color:#ffffff;border-radius:16px;padding:16px;min-height:150px;">
            <div style="font-weight:700;font-size:18px;margin-bottom:8px;">5. Gemini</div>
                        <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">Fetched context is analyzed</div>
                        <div style="font-size:15px;line-height:1.5;">Answer, sources, and any detected risks</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Fetch Trace")
    trace_a, trace_b = st.columns(2)
    with trace_a:
        st.markdown("**Query Tokens**")
        st.json(retrieval_explanation.query_terms)
        st.markdown("**Prioritized Sections**")
        st.json(retrieval_explanation.prioritized_sections)
    with trace_b:
        st.markdown("**Why these sections were fetched**")
        st.caption("Final score = base score + keyword score + red-flag bonus")
        st.json(
            [
                {
                    "section": item["section"],
                    "score": item["score"],
                    "base_score": item["base_score"],
                    "keyword_score": item["keyword_score"],
                    "red_flag_score": item["red_flag_score"],
                    "matched_terms": item["matched_terms"],
                }
                for item in retrieval_explanation.ranked_sections
            ]
        )

    st.markdown("#### Ranked Section Cards")
    for item in retrieval_explanation.ranked_sections:
        st.write(f"`{item['section']}`  score={item['score']}  matched={item['matched_terms'] or ['none']}")
        st.progress(min(int(item["score"] * 50), 100))
        with st.expander(f"Why {item['section']} scored {item['score']}"):
            st.write(
                f"Base score: {item['base_score']} | "
                f"Keyword score: {item['keyword_score']} | "
                f"Red-flag bonus: {item['red_flag_score']}"
            )
            st.code(item["preview"], language="text")

    st.markdown("#### Worked Example: How Data Was Found")
    if retrieval_explanation.ranked_sections:
        top_item = retrieval_explanation.ranked_sections[0]
        fetched_section = next((item for item in retrieved if item.section == top_item["section"]), None)

        example_a, example_b, example_c = st.columns(3)
        with example_a:
            st.markdown("**Step 1. Query Understanding**")
            st.write(f"Question: `{query}`")
            st.write(f"Tokens used for matching: `{retrieval_explanation.query_terms}`")
            st.write(f"Section priority chosen: `{retrieval_explanation.prioritized_sections}`")

        with example_b:
            st.markdown("**Step 2. How It Found Data**")
            st.write(f"Top matched section: `{top_item['section']}`")
            st.write(
                f"Matched terms: `{top_item['matched_terms']}` | "
                f"Final score: `{top_item['score']}`"
            )
            st.code(
                _highlight_terms(top_item["preview"], top_item["matched_terms"], limit=500),
                language="text",
            )

        with example_c:
            st.markdown("**Step 3. What It Fetched**")
            if fetched_section:
                st.write(
                    f"The retriever selected `{fetched_section.section}` "
                    f"and passed `{len(fetched_section.text)}` characters forward."
                )
                st.code(
                    _highlight_terms(fetched_section.text, top_item["matched_terms"], limit=500),
                    language="text",
                )
            else:
                st.write("No fetched section matched the top ranked explanation.")

        st.markdown("**Step 4. How Output Was Produced**")
        output_a, output_b = st.columns(2)
        with output_a:
            st.caption("Context sent to Gemini")
            st.code(
                "\n\n".join(item.text[:260] for item in retrieved) or "No retrieved context",
                language="text",
            )
        with output_b:
            st.caption("Gemini turns fetched context into structured output")
            st.write(
                "The app sends the fetched sections to Gemini, then the analyzer converts the model "
                "response into top risks, categories, red flags, confidence score, and summary."
            )
            st.code(
                json.dumps(
                    {
                        "top_risks": "<list>",
                        "risk_categories": "<object>",
                        "red_flags": "<list>",
                        "confidence_score": "<int>",
                        "summary": "<string>",
                    },
                    indent=2,
                ),
                language="json",
            )

    st.markdown("#### Section Breakdown")
    for item in _section_stats(sections):
        st.write(f"`{item['section']}`  {item['chars']} chars  ({item['share']}%)")
        st.progress(min(int(item["share"]), 100))
        with st.expander(f"Preview: {item['section']}"):
            st.code(item["preview"], language="text")

    st.markdown("#### Parquet + Memory Shape")
    st.code(
        json.dumps(
            {
                "cleaned_text": f"<{len(cleaned_text)} chars>",
                "sections": {name: f"<{len(text)} chars>" for name, text in sections.items() if text},
                "top_chunks": [
                    {"chunk_id": item.chunk_id, "section": item.section, "score": round(item.score, 3)}
                    for item in top_chunks
                ],
                "retrieved_context": [
                    {"section": item.section, "score": round(item.score, 3), "chars": len(item.text)}
                    for item in retrieved
                ],
            },
            indent=2,
        ),
        language="json",
    )


tab_single, tab_compare, tab_store = st.tabs(["Single Report Analysis", "Compare Reports", "Data Store"])

with tab_single:
    st.subheader("Single Report Analysis")
    uploaded_file = st.file_uploader("Upload a report (PDF or HTML)", type=["pdf", "html", "htm"], key="single")
    query = st.text_input("Ask any question", value="What are the main risks?", key="single_query")
    analyze = st.button("Analyze Report", key="analyze")

    if analyze:
        if not uploaded_file:
            st.warning("Please upload a report before running analysis.")
        else:
            with st.spinner("Analyzing report..."):
                if not api_key_value:
                    st.error("Gemini API key is required for answers. Set GEMINI_API_KEY or paste it above.")
                    st.stop()

                sections, cleaned_text = _parse_upload(uploaded_file)
                if sections and cleaned_text:
                    _missing_section_warnings(sections)

                    retriever = RuleBasedRetriever()
                    llm_client = GeminiClient(model=model_name, api_key=api_key_value)
                    analyzer = RiskAnalyzer(llm_client=llm_client, require_llm=True)
                    qa_engine = QAEngine(llm_client=llm_client)
                    store = ParquetStore()
                    chunk_retriever = ChunkRetriever()

                    stored = store.save_document(
                        source_name=uploaded_file.name,
                        file_type=os.path.splitext(uploaded_file.name)[1].lstrip(".").lower(),
                        raw_text=cleaned_text,
                        sections=sections,
                    )
                    chunks = store.load_chunks(stored.doc_id)
                    top_chunks = chunk_retriever.retrieve(query, chunks, top_k=5)
                    qa_result = qa_engine.answer(query, [item.text for item in top_chunks])

                    is_risk_query = any(term in query.lower() for term in ["risk", "risks", "threat", "uncertainty", "red flag"])

                    retrieval_explanation = retriever.explain_retrieval(query, sections)
                    retrieved = retriever.retrieve(query, sections, max_sections=2)
                    analysis = None
                    if is_risk_query:
                        combined_text = "\n\n".join(result.text for result in retrieved)
                        analysis = analyzer.analyze(combined_text)

                    metric_col, score_col, doc_col = st.columns(3)
                    with metric_col:
                        st.metric("Stored Chunks", len(chunks))
                    with score_col:
                        st.metric("Confidence Score", analysis.confidence_score if analysis else 0)
                    with doc_col:
                        st.metric("Doc ID", stored.doc_id)

                    st.markdown("### Answer")
                    if qa_result.answer:
                        st.success(qa_result.answer)
                    else:
                        last_error = getattr(llm_client, "last_error", None)
                        if last_error:
                            st.warning(f"No answer returned ({last_error}).")
                        else:
                            st.warning("No answer returned. Check your Gemini API key and model name.")
                    if qa_result.sources:
                        with st.expander("Answer Sources"):
                            st.write("\n".join(qa_result.sources))

                    with st.expander("Top Chunks Used"):
                        for item in top_chunks:
                            st.write(f"{item.chunk_id}  score={round(item.score, 3)}  section={item.section}")
                            st.code(item.text[:400], language="text")

                    if analysis:
                        st.markdown("### Risk Summary")
                        st.success(analysis.summary)

                        st.markdown("### Risk Categories")
                        st.json(analysis.risk_categories)

                        st.markdown("### Red Flags")
                        st.write(analysis.red_flags or ["None detected"])

                        with st.expander("Top Risks"):
                            st.write("\n".join(analysis.top_risks) or "None detected")

                        with st.expander("Highlighted Risky Sentences"):
                            st.write("\n".join(analysis.risky_sentences[:50]) or "None detected")
                    else:
                        st.info("Risk analysis runs only when the question is about risks.")

                    with st.expander("How Vectorless RAG Stores This Report", expanded=True):
                        _render_vectorless_rag_view(
                            cleaned_text,
                            sections,
                            retrieved,
                            query,
                            retrieval_explanation,
                            top_chunks,
                        )

                    output = {
                        "doc_id": stored.doc_id,
                        "answer": qa_result.answer,
                        "sources": qa_result.sources,
                        "confidence_score": analysis.confidence_score if analysis else 0,
                    }
                    if analysis:
                        output.update(
                            {
                                "top_risks": analysis.top_risks,
                                "risk_categories": analysis.risk_categories,
                                "red_flags": analysis.red_flags,
                                "summary": analysis.summary,
                                "risky_sentences": analysis.risky_sentences,
                            }
                        )
                    _json_download_button(output, "Download JSON")

with tab_compare:
    st.subheader("Compare Reports")
    demo_mode = st.checkbox("Demo Mode (Apple 10-K)", value=False)
    left, right = st.columns(2)

    with left:
        report_a = st.file_uploader("Report A (older)", type=["pdf", "html", "htm"], key="report_a")

    with right:
        report_b = st.file_uploader("Report B (newer)", type=["pdf", "html", "htm"], key="report_b")

    compare = st.button("Compare Reports", key="compare")

    if compare:
        if not report_a or not report_b:
            st.warning("Please upload both reports before comparing.")
        else:
            with st.spinner("Comparing reports..."):
                sections_a, _ = _parse_upload(report_a)
                sections_b, _ = _parse_upload(report_b)

                if sections_a and sections_b:
                    _missing_section_warnings(sections_a)
                    _missing_section_warnings(sections_b)

                    report_old = _build_report_payload(report_a.name, sections_a)
                    report_new = _build_report_payload(report_b.name, sections_b)
                    result = compare_reports(report_old, report_new)

                    metric_col, score_col = st.columns(2)
                    with metric_col:
                        st.metric("New Risks", len(result.new_risks))
                    with score_col:
                        st.metric("Confidence Score", result.confidence_score)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### New Risks")
                        st.write(result.new_risks or ["None detected"])

                        st.markdown("### Risk Intensity Change")
                        st.write(result.risk_intensity_change)

                        st.markdown("### Red Flags Delta")
                        st.write(result.new_red_flags or ["None detected"])

                    with col_b:
                        st.markdown("### Removed Risks")
                        st.write(result.removed_risks or ["None detected"])

                        st.markdown("### Tone Shift")
                        st.write(result.tone_change)

                        st.markdown("### Summary")
                        st.write(result.summary)

                    with st.expander("Highlighted Sentences"):
                        st.markdown("**New risk sentences**")
                        st.write("\n".join(result.highlighted_sentences.get("new", [])) or "None")
                        st.markdown("**Removed risk sentences**")
                        st.write("\n".join(result.highlighted_sentences.get("removed", [])) or "None")

                    output = {
                        "new_risks": result.new_risks,
                        "removed_risks": result.removed_risks,
                        "risk_intensity_change": result.risk_intensity_change,
                        "tone_change": result.tone_change,
                        "new_red_flags": result.new_red_flags,
                        "confidence_score": result.confidence_score,
                        "summary": result.summary,
                    }
                    _json_download_button(output, "Download Comparison JSON")

with tab_store:
    st.subheader("Parquet Data Store")
    store = ParquetStore()
    if hasattr(store, "list_documents"):
        documents = store.list_documents()
    else:
        documents = []
        if os.path.exists(store.docs_path):
            df = pd.read_parquet(store.docs_path)
            df = df.sort_values(by="created_at", ascending=False)
            documents = df.to_dict(orient="records")

    if not documents:
        st.info("No documents stored yet. Upload a report in Single Report Analysis to create Parquet data.")
    else:
        doc_options = {
            f"{doc['source_name']} ({doc['doc_id']})": doc["doc_id"]
            for doc in documents
        }
        selected_label = st.selectbox("Stored documents", list(doc_options.keys()))
        selected_id = doc_options[selected_label]
        doc = store.load_document(selected_id)

        if doc:
            meta_a, meta_b, meta_c = st.columns(3)
            with meta_a:
                st.metric("Doc ID", doc.get("doc_id", ""))
            with meta_b:
                st.metric("File Type", doc.get("file_type", ""))
            with meta_c:
                st.metric("Text Length", doc.get("text_len", 0))

            st.markdown("### Stored Metadata")
            st.json(
                {
                    "source_name": doc.get("source_name"),
                    "created_at": doc.get("created_at"),
                    "text_len": doc.get("text_len"),
                }
            )

            with st.expander("Raw Text Preview"):
                st.code(str(doc.get("raw_text", ""))[:2000], language="text")

            chunks = store.load_chunks(selected_id)
            st.markdown("### Stored Chunks")
            st.write(f"Total chunks: {len(chunks)}")

            for chunk in chunks[:10]:
                st.write(f"{chunk['chunk_id']}  section={chunk['section']}  index={chunk['chunk_index']}")
                st.code(str(chunk.get("text", ""))[:600], language="text")
        else:
            st.warning("Selected document could not be loaded.")

if demo_mode:
    st.info("Demo Mode is enabled, but no demo files are bundled yet.")

