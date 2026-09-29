"""Streamlit UI for the Provenance-Aware Graph RAG demo.

Run:  streamlit run app.py
"""

from __future__ import annotations

import warnings
from datetime import date

import pandas as pd
import streamlit as st

from provrag.config import reference_date, together_model
from provrag.pipeline import ProvenanceGraphRAG
from provrag.provenance import all_sources
from provrag.ranking import SOURCE_TYPE_PRIOR, RankingConfig
from provrag.vocab import PREDICATES

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXAMPLES = [
    "Where is the company Alice works for headquartered?",
    "In which country is Alice's employer headquartered?",
    "Who is the CEO of the company Alice works for?",
    "How many employees does Acme AI have?",
    "What is Alice's job title?",
    "Who works at Acme AI?",
    "When was Acme AI founded?",
    "Tell me about Munich",
]

st.set_page_config(page_title="Provenance-Aware Graph RAG", page_icon="🧭", layout="wide")


@st.cache_resource
def load_pipeline() -> ProvenanceGraphRAG:
    return ProvenanceGraphRAG()


rag = load_pipeline()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    if rag.llm_available:
        st.success(f"LLM configured (Together AI)\n\n`{together_model()}`")
    else:
        st.info("No `TOGETHER_API_KEY` found — using deterministic entity matching "
                "and template answers. Everything else works the same.")
    use_llm_gen = st.toggle("Use LLM for answer generation", value=rag.llm_available,
                            disabled=not rag.llm_available)
    use_llm_parse = st.toggle("Use LLM for entity/relation detection", value=False,
                              disabled=not rag.llm_available)

    st.subheader("Evidence ranking")
    st.caption("Retrieval **prioritization** only — orders evidence by freshness and source type. "
               "It is **not** a truth score and never removes evidence.")
    show_rank = st.toggle("Show ranking", value=True)
    w_fresh = st.slider("Freshness weight", 0.0, 1.0, 0.5, 0.05)
    w_type = st.slider("Source-type weight", 0.0, 1.0, 0.5, 0.05)
    half_life = st.slider("Freshness half-life (days)", 30, 1500, 365, 5)
    ref_date = st.date_input("Reference date ('today')", value=reference_date())
    with st.expander("Source-type priors"):
        st.table(pd.DataFrame(SOURCE_TYPE_PRIOR.items(), columns=["source type", "prior"]))

ranking = RankingConfig(freshness_weight=w_fresh, source_type_weight=w_type,
                        half_life_days=float(half_life), reference_date=ref_date)

# --------------------------------------------------------------------------- #
# Header + question
# --------------------------------------------------------------------------- #
st.title("🧭 Provenance-Aware Graph RAG with RDF")
st.caption(
    "RDF Named Graphs + PROV-O keep track of **where each retrieved fact comes from**. "
    "This demo shows traceability of *external evidence* used to answer a question — "
    "it does **not** explain the internal reasoning of the language model."
)

col_q, col_ex = st.columns([3, 2])
with col_ex:
    example = st.selectbox("Example questions", EXAMPLES, index=0)
with col_q:
    question = st.text_input("Question", value=example)

if not question.strip():
    st.stop()

with st.spinner("Retrieving evidence…"):
    result = rag.ask(question, use_llm_generation=use_llm_gen, use_llm_parsing=use_llm_parse,
                     ranking=ranking, show_priority=show_rank)
r = result.retrieval

# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #
st.subheader("Answer")
mode = result.answer.mode
(st.success if mode.startswith("llm") else st.warning if "failed" in mode else st.info)(
    f"Generation mode: **{mode}**"
)
if result.conflicts:
    st.error("⚠️ Conflicting evidence detected: " + "; ".join(c.describe() for c in result.conflicts))
with st.container(border=True):
    st.markdown(result.answer.text)
if result.answer.guard_note:
    st.caption(f"🛡️ Guard: {result.answer.guard_note}")

# --------------------------------------------------------------------------- #
# Details
# --------------------------------------------------------------------------- #
conflict_edges = {(c.subject, c.predicate_key) for c in result.conflicts}


def dot_graph() -> str:
    lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];',
             'edge [fontname="Helvetica", fontsize=10];']
    if r.start:
        lines.append(f'"{r.start.label}" [fillcolor="#ffe8a3"];')
    edges: dict = {}
    for e in r.evidence:
        edges.setdefault((e.subject_label, e.predicate_key, e.object_label, e.subject), []).append(e.source.short_id)
    for (s, p, o, subj), ids in edges.items():
        color = "#d62728" if (subj, p) in conflict_edges else "#444444"
        lines.append(f'"{s}" -> "{o}" [label="{p}\\n[{", ".join(sorted(ids))}]", color="{color}", fontcolor="{color}"];')
    lines.append("}")
    return "\n".join(lines)


tab_path, tab_ev, tab_sparql, tab_raw, tab_trace, tab_src = st.tabs(
    ["Graph path", "Evidence", "SPARQL", "Raw RDF / provenance", "Pipeline trace", "Sources & dataset"]
)

with tab_path:
    if r.plan:
        schema = " → ".join([f"**{r.start.label}**"] + [f"`{s.render()}` → ?" for s in r.plan])
        st.markdown(f"**Planned traversal:** {schema}")
    elif r.start:
        st.markdown(f"**Describe mode:** all statements about **{r.start.label}**")
    st.caption(f"Entities: {[e.label for e in r.parsed.entities]} · relations: {r.parsed.intents} "
               f"· detection: {r.parsed.method}")
    if r.evidence:
        st.graphviz_chart(dot_graph())
        st.caption("Edge labels show the source id(s) asserting each statement. Red = conflicting values.")
    if r.paths:
        st.markdown("**Instance paths returned by SPARQL** (each hop keeps its own named graph):")
        src_by_graph = {e.graph: e.source.short_id for e in r.evidence}
        for path in r.paths:
            parts = [rag.retriever.label(path[0][0])]
            for _prev, step, graph, node in path:
                parts.append(f"—{step.render()} [{src_by_graph.get(graph, '?')}]→ {rag.retriever.label(node)}")
            st.markdown("- " + " ".join(parts))

with tab_ev:
    rows = []
    for e in r.evidence:
        row = {
            "hop": e.hop, "subject": e.subject_label, "predicate": e.predicate_key, "object": e.object_label,
            "source": e.source.short_id, "publisher": e.source.publisher, "source type": e.source.source_type,
            "published": e.source.published.isoformat(),
            "retrieved": e.source.retrieved.isoformat() if e.source.retrieved else "",
            "conflict": "⚠️" if (e.subject, e.predicate_key) in conflict_edges else "",
        }
        if show_rank:
            row |= {"priority": e.priority, "freshness": e.priority_components.get("freshness"),
                    "type prior": e.priority_components.get("source_type_prior")}
        row |= {"named graph": str(e.graph), "source snippet": e.snippet}
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if show_rank:
            st.caption("Priority = weighted mix of freshness (exponential decay, half-life "
                       f"{half_life} d) and a source-type prior. It only orders evidence; it is not a truth score.")
    else:
        st.write("No evidence retrieved.")

with tab_sparql:
    st.markdown("**Retrieval query** (one `GRAPH ?gN` per hop, so each triple keeps its named graph)")
    st.code(r.sparql or "-- no query", language="sparql")
    if r.rows:
        st.markdown("**Result bindings**")
        st.dataframe(pd.DataFrame([{k: str(v) for k, v in b.items()} for b in r.rows]), hide_index=True)
    st.markdown("**Provenance lookup query**")
    st.code(result.provenance_query or "-- none", language="sparql")

with tab_raw:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retrieved triples in their named graphs (TriG)**")
        st.code(rag.evidence_trig(result) if r.evidence else "", language="turtle")
    with c2:
        st.markdown("**PROV-O metadata for those graphs (Turtle)**")
        st.code(rag.provenance_turtle(r.graphs) if r.graphs else "", language="turtle")
    if result.answer.prompt:
        with st.expander("Prompt sent to the LLM"):
            for m in result.answer.prompt:
                st.markdown(f"**{m['role']}**")
                st.code(m["content"], language="text")

with tab_trace:
    st.dataframe(pd.DataFrame(result.trace), hide_index=True)
    if r.parsed.note:
        st.caption(r.parsed.note)

with tab_src:
    srcs = sorted(all_sources(rag.dataset).values(), key=lambda s: s.short_id)
    st.dataframe(pd.DataFrame([{
        "id": s.short_id, "title": s.title, "publisher": s.publisher, "type": s.source_type,
        "published": s.published.isoformat(), "retrieved": s.retrieved.isoformat() if s.retrieved else "",
        "source URI": str(s.source_uri), "named graph": str(s.graph),
    } for s in srcs]), hide_index=True)
    for rep in rag.reports:
        d = rep.document
        with st.expander(f"[{d.short_id}] {d.title}  ·  {len(rep.facts)} extracted triples"):
            st.code(d.text, language="markdown" if d.fmt == "text" else "text")
            st.markdown("**Extracted facts**")
            st.dataframe(pd.DataFrame([{
                "subject": f.subject, "predicate": f.predicate, "object": f.obj, "from": f.snippet
            } for f in rep.facts]), hide_index=True)
            if rep.skipped:
                st.caption("Skipped: " + " | ".join(rep.skipped))
    st.markdown("**Named graphs in the dataset**")
    st.dataframe(pd.DataFrame(rag.graph_stats()), hide_index=True)
    st.download_button("Download full dataset (TriG)", rag.full_trig(), file_name="provrag_dataset.trig")
    st.caption("Vocabulary: " + ", ".join(f"`ex:{k}` ({s.domain}→{s.range})" for k, s in PREDICATES.items()))
