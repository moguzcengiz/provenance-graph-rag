"""End-to-end pipeline wiring all modules together.

Question -> entity detection -> graph traversal / SPARQL -> relevant triples
-> provenance lookup -> evidence ranking -> answer generation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Dataset

from .config import DATA_DIR, SOURCES_DIR
from .generation import GeneratedAnswer, generate_answer
from .ingestion import ExtractionReport, extract_facts, load_documents, load_registry
from .provenance import describe_provenance, lookup_sources
from .ranking import RankingConfig, rank_evidence
from .rdf_builder import build_dataset, fact_to_triple, graph_uri_for
from .retrieval import RetrievalResult, Retriever, parse_question, parse_question_with_llm
from .vocab import PROVENANCE_GRAPH, bind_prefixes

_AUTO = object()


@dataclass
class PipelineResult:
    question: str
    retrieval: RetrievalResult
    answer: GeneratedAnswer
    provenance_query: str = ""
    trace: list[dict] = field(default_factory=list)

    @property
    def evidence(self):
        return self.retrieval.evidence

    @property
    def conflicts(self):
        return self.retrieval.conflicts


class ProvenanceGraphRAG:
    def __init__(self, data_dir: Path = DATA_DIR, sources_dir: Path = SOURCES_DIR, llm=_AUTO):
        self.registry = load_registry(data_dir / "entities.json")
        self.documents = load_documents(sources_dir)
        self.reports: list[ExtractionReport] = [extract_facts(d, self.registry) for d in self.documents]
        self.dataset: Dataset = build_dataset(self.registry, self.reports)

        snippets = {}
        for rep in self.reports:
            g = graph_uri_for(rep.document.graph_name)
            for f in rep.facts:
                snippets[(g, *fact_to_triple(f))] = f.snippet
        self.retriever = Retriever(self.dataset, self.registry, snippets)

        if llm is _AUTO:
            from .llm import llm_from_env
            llm = llm_from_env()
        self.llm = llm

    @property
    def llm_available(self) -> bool:
        return self.llm is not None

    def ask(
        self,
        question: str,
        use_llm_generation: bool = True,
        use_llm_parsing: bool = False,
        ranking: RankingConfig | None = None,
        show_priority: bool = True,
    ) -> PipelineResult:
        trace = []

        def step(name, detail, t0):
            trace.append({"step": name, "detail": detail, "ms": round((time.perf_counter() - t0) * 1000, 1)})

        # 1. entity + relation detection
        t0 = time.perf_counter()
        if use_llm_parsing and self.llm:
            parsed = parse_question_with_llm(question, self.registry, self.llm)
        else:
            parsed = parse_question(question, self.registry)
        step(
            "1. Entity & relation detection",
            f"method={parsed.method}; entities={[e.label for e in parsed.entities]}; relations={parsed.intents}",
            t0,
        )

        # 2+3. graph traversal via SPARQL -> relevant triples (one item per named graph)
        t0 = time.perf_counter()
        result = self.retriever.retrieve(parsed)
        plan = " -> ".join(s.render() for s in result.plan) or result.mode
        step(
            "2. Graph traversal / SPARQL",
            f"start={result.start.label if result.start else None}; plan={plan}; rows={len(result.rows)}",
            t0,
        )
        trace.append({
            "step": "3. Relevant triples",
            "detail": f"{len(result.evidence)} evidence items from {len(result.graphs)} named graphs",
            "ms": 0.0,
        })

        # 4. provenance lookup
        t0 = time.perf_counter()
        sources, prov_query = lookup_sources(self.dataset, result.graphs)
        for ev in result.evidence:
            ev.source = sources.get(ev.graph)
        step("4. Provenance lookup", f"{len(sources)} sources: {sorted(s.short_id for s in sources.values())}", t0)

        # 5. evidence ranking (prioritization only; nothing is dropped)
        t0 = time.perf_counter()
        result.evidence = rank_evidence(result.evidence, ranking)
        step(
            "5. Evidence ranking",
            f"{len(result.evidence)} items ordered by freshness/source type; "
            f"{len(result.conflicts)} conflict(s) preserved",
            t0,
        )

        # 6. answer generation
        t0 = time.perf_counter()
        answer = generate_answer(
            question, result, self.llm if use_llm_generation else None, show_priority
        )
        step("6. Answer generation", f"mode={answer.mode}", t0)

        return PipelineResult(question, result, answer, prov_query, trace)

    # ------------------------------------------------------------------ #
    # Serialisation helpers for the UI
    # ------------------------------------------------------------------ #

    def evidence_trig(self, result: PipelineResult) -> str:
        """Only the retrieved triples, still inside their original named graphs."""
        out = Dataset()
        bind_prefixes(out)
        for ev in result.evidence:
            out.graph(ev.graph).add(ev.triple)
        return out.serialize(format="trig")

    def provenance_turtle(self, graph_uris) -> str:
        return describe_provenance(self.dataset, graph_uris).serialize(format="turtle")

    def full_trig(self) -> str:
        return self.dataset.serialize(format="trig")

    def graph_stats(self) -> list[dict]:
        rows = []
        for g in self.dataset.graphs():
            n = len(g)
            if n == 0:
                continue
            rows.append({"named graph": str(g.identifier), "triples": n,
                         "kind": "provenance" if g.identifier == PROVENANCE_GRAPH else ""})
        return rows
