"""Command-line entry point: python -m provrag.cli "Where is the company Alice works for headquartered?" """

from __future__ import annotations

import argparse

from .pipeline import ProvenanceGraphRAG


def main() -> None:
    ap = argparse.ArgumentParser(description="Provenance-aware Graph RAG demo")
    ap.add_argument("question", nargs="?", default="Where is the company Alice works for headquartered?")
    ap.add_argument("--no-llm", action="store_true", help="force template-based generation")
    ap.add_argument("--dump-trig", action="store_true", help="print the full RDF dataset and exit")
    args = ap.parse_args()

    rag = ProvenanceGraphRAG(llm=None) if args.no_llm else ProvenanceGraphRAG()
    if args.dump_trig:
        print(rag.full_trig())
        return
    res = rag.ask(args.question)
    print("=" * 78)
    print(res.answer.text)
    print("=" * 78)
    print("Generation mode:", res.answer.mode)
    for t in res.trace:
        print(f"  {t['step']}: {t['detail']}")
    print("\nEvidence:")
    for e in res.evidence:
        print(f"  hop{e.hop} [{e.source.short_id}] {e.subject_label} --{e.predicate_key}--> {e.object_label}"
              f"   ({e.source.publisher}, {e.source.published}, priority={e.priority})")
    print("\nSPARQL:\n" + res.retrieval.sparql)


if __name__ == "__main__":
    main()
