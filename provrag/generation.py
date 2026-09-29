"""Answer generation from retrieved evidence (template-based or LLM-based).

Both generators only see *external evidence* (triples + provenance). The
answer cites sources; it is not an account of how an LLM reasons internally.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .retrieval import Conflict, Evidence, RetrievalResult
from .vocab import PREDICATES


@dataclass
class GeneratedAnswer:
    text: str
    mode: str  # "template" | "llm" | "template (LLM failed: ...)"
    prompt: list[dict] | None = None
    guard_note: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def cite(items: list[Evidence]) -> str:
    ids = sorted({e.source.short_id for e in items if e.source}, key=lambda s: (len(s), s))
    return "[" + ", ".join(ids) + "]"


def _group(items: list[Evidence], key) -> "OrderedDict":
    groups: OrderedDict = OrderedDict()
    for ev in items:
        groups.setdefault(key(ev), []).append(ev)
    return groups


def _latest(items: list[Evidence]):
    return max(e.source.published for e in items if e.source)


def conflict_paragraph(conflict: Conflict, show_priority: bool = True) -> str:
    spec = PREDICATES[conflict.predicate_key]
    values = sorted(conflict.values.items(), key=lambda kv: _latest(kv[1]), reverse=True)
    lines = [
        f"**Sources disagree about {conflict.subject_label}'s {spec.label}.** "
        f"The retrieved evidence contains {len(values)} different values:"
    ]
    for value, items in values:
        srcs = "; ".join(
            f"[{e.source.short_id}] {e.source.publisher} ({e.source.source_type}, "
            f"published {e.source.published.isoformat()})"
            for e in sorted(items, key=lambda e: e.source.published, reverse=True)
        )
        lines.append(f"- **{value}** — {srcs}")
    newest_value, newest_items = values[0]
    newest_date = _latest(newest_items)
    newest_srcs = [e for e in newest_items if e.source.published == newest_date]
    lines.append(
        f"\nThe most recently published statement ({newest_date.isoformat()}, "
        f"{cite(newest_srcs)}) says **{newest_value}**; the other values come from older sources. "
        "All values are reported because the system does not decide which one is true."
    )
    if show_priority and all(e.priority is not None for items in conflict.values.values() for e in items):
        top = max((e for items in conflict.values.values() for e in items), key=lambda e: e.priority)
        lines.append(
            f"\n_Highest retrieval priority: **{top.object_label}** [{top.source.short_id}] "
            f"(score {top.priority:.2f}, based only on freshness and source type — "
            "a retrieval prioritization, not a truth score)._"
        )
    return "\n".join(lines)


def sources_footer(evidence: list[Evidence]) -> str:
    sources = {e.source.short_id: e.source for e in evidence if e.source}
    ordered = sorted(sources.values(), key=lambda s: (len(s.short_id), s.short_id))
    return "**Sources**\n" + "\n".join(f"- {s.citation()}" for s in ordered)


# --------------------------------------------------------------------------- #
# Template generator (no LLM required)
# --------------------------------------------------------------------------- #


def template_answer(result: RetrievalResult, show_priority: bool = True) -> str:
    parsed = result.parsed
    if not parsed.entities:
        return (
            "I could not recognise a known entity in the question, so nothing was retrieved "
            "from the knowledge graph. Try mentioning e.g. Alice Schmidt, Acme AI, Bob Chen or Munich."
        )
    if not result.evidence:
        return f"The knowledge graph contains no evidence about **{result.start.label}** for this question."

    conflict_map = {(c.subject, c.predicate_key): c for c in result.conflicts}
    parts: list[str] = []
    if result.mode == "describe":
        parts.append(
            f"No specific relation was recognised in the question, so here is everything the "
            f"graph states about **{result.start.label}**:"
        )

    for hop, hop_items in _group(result.evidence, lambda e: e.hop).items():
        for (subject, pkey), items in _group(hop_items, lambda e: (e.subject, e.predicate_key)).items():
            if (subject, pkey) in conflict_map:
                parts.append(conflict_paragraph(conflict_map[(subject, pkey)], show_priority))
                continue
            spec = PREDICATES[pkey]
            sentences = []
            for value, v_items in _group(items, lambda e: e.object_label).items():
                n = len({e.graph for e in v_items})
                agree = f" (stated by {n} independent sources)" if n > 1 else ""
                sentences.append(f"{spec.sentence.format(s=v_items[0].subject_label, o=value)} {cite(v_items)}{agree}.")
            parts.append(" ".join(sentences))

    if result.mode == "path" and len(result.plan) > 1:
        final_values = {e.object_label if not result.plan[-1].inverse else e.subject_label
                        for e in result.evidence if e.hop == len(result.plan)}
        if len(final_values) == 1 and result.conflicts:
            parts.append(
                f"Although the sources disagree on an intermediate step, every retrieved path "
                f"leads to **{next(iter(final_values))}**."
            )

    parts.append(sources_footer(result.evidence))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# LLM generator
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are the answer-generation step of a provenance-aware Graph RAG system.
Rules:
1. Answer ONLY from the evidence listed by the user. Do not use outside knowledge.
2. Cite the source id in square brackets (e.g. [S2]) after every factual claim.
3. If the evidence contains different values for the same property (listed under CONFLICTS),
   you MUST state explicitly that the sources disagree, list EVERY conflicting value with its
   source id, publisher and publication date, and must NOT silently pick one value.
   You may point out which statement is the most recently published.
4. The "priority" numbers are a retrieval prioritization based on freshness and source type.
   Never present them as probabilities, confidence or truth.
5. Do not describe or explain your own internal reasoning; just answer with cited evidence.
6. Be concise (max ~170 words). Use Markdown."""


def evidence_block(result: RetrievalResult) -> str:
    lines = []
    for e in result.evidence:
        s = e.source
        pr = f" | priority {e.priority:.2f}" if e.priority is not None else ""
        lines.append(
            f"[{s.short_id}] hop {e.hop}: ({e.subject_label}) --{e.predicate_key}--> ({e.object_label})"
            f" | publisher: {s.publisher} | type: {s.source_type} | published: {s.published.isoformat()}{pr}"
        )
    return "\n".join(lines)


def build_llm_messages(question: str, result: RetrievalResult) -> list[dict]:
    path = " -> ".join([result.start.label] + [s.render() for s in result.plan]) if result.plan else "(describe entity)"
    conflicts = "\n".join(f"- {c.describe()}" for c in result.conflicts) or "- none detected"
    user = (
        f"QUESTION: {question}\n\n"
        f"RETRIEVED GRAPH PATH: {path}\n\n"
        f"EVIDENCE (one line per statement per source):\n{evidence_block(result)}\n\n"
        f"CONFLICTS:\n{conflicts}\n\n"
        "Write the answer now."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def enforce_conflicts(text: str, result: RetrievalResult, show_priority: bool) -> tuple[str, str]:
    """Guard: if the LLM omitted a conflicting value, append the template disclosure."""
    missing = [
        c for c in result.conflicts
        if not all(value.lower() in text.lower() for value in c.values)
    ]
    if not missing:
        return text, ""
    note = "The generated answer omitted conflicting evidence; the disclosure below was added automatically."
    extra = "\n\n".join(conflict_paragraph(c, show_priority) for c in missing)
    return f"{text}\n\n---\n_{note}_\n\n{extra}", note


def generate_answer(
    question: str,
    result: RetrievalResult,
    llm=None,
    show_priority: bool = True,
) -> GeneratedAnswer:
    template = template_answer(result, show_priority)
    if llm is None or not result.evidence:
        return GeneratedAnswer(template, "template")
    messages = build_llm_messages(question, result)
    try:
        text = llm.complete(messages)
        if not text:
            raise ValueError("empty completion")
    except Exception as exc:
        return GeneratedAnswer(template, f"template (LLM failed: {exc})", messages)
    text, note = enforce_conflicts(text, result, show_priority)
    if "**Sources**" not in text:
        text = f"{text}\n\n{sources_footer(result.evidence)}"
    return GeneratedAnswer(text, f"llm ({llm.model})", messages, note)
