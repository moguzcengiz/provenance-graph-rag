"""Retrieval: question -> entities/intents -> path plan -> SPARQL -> evidence.

The retriever never merges facts: each (subject, predicate, object) is returned
once *per named graph* that asserts it, so every piece of evidence keeps the
identity of the graph (and therefore the source) it came from.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDFS

from .ingestion import EntityRegistry
from .provenance import SourceInfo
from .vocab import (
    LITERAL, PREDICATE_BY_URI, PREDICATES, PROVENANCE_GRAPH, REGISTRY_GRAPH, RES,
    SPARQL_PREFIXES, local_name,
)

MAX_HOPS = 4


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EntityMatch:
    key: str
    label: str
    type: str
    surface: str
    offset: int

    @property
    def uri(self) -> URIRef:
        return RES[self.key]


@dataclass
class ParsedQuestion:
    question: str
    entities: list[EntityMatch]
    intents: list[str]
    method: str  # "deterministic" | "llm" | "llm→deterministic fallback"
    note: str = ""


@dataclass(frozen=True)
class Step:
    predicate: str
    inverse: bool = False

    def render(self) -> str:
        return f"^{self.predicate}" if self.inverse else self.predicate


@dataclass
class Evidence:
    hop: int
    subject: URIRef
    predicate: URIRef
    obj: URIRef | Literal
    graph: URIRef
    subject_label: str
    predicate_key: str
    object_label: str
    source: SourceInfo | None = None
    snippet: str = ""
    priority: float | None = None
    priority_components: dict = field(default_factory=dict)

    @property
    def triple(self):
        return (self.subject, self.predicate, self.obj)


@dataclass
class Conflict:
    subject: URIRef
    subject_label: str
    predicate_key: str
    values: "OrderedDict[str, list[Evidence]]"  # object label -> evidence items

    def describe(self) -> str:
        spec = PREDICATES[self.predicate_key]
        return f"{self.subject_label} — {spec.label}: " + " vs. ".join(self.values)


@dataclass
class RetrievalResult:
    parsed: ParsedQuestion
    start: EntityMatch | None = None
    plan: list[Step] = field(default_factory=list)
    mode: str = "none"  # "path" | "describe" | "none"
    sparql: str = ""
    rows: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    paths: list[list[tuple]] = field(default_factory=list)  # [(node, step, graph, node), ...]

    @property
    def graphs(self) -> set[URIRef]:
        return {e.graph for e in self.evidence}


# --------------------------------------------------------------------------- #
# Question parsing
# --------------------------------------------------------------------------- #


def detect_intents(question: str) -> list[str]:
    q = question.lower()
    return [key for key, spec in PREDICATES.items() if any(re.search(p, q) for p in spec.intent_patterns)]


def parse_question(question: str, registry: EntityRegistry) -> ParsedQuestion:
    seen, entities = set(), []
    for key, surface, offset in registry.find_mentions(question):
        if key in seen:
            continue
        seen.add(key)
        rec = registry.by_key[key]
        entities.append(EntityMatch(key, rec.label, rec.type, surface, offset))
    return ParsedQuestion(question, entities, detect_intents(question), "deterministic")


def parse_question_with_llm(question: str, registry: EntityRegistry, llm) -> ParsedQuestion:
    """Ask the LLM to map the question onto the *closed* entity/relation vocabulary.

    The output is validated against the registry; anything unknown is dropped and
    we fall back to deterministic matching if nothing usable remains.
    """
    fallback = parse_question(question, registry)
    entity_list = "\n".join(f"- {r.label} ({r.type})" for r in registry.records)
    relation_list = "\n".join(f"- {k}: {s.label} ({s.domain} -> {s.range})" for k, s in PREDICATES.items())
    prompt = (
        "Map the user question onto a fixed vocabulary. Reply with JSON only, of the form "
        '{"entities": ["<entity label>", ...], "relations": ["<relation key>", ...]}.\n'
        "Use only labels/keys from the lists. Include every relation that the question "
        "needs to traverse (e.g. 'company X works for' implies worksAt).\n\n"
        f"Entities:\n{entity_list}\n\nRelations:\n{relation_list}\n\nQuestion: {question}"
    )
    try:
        raw = llm.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2000)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in LLM output: {raw[:200]!r}")
        data = json.loads(match.group(0))
        entities = []
        for name in data.get("entities", []):
            key = registry.resolve(str(name))
            if key and key not in {e.key for e in entities}:
                rec = registry.by_key[key]
                offset = question.lower().find(rec.label.split()[0].lower())
                entities.append(EntityMatch(key, rec.label, rec.type, str(name), offset if offset >= 0 else 999))
        intents = [r for r in data.get("relations", []) if r in PREDICATES]
        if not entities:
            fallback.method = "llm→deterministic fallback"
            fallback.note = "LLM returned no known entity."
            return fallback
        entities.sort(key=lambda e: e.offset)
        return ParsedQuestion(question, entities, intents or fallback.intents, "llm", f"raw LLM output: {raw}")
    except Exception as exc:  # network, JSON, ...
        fallback.method = "llm→deterministic fallback"
        fallback.note = f"LLM parsing failed: {exc}"
        return fallback


# --------------------------------------------------------------------------- #
# Path planning (schema-guided traversal)
# --------------------------------------------------------------------------- #


def plan_path(start_type: str, intents: list[str], max_hops: int = MAX_HOPS) -> list[Step] | None:
    """Breadth-first search over the *schema* (domain/range of predicates).

    Finds the shortest predicate sequence starting at ``start_type`` that uses
    every intent and ends with one. E.g. Person + {worksAt, headquarters}
    -> [worksAt, headquarters]; Person + {locatedIn}
    -> [worksAt, headquarters, locatedIn].
    """
    target = set(intents)
    if not target:
        return None
    queue: deque[tuple[str, list[Step]]] = deque([(start_type, [])])
    while queue:
        node_type, path = queue.popleft()
        if path and path[-1].predicate in target and target <= {s.predicate for s in path}:
            return path
        if len(path) >= max_hops or node_type == LITERAL:
            continue
        for spec in PREDICATES.values():
            last = path[-1] if path else None
            if spec.domain == node_type and not (last and last.predicate == spec.key and last.inverse):
                queue.append((spec.range, path + [Step(spec.key)]))
            if spec.links_entities and spec.range == node_type and not (
                last and last.predicate == spec.key and not last.inverse
            ):
                queue.append((spec.domain, path + [Step(spec.key, inverse=True)]))
    return None


def path_query(start: URIRef, plan: list[Step]) -> str:
    select = " ".join(f"?n{i} ?g{i}" for i in range(1, len(plan) + 1))
    patterns = []
    for i, step in enumerate(plan, start=1):
        prev = f"res:{local_name(start)}" if i == 1 else f"?n{i - 1}"
        pred = f"ex:{step.predicate}"
        triple = f"?n{i} {pred} {prev} ." if step.inverse else f"{prev} {pred} ?n{i} ."
        patterns.append(f"  # hop {i}: {step.render()}\n  GRAPH ?g{i} {{ {triple} }}")
    order = " ".join(f"?n{i}" for i in range(1, len(plan) + 1))
    body = "\n".join(patterns)
    return f"{SPARQL_PREFIXES}\n\nSELECT {select}\nWHERE {{\n{body}\n}}\nORDER BY {order}"


def describe_query(start: URIRef) -> str:
    s = f"res:{local_name(start)}"
    return f"""{SPARQL_PREFIXES}

SELECT ?s ?p ?o ?g
WHERE {{
  {{ GRAPH ?g {{ {s} ?p ?o }} BIND({s} AS ?s) }}
  UNION
  {{ GRAPH ?g {{ ?s ?p {s} }} BIND({s} AS ?o) }}
  FILTER(?g NOT IN (<{REGISTRY_GRAPH}>, <{PROVENANCE_GRAPH}>))
}}
ORDER BY ?p ?g"""


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


class Retriever:
    def __init__(self, ds: Dataset, registry: EntityRegistry, snippets: dict | None = None):
        self.ds = ds
        self.registry = registry
        self.snippets = snippets or {}  # (graph, s, p, o) -> source sentence/row
        self._labels = {s: str(o) for s, o in ds.graph(REGISTRY_GRAPH).subject_objects(RDFS.label)}

    def label(self, term) -> str:
        if isinstance(term, Literal):
            return str(term)
        return self._labels.get(term, local_name(term))

    def _evidence(self, hop, s, p, o, g) -> Evidence:
        return Evidence(
            hop=hop, subject=s, predicate=p, obj=o, graph=g,
            subject_label=self.label(s),
            predicate_key=local_name(p),
            object_label=self.label(o),
            snippet=self.snippets.get((g, s, p, o), ""),
        )

    def retrieve(self, parsed: ParsedQuestion) -> RetrievalResult:
        result = RetrievalResult(parsed)
        if not parsed.entities:
            return result

        # Try entities in question order; first one with a valid plan wins.
        for ent in parsed.entities:
            plan = plan_path(ent.type, parsed.intents)
            if plan:
                result.start, result.plan, result.mode = ent, plan, "path"
                self._run_path(result)
                if result.evidence:
                    break
        if not result.evidence:
            result.start, result.plan, result.mode = parsed.entities[0], [], "describe"
            self._run_describe(result)

        result.conflicts = find_conflicts(result.evidence)
        return result

    def _run_path(self, result: RetrievalResult) -> None:
        start, plan = result.start.uri, result.plan
        result.sparql = path_query(start, plan)
        result.rows, result.evidence, result.paths = [], [], []
        seen = set()
        for row in self.ds.query(result.sparql):
            binding = {str(k): v for k, v in row.asdict().items()}
            result.rows.append(binding)
            prev, path = start, []
            for i, step in enumerate(plan, start=1):
                node, graph = binding[f"n{i}"], binding[f"g{i}"]
                s, o = (node, prev) if step.inverse else (prev, node)
                p = PREDICATES[step.predicate].uri
                key = (i, s, p, o, graph)
                if key not in seen:
                    seen.add(key)
                    result.evidence.append(self._evidence(i, s, p, o, graph))
                path.append((prev, step, graph, node))
                prev = node
            result.paths.append(path)

    def _run_describe(self, result: RetrievalResult) -> None:
        result.sparql = describe_query(result.start.uri)
        for row in self.ds.query(result.sparql):
            binding = {str(k): v for k, v in row.asdict().items()}
            if binding["p"] not in PREDICATE_BY_URI:
                continue
            result.rows.append(binding)
            result.evidence.append(self._evidence(1, binding["s"], binding["p"], binding["o"], binding["g"]))


def find_conflicts(evidence: list[Evidence]) -> list[Conflict]:
    """Functional predicates with >1 distinct value for the same subject.

    Nothing is removed: a conflict just groups the competing evidence items.
    """
    groups: "OrderedDict[tuple, list[Evidence]]" = OrderedDict()
    for ev in evidence:
        groups.setdefault((ev.subject, ev.predicate_key), []).append(ev)
    conflicts = []
    for (subject, pkey), items in groups.items():
        spec = PREDICATES.get(pkey)
        values: "OrderedDict[str, list[Evidence]]" = OrderedDict()
        for ev in items:
            values.setdefault(ev.object_label, []).append(ev)
        if spec and spec.functional and len(values) > 1:
            conflicts.append(Conflict(subject, items[0].subject_label, pkey, values))
    return conflicts
