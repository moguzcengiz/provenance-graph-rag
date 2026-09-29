"""Ingestion: load source documents + metadata and extract facts from them.

Extraction is deliberately simple and deterministic (regular expressions over
sentences for text sources, a column mapping for the CSV dump). Every
extracted fact keeps the sentence/row it came from, so it can be shown next to
the evidence later.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import DATA_DIR, SOURCES_DIR
from .vocab import PREDICATES


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceDocument:
    short_id: str
    graph_name: str
    path: Path
    fmt: str
    source_uri: str
    title: str
    publisher: str
    publisher_uri: str
    source_type: str
    published: date
    retrieved: date | None
    text: str


@dataclass(frozen=True)
class EntityRecord:
    key: str
    label: str
    type: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedFact:
    subject: str  # entity key
    predicate: str  # predicate key (see vocab.PREDICATES)
    obj: str  # entity key or literal lexical value
    object_is_entity: bool
    snippet: str  # sentence / CSV row the fact was extracted from


@dataclass
class ExtractionReport:
    document: SourceDocument
    facts: list[ExtractedFact] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Entity registry
# --------------------------------------------------------------------------- #


class EntityRegistry:
    """Known entities with aliases; used for extraction and question parsing."""

    def __init__(self, records: list[EntityRecord]):
        self.records = records
        self.by_key = {r.key: r for r in records}
        self._alias_to_key: dict[str, str] = {}
        for r in records:
            for alias in (r.label, *r.aliases):
                self._alias_to_key[alias.lower()] = r.key
        surface_forms = {a for r in records for a in (r.label, *r.aliases)}
        aliases = sorted(surface_forms, key=len, reverse=True)
        # Longest alias first so "Acme AI" wins over "Acme".
        self.alias_pattern = "|".join(re.escape(a) for a in aliases)
        self._mention_re = re.compile(rf"\b(?:{self.alias_pattern})\b", re.IGNORECASE)

    def resolve(self, surface: str) -> str | None:
        return self._alias_to_key.get(surface.strip().lower())

    def find_mentions(self, text: str) -> list[tuple[str, str, int]]:
        """Return (entity_key, surface_form, offset) for each mention, in order."""
        out = []
        for m in self._mention_re.finditer(text):
            key = self.resolve(m.group(0))
            if key:
                out.append((key, m.group(0), m.start()))
        return out


def load_registry(path: Path = DATA_DIR / "entities.json") -> EntityRegistry:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EntityRegistry(
        [EntityRecord(r["key"], r["label"], r["type"], tuple(r.get("aliases", []))) for r in raw]
    )


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def load_documents(sources_dir: Path = SOURCES_DIR) -> list[SourceDocument]:
    manifest = json.loads((sources_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = []
    for m in manifest:
        path = sources_dir / m["file"]
        docs.append(
            SourceDocument(
                short_id=m["short_id"],
                graph_name=m["graph"],
                path=path,
                fmt=m["format"],
                source_uri=m["source_uri"],
                title=m["title"],
                publisher=m["publisher"],
                publisher_uri=m["publisher_uri"],
                source_type=m["source_type"],
                published=date.fromisoformat(m["published"]),
                retrieved=date.fromisoformat(m["retrieved"]) if m.get("retrieved") else None,
                text=path.read_text(encoding="utf-8"),
            )
        )
    return docs


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

# {s}/{o} = captured subject/object entity mention, {x} = uncaptured mention,
# (?P<lit>...) = literal object.
_TEXT_RULES: list[tuple[str, str]] = [
    ("worksAt", r"{s} works (?:at|for) {o}"),
    ("jobTitle", r"{s} works (?:at|for) {x} as (?:an? )?(?P<lit>[A-Z][A-Za-z ]*[A-Za-z])"),
    ("headquarters", r"{s} (?:is|remains) headquartered in {o}"),
    ("headquarters", r"{s} (?:has )?moved its headquarters (?:from {x} )?to {o}"),
    ("ceo", r"{o} is the (?:CEO|chief executive officer) of {s}"),
    ("foundedYear", r"{s} was founded in (?P<lit>\d{4})"),
    (
        "employeeCount",
        r"{s} (?:has|employs) (?:approximately |about |around |roughly )?"
        r"(?P<lit>\d[\d,]*) (?:employees|people|staff)",
    ),
    ("locatedIn", r"{s} is located in {o}"),
]

# CSV attribute name -> predicate key
CSV_ATTRIBUTE_MAP = {
    "hq_city": "headquarters",
    "ceo": "ceo",
    "employees": "employeeCount",
    "founded": "foundedYear",
    "employer": "worksAt",
    "job_title": "jobTitle",
    "country": "locatedIn",
}


def _compile_rules(registry: EntityRegistry) -> list[tuple[str, re.Pattern]]:
    alt = registry.alias_pattern
    compiled = []
    for predicate, template in _TEXT_RULES:
        pattern = (
            template.replace("{s}", rf"\b(?P<s>{alt})\b")
            .replace("{o}", rf"\b(?P<o>{alt})\b")
            .replace("{x}", rf"\b(?:{alt})\b")
        )
        compiled.append((predicate, re.compile(pattern)))
    return compiled


def split_sentences(text: str) -> list[str]:
    sentences = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(line.strip() for line in block.splitlines() if line.strip())
        block = block.lstrip("#* ").strip()
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip())
    return sentences


def _extract_text(doc: SourceDocument, registry: EntityRegistry, report: ExtractionReport):
    rules = _compile_rules(registry)
    for sentence in split_sentences(doc.text):
        for predicate, regex in rules:
            for m in regex.finditer(sentence):
                subject = registry.resolve(m.group("s"))
                if "o" in regex.groupindex:
                    obj, is_entity = registry.resolve(m.group("o")), True
                else:
                    obj, is_entity = m.group("lit").strip(), False
                if subject and obj:
                    report.facts.append(ExtractedFact(subject, predicate, obj, is_entity, sentence))


def _extract_csv(doc: SourceDocument, registry: EntityRegistry, report: ExtractionReport):
    for row in csv.DictReader(io.StringIO(doc.text)):
        snippet = ",".join(row.values())
        predicate = CSV_ATTRIBUTE_MAP.get(row["attribute"])
        subject = registry.resolve(row["entity"])
        if not predicate or not subject:
            report.skipped.append(f"{snippet}  (unmapped attribute or unknown entity)")
            continue
        spec = PREDICATES[predicate]
        if spec.links_entities:
            obj = registry.resolve(row["value"])
            if not obj:
                report.skipped.append(f"{snippet}  (unknown entity '{row['value']}')")
                continue
            report.facts.append(ExtractedFact(subject, predicate, obj, True, snippet))
        else:
            report.facts.append(ExtractedFact(subject, predicate, row["value"].strip(), False, snippet))


def extract_facts(doc: SourceDocument, registry: EntityRegistry) -> ExtractionReport:
    report = ExtractionReport(doc)
    if doc.fmt == "csv":
        _extract_csv(doc, registry, report)
    else:
        _extract_text(doc, registry, report)
    # de-duplicate while keeping first snippet
    seen, unique = set(), []
    for f in report.facts:
        key = (f.subject, f.predicate, f.obj)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    report.facts = unique
    return report
