"""RDF namespaces and the small domain vocabulary used by the demo.

Every predicate carries a little schema information (domain, range, whether it
is functional) which the retriever uses to plan multi-hop traversals.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdflib import Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

BASE = "http://example.org/provrag/"
EX = Namespace(BASE + "ontology/")
RES = Namespace(BASE + "resource/")
GRAPH = Namespace(BASE + "graph/")
AGENT = Namespace(BASE + "agent/")
ACTIVITY = Namespace(BASE + "activity/")
PAV = Namespace("http://purl.org/pav/")

REGISTRY_GRAPH = GRAPH["entity-registry"]
PROVENANCE_GRAPH = GRAPH["provenance"]

PREFIXES = {
    "ex": EX,
    "res": RES,
    "graph": GRAPH,
    "agent": AGENT,
    "activity": ACTIVITY,
    "prov": PROV,
    "dcterms": DCTERMS,
    "pav": PAV,
    "rdf": RDF,
    "rdfs": RDFS,
    "xsd": XSD,
}

SPARQL_PREFIXES = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in PREFIXES.items())

ENTITY_TYPES = ("Person", "Organization", "City", "Country")
LITERAL = "literal"


@dataclass(frozen=True)
class PredicateSpec:
    key: str
    label: str
    domain: str
    range: str
    functional: bool
    sentence: str  # template with {s} and {o}
    intent_patterns: tuple[str, ...]
    datatype: URIRef | None = None

    @property
    def uri(self) -> URIRef:
        return EX[self.key]

    @property
    def links_entities(self) -> bool:
        return self.range != LITERAL


# Order matters: the path planner explores predicates in this order.
PREDICATES: dict[str, PredicateSpec] = {
    spec.key: spec
    for spec in [
        PredicateSpec(
            "worksAt", "works at", "Person", "Organization", False,
            "{s} works at {o}",
            (r"\bwork(s|ing)?\b", r"\bemployer\b", r"\bemployed\b"),
        ),
        PredicateSpec(
            "jobTitle", "job title", "Person", LITERAL, True,
            "{s}'s job title is {o}",
            (r"job title", r"\brole\b", r"\bposition\b"),
            XSD.string,
        ),
        PredicateSpec(
            "headquarters", "headquarters", "Organization", "City", True,
            "{s} is headquartered in {o}",
            (r"headquarter", r"\bbased\b", r"\bhq\b", r"head ?office", r"\blocated\b"),
        ),
        PredicateSpec(
            "locatedIn", "located in", "City", "Country", True,
            "{s} is located in {o}",
            (r"\bcountry\b",),
        ),
        PredicateSpec(
            "ceo", "CEO", "Organization", "Person", True,
            "The CEO of {s} is {o}",
            (r"\bceo\b", r"chief executive", r"\bleads?\b", r"\bruns\b"),
        ),
        PredicateSpec(
            "foundedYear", "founding year", "Organization", LITERAL, True,
            "{s} was founded in {o}",
            (r"\bfounded\b", r"\bestablished\b"),
            XSD.integer,
        ),
        PredicateSpec(
            "employeeCount", "number of employees", "Organization", LITERAL, True,
            "{s} has {o} employees",
            (r"\bemployees\b", r"\bheadcount\b", r"\bstaff\b", r"how many people"),
            XSD.integer,
        ),
    ]
}

PREDICATE_BY_URI: dict[URIRef, PredicateSpec] = {s.uri: s for s in PREDICATES.values()}


def local_name(uri: URIRef | str) -> str:
    text = str(uri)
    for sep in ("#", "/", ":"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


def bind_prefixes(graph) -> None:
    for prefix, ns in PREFIXES.items():
        graph.bind(prefix, ns, override=True)
