"""RDF construction: one named graph per source document + registry + provenance."""

from __future__ import annotations

from datetime import datetime

from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from .ingestion import EntityRegistry, ExtractedFact, ExtractionReport
from .provenance import attach_provenance
from .vocab import EX, GRAPH, PREDICATES, REGISTRY_GRAPH, RES, bind_prefixes


def fact_to_triple(fact: ExtractedFact) -> tuple[URIRef, URIRef, URIRef | Literal]:
    spec = PREDICATES[fact.predicate]
    if fact.object_is_entity:
        obj = RES[fact.obj]
    elif spec.datatype == XSD.integer:
        obj = Literal(int(fact.obj.replace(",", "")), datatype=XSD.integer)
    else:
        obj = Literal(fact.obj)
    return RES[fact.subject], spec.uri, obj


def graph_uri_for(graph_name: str) -> URIRef:
    return GRAPH[graph_name]


def build_dataset(
    registry: EntityRegistry,
    reports: list[ExtractionReport],
    generated_at: datetime | None = None,
) -> Dataset:
    ds = Dataset()
    bind_prefixes(ds)

    # Entity registry (types + labels). Not evidence: only used to name things.
    reg = ds.graph(REGISTRY_GRAPH)
    for r in registry.records:
        reg.add((RES[r.key], RDF.type, EX[r.type]))
        reg.add((RES[r.key], RDFS.label, Literal(r.label)))

    # One named graph per source document.
    for report in reports:
        g_uri = graph_uri_for(report.document.graph_name)
        g = ds.graph(g_uri)
        for fact in report.facts:
            g.add(fact_to_triple(fact))
        attach_provenance(ds, report.document, g_uri, generated_at)
    return ds
