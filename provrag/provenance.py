"""PROV-O provenance: attach metadata to named graphs and look it up again.

Each fact graph G gets (in the dedicated provenance graph):

    G       a prov:Entity ; prov:wasDerivedFrom SRC ; prov:wasAttributedTo PUB ;
            prov:wasGeneratedBy ACT ; prov:generatedAtTime ... ;
            pav:retrievedFrom SRC ; pav:retrievedOn ... ; ex:sourceId "S2" .
    SRC     a prov:Entity, ex:<SourceType> ; dcterms:title ... ;
            dcterms:publisher PUB ; dcterms:issued ... ; ex:sourceType ex:<SourceType> .
    PUB     a prov:Agent, prov:Organization ; rdfs:label ... .
    ACT     a prov:Activity ; prov:used SRC ; prov:generated G ;
            prov:wasAssociatedWith agent:provrag-extractor .
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from rdflib import Dataset, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

from .ingestion import SourceDocument
from .vocab import (
    ACTIVITY, AGENT, EX, PAV, PROVENANCE_GRAPH, SPARQL_PREFIXES, bind_prefixes, local_name,
)

EXTRACTOR_AGENT = AGENT["provrag-extractor"]


@dataclass(frozen=True)
class SourceInfo:
    graph: URIRef
    short_id: str
    source_uri: URIRef
    title: str
    publisher: str
    publisher_uri: URIRef
    source_type: str
    published: date
    retrieved: date | None

    def citation(self) -> str:
        return (
            f"[{self.short_id}] {self.title} — {self.publisher} "
            f"({self.source_type}, published {self.published.isoformat()})"
        )


def attach_provenance(
    ds: Dataset,
    doc: SourceDocument,
    graph_uri: URIRef,
    generated_at: datetime | None = None,
) -> None:
    prov_g = ds.graph(PROVENANCE_GRAPH)
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0)

    src = URIRef(doc.source_uri)
    pub = URIRef(doc.publisher_uri)
    act = ACTIVITY[f"extract-{doc.graph_name}"]
    stype = EX[doc.source_type]

    # the named graph itself
    prov_g.add((graph_uri, RDF.type, PROV.Entity))
    prov_g.add((graph_uri, RDFS.label, Literal(f"Facts extracted from: {doc.title}")))
    prov_g.add((graph_uri, EX.sourceId, Literal(doc.short_id)))
    prov_g.add((graph_uri, PROV.wasDerivedFrom, src))
    prov_g.add((graph_uri, PROV.wasAttributedTo, pub))
    prov_g.add((graph_uri, PROV.wasGeneratedBy, act))
    prov_g.add((graph_uri, PROV.generatedAtTime, Literal(generated_at, datatype=XSD.dateTime)))
    prov_g.add((graph_uri, PAV.retrievedFrom, src))
    if doc.retrieved:
        prov_g.add((graph_uri, PAV.retrievedOn, Literal(doc.retrieved, datatype=XSD.date)))

    # the original source document
    prov_g.add((src, RDF.type, PROV.Entity))
    prov_g.add((src, RDF.type, stype))
    prov_g.add((src, EX.sourceType, stype))
    prov_g.add((src, DCTERMS.title, Literal(doc.title)))
    prov_g.add((src, DCTERMS.publisher, pub))
    prov_g.add((src, PROV.wasAttributedTo, pub))
    prov_g.add((src, DCTERMS.issued, Literal(doc.published, datatype=XSD.date)))

    # the publisher
    prov_g.add((pub, RDF.type, PROV.Agent))
    prov_g.add((pub, RDF.type, PROV.Organization))
    prov_g.add((pub, RDFS.label, Literal(doc.publisher)))

    # the extraction activity
    prov_g.add((act, RDF.type, PROV.Activity))
    prov_g.add((act, PROV.used, src))
    prov_g.add((act, PROV.generated, graph_uri))
    prov_g.add((act, PROV.wasAssociatedWith, EXTRACTOR_AGENT))
    prov_g.add((act, PROV.endedAtTime, Literal(generated_at, datatype=XSD.dateTime)))
    prov_g.add((EXTRACTOR_AGENT, RDF.type, PROV.SoftwareAgent))
    prov_g.add((EXTRACTOR_AGENT, RDFS.label, Literal("provrag deterministic extractor")))


def provenance_query(graph_uris: list[URIRef]) -> str:
    values = " ".join(f"<{g}>" for g in graph_uris)
    return f"""{SPARQL_PREFIXES}

SELECT ?graph ?sourceId ?source ?title ?publisher ?publisherName ?sourceType ?issued ?retrieved
WHERE {{
  GRAPH <{PROVENANCE_GRAPH}> {{
    VALUES ?graph {{ {values} }}
    ?graph prov:wasDerivedFrom ?source ;
           ex:sourceId ?sourceId .
    ?source dcterms:title ?title ;
            dcterms:publisher ?publisher ;
            dcterms:issued ?issued ;
            ex:sourceType ?sourceType .
    ?publisher rdfs:label ?publisherName .
    OPTIONAL {{ ?graph pav:retrievedOn ?retrieved }}
  }}
}}
ORDER BY ?sourceId"""


def lookup_sources(ds: Dataset, graph_uris) -> tuple[dict[URIRef, SourceInfo], str]:
    """Return {graph_uri: SourceInfo} plus the SPARQL query that was used."""
    graph_uris = sorted(set(graph_uris), key=str)
    if not graph_uris:
        return {}, ""
    query = provenance_query(graph_uris)
    out: dict[URIRef, SourceInfo] = {}
    for row in ds.query(query):
        out[row.graph] = SourceInfo(
            graph=row.graph,
            short_id=str(row.sourceId),
            source_uri=row.source,
            title=str(row.title),
            publisher=str(row.publisherName),
            publisher_uri=row.publisher,
            source_type=local_name(row.sourceType),
            published=row.issued.toPython(),
            retrieved=row.retrieved.toPython() if row.retrieved is not None else None,
        )
    return out, query


def all_sources(ds: Dataset) -> dict[URIRef, SourceInfo]:
    prov_g = ds.graph(PROVENANCE_GRAPH)
    graphs = list(prov_g.subjects(PROV.wasDerivedFrom, None))
    graphs = [g for g in graphs if (g, EX.sourceId, None) in prov_g]
    return lookup_sources(ds, graphs)[0]


def describe_provenance(ds: Dataset, graph_uris) -> Graph:
    """Collect the provenance sub-graph (2 hops) around the given named graphs."""
    prov_g = ds.graph(PROVENANCE_GRAPH)
    out = Graph()
    bind_prefixes(out)
    frontier, seen = set(graph_uris), set()
    for _ in range(3):
        nxt = set()
        for node in frontier - seen:
            seen.add(node)
            for p, o in prov_g.predicate_objects(node):
                out.add((node, p, o))
                if isinstance(o, URIRef) and p != RDF.type:
                    nxt.add(o)
        frontier = nxt
    return out
