"""Every source lives in its own named graph, and every graph has PROV-O metadata."""

from rdflib import URIRef
from rdflib.namespace import DCTERMS, PROV, RDF

from provrag.vocab import EX, GRAPH, PAV, PROVENANCE_GRAPH, RES


def fact_graphs(rag):
    return [GRAPH[d.graph_name] for d in rag.documents]


def test_one_named_graph_per_document(rag):
    graphs = fact_graphs(rag)
    assert len(graphs) == len(rag.documents) == len(set(graphs)) == 5
    for g in graphs:
        assert len(rag.dataset.graph(g)) > 0, f"{g} is empty"


def test_facts_stay_in_their_source_graph(rag):
    official = rag.dataset.graph(GRAPH["acme-official-about"])
    news = rag.dataset.graph(GRAPH["techherald-acme-moves-to-munich"])
    assert (RES.AcmeAI, EX.headquarters, RES.Berlin) in official
    assert (RES.AcmeAI, EX.headquarters, RES.Munich) not in official
    assert (RES.AcmeAI, EX.headquarters, RES.Munich) in news
    assert (RES.AcmeAI, EX.headquarters, RES.Berlin) not in news  # "moved from Berlin" is not a fact


def test_every_fact_graph_has_prov_metadata(rag):
    prov = rag.dataset.graph(PROVENANCE_GRAPH)
    for doc in rag.documents:
        g = GRAPH[doc.graph_name]
        src = URIRef(doc.source_uri)
        assert (g, RDF.type, PROV.Entity) in prov
        assert (g, PROV.wasDerivedFrom, src) in prov
        assert (g, PROV.wasAttributedTo, URIRef(doc.publisher_uri)) in prov
        assert (g, PROV.wasGeneratedBy, None) in prov
        assert (g, PAV.retrievedOn, None) in prov
        assert (src, DCTERMS.publisher, URIRef(doc.publisher_uri)) in prov
        assert (src, DCTERMS.issued, None) in prov
        assert (src, EX.sourceType, EX[doc.source_type]) in prov
        activity = prov.value(g, PROV.wasGeneratedBy)
        assert (activity, RDF.type, PROV.Activity) in prov
        assert (activity, PROV.used, src) in prov


def test_provenance_is_not_mixed_into_fact_graphs(rag):
    for doc in rag.documents:
        g = rag.dataset.graph(GRAPH[doc.graph_name])
        assert not list(g.triples((None, PROV.wasDerivedFrom, None)))
