"""Source metadata can be retrieved for a named graph via SPARQL."""

from datetime import date

from provrag.provenance import all_sources, lookup_sources
from provrag.vocab import GRAPH


def test_lookup_news_source(rag):
    g = GRAPH["techherald-acme-moves-to-munich"]
    sources, query = lookup_sources(rag.dataset, [g])
    assert "prov:wasDerivedFrom" in query
    info = sources[g]
    assert info.short_id == "S2"
    assert str(info.source_uri) == "https://techherald.example/2025/06/acme-ai-moves-to-munich"
    assert info.publisher == "Tech Herald"
    assert info.source_type == "NewsArticle"
    assert info.published == date(2025, 6, 15)
    assert info.retrieved == date(2026, 9, 1)


def test_all_sources_have_complete_metadata(rag):
    sources = all_sources(rag.dataset)
    assert {s.short_id for s in sources.values()} == {"S1", "S2", "S3", "S4", "S5"}
    types = {s.short_id: s.source_type for s in sources.values()}
    assert types["S1"] == "OfficialWebsite"
    assert types["S4"] == "DatabaseDump"
    for s in sources.values():
        assert s.publisher and s.title and s.published


def test_every_returned_fact_carries_its_source(alice_result):
    assert alice_result.evidence
    for ev in alice_result.evidence:
        assert ev.source is not None
        assert ev.source.graph == ev.graph
        assert ev.source.publisher and ev.source.published
