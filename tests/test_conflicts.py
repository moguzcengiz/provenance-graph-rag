"""Conflicting facts are preserved, never collapsed, and ranking does not drop them."""

from datetime import date

from provrag.ranking import RankingConfig
from provrag.vocab import RES


def test_headquarters_conflict_preserved(alice_result):
    conflicts = alice_result.conflicts
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.subject == RES.AcmeAI and c.predicate_key == "headquarters"
    assert set(c.values) == {"Berlin", "Munich"}
    assert {e.source.short_id for e in c.values["Berlin"]} == {"S1", "S4"}
    assert {e.source.short_id for e in c.values["Munich"]} == {"S2"}


def test_same_fact_from_two_sources_is_two_evidence_items(alice_result):
    berlin = [e for e in alice_result.evidence if e.obj == RES.Berlin]
    assert len(berlin) == 2
    assert {e.graph for e in berlin} == {b.source.graph for b in berlin}


def test_ranking_orders_but_never_drops(rag):
    for weights in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
        cfg = RankingConfig(freshness_weight=weights[0], source_type_weight=weights[1],
                            reference_date=date(2026, 9, 27))
        res = rag.ask("Where is the company Alice works for headquartered?", ranking=cfg)
        hq = [e for e in res.evidence if e.predicate_key == "headquarters"]
        assert {(e.object_label, e.source.short_id) for e in hq} == {
            ("Berlin", "S1"), ("Berlin", "S4"), ("Munich", "S2")
        }


def test_ranking_prefers_fresh_news_over_old_dump(alice_result):
    hq = {e.source.short_id: e.priority for e in alice_result.evidence if e.hop == 2}
    assert hq["S2"] > hq["S4"]
    assert all(0.0 <= p <= 1.0 for p in hq.values())


def test_other_conflicts(rag, ranking):
    ceo = rag.ask("Who is the CEO of Acme AI?", ranking=ranking)
    assert set(ceo.conflicts[0].values) == {"Bob Chen", "Carol Diaz"}
    emp = rag.ask("How many employees does Acme AI have?", ranking=ranking)
    assert set(emp.conflicts[0].values) == {"95", "120", "180"}


def test_non_functional_predicate_is_not_a_conflict(rag, ranking):
    res = rag.ask("Who works at Acme AI?", ranking=ranking)
    assert res.conflicts == []
    assert {e.subject_label for e in res.evidence} == {"Alice Schmidt", "Bob Chen", "Carol Diaz"}
