"""Multi-hop traversal across named graphs."""

from provrag.retrieval import Step, parse_question, plan_path
from provrag.vocab import EX, RES


def test_plan_two_hops():
    assert plan_path("Person", ["worksAt", "headquarters"]) == [Step("worksAt"), Step("headquarters")]


def test_plan_implicit_hops_for_country():
    assert plan_path("Person", ["locatedIn"]) == [Step("worksAt"), Step("headquarters"), Step("locatedIn")]


def test_plan_inverse_hop():
    assert plan_path("Organization", ["worksAt"]) == [Step("worksAt", inverse=True)]


def test_entity_detection(rag):
    parsed = parse_question("Where is the company Alice works for headquartered?", rag.registry)
    assert [e.key for e in parsed.entities] == ["AliceSchmidt"]
    assert set(parsed.intents) == {"worksAt", "headquarters"}


def test_multi_hop_path_crosses_graphs(alice_result):
    r = alice_result.retrieval
    assert r.mode == "path"
    assert [s.predicate for s in r.plan] == ["worksAt", "headquarters"]
    hop1 = {(e.subject, e.predicate, e.obj) for e in r.evidence if e.hop == 1}
    assert hop1 == {(RES.AliceSchmidt, EX.worksAt, RES.AcmeAI)}
    hop2_objects = {e.obj for e in r.evidence if e.hop == 2}
    assert hop2_objects == {RES.Berlin, RES.Munich}
    # the SPARQL query binds a separate graph variable per hop
    assert "GRAPH ?g1" in r.sparql and "GRAPH ?g2" in r.sparql
    # at least one full path combines facts from two *different* sources
    assert any(len({hop[2] for hop in path}) > 1 for path in r.paths)


def test_three_hop_country_question(rag, ranking):
    res = rag.ask("In which country is Alice's employer headquartered?", ranking=ranking)
    hop3 = {(e.subject, e.obj) for e in res.evidence if e.hop == 3}
    assert hop3 == {(RES.Berlin, RES.Germany), (RES.Munich, RES.Germany)}
    assert "Germany" in res.answer.text
