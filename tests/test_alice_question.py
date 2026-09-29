"""End-to-end check of the flagship question (template mode, no LLM)."""

from provrag.generation import generate_answer
from tests.conftest import ALICE_Q


def test_alice_evidence(alice_result):
    ev = alice_result.evidence
    hop1 = {e.source.short_id for e in ev if e.hop == 1}
    hop2 = {(e.object_label, e.source.short_id) for e in ev if e.hop == 2}
    assert hop1 == {"S3", "S4"}  # profile + legacy CRM say Alice works at Acme AI
    assert hop2 == {("Berlin", "S1"), ("Munich", "S2"), ("Berlin", "S4")}
    assert "S5" not in {e.source.short_id for e in ev}  # gazetteer is irrelevant here


def test_alice_answer_mentions_disagreement(alice_result):
    text = alice_result.answer.text
    assert alice_result.answer.mode == "template"
    assert "Acme AI" in text and "Berlin" in text and "Munich" in text
    assert "disagree" in text.lower()
    for sid in ("[S1]", "[S2]", "S3", "S4"):
        assert sid in text
    assert "not a truth score" in text


class _ForgetfulLLM:
    """Fake LLM that silently collapses the conflict into one value."""
    model = "fake"

    def complete(self, messages, **kw):
        return "Acme AI is headquartered in Munich [S2]."


class _BrokenLLM:
    model = "broken"

    def complete(self, messages, **kw):
        raise RuntimeError("no network")


def test_llm_answer_guard_restores_conflict(alice_result):
    ans = generate_answer(ALICE_Q, alice_result.retrieval, llm=_ForgetfulLLM())
    assert ans.mode.startswith("llm")
    assert ans.guard_note
    assert "Berlin" in ans.text and "disagree" in ans.text.lower()


def test_llm_prompt_contains_all_evidence_and_conflicts(alice_result):
    ans = generate_answer(ALICE_Q, alice_result.retrieval, llm=_ForgetfulLLM())
    user_msg = ans.prompt[1]["content"]
    for sid in ("[S1]", "[S2]", "[S3]", "[S4]"):
        assert sid in user_msg
    assert "Berlin vs. Munich" in user_msg or "Munich vs. Berlin" in user_msg


def test_llm_failure_falls_back_to_template(alice_result):
    ans = generate_answer(ALICE_Q, alice_result.retrieval, llm=_BrokenLLM())
    assert ans.mode.startswith("template (LLM failed")
    assert "disagree" in ans.text.lower()
