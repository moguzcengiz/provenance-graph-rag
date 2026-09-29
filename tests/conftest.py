from datetime import date

import pytest

from provrag.pipeline import ProvenanceGraphRAG
from provrag.ranking import RankingConfig

ALICE_Q = "Where is the company Alice works for headquartered?"


@pytest.fixture(scope="session")
def rag():
    # llm=None -> fully deterministic, no network access
    return ProvenanceGraphRAG(llm=None)


@pytest.fixture
def ranking():
    return RankingConfig(reference_date=date(2026, 9, 27))


@pytest.fixture
def alice_result(rag, ranking):
    return rag.ask(ALICE_Q, ranking=ranking)
