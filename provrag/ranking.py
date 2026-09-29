"""Evidence ranking = *retrieval prioritization*, NOT a truth score.

The score only decides the order in which evidence is presented to the answer
generator / user. It combines two transparent signals:

* freshness     – exponential decay of the source's publication age
* source type   – a fixed, hand-set prior per kind of source

A high score does not mean a statement is true; a fresh news article can be
wrong and an old official page can still be right. Conflicting evidence is
never dropped because of a low score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .config import reference_date
from .retrieval import Evidence

SOURCE_TYPE_PRIOR: dict[str, float] = {
    "OfficialWebsite": 0.90,
    "ReferenceDataset": 0.85,
    "NewsArticle": 0.70,
    "EmployeeProfile": 0.60,
    "DatabaseDump": 0.40,
}
DEFAULT_PRIOR = 0.50


@dataclass
class RankingConfig:
    freshness_weight: float = 0.5
    source_type_weight: float = 0.5
    half_life_days: float = 365.0
    reference_date: date = field(default_factory=reference_date)
    type_priors: dict[str, float] = field(default_factory=lambda: dict(SOURCE_TYPE_PRIOR))


def freshness(published: date, config: RankingConfig) -> float:
    age_days = max((config.reference_date - published).days, 0)
    return 0.5 ** (age_days / config.half_life_days)


def score(ev: Evidence, config: RankingConfig) -> tuple[float, dict]:
    if ev.source is None:
        return 0.0, {"note": "no provenance"}
    fresh = freshness(ev.source.published, config)
    prior = config.type_priors.get(ev.source.source_type, DEFAULT_PRIOR)
    total_w = (config.freshness_weight + config.source_type_weight) or 1.0
    value = (config.freshness_weight * fresh + config.source_type_weight * prior) / total_w
    return round(value, 4), {
        "freshness": round(fresh, 4),
        "age_days": (config.reference_date - ev.source.published).days,
        "source_type_prior": prior,
    }


def rank_evidence(evidence: list[Evidence], config: RankingConfig | None = None) -> list[Evidence]:
    """Annotate every item with a priority and return a *reordered copy* (same items)."""
    config = config or RankingConfig()
    for ev in evidence:
        ev.priority, ev.priority_components = score(ev, config)
    return sorted(evidence, key=lambda e: (e.hop, -(e.priority or 0.0), e.source.short_id if e.source else ""))
