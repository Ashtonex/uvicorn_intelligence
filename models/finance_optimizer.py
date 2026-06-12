from __future__ import annotations
from analytics.nirvana_analytics import capital_allocation

def optimize_capital_allocation() -> dict:
    """Thin wrapper around the centralized capital_allocation logic."""
    return capital_allocation.run()
