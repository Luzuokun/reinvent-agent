"""Independent portfolio shortlist tool — not part of the REINVENT executor.

Cross-modal ranking over existing CSV fields (QED / SA / PAINS / dock / MD
summary when present) using **human-written weight preset ids**. Writes
``output/portfolio/shortlist.csv`` plus a fixed-field campaign one-pager and
a local run index (not CRM).

Human approval (``--approve-portfolio``) is separate from ``--approve-run`` /
``--approve-dock`` / ``--approve-md`` / ``--approve-literature`` /
``--approve-chembl``. No LLM scorer; missing numbers are never invented.
"""

from tools.portfolio.run import PortfolioError, run_portfolio

__all__ = [
    "PortfolioError",
    "run_portfolio",
]
