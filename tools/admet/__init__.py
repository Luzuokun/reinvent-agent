"""Independent ADMET coarse-screen tool — not part of the REINVENT executor.

Offline RDKit / rule-based filters over an existing SMILES CSV using
**human-written rule preset ids**. Writes ``output/admet/admet_scores.csv``.

Human approval (``--approve-admet``) is separate from ``--approve-run`` /
``--approve-dock`` / ``--approve-md`` / ``--approve-literature`` /
``--approve-chembl`` / ``--approve-portfolio``. No docking, MD, or assay
numbers are invented; missing RDKit is a loud failure.
"""

from tools.admet.run import AdmetError, run_admet

__all__ = [
    "AdmetError",
    "run_admet",
]
