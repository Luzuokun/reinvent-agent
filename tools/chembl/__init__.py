"""Independent ChEMBL assay download tool — not part of the REINVENT executor.

Downloads bioactivity rows for a **predefined query id** from the ChEMBL REST
API (``www.ebi.ac.uk`` only). Humans pick the query id; agents must not invent
SQL, curl, or free-form endpoints. Optional local ``smiles_prep`` of the
downloaded CSV is a second, still-local step.

Human approval (``--approve-chembl``) is separate from ``--approve-run`` /
``--approve-dock`` / ``--approve-md`` / ``--approve-literature``.
Missing network is a loud failure — rows are never fabricated.
"""

from tools.chembl.run import ChemblError, run_chembl

__all__ = [
    "ChemblError",
    "run_chembl",
]
