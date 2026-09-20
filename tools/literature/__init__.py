"""Independent literature / research tool — not part of the REINVENT executor.

Query NCBI PubMed (E-utilities) with a human-supplied search string and return
**only sourced entries** (URL and/or PMID and/or DOI). Missing network is a
loud failure. This package never invents papers, never writes a manuscript,
and is never imported by the REINVENT executor.

Human approval (``--approve-literature``) is separate from ``--approve-run`` /
``--approve-dock`` / ``--approve-md``.
"""

from tools.literature.run import LiteratureError, run_literature

__all__ = [
    "LiteratureError",
    "run_literature",
]
