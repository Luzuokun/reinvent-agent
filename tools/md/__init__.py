"""Independent GROMACS MD module — not part of the REINVENT executor loop.

Materialize human-written mdp templates, run a short minimization and/or NVT
chain, and write RMSD/RMSF tables under ``<project>/output/md/``.

Human approval (``--approve-md``) and environment checks are separate from
``--approve-run`` / ``check_environment``. This package is never imported by
the REINVENT executor. No LLM-written TOML or mdp, no arbitrary shell.
Production 100 ns / MM-PBSA are out of scope for this phase.
"""

from tools.md.environment import check_md_environment
from tools.md.run import MdError, run_md

__all__ = [
    "MdError",
    "check_md_environment",
    "run_md",
]
