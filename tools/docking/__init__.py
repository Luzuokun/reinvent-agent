"""Independent docking module — not part of the REINVENT executor loop.

Prepare receptor/ligand, run AutoDock Vina or GNINA with a predefined argv,
and write a score table under ``<project>/output/docking/``.

Human approval (``--approve-dock``) and environment checks are separate from
``--approve-run`` / ``check_environment``. This package is never imported by
the REINVENT executor. No LLM-written TOML, no arbitrary shell, no MD.
"""

from tools.docking.environment import check_docking_environment
from tools.docking.run import DockingError, run_docking

__all__ = [
    "DockingError",
    "check_docking_environment",
    "run_docking",
]
