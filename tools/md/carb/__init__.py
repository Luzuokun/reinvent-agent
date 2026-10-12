"""Amylose–fucoidan MD protocol — isolated from EGFR protein–ligand MD.

``python -m tools.md.carb {doctor|prepare|run|report}``

This package never calls ``tools.md.complex.prepare_complex``. mdp files are
human templates; only allowlisted scalars may be overridden. The default
protocol is a short EM+NVT+NPT smoke test, not 200 ns.
"""

from tools.md.carb.constants import DEFAULT_PROTOCOL, PROTOCOL_ID

__all__ = ["DEFAULT_PROTOCOL", "PROTOCOL_ID"]
