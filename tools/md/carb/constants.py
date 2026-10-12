"""Frozen protocol IDs and paths. Not model-editable."""

from __future__ import annotations

from pathlib import Path

from tools import REPO_ROOT

PROTOCOL_ID = "amylose26_fucoidan12"
PROTOCOL_DIR = REPO_ROOT / "protocols" / "amylose_fucoidan"
TEMPLATES_DIR = PROTOCOL_DIR / "templates"
MDP_DIR = PROTOCOL_DIR / "mdp"
PIN_PATH = PROTOCOL_DIR / "charmm36.pin.json"
FF_ROOT = REPO_ROOT / "forcefields" / "charmm36"

DEFAULT_TEMPLATE_ID = "amylose26_fucoidan12"
ALT_TEMPLATE_ID = "amylose26_fucoidan12_alt"

ALLOWED_TEMPLATE_IDS: tuple[str, ...] = (DEFAULT_TEMPLATE_ID, ALT_TEMPLATE_ID)

DEFAULT_PROTOCOL = "em-nvt-npt-short"
PRODUCTION_PROTOCOL = "production_200ns"
ALLOWED_PROTOCOLS: tuple[str, ...] = (DEFAULT_PROTOCOL, PRODUCTION_PROTOCOL)

DEFAULT_REF_T = 368.0
DEFAULT_BOX_D = 1.2
DEFAULT_SALT_MOL = 0.15
DEFAULT_N_REPLICAS = 3
REPLICA_SEEDS: tuple[int, ...] = (101, 202, 303)

COM_DISTANCE_MIN_NM = 0.8
COM_DISTANCE_MAX_NM = 1.5

MIN_BOX_D = 0.5
MAX_BOX_D = 2.0
MIN_SALT_MOL = 0.0
MAX_SALT_MOL = 1.0
MIN_REPLICAS = 1
MAX_REPLICAS = 5

# Short smoke-test launch cap. 200 ns is only for production_200ns.
MAX_LAUNCH_NSTEPS_SHORT = 10_000
MAX_LAUNCH_NSTEPS_200NS = 100_000_000
PRODUCTION_NSTEPS = 100_000_000  # 200 ns at dt=0.002 ps

OUTPUT_SUBDIR = "carb"
INPUT_MD_SUBDIR = ("input", "md")

AMYLOSE_RESNAME = "AGLC"
FUCOIDAN_RESNAME = "AFUS"
AMYLOSE_N_RESIDUES = 26
FUCOIDAN_N_RESIDUES = 12
AMYLOSE_CHARGE = 0.0
FUCOIDAN_CHARGE = -12.0
FUCOIDAN_ALT_CHARGE = -18.0

DISK_FREE_MIN_BYTES = 2 * 1024 ** 3

ALLOWED_OVERRIDE_KEYS: tuple[str, ...] = (
    "nsteps",
    "dt",
    "ref_t",
    "salt_mol",
    "box_d",
    "n_replicas",
)
