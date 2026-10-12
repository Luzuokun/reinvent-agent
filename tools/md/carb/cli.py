"""``python -m tools.md.carb {doctor|prepare|run|report}``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from tools.md.carb.constants import (
    ALLOWED_PROTOCOLS,
    ALLOWED_TEMPLATE_IDS,
    DEFAULT_BOX_D,
    DEFAULT_PROTOCOL,
    DEFAULT_REF_T,
    DEFAULT_SALT_MOL,
    DEFAULT_TEMPLATE_ID,
    PRODUCTION_PROTOCOL,
    PROTOCOL_ID,
)
from tools.md.carb.doctor import doctor
from tools.md.carb.mdp import MdpError
from tools.md.carb.prepare import CarbPrepareError, prepare_carb
from tools.md.carb.report import report_carb
from tools.md.carb.run import CarbRunError, run_carb
from tools.md.run import MdError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.md.carb",
        description=(
            "Amylose-26 × Fucoidan-12 MD protocol. Isolated from the EGFR "
            "protein-ligand complex prep. Human mdp templates only; allowlisted scalars "
            f"nsteps/dt/ref_t/salt_mol/box_d/n_replicas. Default protocol is "
            f"{DEFAULT_PROTOCOL}, not {PRODUCTION_PROTOCOL}."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check gmx, disk, GPU (optional), templates, pinned CHARMM36")

    prep = sub.add_parser("prepare", help="Assemble bundled templates into <project>/input/md/")
    prep.add_argument("--project", required=True, help="Project directory")
    prep.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE_ID,
        choices=ALLOWED_TEMPLATE_IDS,
        help=f"Bundled oligomer pair (default {DEFAULT_TEMPLATE_ID})",
    )
    prep.add_argument(
        "--box-d",
        type=float,
        default=None,
        dest="box_d",
        help=f"editconf -d box buffer in nm (default {DEFAULT_BOX_D})",
    )
    prep.add_argument(
        "--salt-mol",
        type=float,
        default=None,
        dest="salt_mol",
        help=f"NaCl concentration in mol/L (default {DEFAULT_SALT_MOL})",
    )
    prep.add_argument(
        "--dry-run",
        action="store_true",
        help="Check templates and scalars; do not write files or run gmx",
    )
    prep.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch CHARMM36; use cache only",
    )

    run = sub.add_parser(
        "run",
        help=f"Launch predefined gmx chain (needs --approve-md). Default {DEFAULT_PROTOCOL}.",
    )
    run.add_argument("--project", required=True, help="Project directory")
    run.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
        choices=ALLOWED_PROTOCOLS,
        help=(
            f"Human template chain. Default {DEFAULT_PROTOCOL} (seconds–minutes). "
            f"{PRODUCTION_PROTOCOL} is 200 ns × 3 replicas and must be named explicitly."
        ),
    )
    run.add_argument("--structure", default=None, help="Existing .gro/.pdb under <project>/input/")
    run.add_argument("--topology", default=None, help="Existing .top under <project>/input/")
    run.add_argument("--nsteps", type=int, default=None, help="Allowlisted nsteps override")
    run.add_argument("--dt", type=float, default=None, help="Allowlisted dt override in ps")
    run.add_argument(
        "--ref-t",
        type=float,
        default=None,
        dest="ref_t",
        help=f"Allowlisted ref_t override in K (template default {DEFAULT_REF_T})",
    )
    run.add_argument(
        "--n-replicas",
        type=int,
        default=None,
        dest="n_replicas",
        help="Allowlisted replica count (default 1 for short, 3 for 200 ns)",
    )
    run.add_argument(
        "--gpu",
        action="store_true",
        help="Opt-in gmx mdrun -nb gpu (default is CPU)",
    )
    run.add_argument(
        "--approve-md",
        action="store_true",
        help="Human approval to launch the predefined gmx chain",
    )
    run.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-md",
    )

    report = sub.add_parser("report", help="Fixed analysis pack (RMSD/Rg/COM/hbond/SASA/FEL/heatmap/energy)")
    report.add_argument("--project", required=True, help="Project directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(fetch_forcefield=True)
        elif args.command == "prepare":
            result = prepare_carb(
                args.project,
                template_id=args.template,
                box_d=args.box_d,
                salt_mol=args.salt_mol,
                dry_run=bool(args.dry_run),
                fetch_forcefield=not bool(args.no_fetch),
            )
        elif args.command == "run":
            result = run_carb(
                args.project,
                protocol=args.protocol,
                approve=bool(args.approve_md),
                assume_yes=bool(args.yes),
                nsteps=args.nsteps,
                dt=args.dt,
                ref_t=args.ref_t,
                n_replicas=args.n_replicas,
                gpu=bool(args.gpu),
                structure=args.structure,
                topology=args.topology,
            )
        elif args.command == "report":
            result = report_carb(args.project)
        else:
            parser.error(f"unknown command {args.command}")
            return 2
    except (CarbPrepareError, CarbRunError, MdError, MdpError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(_public(result), indent=2, ensure_ascii=False))
    if not result.get("ok", False) and not result.get("skipped"):
        return 1
    return 0


def _public(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.setdefault("protocol_id", PROTOCOL_ID)
    payload.pop("command_templates", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
