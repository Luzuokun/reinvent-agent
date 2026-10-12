"""Local Gradio UI for the amylose–fucoidan MD protocol.

Launch (optional extra, not required for automation)::

    pip install '.[ui]'   # or: pip install -r requirements-ui.txt
    python -m tools.md.carb.ui

The CLI ``python -m tools.md.carb`` remains the primary automation entry.
This UI calls the same Python functions as the CLI (no shell child process).
Localhost only: no SaaS, no cloud auth, no public share link.
"""

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

INSTALL_HINT = (
    "Gradio is an optional extra for the local carb UI. "
    "Install with: pip install '.[ui]'  or  pip install -r requirements-ui.txt. "
    "CLI stays available: python -m tools.md.carb doctor|prepare|run|report"
)

DEFAULT_PROJECT = "projects/amylose_fucoidan"


def public_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.setdefault("protocol_id", PROTOCOL_ID)
    payload.pop("command_templates", None)
    return payload


def format_result(result: dict[str, Any]) -> str:
    return json.dumps(public_payload(result), indent=2, ensure_ascii=False)


def approve_default_for_protocol(_protocol: str) -> bool:
    """HITL: approve-md is never on by default, including production_200ns."""
    return False


def effective_approve(
    protocol: str,
    approve_md: bool,
    confirm_200ns: bool,
) -> bool:
    """Map UI checkboxes onto ``run_carb(approve=...)``.

    Production 200 ns needs both the approve-md checkbox and an extra
    explicit 200 ns confirmation. Both default off.
    """
    if not approve_md:
        return False
    if protocol == PRODUCTION_PROTOCOL and not confirm_200ns:
        return False
    return True


def ui_doctor() -> str:
    return format_result(doctor(fetch_forcefield=True))


def ui_prepare(
    project: str,
    template_id: str,
    box_d: float | None,
    salt_mol: float | None,
    dry_run: bool,
) -> str:
    result = prepare_carb(
        project,
        template_id=template_id or DEFAULT_TEMPLATE_ID,
        box_d=box_d,
        salt_mol=salt_mol,
        dry_run=bool(dry_run),
        fetch_forcefield=True,
    )
    return format_result(result)


def ui_run(
    project: str,
    protocol: str,
    approve_md: bool,
    confirm_200ns: bool,
    gpu: bool,
    nsteps: float | int | None,
    dt: float | None,
    ref_t: float | None,
    n_replicas: float | int | None,
) -> str:
    chosen = protocol or DEFAULT_PROTOCOL
    approve = effective_approve(chosen, bool(approve_md), bool(confirm_200ns))
    skipped_note = None
    if (
        chosen == PRODUCTION_PROTOCOL
        and bool(approve_md)
        and not bool(confirm_200ns)
    ):
        skipped_note = (
            f"{PRODUCTION_PROTOCOL} not launched: the extra Confirm 200 ns "
            "checkbox is off (HITL). Approve MD alone is not enough."
        )
    result = run_carb(
        project,
        protocol=chosen,
        approve=approve,
        assume_yes=True,
        nsteps=_as_int(nsteps),
        dt=dt,
        ref_t=ref_t,
        n_replicas=_as_int(n_replicas),
        gpu=bool(gpu),
    )
    if skipped_note and result.get("skipped"):
        result = dict(result)
        result["message"] = skipped_note + " " + str(result.get("message") or "")
    return format_result(result)


def ui_report(project: str) -> str:
    return format_result(report_carb(project))


def _as_int(value: float | int | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _safe(fn):
    def wrapped(*args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except (CarbPrepareError, CarbRunError, MdError, MdpError, ValueError, OSError) as exc:
            return json.dumps(
                {
                    "ok": False,
                    "protocol_id": PROTOCOL_ID,
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )

    return wrapped


def build_app():
    """Build the Gradio Blocks app. Imports Gradio lazily."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    safe_doctor = _safe(ui_doctor)
    safe_prepare = _safe(ui_prepare)
    safe_run = _safe(ui_run)
    safe_report = _safe(ui_report)

    with gr.Blocks(title="Amylose–fucoidan MD (local)") as demo:
        gr.Markdown(
            """
# 直链淀粉–岩藻多糖 MD（本机点选）

这是 **本机** 界面，给不写命令的同事点 Doctor / Prepare / Run / Report。
**自动化仍以 CLI 为主**（可选）：

```bash
python -m tools.md.carb doctor
python -m tools.md.carb prepare --project projects/amylose_fucoidan
python -m tools.md.carb run --project projects/amylose_fucoidan --approve-md --yes
python -m tools.md.carb report --project projects/amylose_fucoidan
```

不是 EGFR `--prepare-complex`。不是 SaaS，不开公网分享，不要 200 ns 当默认。
            """
        )
        project = gr.Textbox(label="Project", value=DEFAULT_PROJECT)
        log = gr.Textbox(label="Result JSON", lines=22)

        with gr.Tab("Doctor"):
            gr.Markdown("检查 `gmx`、模板残基数/电荷、钉扎 CHARMM36。")
            doctor_btn = gr.Button("Doctor")
            doctor_btn.click(safe_doctor, inputs=[], outputs=log)

        with gr.Tab("Prepare"):
            template = gr.Dropdown(
                choices=list(ALLOWED_TEMPLATE_IDS),
                value=DEFAULT_TEMPLATE_ID,
                label="Template",
            )
            box_d = gr.Number(label="box_d (nm)", value=DEFAULT_BOX_D)
            salt_mol = gr.Number(label="salt_mol (mol/L)", value=DEFAULT_SALT_MOL)
            dry_run = gr.Checkbox(label="Dry run (do not write files / run gmx)", value=False)
            prepare_btn = gr.Button("Prepare")
            prepare_btn.click(
                safe_prepare,
                inputs=[project, template, box_d, salt_mol, dry_run],
                outputs=log,
            )

        with gr.Tab("Run"):
            protocol = gr.Dropdown(
                choices=list(ALLOWED_PROTOCOLS),
                value=DEFAULT_PROTOCOL,
                label="Protocol (default is short EM→NVT→NPT, not 200 ns)",
            )
            warn_200 = gr.Markdown(
                f"**{PRODUCTION_PROTOCOL} is 200 ns × replicas.** "
                "Approve MD defaults off. Also tick Confirm 200 ns. "
                "Do not use this for a smoke test.",
                visible=False,
            )
            gpu = gr.Checkbox(label="GPU (gmx mdrun -nb gpu)", value=False)
            nsteps = gr.Number(label="nsteps override (optional)", value=None)
            dt = gr.Number(label="dt override ps (optional)", value=None)
            ref_t = gr.Number(label="ref_t override K (optional)", value=None)
            n_replicas = gr.Number(label="n_replicas override (optional)", value=None)
            approve_md = gr.Checkbox(
                label="Approve MD (HITL; required to launch gmx; default off)",
                value=False,
            )
            confirm_200ns = gr.Checkbox(
                label="Confirm 200 ns production (HITL; extra, default off)",
                value=False,
                visible=False,
            )
            run_btn = gr.Button("Run")
            run_btn.click(
                safe_run,
                inputs=[
                    project,
                    protocol,
                    approve_md,
                    confirm_200ns,
                    gpu,
                    nsteps,
                    dt,
                    ref_t,
                    n_replicas,
                ],
                outputs=log,
            )

            def _on_protocol(selected: str):
                is_prod = selected == PRODUCTION_PROTOCOL
                return (
                    gr.update(visible=is_prod),
                    gr.update(value=False),
                    gr.update(visible=is_prod, value=False),
                )

            protocol.change(
                _on_protocol,
                inputs=[protocol],
                outputs=[warn_200, approve_md, confirm_200ns],
            )

        with gr.Tab("Report"):
            gr.Markdown("固定分析包。缺轨迹时标明 skipped，不编造数字。")
            report_btn = gr.Button("Report")
            report_btn.click(safe_report, inputs=[project], outputs=log)

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.md.carb.ui",
        description=(
            "Local Gradio UI for the amylose–fucoidan protocol. "
            "CLI remains the primary automation entry. Localhost only."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1; do not expose as a cloud service)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        print(f"ERROR: {INSTALL_HINT}", file=sys.stderr)
        return 2
    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
