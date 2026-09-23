# EGFR 1M17 (ATP site)

Human-written sampling / docking / short MD project. Target is EGFR kinase
ATP site, PDB **1M17** chain A, co-crystal ligand **AQ4 (erlotinib)**. This
is **not** the C797 covalent site.

Prior (`priors/reinvent.prior`) is a local symlink; do not commit it.

Docking box is 20 Å, centered on AQ4 heavy atoms
(`22.0137, 0.2528, 52.794` Å; 29 heavy atoms). Coordinates are in
`input/docking/box.json` (computed from the PDB, not invented).

```bash
conda activate reinvent4
# symlink a local reinvent.prior (kinase-denovo-design or demo_project)
ln -s /path/to/reinvent.prior projects/egfr_1m17/priors/reinvent.prior

python -m tools.literature --project projects/egfr_1m17 \
  --query "EGFR tyrosine kinase inhibitor" --approve-literature --yes

python main.py --project projects/egfr_1m17 --approve-run --yes

python -m tools.docking --project projects/egfr_1m17 \
  --receptor input/docking/receptor.pdbqt \
  --ligands output/docking/ligands.smi \
  --engine vina --center <from box.json> --size 20 20 20 \
  --max-ligands 51 --approve-dock --yes

python -m tools.md --project projects/egfr_1m17 \
  --prepare-complex \
  --protein input/docking/1m17_chainA.pdb \
  --ligand-pose output/docking/poses/<best>.pdbqt \
  --approve-md --yes

python -m tools.md --project projects/egfr_1m17 \
  --structure input/md/system.gro \
  --topology input/md/system.top \
  --protocol em-nvt-md2ns --gpu --approve-md --yes
```

2 ns MD uses `experiments/md2ns.mdp` (`nsteps=1000000`, `dt=0.002`).
Do not launch 100 ns (`nsteps=50000000`).
