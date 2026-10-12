# 直链淀粉–岩藻多糖 MD（同事三步）

协议 ID：`amylose26_fucoidan12`。入口是独立 CLI
`python -m tools.md.carb`，**不是** EGFR 蛋白–配体的
`python -m tools.md --prepare-complex`。

mdp 只来自人写模板；可改的标量仅限 `nsteps`、`dt`、`ref_t`、`salt_mol`、
`box_d`、`n_replicas`。模型不能写整份 mdp，也不能改 integrator / 约束 /
力场。默认协议是短 **EM→NVT→NPT** 烟测，**不会**启动 200 ns。

预置坐标/拓扑是协议模板（自动化烟测 + 工作流接线），不是晶体结构，也不是
「已证明结合机制」。

## 0. 一次环境

`gmx`（或 `gmx_mpi`）需已在 `PATH`。本模块从不 `pip`/`conda` 安装
GROMACS。CHARMM36 由 doctor/prepare 按钉扎 URL+sha256 拉到
`forcefields/charmm36/`（gitignore）。缓存在时 doctor 可离线。

```bash
conda activate reinvent4   # 或你的 GROMACS 环境
cd /path/to/reinvent-agent
```

## 1. doctor — 检查并拉齐力场

```bash
python -m tools.md.carb doctor
```

会检查：`gmx`、可选 GPU、磁盘、入库的 Amylose-26 / Fucoidan-12
残基数与电荷、钉扎 CHARMM36。缺力场则拉取并校验 sha256；校验失败会大声退出，
不换源。

## 2. prepare — 用预置模板组体系

```bash
python -m tools.md.carb prepare \
  --project projects/amylose_fucoidan
```

默认模板：26×α-D-Glc（α-(1→4)，左手螺旋）+ 12×α-L-Fuc（交替
α-(1→3)/α-(1→4)，全部 C2 硫酸化，净电荷 −12），初始质心距离
0.8–1.5 nm，盒缓冲 1.2 nm，0.15 M NaCl。产物写在
`<project>/input/md/`。

只检查不写：

```bash
python -m tools.md.carb prepare \
  --project projects/amylose_fucoidan \
  --dry-run
```

可选标量：`--box-d`、`--salt-mol`。第二套硫酸化分布：

```bash
python -m tools.md.carb prepare \
  --project projects/amylose_fucoidan \
  --template amylose26_fucoidan12_alt
```

## 3. run + report — 短烟测，再出报告

```bash
python -m tools.md.carb run \
  --project projects/amylose_fucoidan \
  --approve-md --yes

python -m tools.md.carb report \
  --project projects/amylose_fucoidan
```

没有 `--approve-md` 时只物化 mdp、不启动 `gmx`。默认协议
`em-nvt-npt-short`（秒–分钟级）。报告在 `output/carb/`：RMSD、Rg、COM、
hbond、SASA；有轨迹时再加 FEL、接触热图、能量组分解。本机没有
`gmx_MMPBSA` 时报告会标明 **SKIPPED**，不编造 ΔG。

烟测/demo 数字不得写成结合机制已证明。

## 生产 200 ns（必须显式选择）

人写模板 `protocols/amylose_fucoidan/mdp/md_200ns.mdp`
（`nsteps=100000000`，`dt=0.002` → 200 ns，默认 368 K）。三平行种子
101/202/303。作业状态在 `output/carb/job_status.json`。失败不编造轨迹。

```bash
python -m tools.md.carb run \
  --project projects/amylose_fucoidan \
  --protocol production_200ns \
  --approve-md --yes
```

可选 `--gpu`（`mdrun -nb gpu`）。可选 `--n-replicas`（默认 3，上限 5）。

## 不要用的命令

- `python -m tools.md --prepare-complex`（EGFR 蛋白–配体，Amber/ACPYPE）
- 把 200 ns 当默认协议
- 让模型生成 mdp / 残基拓扑 / 任意 shell
