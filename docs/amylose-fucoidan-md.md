# 直链淀粉–岩藻多糖 MD（同事上手）

协议 ID：`amylose26_fucoidan12`。给**第一次用、不写代码**的同事：从建环境到短烟测。

**主入口是本机 UI**：`python -m tools.md.carb.ui`（点选 Doctor / Prepare / Run / Report）。  
CLI（`python -m tools.md.carb …`）留给自动化/脚本，文末可选。  
**不是** EGFR 蛋白–配体的 `python -m tools.md --prepare-complex`。

预置坐标/拓扑在仓库里已打包（`protocols/amylose_fucoidan/templates/`），是协议模板，**不是**晶体结构，也不是「已证明结合机制」。mdp 只来自人写模板；默认短 **EM→NVT→NPT** 烟测，**不会**启动 200 ns。

---

## 1. 克隆仓库

```bash
git clone <本仓库 URL>
cd reinvent-agent
```

已有仓库则 `cd` 到仓库根目录即可。

---

## 2. 新建 conda / mamba 环境（先 create，不要假设已有）

大多数人**还没有** `reinvent4` 环境。为多糖 MD 单独建一个即可（不必复用 REINVENT 环境）：

```bash
# conda 或 mamba 均可
conda create -n reinvent-agent-carb python=3.11 -y
conda activate reinvent-agent-carb
```

---

## 3. 安装本仓库包 + UI 依赖

仍在仓库根目录、已 `conda activate`：

```bash
pip install -e .
pip install '.[ui]'
# 等价：pip install -r requirements.txt && pip install -r requirements-ui.txt
```

---

## 4. GROMACS（`gmx`）— 与 conda 无关

本机 **`gmx`（或 `gmx_mpi`）通常已在系统 `PATH` 上**，由机房/本机单独安装，**独立于 conda**。  
本模块**不会**、也**不要求**你在 conda 里再装一遍 GROMACS。

自己先确认：

```bash
which gmx    # 或 which gmx_mpi
gmx --version
```

找不到时：找本机管理员把系统 GROMACS 加进 `PATH`（例如 `source /usr/local/gromacs/bin/GMXRC`）。  
**不要**默认 `conda install gromacs`；只有你明确知道自己要在环境里另装时才可选。

CHARMM36 力场由 `doctor` / `prepare` 按钉扎 URL+sha256 拉到 `forcefields/charmm36/`（gitignore）；缓存在时可离线。

---

## 5. doctor — 检查环境（含 PATH 上的 `gmx`）

```bash
python -m tools.md.carb doctor
```

会检查：`gmx` 是否在 PATH、可选 GPU、磁盘、入库模板（Amylose-26 / Fucoidan-12 残基数与电荷）、钉扎 CHARMM36。缺力场则拉取并校验；校验失败会退出，不换源。

---

## 6. 启动 UI（推荐）

```bash
python -m tools.md.carb.ui
```

浏览器打开提示的本地地址（默认只绑 `127.0.0.1`，无账号、无云端）。按页签顺序：

1. **Doctor** — 再确认一遍  
2. **Prepare** — 默认项目 `projects/amylose_fucoidan`，默认模板 `amylose26_fucoidan12`（也可选 `amylose26_fucoidan12_alt`）。产物在 `<project>/input/md/`  
3. **Run** — 默认协议 `em-nvt-npt-short`（秒–分钟级）。**Approve MD 默认关**；不勾选只物化 mdp、不启动 `gmx`。勾选后才会真跑  
4. **Report** — 报告在 `output/carb/`（RMSD、Rg、COM、hbond、SASA 等）。没有轨迹不会编造；没有 `gmx_MMPBSA` 会标 **SKIPPED**

**HITL（人为确认）：**

- Approve MD：**默认关闭**  
- 选 `production_200ns` 时还要额外勾选 **Confirm 200 ns**（默认关）；切换到 200 ns 时 Approve MD 会被打回关  
- 失败或跳过时**不编造轨迹 / ΔG**

烟测数字不得写成结合机制已证明。

---

## 7. 生产 200 ns（必须显式选择；日常勿用）

人写模板：`protocols/amylose_fucoidan/mdp/md_200ns.mdp`（200 ns，默认 368 K；三平行种子 101/202/303）。作业状态：`output/carb/job_status.json`。

仅在 UI 里选协议 `production_200ns`，并同时勾选 **Approve MD** + **Confirm 200 ns**。  
或用下面可选 CLI（同样要 `--approve-md`）。

---

## 可选：CLI（自动化）

```bash
python -m tools.md.carb doctor
python -m tools.md.carb prepare --project projects/amylose_fucoidan
python -m tools.md.carb run --project projects/amylose_fucoidan --approve-md --yes
python -m tools.md.carb report --project projects/amylose_fucoidan
```

没有 `--approve-md` 时只物化 mdp、不启动 `gmx`。200 ns：

```bash
python -m tools.md.carb run \
  --project projects/amylose_fucoidan \
  --protocol production_200ns \
  --approve-md --yes
```

可选：`--gpu`、`--n-replicas`（默认 3，上限 5）、`--box-d`、`--salt-mol`、`--dry-run`。

---

## 打包模板在哪

| 内容 | 路径 |
|------|------|
| 坐标 / 拓扑模板 | `protocols/amylose_fucoidan/templates/` |
| 人写 mdp | `protocols/amylose_fucoidan/mdp/` |
| 项目目录 | `projects/amylose_fucoidan/` |

---

## 不要做的事

- `python -m tools.md --prepare-complex`（EGFR / Amber）  
- 假设已有 `reinvent4` 环境而不先 `conda create`  
- 把 `gmx` 默认装进 conda（它应在系统 PATH）  
- 把 200 ns 当默认协议  
- 让模型生成 mdp / 残基拓扑 / 任意 shell  
- 把本机 UI 暴露到公网或加云端鉴权  
- 把烟测/失败结果写成已证明结合或假轨迹  
