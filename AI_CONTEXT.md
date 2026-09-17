# AI_CONTEXT.md

Living context for [reinvent-agent](https://github.com/Luzuokun/reinvent-agent).  
Last updated: **2026-09-16**.

This file is for humans and coding agents. Prefer it over reconstructing intent from chat history.

---

## 1. 总目标 (Overall goal)

一个人也能运转的 **AI 药物发现研究系统**：先把一条可复现的垂直做扎实——

**REINVENT4 → 分析 → 报告**

再逐步长成模块化的 **AI Drug Discovery OS**（文献、化学、对接、MD、Critic），始终 **human-in-the-loop**。

飞轮：

```text
Research  →  Content（handbook / 教程）  →  Agent tools  →  Research
```

相邻仓库：

| 项目 | 角色 |
|------|------|
| [AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab) | Handbook / MkDocs（本仓库不改它） |
| kinase-denovo-design | 真实研究管线；本机 prior 的可选来源 |
| **reinvent-agent**（本仓库） | 包在已有 REINVENT4 外的薄、安全自动化 |

原则：**工具先于 Agent**；确定性工作流先于 LLM；LLM 只产出经 schema 校验的 JSON，绝不直接进 shell。

---

## 2. 当前目标 (Current goal)

**当前：MCP 包装现有 allowlist 工具（v0.6）。** 把 Planner 已有的 8 个步骤暴露给外部 MCP 客户端（Cursor 等），每个 tool 仍调用同一 Python 函数，并沿用同样的路径/步骤限制。Agent **不获得新能力**，只获得新入口。没有 shell 工具，不能改 `reinvent.toml`，不上 docking / MD / 文献，不引入新 Agent 框架。

- `tools/mcp_allowlist.py`：唯一调度面；tool 名 = `ALLOWED_STEPS`；`csv_path` 必须在 `<project>/output/`；`project_dir` 不得逃出仓库根。
- `tools/mcp_server.py`：stdio MCP server（`python -m tools.mcp_server`）；可选依赖 `mcp>=1.9,<2`。
- 内部 CLI / Executor 不变，仍直接调 `check_environment` / `analyze_molecules` 等。
- 多 provider LLM（openai / xai / gemini / openai_compatible）已可用；客户端构造集中在 `agents/llm_client.py`。密钥从环境变量读；启动时加载 gitignored 的仓库根目录 `.env`（已有环境变量优先）。
- **Transfer learning demo**（`projects/demo_tl`）：预定义 CPU TL TOML + 人批；产物是 checkpoint。本地 `tools/smiles_prep.py` 只做校验/去重。文献 / ChEMBL 属于未来的 Research tool，不在 Chemistry executor 里。
- 安全不变量不变。

下一步（尚未做）见第 6 节——不是一次 10-agent 重写。

---

## 3. 已完成工作 (Completed)

### Phase 1 — 确定性 MVP (v0.1)

固定管线（无 LLM）：

`check env → validate project → (approve-run) reinvent → find output → analyze → HTML report + critic`

- `agents/planner.py` / `executor.py` / `critic.py`
- `tools/`：environment、files、reinvent（`shell=False`）、analysis、artifacts
- `projects/demo_project`：Tutorial-01 风格 **CPU sampling**（`num_smiles = 100`），不是 EGFR GPU 大作业
- REINVENT 必须 `--approve-run`，交互确认或 `--yes`

### v0.2 — dry-run 标注

Dry-run / 未真正跑 REINVENT 时，分析的是 **已有 CSV**（例如 `sampled-sample.csv`）。CLI、HTML banner、`result.json` 均标明 `analysis_source: existing_csv`，避免把陈旧统计当成新生成。Critic 对未批准的 run 给 `WARNING`。

### Phase 2 — 可选 LLM Planner (v0.3, PR #1, 已合入 main)

- `agents/plan_schema.py`：步骤 allowlist；拒绝未知 step / 未知 key；`csv_path` 必须已存在且落在 `<project>/output/`
- `agents/llm_planner.py`：OpenAI-compatible Chat Completions（默认）或 Responses API；`--planner llm`
- 非法 plan / 缺 key / provider 错误 → 确定性 fallback
- `docs/phase2-llm-planner.md`；测试全部 mock，CI 不访问网络
- `run_reinvent` **没有** 模型可控参数；启动仍要人批

### Phase 3 — 可选 LLM Critic（PR #2, 已合入 main）

- `agents/critic_schema.py` + `agents/llm_critic.py`
- `--critic {deterministic,llm}`；`docs/phase3-llm-critic.md`
- evidence-only；docking/MD/文献等无证据声称会被 schema 拒绝并 fallback
- 确定性 `CriticAgent.review` 算法未改

### 多 provider LLM（openai / xai / gemini / openai_compatible）

- `agents/llm_client.py`：Planner 与 Critic 共用的 OpenAI-compatible 客户端
- 预设：`openai`（`OPENAI_API_KEY`）、`xai`（`https://api.x.ai/v1` + `XAI_API_KEY`）、`gemini`（Google OpenAI-compat URL + `GEMINI_API_KEY`）、`openai_compatible`（yaml 全指定）
- `--provider`；yaml `provider` / `model` / `base_url` / `api_key_env`
- 测试全部 mock，CI 不访问网络

### CSV 描述符与 HTML 直方图（v0.5）

- SA Score（`rdkit.Contrib.SA_Score`）、PAINS（FilterCatalog）、QED ≥ 阈值 / PAINS-free 计数
- 额外描述符：TPSA / HBD / HBA / 可旋转键和 Lipinski 通过率（仍只基于 CSV，不是对接或活性）
- `reports/report_*.html` 内嵌 QED / MW / logP 三张 SVG 直方图
- Critic 可选 `min_mean_qed`；QED 均值缺失时不编造
- `docs/csv-analysis-report.md`；测试不依赖 GPU / 网络 / API key

### MCP allowlist server（v0.6）

- `tools/mcp_allowlist.py` + `tools/mcp_server.py`：把现有 8 个步骤暴露给 MCP 客户端
- 与 `agents/plan_schema.ALLOWED_STEPS` 同一名单；无 shell / argv / 任意路径
- `analyze_molecules` 的 `csv_path` 必须已存在且落在 `<project>/output/`（与 LLM planner 相同）
- `run_reinvent` 仍要 `approve_run=true`，命令仍是预定义 `reinvent -l … -s … reinvent.toml`，`shell=False`
- `docs/mcp-tools.md`；allowlist 测试不依赖 MCP SDK；若安装了 `mcp` 则加一条 stdio 集成测试
- 可选 extra：`pip install '.[mcp]'`（`mcp>=1.9,<2`）

### Transfer learning demo（CPU）

- `projects/demo_tl`：`run_type = "transfer_learning"`，bundled `input/tl_train.smi`
- `tools/smiles_prep.py`：本地 CSV/SMI → canonical `.smi`（无网络；**不**在 Planner allowlist，由 PI / 未来 Research tool 调用）
- 校验按 `run_type` 分支；Critic 不因缺少 `sampled.csv` 把诚实的 TL 判 FAIL
- 分子统计标注 `analysis_source: tl_training_set`（训练集，不是新采样）

### 测试与文档

- `tests/`：offline、v0.2 artifacts、plan schema、LLM planner、LLM critic、LLM client/providers、executor 安全、MCP allowlist、TL project、envfile、smiles_prep
- `python -m pytest tests/ -q` 不需要 API key / 不需要 GPU

---

## 4. 当前遇到的问题 / 风险 (Open issues)

1. **Agent 可靠性 / scientific bullshit**  
   模型仍可能把计数讲成“发现了药”。Planner/Critic 用 schema + fallback 压风险，但不能替代人读报告。
2. **`reinvent` 不在 PATH**，直到 `conda activate reinvent4`。环境检查会如实报缺失，不会自动安装。
3. **Dry-run 分析陈旧 CSV** 容易让人以为刚生成了一批分子。v0.2 已标注；仍需在 UI/文档里盯着。
4. **LLM 依赖 provider / API key**。缺 key 时默认 fallback，不是静默编造 plan/verdict。未装 `openai` 包同样 fallback。可用 xAI / Gemini / 任意 OpenAI-compatible 端点，不必绑死 OpenAI 配额。`ALL_PROXY=socks://...` 会被跳过并改用 `HTTP(S)_PROXY`。
5. **GPU 作业需要人批准**。本 MVP 的 demo 是 CPU sampling；真 GPU 工作流不自动开跑。
6. **两套 REINVENT4 树**（本机环境，不在本 git 里）：可编辑的 kinase 安装 vs Documents 下的 clone。Agent 只调用 PATH 上的 `reinvent` 可执行文件 + 项目内 `reinvent.toml`。不要假设某棵源码树被 import。
7. **MCP 是可选入口**。未安装 `mcp` 时 CLI 不受影响；`python -m tools.mcp_server` 会提示安装 extra。Cursor 里仍要人确认 tool call；`run_reinvent` 仍要 `approve_run=true`。

---

## 5. 重要决策 (Key decisions)

| 决策 | 含义 |
|------|------|
| Tools before agents | 先有预定义、可测的 tools，再包 Planner/Executor/Critic |
| Deterministic workflow before LLM | 默认路径零网络；LLM 是可选层 |
| Allowlisted plans only | 模型不能发明 step / argv / 任意路径 |
| Never LLM → shell | 模型文本不进 `subprocess`、不进 argv |
| Human approve for reinvent | `--approve-run` + 确认 / `--yes` |
| Planner / Executor / Critic 分离 | 计划、执行、评价三条边界；Phase 3 只加 Critic 的 LLM 层 |
| Narrow MVP | 暂无 PubMed / docking / MD |
| Fallback on LLM failure | 默认回退确定性实现并大声警告；`fallback_on_error: false` 才 hard-fail（exit 2） |
| Multi-provider LLM | 一个 OpenAI-compatible 客户端；provider 由 yaml / `--provider` / env 选择，不把单一厂商写进 Planner/Critic |
| MCP wraps existing tools | 新入口（Cursor 等），不新能力；tool 名 = planner allowlist；无 shell |

---

## 6. 下一步计划 (Next steps)

Phase 3、多 provider LLM、CSV 描述符 / HTML 直方图、MCP allowlist 包装、以及 **CPU transfer learning demo** 已落地。之后 **按模块** 考虑，而不是重写成 10 个互相聊天的 agent：

1. **Research tool**（文献 / ChEMBL 下载）：预定义查询 + 人批，再调用 `smiles_prep`；不要让 LLM 发明 SQL 或 curl
2. 用 TL 后的 checkpoint 再跑 sampling project（TL → sample 两段，仍不许模型改 TOML）
3. 人写的实验预设 ID（Planner 只许输出预设，不得生成 TOML）
4. 更丰富的过滤 / 导出（仍基于已有 CSV）
5. Docking / MD 作为 **独立 tool 模块**（各自的人批与环境），不是塞进当前 Executor
6. 与 handbook（AI-Drug-Discovery-Lab）的链接：教程 ↔ 可运行 tool

明确不做：用 chat 框架替换当前管线；让模型改写 `reinvent.toml`；无人值守“自动发现药物”。

---

## 7. 仓库地图 / 怎么跑

```text
main.py                    CLI：Planning → Execution → Tools → Critic
agents/planner.py          确定性 Planner（默认）
agents/llm_planner.py      可选 LLM Planner
agents/llm_client.py       多 provider OpenAI-compatible 客户端
agents/plan_schema.py      plan allowlist
agents/executor.py         只调用预定义 tools
agents/critic.py           确定性 Critic（默认）
agents/llm_critic.py       可选 LLM Critic
agents/critic_schema.py    critic JSON schema + evidence 摘要
tools/                     environment / files / reinvent / smiles_prep / analysis / artifacts / MCP allowlist
analysis/                  分子统计 + HTML 报告
projects/demo_project      CPU sampling；bundled `output/sampled-sample.csv`
projects/demo_tl           CPU transfer learning；bundled `input/tl_train.smi`
config/agent.yaml          planner / critic / 路径 / 阈值
docs/mcp-tools.md
docs/phase2-llm-planner.md
docs/phase3-llm-critic.md
docs/csv-analysis-report.md
logs/runs/<utc>/result.json
reports/report_*.html
```

环境：

```bash
conda activate reinvent4    # REINVENT 4.8.x + RDKit；否则 reinvent 不在 PATH
cd /path/to/reinvent-agent
pip install -r requirements.txt
# 可选 LLM（同一 openai SDK；密钥来自环境变量或 gitignored `.env`）
pip install 'openai>=1.40'   # 或 pip install '.[llm]'
export OPENAI_API_KEY=...    # 或 XAI_API_KEY / GEMINI_API_KEY
# .env 示例：XAI_API_KEY=...   （已有 export 优先）
# 可选 MCP（Cursor 等外部客户端；同一 allowlist，无 shell）
pip install 'mcp>=1.9,<2'    # 或 pip install '.[mcp]'
python -m tools.mcp_server
```

Prior（真实 sampling 需要，gitignored）：

```bash
mkdir -p projects/demo_project/priors
ln -s /path/to/reinvent.prior projects/demo_project/priors/reinvent.prior
```

常用 CLI：

```bash
# 默认：确定性 plan + 确定性 critic；不启动 REINVENT
python main.py --project projects/demo_project --goal "Dry run"

# 人批后跑预定义 reinvent 命令
python main.py --project projects/demo_project --goal "Sample" --approve-run --yes

# CPU transfer learning（产物是 model，分析的是训练 SMILES）
python -m tools.smiles_prep --project projects/demo_tl \
  --source projects/demo_project/output/sampled-sample.csv
python main.py --project projects/demo_tl \
  --goal "Fine-tune prior on bundled SMILES via transfer learning." \
  --approve-run --yes

# 离线分析 bundled CSV
python main.py --project projects/demo_project --goal "Offline" \
  --skip-reinvent --csv projects/demo_project/output/sampled-sample.csv

# 可选 LLM 层（缺 key 则 fallback）
python main.py --project projects/demo_project --goal "Offline" \
  --planner llm --critic llm --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv

# xAI（无需 OpenAI 配额）
export XAI_API_KEY=...
python main.py --project projects/demo_project --goal "Offline" \
  --planner llm --critic llm --provider xai --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

关键 flag：`--project` `--goal` `--approve-run` `--yes` `--skip-reinvent` `--csv` `--seed` `--config` `--planner {deterministic,llm}` `--critic {deterministic,llm}` `--provider {openai,xai,gemini,openai_compatible}`

Exit：Critic `PASS`/`WARNING` → 0；`FAIL` → 1；LLM hard-fail（fallback 关闭）→ 2。

---

## 8. 非目标 (Non-goals for now)

- 任意 / 未 allowlist 的 shell（含 MCP 入口）
- 无人值守写论文、自动“发现药物”
- 自动 `pip/conda install`
- 模型改写 `reinvent.toml`
- PubMed、ChEMBL 下载、docking、GROMACS/MD（独立 Research / Docking / MD 模块之前）
- CrewAI / LangGraph / LlamaIndex / 多 agent 对话框架
- 把 LLM 输出当命令执行

---

## 维护说明

改管线、安全边界、或阶段目标时同步更新本文件日期与第 2–6 节。细节以代码和 `docs/` 为准。
