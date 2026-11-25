# 部署指南 · Budget Agent

本项目有两种形态，按需选择：

| 形态 | 说明 | 适合 |
|---|---|---|
| **A. GitHub Pages 静态 Mock 主页** | 仅静态 HTML/JS，运行/行程/对比结果均为预置示例，实验数据为真实快照 | 在线展示项目效果 |
| **B. 本地完整版** | 真实执行引擎 + DeepSeek LLM 分解与评分 + 图记忆持久化 + 完整交互 | 开发、实验、答辩演示 |

仓库地址：https://github.com/itemgroup/budget-constrained-agent （工作分支 `develop`）

---

## A. 部署 GitHub Pages 静态主页（约 1 分钟）

`docs/index.html` 即静态 Mock 主页，启用 Pages 即可：

1. 打开仓库 **Settings → Pages**
2. **Build and deployment** 选择 **Deploy from a branch**
3. Branch 选择 **`develop`**，目录选择 **`/docs`**，点击 **Save**
4. 等待部署完成，访问：

   **https://itemgroup.github.io/budget-constrained-agent/**

也可以使用 GitHub CLI 一条命令启用（需管理员权限）：

```bash
gh api repos/itemgroup/budget-constrained-agent/pages -X POST -f "source[branch]=develop" -f "source[path]=/docs"
```

> 更新主页：修改 `docs/index.html` 后推送 `develop`，Pages 会自动重新部署。

---

## B. 本地运行完整版

### 1. 克隆与安装

```powershell
git clone https://github.com/itemgroup/budget-constrained-agent.git
cd budget-constrained-agent
git checkout develop

# Python 3.10+（项目开发环境为 3.14），虚拟环境约定名为 budget_agent_env
python -m venv budget_agent_env
.\budget_agent_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 启动 Web UI（完整交互版）

```powershell
python web_server.py
# 打开 http://127.0.0.1:8000
```

包含：单次运行、行程规划（实时报价单）、四方案对比、服务目录、实验数据、架构说明。

### 3. 命令行运行

```powershell
# 模拟分解器 + 模拟评分（无需任何 API key）
python main.py run --task "调研行业数据，分析市场机会并生成季度业务报告" --budget 0.5 --seed 5

# 真实 LLM（DeepSeek）任务分解 + LLM-Judge 评分
python main.py run --task "调研行业数据，分析市场机会并生成季度业务报告" --budget 0.5 --seed 5 --llm
```

### 4.（可选）接入真实 LLM

1. 在 https://platform.deepseek.com 申请 API key
2. 复制 `.env.example` 为 `.env`，填入：

   ```
   DEEPSEEK_API_KEY=sk-你的key
   ```

3. 加 `--llm` 参数即可；**无 key 时自动回退模拟实现，不会报错**。

### 5. 复现实验数据

```powershell
python main.py experiment all
# 产出 experiments/results/ 下 9 个 CSV（基线 / 消融 / 多预算 / 失败注入）
# 单独运行：experiment baselines | ablation | budget_sweep | failure_test
```

### 6. Demo 脚本速览

| 脚本 | 演示内容 |
|---|---|
| `demo_agent_run.py` | 主循环双案例（正常完成 / 预算耗尽） |
| `demo_travel.py` | 行程规划四档预算（含降级、满房重试、体面终止） |
| `demo_memory.py` | 图记忆：相似任务命中后分解零 LLM 调用（`--fresh` 冷启动对照） |
| `demo_eval.py` | 评估指标与 LLM-Judge |
| `demo_replan.py` | 重规划 RETRY / SKIP / ABORT 全策略 |

---

## Mock 主页与完整版的差异

| 能力 | Mock 主页（Pages） | 本地完整版 |
|---|---|---|
| 页面视觉与分区 | ✅ 一致 | ✅ |
| 运行 / 行程 / 对比结果 | 预置示例 | 真实引擎计算 |
| 实验数据 | 真实实验快照（内嵌） | 实时读取最新 CSV |
| 自定义任务与预算 | 表单仅供预览 | 完全可调 |
| DeepSeek LLM 分解 / 评分 | ❌ | ✅（`--llm`） |
| 图记忆持久化 | ❌（仅架构展示） | ✅（`data/experience_graph.json`） |
