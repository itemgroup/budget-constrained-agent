# 配置与启动

## 一、配置方法

### 1. 运行环境

```powershell
python -m venv budget_agent_env
.\budget_agent_env\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. LLM 配置（两种方式，二选一）

**方式 A：环境变量 / `.env` 文件**

```powershell
copy .env.example .env
```

编辑 `.env`：

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**方式 B：网页前端运行时配置**

1. 启动网页界面，进入「运行」页；
2. 在「真实 LLM 设置」卡片填写 Base URL / API Key / 模型；
3. 点「保存配置」——只保存在本机进程内存，不落盘、不写 `.env`；
4. 点「测试连接」——先做地址合规性校验（必须 https；主机需在白名单内；禁止私网 IP，
   本机回环 `localhost` / `127.0.0.1` 可用 http），再发一次最小真实请求验证可达；
5. 两项都通过后，勾选「使用真实 LLM」（需要参考历史任务时再勾选「参考历史任务」）。

不配置或不勾选时，仍然使用模拟引擎，Mock 演示不受影响。

### 3. 历史任务检索配置

- 记录文件：`data/run_log.jsonl`，每次运行自动追加一条（任务、预算、花费、各步骤服务与结果）；
- **只有真实 LLM 成功参与的运行才会进入检索库**，模拟运行 / 演示数据 / LLM 回退均不计入；
- 默认后端 `tfidf`（纯标准库，离线、零下载、可复现）；
- 可选向量后端：设置环境变量 `RAG_BACKEND=chroma`（需要 `chromadb`，
  索引持久化在 `kb_chroma/`）；
- `RAG_BACKEND=auto`（默认）：`kb_chroma/` 已建好时用 chroma，否则回退 tfidf。

### 4. 预算与实验参数

- 单次预算：网页输入框直接填写（也可点快捷胶囊），或命令行 `--budget`；
- 三档预算（high / medium / low）：`src/evaluation/test_tasks.json` 的 `budget_tiers`。

## 二、启动方法

### 1. 网页界面（完整版）

```powershell
python web_server.py
```

浏览器打开 http://127.0.0.1:8000

### 2. 命令行

```powershell
# 单次运行（默认多 Agent 编排；--llm 启用真实 LLM；--rag 参考历史任务）
python main.py run --task "调研行业数据并生成季度业务报告" --budget 0.5 --llm --rag

# 批量实验
python main.py experiment baselines
python main.py experiment ablation
python main.py experiment budget_sweep
python main.py experiment failure_test
python main.py experiment all
```

### 3. 静态主页（GitHub Pages）

静态页不需要任何配置，也不提供 API Key 输入：

仓库 Settings → Pages → 分支 `develop`、目录 `/docs` → Save，
访问 https://itemgroup.github.io/budget-constrained-agent/
