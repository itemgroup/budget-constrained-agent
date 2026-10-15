# Budget-Constrained Autonomous Agent for Enterprise Tasks

Project skeleton. Entry point: `main.py` (single task / batch experiments).

```
├── main.py                     # Entry point: run single task / batch experiments
├── configs/
│   └── service_config.yaml     # Simulated service price, quality, failure probability table
├── src/
│   ├── budget_manager.py       # Budget tracking, spend deduction, logging
│   ├── service_selector.py     # Score & rank candidate services
│   ├── replanner.py            # Dynamic replan, failure handling, termination logic
│   ├── simulated_services.py   # Simulated tool calls, random failure simulation
│   ├── agent_core.py           # Main agent loop: decompose task -> select service -> execute -> feedback
│   └── evaluation/
│       ├── test_tasks.json     # ToolBench filtered + custom enterprise tasks
│       ├── evaluator.py        # Metrics, baseline runner, ablation runner
│       └── llm_judge.py        # LLM judge for output quality scoring
├── experiments/
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_budget_sweep.py     # High/Medium/Low budget comparison
│   ├── run_failure_test.py
│   └── results/                # CSV logs, outputs saved here
└── docs/
    ├── proposal.md
    └── slides_notes.md
```
