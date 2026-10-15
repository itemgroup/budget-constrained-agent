# **A Budget-Constrained Autonomous Agent for Enterprise Tasks


## 1. Problem and Research Motivation

Large-language-model-powered AI agents can complete multi-step workflows through task understanding, task decomposition, and tool invocation. Enterprise-oriented tasks often need to call multiple external services such as information retrieval, document analysis, and report generation. These services differ in price, output quality, and reliability.

If an agent only pays attention to task completion and does not monitor running costs, early subtasks may use up most of the allocated budget and leave insufficient resources for subsequent work. This project studies how an agent selects services according to task status and remaining budget when given a predefined enterprise task and a fixed budget. It also looks at how the agent adjusts its execution strategy when tool invocations fail or the budget becomes insufficient.

The budget in this work simulates real-world costs produced by invoking external services, including search engines, document-analysis tools, and translation APIs. Higher-quality services can be used when the budget is abundant. When resources become limited, lower-cost alternatives are selected so that later subtasks can continue running.

All services and cost calculations run inside a local simulation environment. It will not connect to real-world payment gateways or corporate financial datasets.

## 2. Proposed Approach

This is an application-oriented project with algorithm-level improvements. We use LangChain or the openJiuwen Agent Platform as the underlying framework and reuse its built-in capabilities for LLM inference, task understanding, task decomposition, and basic tool invocation. Three self-written components are implemented on top of the framework: budget management, budget-aware service selection, and dynamic replanning.

Instead of writing the full tool sequence at the beginning, the agent decides each next step when the program is running. The overall workflow is as follows. After receiving an enterprise task together with a total budget, the agent decomposes the main objective into multiple subtasks. It checks the remaining budget, filters available candidate services, and makes service selections. Then it runs simulated service calls and collects execution feedback.

When the task finishes, outputs are returned. If the task is not finished, the agent responds to three runtime conditions: normal execution, tool failure, and budget shortage. It may continue running, switch services for replanning, or use cheaper services. If no valid alternatives exist, the agent stops and returns partial outputs that have already been generated.

### 2.1 Budget Management

The budget-management component records total allocated budget, consumed expenditure, and remaining budget for each individual task instance. It updates spending records after every simulated service call and writes down the corresponding subtask, invoked service, and execution outcome.

For example, given an initial budget of $0.50, calling a $0.20 search service makes the remaining budget $0.30. Subsequent service selections are based on this updated value. Service prices are stored in configuration files, so pricing, budget values, and failure-simulation parameters can be easily modified.

### 2.2 Budget-Aware Service Selection

For each pending subtask, the system first filters out candidate services that can satisfy the requirement. For instance, information-retrieval subtasks may select from Search-Lite, Search-Pro, and Search-Premium.

Service selection considers service quality, price, remaining budget, and subtask priority. A simple scoring mechanism gives different weights to quality and cost dimensions. Candidate services are ranked by their calculated scores before one service is chosen for invocation.

### 2.3 Dynamic Replanning and Task Termination

After each tool invocation finishes, the agent decides its next move based on the returned results and remaining budget:

- If the tool successfully finishes the current subtask: proceed to subsequent subtasks.
- If tool invocation fails: do not call the faulty service again. Trigger replanning with alternative candidates if the remaining budget is enough; terminate the task if there are no viable substitutes.
- If the budget is insufficient: try degraded execution using lower-cost services. Terminate and return intermediate outputs when no appropriate services can be found.

## 3. Implementation and Evaluation Plan

### 3.1 Development Schedule

| Time              | Main Tasks                                                   |
| ----------------- | ------------------------------------------------------------ |
| Before 2026-10-09 | Complete and submit the project proposal                     |
| 10.10-10.31       | Set up agent runtime environment, implement budget-management module and configure simulated services |
| 11.01-11.10       | Develop service-selection, replanning and termination modules; debug the full execution pipeline |
| 11.11-11.14       | Run experiments, organise experimental data and prepare presentation slides |
| 2026-11-15 23:59  | Submit presentation slides                                   |
| 11.20 / 11.27     | In-class project presentation and Q&A session                |
| 11.21-11.28       | Compose final report; organise source code and README documentation |
| 2026-11-29 23:59  | Submit final report and source-code package                  |

### 3.2 Simulated Service Environment

A local simulation environment is built. Each simulated service is assigned a simulated price, quality score and failure probability. Each invocation returns simulated outputs following the preset configuration and deducts the corresponding simulated cost. These parameters are only for experimental use and do not reflect pricing or performance of real commercial services.

| Service Category  | Service Name     | Simulated Price | Quality Score | Failure Probability |
| ----------------- | ---------------- | --------------- | ------------- | ------------------- |
| Search            | Search-Lite      | $0.05           | 0.65          | 20%                 |
| Search            | Search-Pro       | $0.20           | 0.85          | 10%                 |
| Search            | Search-Premium   | $0.50           | 0.95          | 5%                  |
| Document Analysis | Analysis-Lite    | $0.05           | 0.60          | 18%                 |
| Document Analysis | Analysis-Pro     | $0.20           | 0.82          | 9%                  |
| Translation       | Translation-Lite | $0.05           | 0.62          | 15%                 |
| Translation       | Translation-Pro  | $0.15           | 0.88          | 7%                  |

### 3.3 Dataset and Test Tasks

Test cases come from two sources: ToolBench and self-constructed enterprise-style tasks. From ToolBench we filter multi-step tool-use tasks and add custom parameters including price, quality, failure probability and assigned budget for simulation. We additionally build 20-30 office-oriented test tasks covering information gathering, document summarisation, multi-document comparison and simple business-report drafting. All tasks need multiple sequential steps to finish.

Three budget tiers are assigned for every test case: high ($1.00), medium ($0.50) and low ($0.25). We compare agent behaviours under different resource constraints.

### 3.4 Comparison Baselines

Three baseline agents are implemented for side-by-side comparison against our proposed agent:

- **Standard Agent:** runs without budget constraints; invokes tools following task requirements until completion or failure.
- **Minimum-Cost Agent:** tracks consumed resources but always selects the cheapest valid service and does not care about output quality.
- **Budget Hard-Stop Agent:** keeps spending records and halts execution once budget cannot cover the next step; it has no service downgrade or replanning logic.
- **Proposed Agent:** integrates budget management, budget-aware service selection, failure-driven replanning and termination control.

### 3.5 Evaluation Metrics

Experiments assess both task-completion performance and resource consumption. We record task success rate under budget constraints, average simulated expenditure, over-budget ratio and tool-call frequency, to find out redundant repeated invocations. LLM-Judge scoring with limited manual sampling evaluates completeness and factual correctness of textual outputs. Cost-effectiveness across different agents is analysed by putting output quality and actual resource consumption together.

### 3.6 Experimental Design

- **Multi-Budget Comparison:** Run the same test set under high, medium and low budget settings, to observe how resource limits change service-selection patterns and overall task performance.
- **Baseline Comparison:** Evaluate the proposed agent alongside three baselines using identical tasks and budget configurations, collect metrics for task completion, resource usage and output quality.
- **Tool-Failure Test:** Inject simulated tool-call failures within the environment. Observe whether the agent switches to alternative services and quantify extra overhead that brought by failure handling.
- **Ablation Study:** Disable the budget-aware service-selection module and dynamic-replanning module separately. Compare ablated variants against the full-feature agent, to see each component’s influence on success rate, cost and output quality.

### 3.7 Case Analysis

Several representative test cases are analysed qualitatively. Scenarios include picking high-quality services under sufficient budget, switching to low-cost options as budget shrinks, substituting services after tool failures, and safe termination with partial outputs once budget is exhausted. Case studies show agent runtime logic and help people understand quantitative experimental results.

### 3.8 Reproducibility

Complete Python source code together with a README file will be submitted. The README tells users runtime environment, dependency installation, parameter settings and experiment-execution instructions. Service parameters and test cases are stored in configuration files. Core logic is added necessary comments. Experimental configurations and test suites are provided to make sure key experiments can be reproduced.

## 4. Expected Project Outcomes

This project delivers a budget-constrained autonomous agent for simulated enterprise tasks. Given a task and fixed budget, the agent decomposes objectives, selects services according to remaining resources and adjusts runtime behaviour based on tool-execution feedback.

When tool call meets failure, the agent attempts service substitution. When budget becomes tight, cheaper alternatives are prioritised. The agent terminates and returns partial outputs if can not find feasible services.

We compare our implementation against the Standard Agent, Minimum-Cost Agent and Budget Hard-Stop Agent. Ablation studies are conducted to find out contributions brought by budget-aware service selection and dynamic replanning.

Final deliverables contain the project proposal, presentation slides, final report, Python source code, README, experimental configurations and raw experimental results.