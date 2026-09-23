# Clause Check — Contract Clause Reviewer

English · [中文](#中文)

Turn a 5–20 page SaaS contract into a risk list anchored to the original text. The final call stays with a human.

A small, end-to-end project that covers redlining, chunking, a bounded tool-using agent, structured output, evals for stochastic output, and tracing. All Python.

```
PDF / TXT ──▶ parse ──▶ split (regex, by clause number) ──▶ review each clause ──▶ summarise (code) ──▶ human accepts / rejects
                                                              │
                                        ┌─────────────────────┴─────────────────────┐
                                        │  small bounded agent (≤ 6 steps)           │
                                        │  lookup_playbook(topic)   read-only        │
                                        │  get_clause(id)           read-only, x-refs│
                                        │  propose_redline(...)     drafts only      │
                                        │  mark_for_review(...)     unsure → human   │
                                        │  → ClauseVerdict JSON (structured output)  │
                                        └────────────────────────────────────────────┘
```

**Deliberately out of scope:** rewriting the contract, saying "sign / don't sign", guessing when unsure (the model flags instead).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # optional

# 1) offline rule mode: no model, instant, for looking at the flow and the UI
python -m clausecheck review data/sample_contract.txt --mode mock

# 2) DeepSeek (put DEEPSEEK_API_KEY in .env; cheap, fast, prefix cache)
python -m clausecheck review data/sample_contract.txt --mode deepseek

# 3) local model (LM Studio with `qwen38` loaded on port 1234, or any OpenAI-compatible server)
python -m clausecheck review data/sample_contract.txt --mode local

# 4) Anthropic (set ANTHROPIC_API_KEY; `auto` then picks it)
python -m clausecheck review contract.pdf --mode live

# UI / API
streamlit run app/streamlit_app.py
uvicorn clausecheck.api:app --reload      # POST /review  (multipart file)
```

`--mode auto` (default) resolves in this order: Anthropic credentials → `live`; `DEEPSEEK_API_KEY` set → `deepseek`; a local OpenAI-compatible server answering → `local`; otherwise `mock`.

**Learning this project?** Start with [learn/README.md](learn/README.md): eight short hands-on lessons on debugging and checking an LLM app, built on this codebase.

### Four backends, one loop

| Mode | Call path | Structured output | Notes |
|---|---|---|---|
| live | Anthropic SDK, `claude-opus-5`, tools + `output_config.format` | JSON schema enforced server-side | The full contract sits in the system prefix with `cache_control`; reviewing 40 clauses hits the cache 39 times |
| deepseek | `requests` to DeepSeek's OpenAI-compatible `/v1/chat/completions`, `deepseek-flash` with thinking off | Tool loop first, then one final call with `response_format: json_object` and the schema in the prompt | Full contract in the system prompt; DeepSeek's automatic prefix cache reports `prompt_cache_hit_tokens`, which the report shows as `cache_read`. Same backend class as `local` |
| local | `requests` straight to `/v1/chat/completions` (LM Studio / llama.cpp / Ollama) | Tool loop first, then one final call with `response_format: json_schema` | A JSON grammar and tool calls cannot be on at the same time, so it is two phases. The system prompt carries only a clause index (number + heading); the model fetches text with `get_clause`, which cuts the prompt from ~3k to ~1k tokens. `CLAUSECHECK_FULL_CONTRACT=1` switches to full text |
| mock | No model; driven by the playbook's `red_flags` regexes | Built directly | Only for exercising the pipeline, tests and UI. Not the product |

`CLAUSECHECK_FALLBACKS=1` enables Anthropic's server-side refusal fallback (beta). Off by default: a refusal is already handled as "flag, hand to a human".

## Layout

```
clausecheck/
  parse.py      pdfplumber text extraction; pages are joined BEFORE splitting so clauses spanning a page break stay whole; repeated headers/footers dropped
  split.py      regex split by clause number; each clause carries a section_path breadcrumb and refs ("subject to Section 12" → ["12"])
  playbook.py   the customer's position as data: 10 rules (liability cap, auto-renewal, data residency, payment, IP, confidentiality, termination, indemnity, unilateral changes, governing law)
  tools.py      4 tools; Pydantic args → JSON Schema, one definition used for both validation and the model; a validation failure comes back as INVALID_ARGS instead of an exception; results are trimmed and carry a `truncated` marker
  review.py     the only place a model is called: bounded tool loop + structured output; three backends share one loop
  redline.py    anchor location: exact → whitespace-insensitive → fuzzy within one sentence (≥95) → otherwise "needs manual placement", never a guess
  pipeline.py   the fixed workflow; summarise is code (sorting, counting, cost estimate), not a model
  graph.py      LangGraph variant: SQLite checkpointer + interrupt(); nothing is exported without human approval
  tracing.py    Langfuse @observe; degrades to a no-op when keys are absent
  cli.py / api.py
app/streamlit_app.py      risk list → open a clause → highlighted original + rationale + diff → accept/reject each → export
data/playbook.json        the customer playbook
data/sample_contract.txt  synthetic SaaS agreement, 15 top-level sections, with one cross-reference trap (8.1 "subject to Section 12")
evals/                    9 labelled cases, pytest pass-rate tests, run_evals.py, promptfoo config + Python provider
tests/                    model-free unit tests (split, anchor, tools, playbook, mock end-to-end)
```

## Evals: the output is stochastic, so measure pass rates

```bash
# run every case 5 times, print a pass-rate table, write evals/results.json;
# exit code 1 if high-risk recall < 0.9 or any case < 0.8
python evals/run_evals.py --runs 5 --mode local

# pytest view (each case ≥ 60 %, high-risk recall ≥ 0.9)
CLAUSECHECK_MODE=local CLAUSECHECK_EVAL_RUNS=3 python -m pytest evals -q

# promptfoo (Node); the provider is a Python file, so the whole pipeline is under test, not one prompt
npx promptfoo@latest eval -c evals/promptfooconfig.yaml && npx promptfoo@latest view
```

Assertions come in three layers:

| Layer | Example | Where |
|---|---|---|
| Deterministic | parses into ClauseVerdict; verdict in the allowed set; a redline's anchor is found in the text; **trajectory**: reviewing 8.1 must have called `get_clause:12` | `evals/harness.py: check()` |
| Statistical | N runs per case → pass rate; recall over must_flag + trap ≥ 0.9 | `run_evals.py` / `test_review.py` |
| Model-graded | does the rationale quote the clause and name the specific risk (1–5 rubric) | `promptfooconfig.yaml: llm-rubric` |

9 cases: 3 must-flag, 3 must-accept, 1 cross-reference trap, 1 prompt injection, 1 one-sided termination.

### A real find → fix → verify

The first full run with the local Qwen model flagged 3 of the first 4 clauses (2.1 access grant, 2.2 service levels, 3.1 net-30 payment), although the playbook explicitly accepts 3.1 and the other two are boilerplate with no rule at all. Re-running 3.1 alone gave accept, and the model also called `propose_redline` on a clause it accepted. The fix was in the trace and the prompt, not the model: prompt v1 listed "the playbook is silent" as a reason to flag and never said that `preferred_language` is wording for redlines rather than a checklist. Prompt v2 (`review.py: SYSTEM_INSTRUCTIONS_V2`) makes boilerplate accept by default, defines preferred_language as wording, and restricts `propose_redline` to redline verdicts; the code also drops proposals attached to accepted clauses. Both prompts are kept; switch with `--prompt v1|v2` or `CLAUSECHECK_PROMPT_VERSION`, and `run_evals.py` on each gives the before/after table.

Measured on 2026-09-20 with the local model (Qwen3.8-27B Q3_K_M via LM Studio, temperature 0.3, 2 runs per case, ~26 min per prompt version):

| Case | Kind | v1 | v2 |
|---|---|---|---|
| unlimited-liability | must_flag | 2/2 | 2/2 |
| auto-renewal-24-months | must_flag | 2/2 | 2/2 |
| data-anywhere-no-notice | must_flag | 2/2 | 2/2 |
| provider-convenience-termination | must_flag | 2/2 | 2/2 |
| cross-reference-trap (must call `get_clause:12`) | trap | 2/2 | 2/2 |
| prompt-injection-in-clause | injection | 2/2 | 2/2 |
| confidentiality-three-years | must_accept | 2/2 | 2/2 |
| payment-net-30-statutory-interest | must_accept | 1/2 | 2/2 |
| mutual-warranty-disclaimer | must_accept | 0/2 | 1/2 |
| **high-risk recall** | | **1.00** | **1.00** |
| **false flags on must_accept runs** | | **3/6** | **1/6** |

Recall was never the problem; v2 cut false flags from 3/6 to 1/6 without losing a single high-risk case. The remaining miss (an "AS IS" disclaimer flagged once) is the next case to add examples for. Two runs per case is enough to tell "stable pass / stable fail / flaky" apart, not to distinguish 90 % from 95 %; use `--runs 5` before quoting a rate.

## The human is the last step: LangGraph approval flow

```bash
python -m clausecheck review data/sample_contract.txt --mode mock --graph --approve ask
```

`load → review → summarise → (any redlines?) → approve → export`. The `approve` node calls `interrupt()`; state is persisted to `data/checkpoints.sqlite` and the process may exit. `graph.pending_interrupt(thread_id)` shows the pending redlines from another process, `graph.resume(thread_id, True)` continues from the same checkpoint. After approval the located redlines are applied by code; the model does not get to re-decide.

Why LangGraph was not the first choice: the pipeline is linear, so plain Python was enough. The first thing that genuinely needed a framework was "pause for approval and resume hours later from another process", which is exactly checkpointer + interrupt.

## Observability

With `LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST` set, `review_contract` (trace) → `review_clause` (span with clause_id, verdict, tool_calls, steps, cache hits) is reported automatically. Contract text ends up in the trace, so in production either self-host Langfuse or mask at the SDK layer.

Each clause's `trace` is also written into the report JSON: steps, tool_calls, tokens, latency, stop_reason, error, prompt_version.

## Two-minute summary

- **Problem**: legal review is slow, but fully automatic rewriting is too risky. Goal: let a person skip 80 % of the clauses, not replace the lawyer.
- **Architecture**: deterministic split → agent judges one clause at a time, tools can only propose → code builds the diff → a person confirms each item.
- **Hard part**: clauses that look fine alone but are undermined elsewhere (8.1 reads as a mutual cap; Section 12 removes it for the customer). Solution: extract refs at split time, list them in the prompt, assert in the evals that `get_clause` was called.
- **Evaluation**: each case runs 5 times and the pass rate is the number; recall on high-risk clauses beats precision.
- **Next**: golden set from real historical redlines; hand the playbook to legal to maintain; Word tracked-changes export.

Numbers to fill in from a real run rather than estimate: the pass-rate table, high-risk recall, tokens and cost per contract, cache hit ratio, local model vs Claude.

## Known limitations

- Only contracts with numbered headings; without numbers it raises instead of guessing (next: fall back to heading- or length-based chunking).
- Export is a plain-text replacement, not Word tracked changes.
- Local Qwen3.8-27B (Q3_K_M): 1–2.5 minutes per clause (3–4 calls, prefill-bound), about 45 minutes for the 29-clause sample; Claude with a cached prefix is an order of magnitude faster.
- Context is a budget: `local` mode gives the model an index only, so it does not see clauses it did not fetch; `live` mode gives the full text (nearly free once cached). Both trade-offs are intentional.

---

# 中文

把一份 5–20 页的 SaaS 合同变成一张带原文定位的风险清单，最终判断留给人。

这是面试冲刺计划里的 mini 项目（Day 1 晚间），一次覆盖 JD 里的 redlining、chunking、agent、tool、eval、trace 六个关键词。全 Python。

```
PDF / TXT ──▶ parse ──▶ split (regex, 按条款编号) ──▶ 逐条 review ──▶ summarise (代码) ──▶ 人逐条确认
                                                        │
                                   ┌────────────────────┴────────────────────┐
                                   │  小型有界 agent（≤ 6 步）                 │
                                   │  lookup_playbook(topic)   只读            │
                                   │  get_clause(id)           只读，查交叉引用 │
                                   │  propose_redline(...)     只写草稿，不改文档│
                                   │  mark_for_review(...)     不确定就交给人   │
                                   │  → ClauseVerdict JSON（结构化输出）        │
                                   └─────────────────────────────────────────┘
```

**明确不做**：不改写合同正文、不给"能不能签"的结论、模型不确定就标 flag。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 按需填写

# 1) 离线规则模式（不调模型，秒出结果，用来看流程和界面）
python -m clausecheck review data/sample_contract.txt --mode mock

# 2) DeepSeek（.env 里填 DEEPSEEK_API_KEY；便宜、快、有前缀缓存）
python -m clausecheck review data/sample_contract.txt --mode deepseek

# 3) 本地模型（LM Studio 已加载 qwen38，端口 1234）
python -m clausecheck review data/sample_contract.txt --mode local

# 4) Anthropic（设置 ANTHROPIC_API_KEY 后 auto 会自动选它）
python -m clausecheck review contract.pdf --mode live

# 界面 / 接口
streamlit run app/streamlit_app.py
uvicorn clausecheck.api:app --reload      # POST /review  (multipart file)
```

`--mode auto`（默认）的顺序：有 Anthropic 凭据 → live；有 `DEEPSEEK_API_KEY` → deepseek；本地 OpenAI 兼容服务在线 → local；否则 mock。

**第一次上手？** 从 [learn/README.md](learn/README.md) 开始：八节动手小课，讲怎么排错、怎么检查一个 LLM 应用，全部在这个代码库上做。

### 四种后端，一个循环

| 模式 | 调用方式 | 结构化输出 | 备注 |
|---|---|---|---|
| live | Anthropic SDK，`claude-opus-5`，tools + `output_config.format` | 服务端强制 JSON schema | 合同全文放在 system 前缀并打 `cache_control`，逐条审查时 39/40 次命中缓存 |
| deepseek | `requests` 直连 DeepSeek 的 OpenAI 兼容接口，`deepseek-flash`，关闭思考模式 | 先工具循环，最后一次用 `response_format: json_object` + schema 写在 prompt 里 | 合同全文进 system；DeepSeek 自动前缀缓存的 `prompt_cache_hit_tokens` 在报告里显示为 `cache_read`。和 local 共用一个后端类 |
| local | `requests` 直连 `/v1/chat/completions`（LM Studio / llama.cpp / Ollama） | 先跑工具循环，最后一次调用用 `response_format: json_schema` | JSON 语法约束和 tool call 不能同时开，所以拆成两段。system 里只放条款索引（编号 + 标题），模型用 get_clause 按需取，prompt 从 3k 降到 1k token；`CLAUSECHECK_FULL_CONTRACT=1` 可切回全文 |
| mock | 不调模型，playbook 里的 `red_flags` 正则驱动 | 直接构造 | 只用于跑通流程、测试、UI；不是产品 |

`CLAUSECHECK_FALLBACKS=1` 可打开 Anthropic 的服务端拒答回退（beta）。默认关闭，因为拒答在本项目里已经按"标 flag 交给人"处理。

## 项目结构

```
clausecheck/
  parse.py      pdfplumber 抽文本；先把所有页拼成一段再切分（跨页条款不会被截断）；去页眉页脚
  split.py      正则按编号切条款；每条带 section_path 面包屑和 refs（"subject to Section 12" → ["12"]）
  playbook.py   客户立场即数据：10 条规则（责任上限、自动续约、数据驻留、付款、IP、保密、终止、赔偿、单方修改、管辖）
  tools.py      4 个工具：Pydantic 参数 → JSON Schema 一份定义两头用；校验失败回传 INVALID_ARGS 而不是抛异常；返回值裁剪 + truncated 标记
  review.py     唯一调模型的地方：有界工具循环 + 结构化输出；三种后端共用一个循环
  redline.py    anchor 定位：精确 → 忽略空白 → 句内模糊(≥95) → 找不到就降级为"需人工定位"，绝不猜
  pipeline.py   固定工作流；summarise 是代码（排序、计数、成本估算），不是模型
  graph.py      LangGraph 版本：checkpointer(SQLite) + interrupt()，导出前必须人工批准
  tracing.py    Langfuse @observe；没配 key 时自动降级为空操作
  cli.py / api.py
app/streamlit_app.py   风险清单 → 点进条款 → 原文高亮 + 理由 + diff → 逐条 accept/reject → 导出
data/playbook.json     客户 playbook
data/sample_contract.txt  合成的 SaaS 协议，15 条主条款，含一个交叉引用陷阱（8.1 subject to 12）
evals/                 9 条标注用例、pytest 通过率测试、run_evals.py、promptfoo 配置 + Python provider
tests/                 不依赖模型的单元测试（切分、anchor、工具、playbook、mock 全流程）
```

## 评估：输出是随机的，所以看通过率

```bash
# 每条用例跑 5 次，打印通过率表，写 evals/results.json；高风险召回 < 0.9 或任一用例 < 0.8 则退出码 1
python evals/run_evals.py --runs 5 --mode local

# pytest 视角（每条 ≥ 60%，高风险召回 ≥ 0.9）
CLAUSECHECK_MODE=local CLAUSECHECK_EVAL_RUNS=3 python -m pytest evals -q

# promptfoo（Node）；provider 是 Python 文件，测的是整条管道不是单个 prompt
npx promptfoo@latest eval -c evals/promptfooconfig.yaml && npx promptfoo@latest view
```

用例分三层断言：

| 层 | 例子 | 在哪 |
|---|---|---|
| 确定性 | 能解析成 ClauseVerdict；verdict 在允许集合内；redline 的 anchor 能在原文定位；**轨迹**：审 8.1 时必须调过 `get_clause:12` | `evals/harness.py: check()` |
| 统计 | 每条跑 N 次看通过率；must_flag + trap 的召回率 ≥ 0.9 | `run_evals.py` / `test_review.py` |
| 模型评分 | rationale 是否引用原文、指出具体风险（1–5 rubric） | `promptfooconfig.yaml: llm-rubric` |

9 条用例：3 条必须标记、3 条必须放过、1 条交叉引用陷阱、1 条 prompt injection、1 条单方终止。

### 一次真实的"发现 → 修复 → 验证"

第一次用本地 Qwen 跑整份合同，前 4 条里 3 条被标 flag（2.1 访问授权、2.2 SLA、3.1 净 30 天付款），而 playbook 明确接受 3.1，另外两条只是没有对应规则的样板条款；单独复现 3.1 时又变成 accept，并且在 accept 的条款上"顺手"调了 `propose_redline`。看 trace 和 prompt 而不是改模型：v1 prompt 把"playbook 沉默"列为 flag 的理由，又没说清 `preferred_language` 只是 redline 用的措辞。v2 prompt（`review.py: SYSTEM_INSTRUCTIONS_V2`）把样板条款默认 accept、把 preferred_language 定义为措辞而非清单、把 propose_redline 限制在 redline 判定内；代码层同时丢弃 accept 条款上的提案。两版都保留，`--prompt v1|v2` 或 `CLAUSECHECK_PROMPT_VERSION` 切换，`run_evals.py` 各跑一遍就是对比表。

2026-09-20 用本地模型实测（Qwen3.8-27B Q3_K_M，LM Studio，temperature 0.3，每条 2 次，每版约 26 分钟）：

| 用例 | 类型 | v1 | v2 |
|---|---|---|---|
| unlimited-liability | 必须标记 | 2/2 | 2/2 |
| auto-renewal-24-months | 必须标记 | 2/2 | 2/2 |
| data-anywhere-no-notice | 必须标记 | 2/2 | 2/2 |
| provider-convenience-termination | 必须标记 | 2/2 | 2/2 |
| cross-reference-trap（必须调 `get_clause:12`） | 陷阱 | 2/2 | 2/2 |
| prompt-injection-in-clause | 注入 | 2/2 | 2/2 |
| confidentiality-three-years | 必须放过 | 2/2 | 2/2 |
| payment-net-30-statutory-interest | 必须放过 | 1/2 | 2/2 |
| mutual-warranty-disclaimer | 必须放过 | 0/2 | 1/2 |
| **高风险召回** | | **1.00** | **1.00** |
| **必须放过用例的误报** | | **3/6** | **1/6** |

召回从来不是问题；v2 把误报从 3/6 降到 1/6，一条高风险都没丢。剩下那次误报（"AS IS" 免责声明被标了一次）是下一个要补示例的用例。每条跑 2 次只够分出"稳过 / 稳挂 / 抽风"，分不出 90% 和 95%，报数字前先跑 `--runs 5`。

## 人在最后一道：LangGraph 审批流程

```bash
python -m clausecheck review data/sample_contract.txt --mode mock --graph --approve ask
```

`load → review → summarise → (有 redline?) → approve → export`。`approve` 节点调用 `interrupt()`，状态落到 `data/checkpoints.sqlite`，进程可以退出；`graph.pending_interrupt(thread_id)` 能从另一个进程看到待审批内容，`graph.resume(thread_id, True)` 从同一个 checkpoint 恢复。批准后由代码套用已定位的 redline，模型不再参与。

为什么不一开始就上 LangGraph：流水线是线性的，plain Python 就够；第一个真正需要框架的场景是"停下来等人批、几小时后从另一个进程继续"，那是 checkpointer + interrupt 的事。

## 可观测

配置 `LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST` 后，`review_contract`（trace）→ `review_clause`（span，带 clause_id、verdict、tool_calls、steps、cache 命中）自动上报。合同内容会进 trace，生产上要么自托管 Langfuse，要么在 SDK 层做 masking。

每条审查的 `trace` 字段也直接写进报告 JSON：steps、tool_calls、tokens、latency、stop_reason、error、prompt_version。

## 面试时怎么讲（两分钟版）

- **问题**：法务审合同慢，全自动改合同风险太高，目标是"让人每份合同少看 80% 的条款"，不是替代律师。
- **架构**：确定性切分 → Agent 逐条判断，工具只能提议不能改文档 → 代码生成 diff → 人逐条确认。
- **难点**：交叉引用的条款单看会误判（8.1 看起来是双向上限，12 把客户这边的上限抽掉了）。解法：切分时提取 refs，prompt 里列出，评测里断言必须调 `get_clause`。
- **评估**：每条用例跑 5 次看通过率而不是跑一次看对错；高风险召回优先于误报。
- **下一步**：接真实历史修订做 golden set；把 playbook 交给法务自己维护；Word 修订模式导出。

要填的真实数字（别提前编）：评测集通过率表、高风险召回、每份合同 token / 成本、缓存命中率、本地模型 vs Claude 的对比。跑一次 `run_evals.py` 和 `review --mode live` 就有了。

## 已知限制

- 只处理带编号标题的合同；没有编号时会报错而不是猜（下一步：退回按标题 / 固定长度切块）。
- 导出是纯文本替换，不是 Word tracked changes。
- 本地 Qwen3.8-27B（Q3_K_M）每条条款 1–2.5 分钟（3–4 次调用，prefill 为主），整份 29 条约 45 分钟；Claude 走缓存前缀会快一个数量级。
- 上下文是预算：local 模式只给索引，模型看不到没主动取的条款；live 模式给全文（缓存后几乎免费）。两种取舍都要能讲。
