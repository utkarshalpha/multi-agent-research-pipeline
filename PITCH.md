# Multi-Agent Research Pipeline — Pitch & Prep Doc

> Private prep for portfolio review / technical interview. Everything below is grounded in the current code on disk; file:line references are real and verified. Honest status: **implemented and verified in `MOCK_MODE`; not yet validated against live Anthropic/Tavily/arXiv (needs keys).**

---

## 1. 30-Second Elevator Pitch

A FastAPI service that turns one research question into a fully-cited Markdown report by orchestrating four specialized agents — **Planner → Researcher → Critic → Writer** — as a LangGraph state machine with a self-healing retry loop. The Critic scores every piece of evidence; when it's weak, a conditional edge loops back to the Researcher, which *reformulates* its queries from the feedback instead of repeating the same search. The whole pipeline runs end-to-end with **zero network calls and zero API keys** thanks to a deterministic `MOCK_MODE`, so it's fully testable and CI-able. The LLM is a swappable backend behind one factory function — the project is the orchestration, not the model.

---

## 2. 2-Minute Pitch

**The problem.** LLMs hallucinate facts and citations. A single prompt that says "research X and cite your sources" gives you confident prose with invented URLs and no way to tell good evidence from bad. You need *retrieval* (fresh, citable sources) plus *judgment* (is this evidence actually relevant, credible, and recent?) plus *self-correction* (if the evidence is weak, go get better evidence).

**The solution.** Four agents, each owning one job, communicating only through a typed shared state:
- **Planner** decomposes the question into 3–5 self-contained sub-questions (`agents/planner.py:22-47`).
- **Researcher** gathers evidence per sub-question — checks a semantic cache, then hits Tavily web search and arXiv in parallel (`agents/researcher.py:154-196`).
- **Critic** scores every result on relevance (LLM-judged), credibility, and recency (deterministic), and decides pass/fail (`agents/critic.py:121-208`).
- **Writer** synthesizes a structured, inline-cited report and back-fills the citation list from sources actually gathered (`agents/writer.py:95-145`).

When the Critic fails the evidence, a conditional edge loops back to the Researcher until a retry budget is spent (`graph/edges.py:11-37`, `MAX_RETRIES = 2`).

**Why it's more than an API wrapper.** The value is in the orchestration and the guardrails: a real graph/state-machine with a cyclic retry loop a linear chain can't express; structured Pydantic outputs at every hop instead of regex parsing; a semantic cache to cut cost/latency; tenacity backoff scoped to *only* transient errors; graceful degradation of every external dependency; and groundedness-aware evaluation that catches hallucinated citations. The LLM itself sits behind one `get_llm()` factory (`config.py:139-177`) — swap Sonnet for Opus, or the real model for a mock, with no agent changes.

**The honest status.** The system is **implemented and provably correct end-to-end in `MOCK_MODE`** — the real LangGraph, agents, critic loop, cache, and API all run unchanged against deterministic offline mocks (61 tests green on every push). It has **not yet been run against live Anthropic/Tavily/arXiv** — that's a key-swap and a validation pass, not a rewrite. `MODEL_LABEL` even tags mock runs explicitly, e.g. `mock (claude-sonnet-4-6)` (`config.py:59`), so offline results are never mistaken for live ones.

---

## 3. The Story to Tell

This is the narrative spine — tell it as a journey, because it shows engineering judgment, not just feature output.

**Act 1 — A well-built but UNVALIDATED skeleton.** Every module was implemented, zero stubs — but the live pipeline had never run once. No API keys, an empty `.env`, no eval results, and the "demo" was a static JSON fixture (`examples/sample_response.json`). A multi-agent audit established exactly that: complete on paper, unproven in practice.

**Act 2 — Made provably correct offline, without keys.** Rather than wait for credentials, I built a deterministic `MOCK_MODE` (`config.py:43`, `tools/mocks.py`): a mock LLM that *introspects any Pydantic structured-output schema* and synthesizes a valid instance, plus mock Tavily/arXiv and hash-derived 384-dim embeddings. This drives the **real** LangGraph end-to-end with zero network. Determinism (SHA-256 of inputs, no `uuid4`/randomness in the mocks) makes every test reproducible.

**Act 3 — Fixed the real bugs the audit found.** Temperature passed to model families that 400 on it; an over-broad retry predicate that burned ~60s of backoff retrying 400/401 four times; a dead tenacity retry on Tavily that never actually fired; duplicate Qdrant cache points from `uuid4`. (War stories in §5.)

**Act 4 — Made the critic loop genuinely self-healing.** A retry that re-runs the identical query is theatre. Now retries bypass the cache, reformulate each query from the Critic's feedback, and surface zero-result sub-questions as `unanswered_questions` instead of dropping them silently.

**Act 5 — Hardened, tested, shipped.** API contracts in `schemas/models.py`, sanitized 500s, a 504 timeout, optional `X-API-Key` auth, per-IP rate limiting; tests grown from 3 → **61**; GitHub Actions CI that runs the offline suite, then builds, smoke-tests (a *real* `POST /research` against the running container), and publishes a Docker image to GHCR; and eval quality scoring including groundedness.

**Reframe "no API keys" as a strength, not a gap.** Building deterministic mocks that satisfy the real interfaces forced clean seams everywhere — the LLM, search, and embeddings are all swappable behind narrow factory functions. The payoff: the entire pipeline is **reproducible, unit-testable, and runs in CI keyless**, which most LLM projects can't claim. Going live is `MOCK_MODE=false` plus three keys.

---

## 4. Architecture at a Glance

### Request lifecycle: `POST /research` → response
1. **Auth + rate limit** via FastAPI dependencies (`main.py:181`): constant-time `X-API-Key` check (`main.py:52-64`) and a per-IP sliding-window limiter (`main.py:85-104`). Both no-op when their settings are empty/0.
2. **Body validation** into `ResearchRequest` (`query`, `min_length=3`) (`schemas/models.py:110-113`).
3. **Run setup**: mint `run_id = uuid4` (`main.py:193`), fetch the cached compiled graph (`main.py:196`), attach a `UsageMetadataCallbackHandler` for token accounting (`main.py:197`).
4. **Invoke under a hard timeout**: `asyncio.wait_for(graph.ainvoke(...), timeout=RESEARCH_TIMEOUT_SECONDS)` (default 300s) (`main.py:215-217`, `config.py:97`).
5. **Error mapping**: `TimeoutError` → **504**; any other exception → **sanitized 500** carrying only the `run_id` (`main.py:218-231`).
6. **Response assembly**: pull `final_report`, citations (preferring the writer's `all_citations`, else de-duped source URLs — `_extract_citations`, `main.py:166-175`), aggregate token usage (`main.py:243-248`), and return `ResearchResponse(run_id, report, citations, metadata)` (`main.py:262-264`).

### The four agents

| Agent | Node | Does | Structured output |
|-------|------|------|-------------------|
| **Planner** | `agents/planner.py:22-47` | Query → 3–5 self-contained sub-questions | `SubQuestions` (`questions`, `min_length=3, max_length=5`, `schemas/models.py:15-24`) |
| **Researcher** | `agents/researcher.py:199-289` | Per sub-question: cache lookup → parallel Tavily + arXiv → normalize → cache write-back; on retry, re-research only weak/unanswered questions with reformulated queries, cache bypassed | Produces `list[ResearchResult]` (no LLM schema bound — it's the tool/orchestration agent) |
| **Critic** | `agents/critic.py:121-208` | Scores each result `0.5·relevance + 0.25·credibility + 0.25·recency`; relevance LLM-judged, the rest deterministic; decides `overall_pass` | LLM returns private `_RelevanceAssessment`; node emits `CritiqueResult` |
| **Writer** | `agents/writer.py:95-145` | Selects top sources, synthesizes a structured report with inline `[n]` citations, back-fills `all_citations`, renders Markdown, stub fallback on empty | `FinalReport` (`title`, `summary`, `sections`, `all_citations`) |

### State machine + retry loop
- **State**: `AgentState` is a `TypedDict(total=False)` (`graph/state.py:16-30`) threaded through all four nodes; each returns a partial dict LangGraph merges (last-write-wins, no custom reducers — every field is fully owned by one node). Key fields: `query`, `sub_questions`, `research_results`, `critique`, `final_report`, `retry_count`, `run_id`, `unanswered_questions`.
- **Wiring**: `planner → researcher → critic`, then a conditional edge, and `writer → END` (`graph/graph.py:38-46`).
- **Retry edge**: `should_retry` returns `"researcher"` iff `critique is not None and not critique.overall_pass and retry_count < MAX_RETRIES`, else `"writer"` (`graph/edges.py:30-37`). `MAX_RETRIES = 2` (`config.py:88`).
- **Compilation**: built once with a `MemorySaver` checkpointer (keyed by `thread_id = run_id`), cached via `@lru_cache(maxsize=1)` (`graph/graph.py:51-60`).

### Tool & memory layer (every dependency degrades gracefully)
- **Tavily** (`tools/search.py`): tenacity (3 attempts), per-call `asyncio.wait_for` timeout; missing package / timeout / exhausted retries → `[]` (`tools/search.py:40-101`).
- **arXiv** (`tools/arxiv.py`): blocking lib wrapped in `asyncio.to_thread`; any error → `[]` (`tools/arxiv.py:50-75`).
- **Qdrant semantic cache** (`tools/vector_store.py`): local fastembed `BAAI/bge-small-en-v1.5` (384-dim) embeddings; cosine search with `score_threshold = 0.85`; deterministic `uuid5(question|url)` point IDs; every op degrades to a cache-miss/no-op on error (`tools/vector_store.py:142-153`).
- **Redis short-term memory** (`memory/redis_store.py`): per-run audit trail under a `run:<run_id>` hash with 24h TTL; if absent, every method is a silent no-op (`memory/redis_store.py:42-52`).
- **LLM resilience** (`config.py`): `get_llm` shared via `@lru_cache`, omits sampling params for families that reject them; `ainvoke_with_retry` retries only transient errors (`config.py:189-215`).

### Compact file map
| Path | Responsibility |
|------|----------------|
| `main.py` | FastAPI app: `POST /research`, `/health`, `/sample-response`, static UI; auth + rate-limit deps; lifespan; timeout, token aggregation, response assembly |
| `config.py` | Settings, `get_llm` factory, `ainvoke_with_retry`, LangSmith, `MOCK_MODE`, thresholds |
| `graph/state.py` | `AgentState` TypedDict — shared contract |
| `graph/graph.py` | Builds + compiles the `StateGraph` (`MemorySaver`, lru_cache) |
| `graph/edges.py` | `should_retry` conditional-edge decision |
| `agents/planner.py` | Query → sub-questions |
| `agents/researcher.py` | Cache + parallel fetch, retry reformulation, `unanswered_questions` |
| `agents/critic.py` | Relevance (LLM) + credibility + recency → `CritiqueResult` |
| `agents/writer.py` | Source selection + synthesis → `FinalReport`, Markdown, stub fallback |
| `schemas/models.py` | All Pydantic v2 contracts |
| `tools/search.py` · `tools/arxiv.py` · `tools/vector_store.py` | Tavily / arXiv / Qdrant+fastembed |
| `tools/mocks.py` | Deterministic offline mocks + `force-retry` test hook |
| `memory/redis_store.py` | Per-run audit trail + `redis_store` singleton |

---

## 5. Engineering Decisions You Can Defend

### Decision → Why → Tradeoff

| # | Decision | Why | Tradeoff |
|---|----------|-----|----------|
| 1 | **One swappable LLM factory** — every agent gets its model from `get_llm()` (`config.py:139-177`), never instantiating a client directly | Value is the orchestration, not the model; model/temp/tokens/retry configured in exactly one place; mock↔live, Sonnet↔Opus swap with no agent changes | Caching by `(temperature, max_tokens)` means config changes need a fresh process; all agents share one tier (no per-agent routing yet) |
| 2 | **Deterministic offline mode via Pydantic introspection** — `MockChatModel.with_structured_output(Schema)` synthesizes a valid instance by walking `model_fields`/`Literal`/`Union`/`list`/`min_length` (`tools/mocks.py:298-404`) | Makes the whole pipeline provably correct end-to-end with zero network/keys; SHA-256-based determinism makes tests reproducible | Proves the *wiring* is correct, not that *prompts* elicit good answers from a real model; generator must track exotic new field types |
| 3 | **Critic = LLM relevance + deterministic credibility/recency, weighted 0.5/0.25/0.25** (`agents/critic.py:33-35`) | Only the subjective dimension (does this answer the question?) needs an LLM; credibility/recency are cheaper, explainable, reproducible pure functions; relevance weighted highest because credible-but-irrelevant is useless | Credibility is a ~30-domain allowlist heuristic, not a reputation signal; recency drifts via `date.today()`; weights & the 0.7 gate are hand-tuned |
| 4 | **Feedback-driven self-healing retry** — `overall_pass` true only if no result is below threshold AND zero `unanswered_questions`; retries bypass cache and reformulate queries (`agents/critic.py:190`, `agents/researcher.py:220-264`) | A retry that re-runs the identical query returns identical evidence — theatre. Reformulation + cache-bypass makes the loop actually self-correcting; weak coverage is surfaced, not dropped | Reformulation is deterministic keyword augmentation (robust, testable, mock-identical) but less creative than an LLM rewrite; budget fixed at 2 |
| 5 | **Semantic cache at cosine ≥ 0.85** (`config.py:82`, `tools/vector_store.py:126-150`) | Near-duplicate questions reuse prior evidence → less latency/cost; high threshold avoids serving loosely-related hits | 0.85 is a hand-chosen precision/recall point; no TTL, so evidence can age (mitigated by retries bypassing it) |
| 6 | **Provider-correct sampling + fail-fast retry predicate** — strip `temperature` for opus-4-7/4-8/fable/mythos; retry only connection/timeout/429/5xx (`config.py:130-136, 189-201`) | Newer families reject sampling params with HTTP 400; retrying 4xx burns ~60s of backoff on a request that can never succeed | The marker tuple is a string-match heuristic to maintain as Anthropic ships tiers |
| 7 | **Deterministic `uuid5` cache point IDs** (`tools/vector_store.py:180-184`) | Idempotent upserts — same (question, source) → one stable point instead of duplicates on every re-research | Identity pinned to exact question+URL; reworded question + same URL → distinct point (fine — vector lookup dedupes semantically) |
| 8 | **Groundedness-aware evals** — `quality_score` = equal-weighted mean of 6 components incl. groundedness + coverage, pass at ≥ 0.7 (`evals/run_evals.py:64-68, 131-152`) | A fast/cheap pipeline that hallucinates citations is worse than useless; groundedness catches fabricated sources, coverage catches dropped sub-questions | Measures structure + provenance, not answer *correctness* (needs a gold set / LLM judge) |
| 9 | **Defense-in-depth API hardening** — optional key auth (`secrets.compare_digest`), per-IP sliding-window limit, 504 timeout, sanitized 500 (`main.py:52-104, 178-231`) | The things an interviewer expects on a public endpoint: timing-safe compare, abuse protection, runaway bound, no stack-trace/secret leakage | Single static key (not JWT); rate limiter is per-process in-memory — deliberately scoped to single-instance demo |

### Bug-fix war stories (before → after)
- **Temperature 400s on newer families.** Before: `temperature` passed unconditionally, so opus-4-7/4-8, fable, mythos rejected every call with HTTP 400. After: `_accepts_temperature()` gates the kwarg by model-family marker (`config.py:130-136, 175-176`).
- **Over-broad retry predicate.** Before: tenacity retried *any* exception 4×, so a 400/401/403/404 cost ~60s of backoff before failing anyway. After: `_RETRYABLE_ANTHROPIC_ERRORS` restricts retries to connection/timeout/429/5xx (`config.py:189-201`).
- **Dead Tavily retry.** Before: tenacity wrapped a function that swallowed its own exceptions, so the decorator never saw a failure and never retried. After: the inner `_search_with_retry` deliberately *raises* (timeouts included) so `stop_after_attempt(3)` fires; the public `tavily_search` converts exhausted retries into `[]` (`tools/search.py:40-101`).
- **Duplicate cache points from `uuid4`.** Before: each upsert minted a fresh `uuid4` ID, so re-research duplicated points forever. After: deterministic `uuid5(NAMESPACE_URL, "{question}|{source_url}")` overwrites in place (`tools/vector_store.py:180-184`).
- **Silent retry loop.** Before: a failed critique looped back and re-ran identical queries against a cache that returned the same evidence. After: retries bypass the cache, reformulate from feedback, and zero-result sub-questions become `unanswered_questions` (`agents/researcher.py:220-283`).

---

## 6. Live Demo Script

**Lead with the metrics and the architecture, not the prose. The report content is mock-generated; the *pipeline behavior* is the product.**

### Start it in mock mode (no keys, no network)

PowerShell (Windows):
```powershell
$env:MOCK_MODE = "true"; $env:QDRANT_PATH = ":memory:"
.\.codex_venv\Scripts\python.exe -m uvicorn main:app --port 8000
```

bash:
```bash
MOCK_MODE=true QDRANT_PATH=':memory:' python -m uvicorn main:app --port 8000
```

One-command Docker alternative (the published image — boots keyless in mock mode). Single line, so it runs as-is in PowerShell, cmd, or bash:
```
docker run -p 8000:8000 -e MOCK_MODE=true -e QDRANT_PATH=":memory:" ghcr.io/utkarshalpha/multi-agent-research-pipeline:latest
```

### Drive the demo console
Open **http://127.0.0.1:8000/**. Two buttons: **Run Research** and **Load Sample** (`static/index.html:47-48`).

1. **Open with the architecture, not the output.** "This is four agents — Planner, Researcher, Critic, Writer — wired as a LangGraph state machine with a self-healing retry loop. The model is a *swappable backend* behind one factory; right now it's running fully offline in mock mode, which is why `/health` says `mock (claude-sonnet-4-6)`."
2. **Click *Load Sample* first.** It hits `GET /sample-response` and renders the saved RAG-vs-fine-tuning report instantly with no token spend — "this is the shape of a real report: structured sections, inline `[n]` citations, a sources list." (`main.py:156-163`, `examples/sample_response.json`)
3. **Click *Run Research* on a normal query** (e.g. *"tradeoffs between RAG and fine-tuning"*). Point at the **metadata**: `model`, `latency_seconds`, `retry_count`, `num_sub_questions`, `num_sources`, `num_citations`, `unanswered_questions`. "Lead with the pipeline metrics — this is the orchestration doing its job." (`main.py:250-259`)
4. **Run a `force-retry` query** — include the literal string `force-retry` (e.g. *"force-retry: state of vector databases"*). The mock Critic scores ~0.1, forcing the conditional edge to loop back to the Researcher; watch `retry_count` climb to its cap. "This is the self-healing loop — on retry it bypasses the cache and reformulates the query from feedback, bounded by `MAX_RETRIES = 2`." (`tools/mocks.py:37, 375`, `graph/edges.py:11-37`)
5. **Open `/docs`.** Show the OpenAPI schema — the typed `ResearchRequest`/`ResearchResponse` contracts and the auth/rate-limit dependencies. "Structured outputs aren't just internal; the HTTP contract is the same Pydantic source of truth."

> Honesty line to keep ready: "These latency/cost numbers are mock-mode — token usage reads 0 because the mock doesn't fabricate usage. Going live is a key-swap; the orchestration you're watching is unchanged."

---

## 7. Tough Questions & Strong Answers

**Q: Did you actually run it live?**
No — and I'm explicit about that. It's implemented and **verified in `MOCK_MODE`**, with 61 tests green on every push, but never run against real Anthropic/Tavily/arXiv because that needs keys. The whole point of `MOCK_MODE` is that the *real* graph, agents, critic loop, cache, and API run unchanged offline. `MODEL_LABEL` tags mock runs (e.g. `mock (claude-sonnet-4-6)`, `config.py:59`) so I can never confuse the two. Going live is `MOCK_MODE=false` + three keys + a validation pass.

**Q: So it's just a Claude wrapper?**
The opposite. The model is one swappable line behind `get_llm()` (`config.py:139-177`). The project is everything around it: a cyclic state machine with a feedback-driven retry loop, a three-dimension critic where only relevance is the LLM, a semantic cache, structured Pydantic I/O at every hop, tenacity scoped to transient errors, graceful degradation of every dependency, and groundedness-aware evals. Swap Claude for any chat model and the orchestration is identical.

**Q: Why LangGraph and not a simple `while` loop?**
The core requirement is a *conditional cycle* — Critic loops back to Researcher when evidence is weak, bounded by a budget. A linear `a→b→c` chain can't express that. LangGraph gives me cycles, conditional routing via `add_conditional_edges` (`graph/graph.py:41-45`), a typed shared state, and a checkpointer that persists each run under its `thread_id` for inspection/resumability. It also keeps each agent independently testable.

**Q: How does the retry loop avoid infinite loops / repeating the same search?**
Two mechanisms. (1) **Budget gate**: `should_retry` only loops back while `retry_count < MAX_RETRIES` (= 2); after that it always proceeds to the Writer (`graph/edges.py:30-37`). (2) **Behavior change**: a retry isn't a re-run — the Researcher re-researches only the low-confidence + previously-unanswered sub-questions, *bypasses the cache* (`use_cache=False`, `agents/researcher.py:243`), and reformulates each query from the Critic's feedback (`_reformulate_query`, distilled keywords rotated by attempt, `agents/researcher.py:59-98`). So each retry sends a genuinely different query.

**Q: How do you know the citations aren't hallucinated?**
Two layers. The Writer's prompt forbids inventing sources/numbers and back-fills `all_citations` from the actually-provided source URLs if the model under-populates them (`agents/writer.py:137-138`). Then the eval harness measures **groundedness**: the fraction of cited URLs that actually appear among the gathered source URLs; any cited URL not gathered is flagged `ungrounded` and drags the score down (`evals/run_evals.py:131-135`). That directly catches fabricated citations.

**Q: Why build a mock mode instead of just using a key?**
Because it made the system *provably correct and reproducible* before any key existed, and it forced clean swappable seams (LLM, search, embeddings each behind a narrow factory). The payoff is permanent: the full pipeline is unit-testable and runs in CI keyless — including a real `POST /research` smoke test against the built container. Most LLM projects can't run their own integration path in CI. Determinism (SHA-256 of inputs, no randomness) means assertions are stable.

**Q: How would this scale / go to production?**
The seams are already there. Swap the in-process `MemorySaver` checkpointer for a Redis/Postgres saver for multi-replica state; move the in-memory rate limiter to Redis (already a dependency); point `QDRANT_URL` at a managed Qdrant server instead of embedded mode (one env var, `config.py:64-68`); set `PIPELINE_API_KEY` and `RATE_LIMIT_PER_MINUTE`. The Dockerfile/compose and GHCR image already exist. The hard part — the orchestration and contracts — doesn't change.

**Q: What's the hardest bug you fixed?**
The dead Tavily retry. Tenacity was wrapping a function that swallowed its own exceptions, so the decorator never observed a failure and *never actually retried* — it looked resilient but wasn't. The fix was subtle: make the inner `_search_with_retry` deliberately raise (including `asyncio.wait_for` timeouts) so `stop_after_attempt(3)` fires, while keeping the public `tavily_search` converting exhausted retries into `[]` for graceful degradation (`tools/search.py:40-101`). It taught me that "wrapped in retry" and "actually retries" are different claims.

**Q: Why structured output instead of parsing free text?**
Each agent binds a Pydantic model with `.with_structured_output(Schema)`, so the model returns a validated, typed object (via Anthropic tool/JSON-schema calling) instead of prose I'd regex. No format drift, field-level validation (e.g. exactly 3–5 sub-questions, `schemas/models.py:18-24`), and the same models flow straight into graph state and the HTTP response. The Critic even defends against shape drift — it pads/truncates `relevance_scores` to the result count and clamps each to [0,1] (`agents/critic.py:166-173`).

**Q: How is the cache invalidated?**
It isn't time-invalidated — there's no TTL on the Qdrant cache, which is a known tradeoff (evidence can age). What I do instead: idempotent upserts via deterministic `uuid5(question|url)` IDs so re-researching a source overwrites rather than duplicates (`tools/vector_store.py:180-184`), and the retry path explicitly *bypasses* the cache (`use_cache=False`) so weak evidence is never re-served. For production I'd add a TTL or a recency-weighted eviction.

**Q: Why deterministic keyword reformulation instead of an LLM rewrite on retry?**
Cost, robustness, and testability. An LLM rewrite is an extra call that can itself fail or drift; keyword augmentation distilled from feedback (`_refinement_terms`, ≤4 stopword-filtered keywords, rotated by attempt) is free, deterministic, and works *identically in `MOCK_MODE`* so the retry loop is testable end-to-end (`agents/researcher.py:59-98`). The tradeoff is it's less creative than an LLM rewrite — an easy upgrade if it ever underperforms live.

**Q: Why 384-dim local embeddings when the spec said 1536?**
Deliberate trade-off. Local fastembed `BAAI/bge-small-en-v1.5` (384-dim) needs no API key and runs on CPU, which keeps the whole project runnable keyless (`config.py:74-82`). `VECTOR_SIZE` is configurable — swap in OpenAI's 1536-dim embedder by changing two env vars together. The cache mechanics (cosine, 0.85 threshold) are identical.

**Q: Why is only relevance LLM-judged, not credibility and recency?**
Because only relevance is genuinely subjective — "does this text answer the sub-question?" needs a model. Credibility (arXiv=1.0, ~30-domain allowlist + `.gov`/`.edu`=0.8, else 0.5) and recency (≤1yr=1.0, ≤3yr=0.7, older=0.4) are cheaper, explainable, reproducible pure functions (`agents/critic.py:65-93`). Isolating the LLM to one of three dimensions makes the score auditable and the cost lower.

**Q: What would you build next?**
In order: (1) live validation — real keys, a full 20-question eval run, real `results.json` numbers; (2) answer-correctness grading (gold set or LLM judge) on top of the existing structure/groundedness scoring; (3) per-agent model routing (cheap planner, strong writer) — trivial given the `get_llm()` factory; (4) a recorded walkthrough and a hosted deploy.

---

## 8. Concept Study Guide

Self-study map — concept → where it lives → probing Q&A.

**1. RAG vs Fine-Tuning** (the demo's own subject). RAG injects fresh, citable evidence at inference; fine-tuning bakes in *behavior/format/tone* but is poor for fast-changing facts. This *is* a RAG system: retrieves (Tavily + arXiv + semantic cache), grounds in `[n]` citations, never fine-tunes. *Nuance to volunteer:* it's **tool-augmented retrieval** (live web/arXiv), and the embedding-retrieval piece is the **semantic cache** (`tools/vector_store.py`). *Where:* `examples/sample_response.json`, groundedness in `evals/run_evals.py:131-135`. *Q: When fine-tune instead?* When you need consistent behavior/format (always emit the `FinalReport` shape, a brand voice), not fresh facts.

**2. Multi-agent orchestration & graph-vs-chain.** Four specialists communicate only through typed state, each independently testable. A linear chain can't express the Critic→Researcher cycle with a budget. *Where:* the four `*_node` functions; `graph/graph.py:29-48`; `graph/edges.py`. *Q: How do agents share data?* Through `AgentState`; each node returns a partial update LangGraph merges — no direct calls.

**3. LangGraph: StateGraph, conditional edges, checkpointer.** `StateGraph(AgentState)` defines nodes over typed state; `add_conditional_edges` routes by a function's return; `MemorySaver` persists per `thread_id` (= `run_id`). *Where:* `graph/graph.py:31, 41-45, 58`; `thread_id` set at `main.py:199`. *Q: Why no reducers?* Every field is fully owned/replaced by one node, so last-write-wins is correct.

**4. Structured output (`with_structured_output` + Pydantic).** Validated typed objects instead of regex; schema enforced by the model + Pydantic; same models flow into state and the API. *Where:* `schemas/models.py`; `agents/planner.py:36`, `agents/critic.py:108`, `agents/writer.py:121`; invoked through `config.ainvoke_with_retry`. *Q: Wrong shape?* Critic pads/truncates/clamps (`agents/critic.py:166-173`).

**5. Semantic caching (embeddings, cosine, vector DB).** Embed each sub-question, query Qdrant above a cosine threshold; hit skips the network, miss fetches + writes back. 384-dim fastembed `bge-small-en-v1.5`, cosine, threshold 0.85. *Where:* `config.py:78-82`; `tools/vector_store.py:76-196`; cache use at `agents/researcher.py:154-196`. *Q: Why bypass on retry?* The retry exists because evidence was weak — must fetch genuinely fresh results.

**6. Async Python + FastAPI.** `asyncio.gather` for concurrent awaitables (Tavily + arXiv); `asyncio.to_thread` to move blocking libs (fastembed ONNX, sync `arxiv`) off the loop; `lifespan` for paired startup/teardown. *Where:* `agents/researcher.py:173`; `tools/vector_store.py:99`, `tools/arxiv.py:69`; `main.py:110-123`. *Q: Why `return_exceptions=True`?* One tool failing shouldn't kill the other — it degrades to `[]`.

**7. Resilience (tenacity, graceful degradation, timeouts).** Retry only transient errors with backoff; everything optional degrades rather than crashes; a hard wall-clock timeout bounds latency. *Where:* `config.py:189-201` (LLM retry), `tools/search.py:40-101` (Tavily), `memory/redis_store.py` + `tools/vector_store.py` (no-op/cache-miss), `main.py:215-225` (504). *Q: SDK already retries — why tenacity?* App-level control scoped to transient errors so 4xx fail fast instead of burning ~60s.

**8. Testing without external deps.** `MOCK_MODE` swaps LLM/Tavily/arXiv/embedder for deterministic hash-derived stand-ins; Qdrant `:memory:`; Redis degrades; FastAPI `TestClient` exercises real routes in-process. *Where:* `tools/mocks.py`; tests set env before import (`tests/test_research_endpoint.py:16-17`); `force-retry` drives the loop. *Q: Why env before imports?* `config.py` reads env once at import time and `load_dotenv()` won't override existing vars.

**9. Evals & groundedness.** 20 questions (4 domains × 5) → `quality_score` from 6 components incl. groundedness (cited URLs ⊆ gathered URLs) and coverage; pass at 0.7; results tagged `mock: true/false`. *Where:* `evals/run_evals.py:64-171`; `evals/eval_set.json`. *Q: Catch hallucinated citations?* Set-intersect cited URLs against gathered ones; any miss is ungrounded.

**10. Docker + CI/CD + GHCR.** Slim Python 3.12 image, non-root user, stdlib `/health` HEALTHCHECK; compose brings up app + Redis + Qdrant; CI runs the offline suite, then on `main` builds, smoke-tests a real `POST /research` keyless, and pushes `:${sha}` + `:latest` to GHCR. *Where:* `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`. *Q: What does the smoke test prove?* The *published* image boots and serves a real end-to-end run, not just that unit tests pass.

---

## 9. Numbers & Facts to Memorize

All verified against the code on disk.

| Fact | Value | Source |
|------|-------|--------|
| Agents | 4 (Planner → Researcher → Critic → Writer) | `graph/graph.py:33-36` |
| Sub-questions per query | 3–5 | `schemas/models.py:18-24` |
| `MAX_RETRIES` | 2 | `config.py:88` |
| `CONFIDENCE_PASS_THRESHOLD` | 0.7 | `config.py:87` |
| Critic weights | 0.5 relevance / 0.25 credibility / 0.25 recency | `agents/critic.py:33-35` |
| Credibility tiers | arXiv 1.0 · known/`.gov`/`.edu` 0.8 · else 0.5 | `agents/critic.py:65-74` |
| Recency tiers | ≤1yr 1.0 · ≤3yr 0.7 · older 0.4 · unknown 0.6 | `agents/critic.py:77-93` |
| `CACHE_SIMILARITY_THRESHOLD` (cosine) | 0.85 | `config.py:82` |
| Embedding model / dims | `BAAI/bge-small-en-v1.5` / 384 | `config.py:78-79` |
| Default model / configurable | `claude-sonnet-4-6` / `claude-opus-4-8` | `config.py:52` |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | 0.0 / 4096 (Writer bumps to ≥8000) | `config.py:54-55`, `agents/writer.py:121` |
| LLM retry | 4 attempts, exp backoff min=2 max=30, transient errors only | `config.py:196-201` |
| Tavily retry / timeout | 3 attempts / 10s | `tools/search.py:40-62`, `config.py:90` |
| `RESEARCH_TIMEOUT_SECONDS` → 504 | 300 | `config.py:97`, `main.py:222-225` |
| Per-question result cap | 2 web + 2 arXiv | `agents/researcher.py:34-38` |
| Tests | **61** (green, offline) | `tests/` |
| Eval set | 20 questions (4 domains × 5), `quality_score` pass ≥ 0.7 | `evals/eval_set.json`, `evals/run_evals.py:64-68` |
| Eval quality components | 6 (report present, ≥200 chars, ≥2 sections, citations present, groundedness, coverage) | `evals/run_evals.py:143-152` |
| Pricing table (per Mtok) | opus-4-8 = (5, 25), sonnet-4-6 = (3, 15) | `evals/run_evals.py:71-77` |
| Checkpointer | `MemorySaver`, graph cached `@lru_cache(maxsize=1)` | `graph/graph.py:51-58` |
| Redis run TTL | 24h | `memory/redis_store.py:24` |
| GHCR image | `ghcr.io/utkarshalpha/multi-agent-research-pipeline:latest` | `.github/workflows/ci.yml:48`, `git remote` |
| CI flow | offline tests → build → keyless smoke `POST /research` → push `:sha` + `:latest` | `.github/workflows/ci.yml` |

---

## 10. Known Limitations (own them)

Each gets honest framing and a concrete next step.

1. **Not yet validated live.** Status is implemented + verified in `MOCK_MODE`, not against real Anthropic/Tavily/arXiv. *Frame:* the orchestration is provably correct offline; the LLM is one swappable factory line (`config.py:139-177`); `MODEL_LABEL` tags mock runs so they're never mistaken for live. *Next:* run with keys + a full 20-question eval, record real `results.json`.

2. **Mock token counts are 0.** `MockChatModel.ainvoke` returns an `AIMessage` with no `usage_metadata` (`tools/mocks.py`), so token/cost aggregation reads 0 in mock (`main.py:243-248`). *Frame:* that's honesty, not a gap — cost is reported as 0, never fabricated; `results.json` records `mock: true`; the real `_PRICING` table is already wired (`evals/run_evals.py:71-77`). *Next:* it populates automatically on the live run.

3. **Embedded Qdrant is single-process.** With `QDRANT_PATH` set, Qdrant runs in-process behind a file lock — one writer at a time (`config.py:64-68`). *Frame:* the same code path supports a networked server via `QDRANT_URL` (one env var), and every cache call degrades to a miss rather than crashing (`tools/vector_store.py:142-153`). *Next:* point at a managed/containerized Qdrant for prod.

4. **Rate limiter is in-memory, single-instance.** `_rate_limit_windows` is a per-process dict (`main.py:48`), not shared across replicas. *Frame:* explicitly scoped to the single-instance demo, with a memory-leak guard evicting stale IPs (`_prune_rate_limit_windows`, `main.py:67-82`). *Next:* move the window to Redis (already a project dependency).

5. **Critic credibility is heuristic.** A static ~30-domain allowlist + `.gov`/`.edu` + arXiv=1.0 (`agents/critic.py:38-74`), not a live reputation signal. *Frame:* deterministic, explainable, free, and isolated to one of three dimensions — only relevance uses the LLM. *Next:* swap a reputation API into `_credibility` (a localized change).

6. **No answer-correctness grading.** Evals score structure, groundedness, and coverage — not whether claims are factually true (`evals/run_evals.py:96-171`). *Frame:* groundedness already catches the highest-risk failure (hallucinated citations) with no gold answers needed. *Next:* add a labeled gold set or an LLM judge — a clearly-scoped step, not a hidden gap.

7. **In-process checkpointer / single model tier.** `MemorySaver` is in-memory and all agents share one model tier. *Frame:* both are deliberate single-instance-demo choices; the `get_llm()` factory and the checkpointer interface make both trivial to swap. *Next:* Redis/Postgres saver + per-agent model routing (cheap planner, strong writer).
