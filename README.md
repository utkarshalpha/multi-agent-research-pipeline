# Multi-Agent Research Pipeline

An AI research assistant backend that turns one research question into a
structured, cited Markdown report. It uses FastAPI for the API, LangGraph for
agent orchestration, Anthropic Claude for reasoning, Tavily and arXiv for
research, Qdrant for semantic caching, and optional Redis for run memory.

The portfolio demo console is served at:

```text
http://127.0.0.1:8000/
```

FastAPI's Swagger docs remain available at:

```text
http://127.0.0.1:8000/docs
```

![Demo console](docs/assets/demo-console.png)

## What It Does

The app runs a four-agent workflow:

| Agent | Responsibility |
| --- | --- |
| Planner | Breaks the user query into 3 to 5 focused sub-questions. |
| Researcher | Searches cache, Tavily, and arXiv for evidence. |
| Critic | Scores relevance, credibility, and recency. |
| Writer | Produces a cited Markdown report. |

If the Critic finds weak evidence, the graph loops back to the Researcher for a
limited retry pass before writing the final report.

## Portfolio Status

This is a strong AI/backend portfolio project. It is more advanced than a basic
chatbot because it demonstrates:

- Multi-agent orchestration with LangGraph.
- Tool use across web search, arXiv, vector search, and Redis memory.
- Structured Pydantic request and response contracts.
- Semantic caching with embedded or remote Qdrant.
- A self-healing critic loop.
- A FastAPI surface plus a browser demo console.
- A repeatable eval harness.

The project is now presentable as a portfolio demo. The remaining upgrades are
deployment, a recorded walkthrough, and real eval-result screenshots after the
API keys are configured.

## Architecture

Read the detailed architecture guide:

```text
docs/ARCHITECTURE.md
```

High-level flow:

```text
Query -> Planner -> Researcher -> Critic -> Writer -> Cited report
                   ^             |
                   |             v
              retry weak evidence
```

## Project Layout

```text
research_pipeline/
  main.py                    FastAPI app, API routes, demo static hosting
  config.py                  Environment settings, LLM factory, retry helper
  agents/                    Planner, Researcher, Critic, Writer nodes
  graph/                     LangGraph state, graph builder, retry edge
  tools/                     Tavily, arXiv, Qdrant vector cache helpers
  memory/                    Optional Redis run memory
  schemas/                   Pydantic models
  static/                    Portfolio demo console
  examples/sample_response.json
  docs/ARCHITECTURE.md
  evals/                     Eval set and runner
  tests/                     Route and demo contract tests
  docker-compose.yml         Optional Redis and Qdrant services
```

## Quick Start

PowerShell:

```powershell
cd C:\Users\Utkarsh\Downloads\MULTILANG\research_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set at least these keys in `.env` for live research:

```text
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

Start the app:

```powershell
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

## Local Infrastructure

The project can run without Docker if `.env` uses embedded Qdrant:

```text
QDRANT_PATH=./qdrant_local
```

Redis is optional. If Redis is not running, the app logs that short-term memory
is disabled and continues.

To run Redis and remote Qdrant with Docker:

```powershell
docker-compose up -d
```

Then set:

```text
QDRANT_PATH=
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

## API Usage

```powershell
curl -Method POST http://127.0.0.1:8000/research `
  -ContentType "application/json" `
  -Body '{"query":"What are the key differences between RAG and fine-tuning for LLM applications?"}'
```

The response shape is:

```json
{
  "run_id": "uuid",
  "report": "# Markdown report",
  "citations": ["https://source.example"],
  "metadata": {
    "model": "claude-sonnet-4-6",
    "latency_seconds": 18.4,
    "retry_count": 1,
    "num_sub_questions": 4,
    "num_sources": 9,
    "num_citations": 3,
    "token_usage": {
      "input_tokens": 41200,
      "output_tokens": 3100,
      "total_tokens": 44300
    }
  }
}
```

## Sample Demo Response

A saved sample response is available at:

```text
GET /sample-response
```

It is stored in:

```text
examples/sample_response.json
```

The browser console uses this sample so the portfolio UI can be shown without
spending tokens or depending on live API keys.

## Tests

Run the lightweight route tests:

```powershell
python -m unittest discover -s tests
```

Run the eval harness after API keys are configured:

```powershell
python -m evals.run_evals
```

The eval runner writes:

```text
evals/results.json
```

## Deployment Notes

For a portfolio deployment:

1. Deploy FastAPI with Uvicorn or Gunicorn plus Uvicorn workers.
2. Use managed Redis if you want persistent run memory.
3. Use embedded Qdrant only for local demos; use a managed or containerized
   Qdrant instance for production.
4. Store API keys as secrets, not in the repository.
5. Add a short demo video showing the console, `/docs`, and one successful run.
