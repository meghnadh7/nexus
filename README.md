# Nexus — Autonomous DevOps Intelligence Platform

> A production-grade, multi-agent AI system that autonomously monitors GitHub, Slack, Jira, and Datadog — reviewing PRs, triaging incidents, answering developer questions via RAG, and triggering deployments.

---

## What Nexus Does

| Trigger | What Nexus Does |
|---------|----------------|
| PR opened on GitHub | Fetches diff → RAG context lookup → GPT-4o review → posts comment |
| Datadog alert fires | Fetches alert → searches runbooks → GPT-4o RCA → posts to Slack + creates Jira ticket |
| `@Nexus` mention in Slack | HyDE RAG search → synthesizes answer → replies in thread |
| Deploy request via API | Triggers GitHub Actions → polls status → posts result to Slack |

---

## Architecture

```
GitHub ──────────────────────────────────────────────┐
Slack  ──► Webhook Endpoints ──► LangGraph Orchestrator ──► MCP Servers ──► GitHub API
Datadog ─────────────────────────│                                      ──► Slack API
                                  │                                      ──► Jira API
                              Postgres                                   ──► Datadog API
                           (Checkpointing)
                                  │
                              Pinecone
                            (RAG / HyDE)
```

**Key design decisions:**
- **LangGraph StateGraph** — all agent logic as a directed graph with conditional routing
- **MCP (Model Context Protocol)** — each integration runs as an isolated FastAPI microservice
- **HyDE RAG** — generates a hypothetical answer first, embeds that, then queries Pinecone for higher recall
- **AsyncPostgresSaver** — persists graph state for human-in-the-loop approval flows
- **HMAC webhook verification** — GitHub and Slack signatures validated on every request

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `orchestrator` | 8000 | LangGraph multi-agent brain — FastAPI + webhook handlers |
| `mcp-github` | 8001 | GitHub MCP server — PR diffs, comments, workflow triggers |
| `mcp-slack` | 8002 | Slack MCP server — messages, threads, channel history |
| `mcp-jira` | 8003 | Jira MCP server — ticket creation, transitions, comments |
| `mcp-datadog` | 8004 | Datadog MCP server — alerts, metrics, monitor muting |
| `postgres` | 5432 | LangGraph checkpoint store |
| `redis` | 6379 | Rate limiting and caching |
| `rag-indexer` | — | One-shot job — chunks, embeds, upserts codebase to Pinecone |

---

## Tech Stack

- **LLM:** GPT-4o via LangChain / LangGraph 0.2.x
- **RAG:** Pinecone (serverless) + OpenAI `text-embedding-3-large` + HyDE
- **Orchestration:** LangGraph `StateGraph` with `AsyncPostgresSaver` checkpointing
- **APIs:** FastAPI + uvicorn (async throughout)
- **Observability:** structlog (JSON) + Prometheus metrics + LangSmith tracing
- **Infrastructure:** Docker Compose (dev) + Helm chart (Kubernetes/prod)
- **Retries:** tenacity (exponential backoff on all external calls)
- **CI/CD:** GitHub Actions (CI + CD + RAG re-index + evals)

---

## Quick Start

### Prerequisites
- Docker Desktop
- OpenAI API key
- GitHub personal access token (`repo` + `workflow` scopes)

### 1. Clone and configure

```bash
git clone https://github.com/meghnadh7/nexus.git
cd nexus
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Start all services

```bash
docker compose up --build
```

### 3. Verify everything is healthy

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "nexus-orchestrator"}
```

### 4. Expose to the internet (for GitHub webhooks)

```bash
ngrok http 8000
```

### 5. Add GitHub webhook

- Go to your repo → Settings → Webhooks → Add webhook
- Payload URL: `https://<your-ngrok-url>/webhook/github`
- Content type: `application/json`
- Secret: your `GITHUB_WEBHOOK_SECRET` from `.env`
- Events: Send me everything ✅

### 6. Index your codebase into Pinecone

```bash
docker compose --profile indexer run --rm --user root rag-indexer sh -c "
  apt-get update -qq && apt-get install -y git -qq &&
  git clone https://github.com/<your-org>/<your-repo> /tmp/repo &&
  python -m rag_indexer.main --repo-path /tmp/repo
"
```

Now open a PR — Nexus will automatically review it and post a comment. ✅

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Powers all LLM calls and embeddings |
| `GITHUB_TOKEN` | Read PRs, post comments, trigger workflows |
| `GITHUB_WEBHOOK_SECRET` | HMAC verification for incoming webhooks |
| `SLACK_BOT_TOKEN` | Post messages and read channel history |
| `SLACK_SIGNING_SECRET` | Verify incoming Slack webhooks |
| `JIRA_BASE_URL` | Your Atlassian site URL |
| `JIRA_EMAIL` | Jira account email |
| `JIRA_API_TOKEN` | Jira API token |
| `PINECONE_API_KEY` | RAG vector storage |
| `DATADOG_API_KEY` | Fetch alerts and metrics |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |

---

## Agent Graph

```
START
  │
  ▼
classify ──► pr_opened ──► rag ──► code_review ──► finalize
          ├► incident  ──► rag ──► incident    ──► finalize
          ├► slack_mention ──► rag ──► finalize
          └► deploy_request ──► human_approval ──► deploy ──► finalize
```

Human-in-the-loop approval is supported for deployments — set `HUMAN_IN_THE_LOOP=true` to require manual approval before any deploy.

---

## Project Structure

```
nexus/
├── services/
│   ├── orchestrator/          # LangGraph brain
│   │   ├── src/orchestrator/
│   │   │   ├── agents/        # code_review, incident, rag, deploy
│   │   │   ├── graph/         # StateGraph, nodes, routing
│   │   │   ├── memory/        # Postgres checkpointer
│   │   │   └── tools/         # MCP client
│   │   └── tests/
│   ├── mcp-github/            # GitHub MCP server
│   ├── mcp-slack/             # Slack MCP server
│   ├── mcp-jira/              # Jira MCP server
│   ├── mcp-datadog/           # Datadog MCP server
│   └── rag-indexer/           # Codebase chunker + embedder
├── infra/helm/nexus/          # Kubernetes Helm chart
├── evals/                     # LangSmith eval harness
├── .github/workflows/         # CI, CD, evals, RAG re-index
├── docker-compose.yml
└── Makefile
```

---

## Makefile Commands

```bash
make dev          # Start all services
make test         # Run unit tests
make lint         # Run ruff + mypy
make build        # Build Docker images
make index        # Re-index codebase into Pinecone
make eval         # Run LangSmith evals
make logs         # Tail all service logs
```

---

## Kubernetes Deployment

A full Helm chart is included at `infra/helm/nexus/` with:
- HPA (autoscaling) for the orchestrator
- StatefulSets for Postgres and Redis
- CronJob for scheduled RAG re-indexing
- Ingress with TLS
- Secrets template

```bash
helm upgrade --install nexus ./infra/helm/nexus \
  -f infra/helm/nexus/values-prod.yaml \
  --set secrets.openaiApiKey=$OPENAI_API_KEY
```

---

## Evaluations

LangSmith eval harness included for code review and incident response quality:

```bash
make eval
```

Eval datasets in `evals/datasets/` cover: SQL injection detection, async/blocking code, hardcoded secrets, N+1 queries, memory incidents, disk alerts.

---

## License

MIT
