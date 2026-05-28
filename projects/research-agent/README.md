# Research Agent

## Overview

Research Agent is an autonomous research system powered by LangGraph that takes a topic, plans a research strategy, executes web searches, critiques its own findings, and synthesizes a structured report. Built for analysts and engineers who need rapid literature reviews and technical surveys without manually visiting dozens of sources. The system uses a Reflexion loop to self-correct when initial findings are insufficient.

## Architecture

```
[Topic input]
       |
       v
[Planner Node] ---> research questions + search strategy
       |
       v
[Executor Node] ---> [Tavily Search] + [BeautifulSoup Scraper]
       |                        |
       |                        v
       |                  [Raw sources]
       v
[Critic Node] ---> quality score + gap identification
       |
       |  (score < threshold)
       v
[Reflexion Node] ---> revised search queries ---> back to Executor
       |
       |  (score >= threshold)
       v
[Synthesizer Node] ---> structured report (Pydantic schema)
       |
       v
[Final report output]
```

## Tech stack

| Layer | Tool | Why this tool |
|-------|------|---------------|
| Orchestration | LangGraph | Explicit state machine with checkpointing and conditional routing |
| LLM | OpenRouter (OpenAI-compatible) | Strong reasoning for planning, critique, and synthesis |
| Web search | Tavily API | Purpose-built search API for LLM agents with structured results |
| HTML parsing | BeautifulSoup4 | Extracts clean text from scraped web pages |
| Schema validation | Pydantic v2 | Structured report output with field validation |
| Tracing | LangSmith | Full trace of every node execution for debugging |

## Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Planner node generating research questions from topic | 🔲 Pending |
| M2 | Executor node with Tavily search and BeautifulSoup scraping | 🔲 Pending |
| M3 | Critic node scoring source quality and identifying gaps | 🔲 Pending |
| M4 | Reflexion loop with revised queries on low-quality results | 🔲 Pending |
| M5 | Synthesizer node producing Pydantic-validated structured report | 🔲 Pending |
| M6 | LangSmith tracing and RAGAS evaluation on report faithfulness | 🔲 Pending |

## How to run

```powershell
cd projects\research-agent
copy ..\..\.env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set TAVILY_API_KEY and OPENROUTER_API_KEY in .env
python main.py --topic "Compare vector databases for RAG in 2026"
```

## Evaluation targets

| Metric | Target |
|--------|--------|
| Report faithfulness (RAGAS) | ≥ 0.85 |
| Source coverage (topics addressed / planned) | ≥ 90% |
| Reflexion improvement (critic score delta) | ≥ +0.15 |
| End-to-end completion time | ≤ 5 minutes |

## Known limitations

- Web search limited to Tavily's indexed sources; no direct URL input or PDF retrieval in this POC.
- Maximum 3 Reflexion iterations to control API cost; complex topics may be under-researched.
- Report output is Markdown only; no PDF or slide generation.
