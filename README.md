# MedReason Graph

MedReason Graph is a research prototype for auditable medical reasoning. It turns retrieved medical references into structured evidence claims, builds an evidence graph, ranks a differential diagnosis, and verifies that every visible reasoning step cites graph-backed evidence.

It is not an autonomous doctor and does not provide clinical advice. It is a technical project for evidence-grounded diagnosis support and medical reasoning experiments.

## What It Builds

```text
Patient case
  -> normalized findings
  -> hybrid retrieval
  -> evidence claims
  -> evidence graph
  -> ranked differential
  -> verifier report
```

The implementation is dependency-light for v1: pure Python, deterministic BM25-like retrieval, lexical semantic matching, source-aware scoring, graph export, CLI, tests, and an optional FastAPI adapter.

## Quick Start

```bash
cd /Users/akoroth/Desktop/projects/medreason_graph
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m unittest discover -s tests -v
```

For PDF corpus ingestion, install the optional corpus dependency:

```bash
pip install -e ".[corpus]"
```

For FAISS vector retrieval, install the optional vector dependency:

```bash
pip install -e ".[vector]"
```

For medical transformer encoders such as MedCPT, SapBERT, or BioClinicalBERT:

```bash
pip install -e ".[medical-encoders]"
```

For the OpenAI evidence-claim extractor adapter:

```bash
pip install -e ".[llm-openai]"
```

Or use the task runner:

```bash
make test
make demo
```

Analyze the sample chest pain case:

```bash
medreason-graph ingest examples/corpus --out /tmp/medreason_corpus.json --source-type guideline
medreason-graph analyze --corpus /tmp/medreason_corpus.json --case examples/cases/chest_pain.json --json
```

Text output:

```bash
medreason-graph analyze --corpus /tmp/medreason_corpus.json --case examples/cases/chest_pain.json
```

## CLI

```text
medreason-graph ingest PATH --out corpus.json [--source-type guideline]
medreason-graph analyze --corpus corpus.json --case case.json [--json] [--evidence-extractor deterministic|llm] [--llm-command CMD] [--config config/default_clinical_config.json] [--verbose]
medreason-graph graph --corpus corpus.json --case case.json --format json|cytoscape|dot --out graph.json [--evidence-extractor deterministic|llm] [--llm-command CMD]
medreason-graph download-sources --manifest sources/open_medical_sources.json --out-dir data/open_sources
medreason-graph build-open-corpus --downloaded-manifest data/open_sources/downloaded_manifest.json --out data/open_corpus/open_medical_corpus.json
medreason-graph index-corpus --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.sqlite
medreason-graph index-faiss --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.faiss
medreason-graph index-faiss --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus_medcpt.faiss --embedding-preset medcpt
PYTHONPATH=src python3 scripts/generate_real_outputs.py --corpus data/open_corpus/open_medical_corpus.json --retriever sqlite --index data/open_corpus/open_medical_corpus.sqlite
medreason-graph evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --cases evaluation/retrieval_cases.json --k 5
medreason-graph evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --retriever sqlite --index data/open_corpus/open_medical_corpus.sqlite
medreason-graph evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --retriever faiss --index data/open_corpus/open_medical_corpus.faiss
medreason-graph evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --retriever faiss --index data/open_corpus/open_medical_corpus_medcpt.faiss
medreason-graph graph-store-build --analysis analysis.json --out data/graphs/case.sqlite
medreason-graph graph-query evidence-for --graph data/graphs/case.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query evidence-against --graph data/graphs/case.sqlite --condition "gastroesophageal reflux disease"
medreason-graph graph-query missing-tests --graph data/graphs/case.sqlite --condition "pulmonary embolism"
medreason-graph graph-query source-spans --graph data/graphs/case.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query explain-rank --graph data/graphs/case.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query verifier-failures --graph data/graphs/case.sqlite
```

`ingest` supports Markdown, TXT, JSON, HTML, DOCX, and PDF. PDF extraction requires the optional `corpus` dependency.

## Open Corpus

The seed open corpus manifest is [sources/open_medical_sources.json](sources/open_medical_sources.json). The downloader only accepts HTTPS URLs from allowlisted official domains and records license/provider/checksum metadata.

Build the seed corpus:

```bash
make download-open-corpus
make build-open-corpus
```

Downloaded raw files go under `data/open_sources/raw/`. The ingested corpus is written to `data/open_corpus/open_medical_corpus.json`.

Generate auditable reports from the real downloaded corpus:

```bash
make real-output-demo PYTHON=.venv/bin/python
```

By default this writes reports, analysis JSON, graph stores, and verifier-failure files to `/tmp/medreason_real_outputs` for:

- [examples/cases/chest_pain_real_demo.json](examples/cases/chest_pain_real_demo.json)
- [examples/cases/dyspnea_real_demo.json](examples/cases/dyspnea_real_demo.json)

Evaluate retrieval on the seed cases:

```bash
make evaluate-open-corpus
make index-open-corpus
make evaluate-open-corpus-sqlite
make index-open-corpus-faiss
make evaluate-open-corpus-faiss
make index-open-corpus-medcpt
make evaluate-open-corpus-medcpt
```

The default retriever is in-memory and dependency-free. SQLite FTS5 can be used as a persistent BM25-style backend:

```bash
medreason-graph index-corpus --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.sqlite
medreason-graph analyze --corpus data/open_corpus/open_medical_corpus.json --case examples/cases/chest_pain.json --retriever sqlite --index data/open_corpus/open_medical_corpus.sqlite
```

FAISS vector retrieval uses a deterministic local hashing embedding in v1. This gives a real vector index path without downloading a model:

```bash
medreason-graph index-faiss --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.faiss
medreason-graph analyze --corpus data/open_corpus/open_medical_corpus.json --case examples/cases/chest_pain.json --retriever faiss --index data/open_corpus/open_medical_corpus.faiss
```

The embedding implementation is intentionally swappable; a clinical embedding model can replace it later without changing the retrieval backend interface.

Recommended medical encoder preset:

```bash
medreason-graph index-faiss \
  --corpus data/open_corpus/open_medical_corpus.json \
  --out data/open_corpus/open_medical_corpus_medcpt.faiss \
  --embedding-preset medcpt \
  --batch-size 8 \
  --document-max-length 256 \
  --query-max-length 64
```

Available presets:

- `medcpt`: query/document encoder pair for biomedical retrieval.
- `sapbert`: biomedical entity representation encoder.
- `bioclinicalbert`: clinical-note language encoder.

`--verbose` writes structured JSON logs to stderr for ingestion, retrieval, evidence extraction, and analysis events.

## Evidence Claims

Retrieved passages are converted into structured `EvidenceClaim` records before they can affect ranking or reasoning. Each accepted claim carries:

- a strict schema version and claim type;
- source span start/end offsets into the original chunk text;
- a source-text hash;
- extraction confidence;
- extraction method.

The deterministic extractor currently handles `supports`, `argues_against`, `requires_test`, `rules_out`, and `red_flag` claim types. Spanless or schema-invalid claims are rejected before graph construction. For textbook-style chapters, condition context can come from the source title or section path, while findings and tests still have to appear in the cited source span.

For richer extraction, use the LLM command adapter:

```bash
medreason-graph analyze \
  --corpus data/open_corpus/open_medical_corpus.json \
  --case examples/cases/chest_pain.json \
  --evidence-extractor llm \
  --llm-command ".venv/bin/python scripts/openai_claim_extractor.py" \
  --llm-fallback-to-deterministic
```

The command receives a JSON payload on stdin with the patient case, retrieved source chunk, and evidence-claim schema. It must write JSON to stdout:

```json
{
  "claims": [
    {
      "claim_type": "supports",
      "condition": "acute coronary syndrome",
      "finding": "chest pain",
      "polarity": "supports",
      "strength": "moderate",
      "exact_quote": "Myocardial ischemia can present as chest pain.",
      "extraction_confidence": 0.82
    }
  ]
}
```

The validator accepts only claims whose quote or offsets match the retrieved source text exactly.

The included [scripts/openai_claim_extractor.py](scripts/openai_claim_extractor.py) adapter reads `OPENAI_API_KEY` from `.env` or the process environment. You can override the model with:

```bash
OPENAI_MODEL=gpt-4.1-mini
```

## Graph Store

Persist an analysis result into a queryable SQLite evidence graph:

```bash
medreason-graph analyze \
  --corpus /tmp/medreason_corpus.json \
  --case examples/cases/chest_pain.json \
  --json > /tmp/chest_pain_analysis.json

medreason-graph graph-store-build \
  --analysis /tmp/chest_pain_analysis.json \
  --out /tmp/chest_pain_graph.sqlite
```

Query the persisted graph without re-running retrieval or the LLM:

```bash
medreason-graph graph-query evidence-for --graph /tmp/chest_pain_graph.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query reasoning --graph /tmp/chest_pain_graph.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query explain-rank --graph /tmp/chest_pain_graph.sqlite --condition "acute coronary syndrome"
medreason-graph graph-query verifier-failures --graph /tmp/chest_pain_graph.sqlite
```

The analyzer runs claim-level verification before ranking. Claims that fail source-span entailment checks are kept in `claim_verifications` for audit, but are not allowed to support differential ranking or reasoning steps.

## Configuration

Clinical dictionaries and weights live in [config/default_clinical_config.json](config/default_clinical_config.json).

Config controls:

- concepts, synonyms, and semantic type;
- source quality weights;
- evidence strength weights;
- high-risk condition urgency;
- dangerous alternatives;
- ambiguous abbreviation candidates and context cues;
- section classification keywords.

Pass a custom config with:

```bash
medreason-graph analyze --corpus corpus.json --case case.json --config custom_config.json
```

Terminology normalization reports whether a term matched by canonical form, synonym, abbreviation, contextual abbreviation, phrase, or not at all. Ambiguous abbreviations such as `PE` are not silently normalized without enough context.

## Graph Exports

Export the evidence graph for inspection:

```bash
make graph-dot
make graph-cytoscape
```

## Patient Case Input

```json
{
  "case_id": "case_001",
  "patient": {"age": 45, "sex": "male"},
  "chief_complaint": "crushing chest pain",
  "findings": [
    {"type": "symptom", "name": "left arm radiation", "status": "present"},
    {"type": "symptom", "name": "diaphoresis", "status": "present"},
    {"type": "test", "name": "ECG", "status": "missing"}
  ],
  "free_text": "45M with crushing chest pain, sweating, nausea, and left arm radiation."
}
```

## Output Guarantees

- Every displayed reasoning step must cite at least one evidence claim.
- Evidence claims retain source type, document title, section path, and paragraph location.
- The verifier reports unsupported claims, citation polarity mismatches, missing patient facts, and dangerous alternatives checked.
- High-risk complaints such as chest pain surface dangerous-miss checks.

## API

An optional FastAPI app is available:

```bash
pip install -e ".[api]"
uvicorn medreason_graph.api:create_app --factory --reload
```

Endpoints:

```text
POST /sources/ingest
POST /cases/analyze
GET  /cases/{case_id}/graph
GET  /cases/{case_id}/evidence
GET  /cases/{case_id}/audit
```
