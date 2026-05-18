# MedReason Graph Roadmap

This roadmap turns the current runnable prototype into a serious auditable medical reasoning search system. Each stage should end with tests, demo data, and a working CLI/API path before moving forward.

## Stage 0: Current Prototype Baseline

Status: complete.

- Section-aware ingestion for Markdown/TXT/JSON sources.
- Dependency-light hybrid lexical retrieval with medical synonym expansion.
- Evidence claim extraction from retrieved passages.
- Evidence graph with patient facts, conditions, claims, sources, and reasoning steps.
- Differential ranking, dangerous-miss checks, verifier, CLI, optional API adapter.
- Sample chest-pain corpus and unit tests.

Acceptance:

- `python3 -m unittest discover -s tests -v` passes.
- Sample chest-pain case returns cited reasoning and `Verifier: PASS`.

## Stage 1: Project Hardening

Status: complete.

Goal: make the prototype easier to run, inspect, and extend.

- Add a Makefile or task runner for `test`, `demo`, `lint`, and `api`.
- Add structured logging for ingestion, retrieval, extraction, ranking, and verification.
- Add golden JSON outputs for sample cases.
- Add graph export examples for Cytoscape/Graphviz-compatible viewers.
- Add config files for source-type weights, condition urgency, dangerous alternatives, and synonym mappings.
- Replace hardcoded clinical lexicon entries with loadable YAML/JSON dictionaries.

Acceptance:

- Fresh checkout can run tests and demo with one documented command.
- Config changes do not require editing Python source.
- Golden output tests catch accidental ranking/verifier regressions.

## Stage 2: Real Corpus Ingestion

Status: complete.

Goal: ingest realistic medical sources while preserving auditability.

- Add PDF extraction with page numbers using `pypdf` or `PyMuPDF`.
- Add HTML ingestion for clinical guideline pages.
- Add DOCX ingestion if needed for local notes or textbook exports.
- Preserve source metadata: title, author, publisher, date, version, URL/path, page, section, paragraph.
- Improve section classification for clinical structures:
  `definition`, `epidemiology`, `risk_factors`, `symptoms`, `exam`, `diagnosis`, `tests`, `treatment`, `contraindications`, `red_flags`.
- Add duplicate detection for repeated guideline mirrors or copied textbook sections.

Acceptance:

- Ingest at least 3 source formats: Markdown, PDF, HTML.
- Every extracted chunk has a stable source pointer.
- CLI can show the exact source location for any evidence claim.

## Stage 2.5: Allowlisted Open Source Acquisition

Status: complete.

Goal: download a small legal seed corpus from official/open medical sources and build a local corpus artifact.

- Added an allowlisted downloader for official HTTPS domains.
- Added [sources/open_medical_sources.json](sources/open_medical_sources.json) with NCBI Bookshelf and NHLBI seed sources.
- Downloads raw files to `data/open_sources/raw/`.
- Records provider, license, URL, checksum, content type, byte size, and download timestamp in `data/open_sources/downloaded_manifest.json`.
- Builds `data/open_corpus/open_medical_corpus.json` from downloaded sources.
- Keeps downloaded corpus artifacts ignored by git to avoid accidentally redistributing source content.

Acceptance:

- `make download-open-corpus` downloads all seed sources.
- `medreason-graph build-open-corpus` builds a corpus JSON from the downloaded manifest.
- Corpus chunks retain source manifest ID, provider, license, URL, and checksum metadata.

## Stage 3: Medical Terminology Layer

Status: complete.

Goal: make matching medically robust instead of relying on small hand-written synonyms.

- Integrate a terminology table with canonical concepts, synonyms, abbreviations, and semantic type.
- Add optional UMLS/SNOMED/RxNorm/LOINC mapping imports where licensing allows.
- Add abbreviation disambiguation by context, e.g. `PE` as pulmonary embolism vs physical exam.
- Normalize labs, vitals, medications, diseases, symptoms, anatomy, demographics, and procedures.
- Track whether a concept came from exact match, synonym match, abbreviation, or model extraction.

Acceptance:

- `heart attack`, `MI`, and `myocardial infarction` map to the same concept.
- Ambiguous abbreviations are flagged rather than silently normalized.
- Retrieval tests cover synonyms, abbreviations, negation, and absent findings.

## Stage 4: Production Retrieval Stack

Status: in progress; Stage 4A, 4B, and 4C complete.

Goal: move from toy hybrid search to a real multi-channel retriever.

- Stage 4A completed:
  query decomposition, fused retrieval scoring, section boosts, source-condition boosts, corpus noise filtering, duplicate finding dampening, and retrieval evaluation metrics.
- Stage 4B completed:
  pluggable retrieval backend protocol/factory, persistent SQLite FTS5 index, CLI index/search commands, analyzer backend selection, and SQLite retrieval evaluation.
- Stage 4C completed:
  FAISS vector backend, deterministic local hashing embeddings, MedCPT/SapBERT/BioClinicalBERT transformer presets, FAISS index builder, CLI/Makefile support, analyzer backend selection, and FAISS retrieval evaluation.
- Add BM25 search using OpenSearch, Elasticsearch, or Tantivy if external service scale is needed.
- Add dense vector retrieval using PostgreSQL + pgvector, FAISS, or Qdrant.
- Add medically tuned embeddings such as BioClinicalBERT/SapBERT-style encoders.
- Implement retrieval fusion:
  `keyword score + vector score + source quality + section relevance + freshness`.
- Add query decomposition from patient case into:
  `present findings`, `absent findings`, `missing tests`, `red flags`, `dangerous alternatives`.
- Add source filters by type, date, section, specialty, condition, and evidence quality.

Acceptance:

- Retrieval returns both exact matches and semantically relevant evidence.
- Top-k retrieval is explainable with score components.
- Retrieval quality is measured using recall@k and MRR on a labeled test set.

## Stage 5: Evidence Claim Extraction

Status: in progress; Stage 5A and 5B complete.

Goal: extract higher-quality structured evidence from source passages.

- Stage 5A completed:
  evidence claim JSON schema, extraction confidence/method fields, exact source-span offsets, source-text hashes, schema/span validation before claim acceptance, `requires_test` claim typing, `rules_out` detection, and "does not exclude/rule out" handling.
- Stage 5B completed:
  pluggable evidence extractor selection, command-based LLM adapter, strict JSON input/output contract, LLM claim validation against exact source text, optional deterministic fallback, CLI/API controls, and tests that reject non-verbatim LLM claims.
- Replace remaining simple cue-based extraction with broader deterministic and LLM-backed structured extraction.
- Add direct provider adapters or fine-tuned small model support for emitting evidence claims in the same strict schema.
- Add review workflow and calibration for extraction confidence.
- Represent:
  `supports`, `argues_against`, `requires_test`, `rules_in`, `rules_out`, `contraindicates`, `red_flag`, `treatment_recommends`.
- Expand negation and uncertainty handling.
- Expand source-span validation to API/review workflows.

Acceptance:

- Extracted claims validate against JSON schema.
- No claim is accepted without a source span.
- Manual review set reaches agreed precision target before using extracted claims in ranking.

## Stage 6: Evidence Graph Engine

Status: in progress; Stage 6A and 6B complete.

Goal: make the graph a first-class reasoning layer, not just an export.

- Stage 6A completed:
  SQLite graph store for analysis cases, differential items, missing evidence, evidence claims, reasoning steps, reasoning-evidence links, graph nodes, graph edges, and verifier status.
- Stage 6B completed:
  CLI query layer for `evidence-for`, `evidence-against`, `missing-tests`, `reasoning`, `source-spans`, and `explain-rank`.
- Add graph queries for dangerous alternatives and cross-case/corpus-level evidence.
- Add richer provenance queries from every reasoning step to evidence claims and source passages.
- Add conflict representation when sources disagree.
- Add source quality and date weighting directly into graph queries.
- Add graph diffing when corpus versions change.

Acceptance:

- Every differential item can be reconstructed from graph queries.
- The graph can explain why a condition ranked above another.
- Corpus updates preserve prior audit trails.

## Stage 7: Reasoning and Verifier Upgrade

Status: in progress; Stage 7A complete.

Goal: enforce valid, auditable reasoning instead of generating plausible explanations.

- Stage 7A completed:
  deterministic claim-level entailment verification, failed-claim filtering before ranking/reasoning, persisted claim verification status, and graph query support for verifier failures.
- Keep visible reasoning structured, not hidden chain-of-thought.
- Add a reasoning planner that builds:
  `problem representation`, `evidence for`, `evidence against`, `missing evidence`, `dangerous misses`, `next questions/tests`.
- Add verifier checks:
  source supports cited claim,
  patient fact exists,
  absent findings are not treated as present,
  missing tests are not hallucinated,
  recommendations match source type and section,
  dangerous alternatives were checked.
- Add NLI or LLM-based citation entailment verification.
- Block final output when verifier fails unless explicitly requested as debug output.

Acceptance:

- 100% of displayed reasoning steps cite evidence.
- Verifier catches intentionally unsupported, contradicted, or misquoted reasoning.
- High-risk cases always include dangerous-miss checks.

## Stage 8: Medical Case Evaluation Suite

Goal: measure reasoning quality, not just final diagnosis accuracy.

- Build synthetic and de-identified case sets for chest pain, headache, dyspnea, fever, abdominal pain, syncope, and altered mental status.
- Define expected:
  top differentials,
  dangerous misses,
  missing critical tests,
  evidence categories,
  unacceptable claims.
- Track metrics:
  diagnosis recall@k,
  dangerous-miss recall,
  unsupported-claim rate,
  citation precision,
  verifier pass rate,
  retrieval recall@k.
- Add regression tests for known failure modes.

Acceptance:

- Evaluation suite runs locally and produces a score report.
- New changes cannot increase unsupported-claim rate without failing tests.
- Dangerous-miss recall is tracked separately from diagnosis ranking.

## Stage 9: UI and Review Workflow

Goal: make auditability visible.

- Build a small web UI with:
  case input,
  ranked differential table,
  evidence-for/evidence-against panels,
  missing evidence panel,
  dangerous-miss panel,
  graph view,
  verifier report.
- Add source preview with highlighted cited spans.
- Add filters by source type, date, section, and evidence polarity.
- Add reviewer actions:
  approve claim,
  reject claim,
  mark source conflict,
  request more evidence.

Acceptance:

- A reviewer can inspect every reasoning step back to source text.
- The graph view shows patient facts, evidence claims, conditions, and source passages.
- Reviewer feedback is stored for later evaluation or model training.

## Stage 10: Fine-Tuned Small LLM

Goal: use fine-tuning for structure and discipline, not as the source of truth.

- Fine-tune for:
  structured differential output,
  concise medical summaries,
  evidence claim extraction,
  verifier-style critique,
  citation discipline.
- Do not fine-tune medical facts as the main knowledge source.
- Generate training data from reviewed evidence claims and accepted reasoning traces.
- Add strict schemas and reject invalid model outputs.
- Compare fine-tuned model against deterministic and prompt-only baselines.

Acceptance:

- Fine-tuned model improves schema validity, extraction quality, or explanation consistency.
- Medical facts still come from retrieval and graph-backed evidence.
- Verifier remains mandatory after model output.

## Stage 11: Safety, Governance, and Clinical Boundaries

Goal: make limitations explicit and prevent unsafe behavior.

- Add safety policy for emergency symptoms, uncertainty, self-harm, pregnancy, pediatrics, drug dosing, and contraindications.
- Add clear non-autonomous clinical decision-support posture.
- Add audit logs for corpus version, model version, retrieval results, graph state, and verifier result.
- Add PHI handling rules and local-only/de-identified test modes.
- Add red-team tests for hallucinated diagnosis, missing emergency advice, fabricated citations, and overconfident treatment recommendations.

Acceptance:

- No output is shown without corpus version, evidence provenance, and verifier status.
- Emergency/high-risk cases trigger conservative safety language.
- Fabricated citations and unsupported recommendations fail tests.

## Stage 12: Deployment and Operations

Goal: make the system maintainable beyond local experiments.

- Containerize API, worker, database, vector store, and graph store.
- Add background ingestion jobs and corpus versioning.
- Add database migrations.
- Add health checks for retrieval, graph, verifier, and model services.
- Add monitoring for latency, verifier failures, retrieval misses, and unsupported-claim attempts.
- Add backup/restore for corpus index and graph store.

Acceptance:

- Local Docker Compose starts the full stack.
- API can analyze a case against a persisted corpus.
- Operational metrics expose retrieval and verifier health.

## Final Target

The mature system should answer a medical case with:

- ranked differential diagnosis,
- evidence for and against each condition,
- missing critical evidence,
- dangerous alternatives checked,
- cited source spans,
- graph-backed reasoning steps,
- machine-readable verifier report,
- reproducible audit trail.

The core rule remains: no reasoning step becomes user-visible unless it is backed by source-linked evidence and passes verification.
