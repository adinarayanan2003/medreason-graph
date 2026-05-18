.PHONY: test demo demo-json graph-dot graph-cytoscape compile download-open-corpus build-open-corpus index-open-corpus index-open-corpus-faiss index-open-corpus-medcpt evaluate-open-corpus evaluate-open-corpus-sqlite evaluate-open-corpus-faiss evaluate-open-corpus-medcpt

PYTHON ?= python3
PYTHONPATH := src
CORPUS := /tmp/medreason_corpus.json

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli ingest examples/corpus --out $(CORPUS) --source-type guideline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli analyze --corpus $(CORPUS) --case examples/cases/chest_pain.json

demo-json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli ingest examples/corpus --out $(CORPUS) --source-type guideline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli analyze --corpus $(CORPUS) --case examples/cases/chest_pain.json --json

graph-dot:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli ingest examples/corpus --out $(CORPUS) --source-type guideline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli graph --corpus $(CORPUS) --case examples/cases/chest_pain.json --format dot --out /tmp/medreason_graph.dot

graph-cytoscape:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli ingest examples/corpus --out $(CORPUS) --source-type guideline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli graph --corpus $(CORPUS) --case examples/cases/chest_pain.json --format cytoscape --out /tmp/medreason_graph_cytoscape.json

download-open-corpus:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli download-sources --manifest sources/open_medical_sources.json --out-dir data/open_sources

build-open-corpus:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli build-open-corpus --downloaded-manifest data/open_sources/downloaded_manifest.json --out data/open_corpus/open_medical_corpus.json

index-open-corpus:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli index-corpus --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.sqlite

index-open-corpus-faiss:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli index-faiss --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus.faiss

index-open-corpus-medcpt:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli index-faiss --corpus data/open_corpus/open_medical_corpus.json --out data/open_corpus/open_medical_corpus_medcpt.faiss --embedding-preset medcpt --batch-size 8 --document-max-length 256 --query-max-length 64

evaluate-open-corpus:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --cases evaluation/retrieval_cases.json --k 5

evaluate-open-corpus-sqlite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --cases evaluation/retrieval_cases.json --k 5 --retriever sqlite --index data/open_corpus/open_medical_corpus.sqlite

evaluate-open-corpus-faiss:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --cases evaluation/retrieval_cases.json --k 5 --retriever faiss --index data/open_corpus/open_medical_corpus.faiss

evaluate-open-corpus-medcpt:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medreason_graph.cli evaluate-retrieval --corpus data/open_corpus/open_medical_corpus.json --cases evaluation/retrieval_cases.json --k 5 --retriever faiss --index data/open_corpus/open_medical_corpus_medcpt.faiss
