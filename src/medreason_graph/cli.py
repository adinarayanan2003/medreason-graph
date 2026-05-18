from __future__ import annotations

import argparse
import json
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.config import load_and_apply_config
from medreason_graph.evaluation import evaluate_retrieval
from medreason_graph.faiss_retrieval import FAISSRetriever, build_faiss_index
from medreason_graph.graph import export_cytoscape, export_graphviz_dot, graph_to_json
from medreason_graph.ingestion import ingest_path
from medreason_graph.logging_utils import configure_logging
from medreason_graph.models import PatientCase
from medreason_graph.reporter import render_text
from medreason_graph.retrieval_backend import close_retriever, create_retriever
from medreason_graph.source_downloader import build_downloaded_corpus, download_allowlisted_sources
from medreason_graph.sqlite_retrieval import SQLiteFTSRetriever, build_sqlite_fts_index
from medreason_graph.storage import load_chunks, save_chunks


def main() -> int:
    parser = argparse.ArgumentParser(prog="medreason-graph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest medical source files into a JSON corpus.")
    ingest.add_argument("path")
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--source-type", default="unknown")
    ingest.add_argument("--config")
    ingest.add_argument("--verbose", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Analyze a patient case against a corpus.")
    analyze.add_argument("--corpus", required=True)
    analyze.add_argument("--case", required=True)
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--retriever", choices=("memory", "sqlite", "faiss"), default="memory")
    analyze.add_argument("--index")
    analyze.add_argument("--config")
    analyze.add_argument("--verbose", action="store_true")

    graph = subparsers.add_parser("graph", help="Export the evidence graph for a case.")
    graph.add_argument("--corpus", required=True)
    graph.add_argument("--case", required=True)
    graph.add_argument("--out", required=True)
    graph.add_argument("--format", choices=("json", "cytoscape", "dot"), default="json")
    graph.add_argument("--retriever", choices=("memory", "sqlite", "faiss"), default="memory")
    graph.add_argument("--index")
    graph.add_argument("--config")
    graph.add_argument("--verbose", action="store_true")

    download = subparsers.add_parser("download-sources", help="Download allowlisted open medical sources.")
    download.add_argument("--manifest", default="sources/open_medical_sources.json")
    download.add_argument("--out-dir", default="data/open_sources")
    download.add_argument("--delay-seconds", type=float, default=0.5)
    download.add_argument("--verbose", action="store_true")

    build_corpus = subparsers.add_parser("build-open-corpus", help="Build corpus JSON from downloaded open sources.")
    build_corpus.add_argument("--downloaded-manifest", default="data/open_sources/downloaded_manifest.json")
    build_corpus.add_argument("--out", default="data/open_corpus/open_medical_corpus.json")
    build_corpus.add_argument("--verbose", action="store_true")

    evaluate = subparsers.add_parser("evaluate-retrieval", help="Evaluate differential retrieval quality on labeled cases.")
    evaluate.add_argument("--corpus", required=True)
    evaluate.add_argument("--cases", default="evaluation/retrieval_cases.json")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--retriever", choices=("memory", "sqlite", "faiss"), default="memory")
    evaluate.add_argument("--index")
    evaluate.add_argument("--config")
    evaluate.add_argument("--verbose", action="store_true")

    index_corpus = subparsers.add_parser("index-corpus", help="Build a SQLite FTS5 retrieval index from corpus JSON.")
    index_corpus.add_argument("--corpus", required=True)
    index_corpus.add_argument("--out", required=True)
    index_corpus.add_argument("--verbose", action="store_true")

    index_faiss = subparsers.add_parser("index-faiss", help="Build a FAISS vector retrieval index from corpus JSON.")
    index_faiss.add_argument("--corpus", required=True)
    index_faiss.add_argument("--out", required=True)
    index_faiss.add_argument("--dim", type=int, default=384)
    index_faiss.add_argument("--embedding-preset", choices=("hash", "medcpt", "sapbert", "bioclinicalbert"), default="hash")
    index_faiss.add_argument("--query-model")
    index_faiss.add_argument("--document-model")
    index_faiss.add_argument("--pooling", choices=("cls", "mean"))
    index_faiss.add_argument("--query-max-length", type=int)
    index_faiss.add_argument("--document-max-length", type=int)
    index_faiss.add_argument("--batch-size", type=int, default=8)
    index_faiss.add_argument("--verbose", action="store_true")

    search_index = subparsers.add_parser("search-index", help="Search a retrieval index directly.")
    search_index.add_argument("--index", required=True)
    search_index.add_argument("--query", required=True)
    search_index.add_argument("--retriever", choices=("sqlite", "faiss"), default="sqlite")
    search_index.add_argument("--top-k", type=int, default=5)
    search_index.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    configure_logging(verbose=getattr(args, "verbose", False))
    if getattr(args, "config", None):
        load_and_apply_config(args.config)
    if args.command == "download-sources":
        manifest = download_allowlisted_sources(args.manifest, args.out_dir, delay_seconds=args.delay_seconds)
        print(f"Downloaded {len(manifest['sources'])} sources into {args.out_dir}")
        return 0

    if args.command == "build-open-corpus":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        chunk_count = build_downloaded_corpus(args.downloaded_manifest, args.out)
        print(f"Built corpus with {chunk_count} chunks at {args.out}")
        return 0

    if args.command == "evaluate-retrieval":
        chunks = load_chunks(args.corpus)
        retriever = _retriever_for_args(args, chunks)
        print(json.dumps(evaluate_retrieval(chunks, args.cases, k=args.k, retriever=retriever), indent=2))
        _close_retriever(retriever)
        return 0

    if args.command == "index-corpus":
        chunks = load_chunks(args.corpus)
        build_sqlite_fts_index(chunks, args.out)
        print(f"Indexed {len(chunks)} chunks into {args.out}")
        return 0

    if args.command == "index-faiss":
        chunks = load_chunks(args.corpus)
        build_faiss_index(
            chunks,
            args.out,
            dim=args.dim,
            embedding_preset=args.embedding_preset,
            query_model=args.query_model,
            document_model=args.document_model,
            pooling=args.pooling,
            query_max_length=args.query_max_length,
            document_max_length=args.document_max_length,
            batch_size=args.batch_size,
        )
        print(f"Indexed {len(chunks)} chunks into {args.out}")
        return 0

    if args.command == "search-index":
        from medreason_graph.query import QueryPart

        retriever = SQLiteFTSRetriever(args.index) if args.retriever == "sqlite" else FAISSRetriever(args.index)
        hits = retriever.fused_search(
            [
                QueryPart(
                    label="debug",
                    text=args.query,
                    weight=1.0,
                    section_boosts={},
                    condition_boosts=set(),
                )
            ],
            top_k=args.top_k,
        )
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        retriever.close()
        return 0

    if args.command == "ingest":
        chunks = ingest_path(args.path, source_type=args.source_type)
        save_chunks(chunks, args.out)
        print(f"Ingested {len(chunks)} chunks into {args.out}")
        return 0

    if args.command in {"analyze", "graph"}:
        chunks = load_chunks(args.corpus)
        case = PatientCase.from_dict(json.loads(Path(args.case).read_text(encoding="utf-8")))
        retriever = _retriever_for_args(args, chunks)
        result = MedReasonAnalyzer(chunks, retriever=retriever).analyze(case)
        _close_retriever(retriever)
        if args.command == "graph":
            if args.format == "cytoscape":
                output = json.dumps(export_cytoscape(result.graph), indent=2)
            elif args.format == "dot":
                output = export_graphviz_dot(result.graph)
            else:
                output = graph_to_json(result.graph)
            Path(args.out).write_text(output, encoding="utf-8")
            print(f"Wrote graph to {args.out}")
            return 0
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_text(result))
        return 0

    return 2


def _retriever_for_args(args, chunks):
    kind = getattr(args, "retriever", "memory")
    if kind in {"sqlite", "faiss"} and not getattr(args, "index", None):
        raise SystemExit(f"--index is required when --retriever {kind} is used")
    return create_retriever(kind, chunks, getattr(args, "index", None))


def _close_retriever(retriever) -> None:
    close_retriever(retriever)


if __name__ == "__main__":
    raise SystemExit(main())
