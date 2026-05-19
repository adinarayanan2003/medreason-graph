from __future__ import annotations

import argparse
import json
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.graph_store import build_graph_store, query_verifier_failures
from medreason_graph.models import PatientCase
from medreason_graph.reporter import render_text
from medreason_graph.retrieval_backend import close_retriever, create_retriever
from medreason_graph.storage import load_chunks


DEFAULT_CASES = (
    "examples/cases/chest_pain_real_demo.json",
    "examples/cases/dyspnea_real_demo.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate auditable reports from the downloaded open medical corpus.")
    parser.add_argument("--corpus", default="data/open_corpus/open_medical_corpus.json")
    parser.add_argument("--out-dir", default="data/real_outputs")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--retriever", choices=("memory", "sqlite", "faiss"), default="memory")
    parser.add_argument("--index")
    parser.add_argument("--evidence-extractor", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument("--llm-command")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-fallback-to-deterministic", action="store_true")
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    case_paths = [Path(item) for item in (args.cases or DEFAULT_CASES)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.corpus)

    manifest = []
    for case_path in case_paths:
        case = PatientCase.from_dict(json.loads(case_path.read_text(encoding="utf-8")))
        retriever = create_retriever(args.retriever, chunks, args.index)
        try:
            result = MedReasonAnalyzer(
                chunks,
                retriever=retriever,
                evidence_extractor=args.evidence_extractor,
                llm_command=args.llm_command,
                llm_timeout_seconds=args.llm_timeout_seconds,
                llm_fallback_to_deterministic=args.llm_fallback_to_deterministic,
            ).analyze(case, top_k=args.top_k)
        finally:
            close_retriever(retriever)

        case_stem = case_path.stem
        analysis_path = out_dir / f"{case_stem}.analysis.json"
        report_path = out_dir / f"{case_stem}.report.txt"
        graph_path = out_dir / f"{case_stem}.graph.sqlite"
        verifier_failures_path = out_dir / f"{case_stem}.verifier_failures.json"

        analysis_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        report_path.write_text(render_text(result), encoding="utf-8")
        build_graph_store(result, graph_path)
        verifier_failures_path.write_text(json.dumps(query_verifier_failures(graph_path), indent=2), encoding="utf-8")
        manifest.append(
            {
                "case_id": result.case_id,
                "case": str(case_path),
                "analysis": str(analysis_path),
                "report": str(report_path),
                "graph": str(graph_path),
                "verifier_failures": str(verifier_failures_path),
                "verifier_passed": result.verifier.passed,
                "differential_count": len(result.differential),
                "accepted_claim_count": len(result.evidence_claims),
                "claim_verification_count": len(result.claim_verifications),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} real-output demo reports to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
