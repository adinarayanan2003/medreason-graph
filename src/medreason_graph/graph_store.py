from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from medreason_graph.models import AnalysisResult

SCHEMA_VERSION = 1


def build_graph_store(result: AnalysisResult, path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _init_schema(conn)
        _clear_case(conn, result.case_id)
        _insert_result(conn, result)


def load_analysis_result(path: str | Path) -> AnalysisResult:
    return AnalysisResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def query_evidence(path: str | Path, *, condition: str, polarity: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, claim_type, condition, finding, polarity, strength, source_title,
                   source_type, section_path_json, paragraph_index, sentence,
                   source_span_start, source_span_end, source_text_hash,
                   extraction_confidence, extraction_method
            FROM evidence_claims
            WHERE lower(condition) = lower(?) AND polarity = ?
            ORDER BY strength_rank DESC, extraction_confidence DESC, id
            """,
            (condition, polarity),
        ).fetchall()
    return [_claim_row_to_dict(row) for row in rows]


def query_missing_tests(path: str | Path, *, condition: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT condition, test_name
            FROM missing_evidence
            WHERE lower(condition) = lower(?)
            ORDER BY test_name
            """,
            (condition,),
        ).fetchall()
    return [dict(row) for row in rows]


def query_reasoning(path: str | Path, *, condition: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, condition, statement, polarity, patient_facts_json
            FROM reasoning_steps
            WHERE lower(condition) = lower(?)
            ORDER BY id
            """,
            (condition,),
        ).fetchall()
        result = []
        for row in rows:
            evidence_rows = conn.execute(
                """
                SELECT evidence_id
                FROM reasoning_step_evidence
                WHERE step_id = ?
                ORDER BY evidence_id
                """,
                (row["id"],),
            ).fetchall()
            item = dict(row)
            item["patient_facts"] = json.loads(item.pop("patient_facts_json"))
            item["uses_evidence"] = [evidence["evidence_id"] for evidence in evidence_rows]
            result.append(item)
    return result


def query_source_spans(path: str | Path, *, condition: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, condition, finding, polarity, source_title, source_type,
                   section_path_json, paragraph_index, source_span_start,
                   source_span_end, source_text_hash, sentence
            FROM evidence_claims
            WHERE lower(condition) = lower(?)
            ORDER BY source_title, paragraph_index, source_span_start, id
            """,
            (condition,),
        ).fetchall()
    return [_source_span_row_to_dict(row) for row in rows]


def query_explain_rank(path: str | Path, *, condition: str) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT condition, rank, urgency, score, confidence
            FROM differential
            WHERE lower(condition) = lower(?)
            """,
            (condition,),
        ).fetchone()
    if row is None:
        return {"condition": condition, "found": False}
    return {
        **dict(row),
        "found": True,
        "evidence_for": query_evidence(path, condition=condition, polarity="supports"),
        "evidence_against": query_evidence(path, condition=condition, polarity="argues_against"),
        "missing_tests": query_missing_tests(path, condition=condition),
        "reasoning_steps": query_reasoning(path, condition=condition),
    }


def query_verifier_failures(path: str | Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT v.claim_id, v.supported, v.reasons_json, v.verifier_method,
                   c.claim_type, c.condition, c.finding, c.polarity, c.sentence,
                   c.source_title, c.source_span_start, c.source_span_end
            FROM claim_verifications v
            LEFT JOIN evidence_claims c ON c.id = v.claim_id
            WHERE v.supported = 0
            ORDER BY v.claim_id
            """
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["reasons"] = json.loads(item.pop("reasons_json"))
        item["supported"] = bool(item["supported"])
        result.append(item)
    return result


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cases (
          case_id TEXT PRIMARY KEY,
          problem_representation TEXT NOT NULL,
          verifier_passed INTEGER NOT NULL,
          verifier_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS differential (
          case_id TEXT NOT NULL,
          condition TEXT NOT NULL,
          rank INTEGER NOT NULL,
          urgency TEXT NOT NULL,
          score REAL NOT NULL,
          confidence TEXT NOT NULL,
          PRIMARY KEY (case_id, condition),
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS missing_evidence (
          case_id TEXT NOT NULL,
          condition TEXT NOT NULL,
          test_name TEXT NOT NULL,
          PRIMARY KEY (case_id, condition, test_name),
          FOREIGN KEY (case_id, condition) REFERENCES differential(case_id, condition) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evidence_claims (
          id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          claim_type TEXT NOT NULL,
          condition TEXT NOT NULL,
          finding TEXT,
          polarity TEXT NOT NULL,
          strength TEXT NOT NULL,
          strength_rank INTEGER NOT NULL,
          source_id TEXT NOT NULL,
          source_type TEXT NOT NULL,
          source_title TEXT NOT NULL,
          section_path_json TEXT NOT NULL,
          paragraph_index INTEGER NOT NULL,
          sentence TEXT NOT NULL,
          source_span_start INTEGER NOT NULL,
          source_span_end INTEGER NOT NULL,
          source_text_hash TEXT NOT NULL,
          extraction_confidence REAL NOT NULL,
          extraction_method TEXT NOT NULL,
          schema_version TEXT NOT NULL,
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS claim_verifications (
          claim_id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          supported INTEGER NOT NULL,
          reasons_json TEXT NOT NULL,
          verifier_method TEXT NOT NULL,
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS reasoning_steps (
          id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          condition TEXT NOT NULL,
          statement TEXT NOT NULL,
          polarity TEXT NOT NULL,
          patient_facts_json TEXT NOT NULL,
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS reasoning_step_evidence (
          step_id TEXT NOT NULL,
          evidence_id TEXT NOT NULL,
          PRIMARY KEY (step_id, evidence_id),
          FOREIGN KEY (step_id) REFERENCES reasoning_steps(id) ON DELETE CASCADE,
          FOREIGN KEY (evidence_id) REFERENCES evidence_claims(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS graph_nodes (
          case_id TEXT NOT NULL,
          id TEXT NOT NULL,
          kind TEXT NOT NULL,
          label TEXT NOT NULL,
          attrs_json TEXT NOT NULL,
          PRIMARY KEY (case_id, id),
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS graph_edges (
          case_id TEXT NOT NULL,
          source TEXT NOT NULL,
          type TEXT NOT NULL,
          target TEXT NOT NULL,
          attrs_json TEXT NOT NULL,
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_evidence_condition_polarity
          ON evidence_claims(condition, polarity);
        CREATE INDEX IF NOT EXISTS idx_claim_verifications_supported
          ON claim_verifications(supported);
        CREATE INDEX IF NOT EXISTS idx_reasoning_condition
          ON reasoning_steps(condition);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def _clear_case(conn: sqlite3.Connection, case_id: str) -> None:
    conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))


def _insert_result(conn: sqlite3.Connection, result: AnalysisResult) -> None:
    conn.execute(
        """
        INSERT INTO cases(case_id, problem_representation, verifier_passed, verifier_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            result.case_id,
            result.problem_representation,
            1 if result.verifier.passed else 0,
            json.dumps(result.verifier.to_dict(), sort_keys=True),
        ),
    )
    for item in result.differential:
        conn.execute(
            """
            INSERT INTO differential(case_id, condition, rank, urgency, score, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (result.case_id, item.condition, item.rank, item.urgency, item.score, item.confidence),
        )
        for missing in item.missing_evidence:
            conn.execute(
                """
                INSERT OR IGNORE INTO missing_evidence(case_id, condition, test_name)
                VALUES (?, ?, ?)
                """,
                (result.case_id, item.condition, missing),
            )
    for claim in result.evidence_claims:
        conn.execute(
            """
            INSERT INTO evidence_claims(
              id, case_id, claim_type, condition, finding, polarity, strength,
              strength_rank, source_id, source_type, source_title, section_path_json,
              paragraph_index, sentence, source_span_start, source_span_end,
              source_text_hash, extraction_confidence, extraction_method, schema_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                result.case_id,
                claim.claim_type,
                claim.condition,
                claim.finding,
                claim.polarity,
                claim.strength,
                _strength_rank(claim.strength),
                claim.source_id,
                claim.source_type,
                claim.source_title,
                json.dumps(claim.section_path),
                claim.paragraph_index,
                claim.sentence,
                claim.source_span_start,
                claim.source_span_end,
                claim.source_text_hash,
                claim.extraction_confidence,
                claim.extraction_method,
                claim.schema_version,
            ),
        )
    for verification in result.claim_verifications:
        conn.execute(
            """
            INSERT INTO claim_verifications(claim_id, case_id, supported, reasons_json, verifier_method)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                verification.claim_id,
                result.case_id,
                1 if verification.supported else 0,
                json.dumps(verification.reasons),
                verification.verifier_method,
            ),
        )
    for step in result.reasoning_steps:
        conn.execute(
            """
            INSERT INTO reasoning_steps(id, case_id, condition, statement, polarity, patient_facts_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                step.id,
                result.case_id,
                step.condition,
                step.statement,
                step.polarity,
                json.dumps(step.patient_facts),
            ),
        )
        for evidence_id in step.uses_evidence:
            conn.execute(
                """
                INSERT OR IGNORE INTO reasoning_step_evidence(step_id, evidence_id)
                VALUES (?, ?)
                """,
                (step.id, evidence_id),
            )
    for node in result.graph.nodes:
        attrs = {key: value for key, value in node.items() if key not in {"id", "kind", "label"}}
        conn.execute(
            """
            INSERT INTO graph_nodes(case_id, id, kind, label, attrs_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result.case_id,
                str(node.get("id", "")),
                str(node.get("kind", "")),
                str(node.get("label", "")),
                json.dumps(attrs, sort_keys=True),
            ),
        )
    for edge in result.graph.edges:
        attrs = {key: value for key, value in edge.items() if key not in {"source", "type", "target"}}
        conn.execute(
            """
            INSERT INTO graph_edges(case_id, source, type, target, attrs_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result.case_id,
                str(edge.get("source", "")),
                str(edge.get("type", "")),
                str(edge.get("target", "")),
                json.dumps(attrs, sort_keys=True),
            ),
        )


def _claim_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["section_path"] = json.loads(item.pop("section_path_json"))
    return item


def _source_span_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["section_path"] = json.loads(item.pop("section_path_json"))
    return item


def _strength_rank(strength: str) -> int:
    return {"weak": 1, "moderate": 2, "strong": 3}.get(strength, 0)
