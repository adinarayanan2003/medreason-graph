# Graph Export Examples

The CLI can export the same evidence graph in three formats:

```bash
make graph-dot
make graph-cytoscape
```

Direct commands:

```bash
PYTHONPATH=src python3 -m medreason_graph.cli ingest examples/corpus --out /tmp/medreason_corpus.json --source-type guideline
PYTHONPATH=src python3 -m medreason_graph.cli graph --corpus /tmp/medreason_corpus.json --case examples/cases/chest_pain.json --format json --out /tmp/medreason_graph.json
PYTHONPATH=src python3 -m medreason_graph.cli graph --corpus /tmp/medreason_corpus.json --case examples/cases/chest_pain.json --format cytoscape --out /tmp/medreason_graph_cytoscape.json
PYTHONPATH=src python3 -m medreason_graph.cli graph --corpus /tmp/medreason_corpus.json --case examples/cases/chest_pain.json --format dot --out /tmp/medreason_graph.dot
```

Use the Cytoscape JSON output for web graph viewers. Use DOT output with Graphviz:

```bash
dot -Tpng /tmp/medreason_graph.dot -o /tmp/medreason_graph.png
```

