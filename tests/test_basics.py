import pytest
import srsly
from pathlib import Path 
from prodigy_lunr import index, fetch, spans_lunr_manual, textcat_lunr_manual, ner_lunr_manual


@pytest.mark.parametrize("query", ["download", "query", "benchmark"])
def test_smoke(tmpdir, query):
    """Just a minimum viable smoketest."""
    examples_path = Path("tests/datasets/new-dataset.jsonl")
    index_path = tmpdir / "new-dataset.gz.jsonl"
    fetch_path = tmpdir / "fetched.jsonl"

    # Ensure fetch works as expected
    index(examples_path, index_path)
    for query in ["benchmarks", "download"]:
        fetch(examples_path, index_path, fetch_path, query=query)
        fetched_examples = list(srsly.read_jsonl(fetch_path))
        for ex in fetched_examples:
            assert ex['meta']['query'] == query
    
    # Also ensure the helpers do not break
    recipes = [
        textcat_lunr_manual("xxx", examples_path, index_path, labels="foo,bar", query=query),
        ner_lunr_manual("xxx", "blank:en", examples_path, index_path, labels="foo,bar", query=query),
        spans_lunr_manual("xxx", "blank:en", examples_path, index_path, labels="foo,bar", query=query)
    ]
    for recipe in recipes:
        for ex in recipe['stream']:
            assert ex['meta']['query'] == query
