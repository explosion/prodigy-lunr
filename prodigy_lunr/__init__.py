from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional

import srsly

from prodigy import recipe
from prodigy.recipes.textcat import manual as textcat_manual
from prodigy.recipes.ner import manual as ner_manual
from prodigy.recipes.spans import manual as spans_manual
from lunr import lunr
from lunr.index import Index


@recipe(
    "lunr.text.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path of trained index", "positional", None, Path),
    # fmt: on
)
def index(source: Path, index_path: Path):
    """Builds an HSNWLIB index on example text data."""
    # Store sentences as a list, not perfect, but works.
    documents = [{"idx": i, **ex} for i, ex in enumerate(srsly.read_jsonl(source))]
    # Create the index
    index = lunr(ref='idx', fields=('text',), documents=documents)
    # Store it on disk
    srsly.write_gzip_json(index_path, index.serialize(), indent=0)


@recipe(
    "lunr.text.fetch",
    # fmt: off
    source=("Path to text source that has been indexed", "positional", None, str),
    index_path=("Path to index", "positional", None, Path),
    out_path=("Path to write examples into", "positional", None, Path),
    query=("ANN query to run", "option", "q", str),
    n=("Max number of results to return", "option", "n", int),
    # fmt: on
)
def fetch(source: Path, index_path: Path, out_path: Path, query:str, n:int=200):
    """Fetch a relevant subset using a HNSWlib index."""
    if not query:
        raise ValueError("must pass query")
    
    documents = [{"idx": i, **ex} for i, ex in enumerate(srsly.read_jsonl(source))]
    index = Index.load(srsly.read_gzip_json(index_path))
    results = index.search(query)[:n]

    def to_prodigy_examples(results):
        for res in results:
            ex = documents[int(res['ref'])]
            ex['meta'] = {
                'score': res['score'], 'query': query
            }
            yield ex

    srsly.write_jsonl(out_path, to_prodigy_examples(results))


@recipe(
    "textcat.lunr.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    # fmt: on
)
def textcat_lunr_manual(
    dataset: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    exclusive:bool = False
):
    """Run textcat.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        return textcat_manual(dataset, stream, label=labels.split(","), exclusive=exclusive)


@recipe(
    "ner.lunr.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    patterns=("Path to match patterns file", "option", "pt", Path),
    # fmt: on
)
def ner_lunr_manual(
    dataset: str,
    nlp: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    patterns: Optional[Path] = None,
):
    """Run ner.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        ner_manual(dataset, nlp, stream, label=labels, patterns=patterns)


@recipe(
    "spans.lunr.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    nlp=("spaCy model to load", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    patterns=("Path to match patterns file", "option", "pt", Path),
    # fmt: on
)
def spans_lunr_manual(
    dataset: str,
    nlp: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    patterns: Optional[Path] = None,
):
    """Run spans.manual using a query to populate the stream."""
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        fetch(examples, index_path, out_path=tmpfile.name, query=query)
        stream = list(srsly.read_jsonl(tmpfile.name))
        spans_manual(dataset, nlp, stream, label=labels, patterns=patterns)
