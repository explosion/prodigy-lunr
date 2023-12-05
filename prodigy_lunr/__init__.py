import spacy 
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional

import srsly

from prodigy import recipe
from prodigy.recipes.textcat import manual as textcat_manual
from prodigy.recipes.ner import manual as ner_manual
from prodigy.recipes.spans import manual as spans_manual
from prodigy.util import log
from .util import SearchIndex, JS, CSS, HTML, stream_reset_calback


@recipe(
    "lunr.text.index",
    # fmt: off
    source=("Path to text source to index", "positional", None, str),
    index_path=("Path of trained index", "positional", None, Path),
    # fmt: on
)
def index(source: Path, index_path: Path):
    """Builds a LUNR index on example text data."""
    # Store sentences as a list, not perfect, but works.
    log("RECIPE: Calling `lunr.text.index`")
    index = SearchIndex(source, index_path=index_path)
    index.build_index()
    index.store_index(index_path)


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
    """Fetch a relevant subset using a LUNR index."""
    log("RECIPE: Calling `lunr.text.fetch`")
    if not query:
        raise ValueError("must pass query")
    
    index = SearchIndex(source, index_path=index_path)
    new_examples = index.new_stream(query=query, n=n)
    srsly.write_jsonl(out_path, new_examples)


@recipe(
    "textcat.lunr.manual",
    # fmt: off
    dataset=("Dataset to save answers to", "positional", None, str),
    examples=("Examples that have been indexed", "positional", None, str),
    index_path=("Path to trained index", "positional", None, Path),
    labels=("Comma seperated labels to use", "option", "l", str),
    query=("ANN query to run", "option", "q", str),
    exclusive=("Labels are exclusive", "flag", "e", bool),
    n=("Number of items to retreive via query", "option", "n", int),
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
    # fmt: on
)
def textcat_lunr_manual(
    dataset: str,
    examples: Path,
    index_path: Path,
    labels:str,
    query:str,
    exclusive:bool = False,
    n:int = 200,
    allow_reset: bool = False
):
    """Run textcat.manual using a query to populate the stream."""
    log("RECIPE: Calling `textcat.ann.manual`")
    index = SearchIndex(source=examples, index_path=index_path)
    stream = index.new_stream(query, n=n)
    components = textcat_manual(dataset, stream, label=labels.split(","), exclusive=exclusive)
    
    # Only update the components if the user wants to allow the user to reset the stream
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(index, n=n)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components


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
    n=("Number of items to retreive via query", "option", "n", int),
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
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
    n:int = 200,
    allow_reset:bool = False,
):
    """Run ner.manual using a query to populate the stream."""
    log("RECIPE: Calling `ner.ann.manual`")
    if "blank" in nlp:
        spacy_mod = spacy.blank(nlp.replace("blank:", ""))
    else:
        spacy_mod = spacy.load(nlp)
    index = SearchIndex(source=examples, index_path=index_path)
    stream = index.new_stream(query, n=n)
    
    # Only update the components if the user wants to allow the user to reset the stream
    components = ner_manual(dataset, spacy_mod, stream, label=labels.split(","), patterns=patterns)
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(index, n=n)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components


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
    n=("Number of items to retreive via query", "option", "n", int),
    allow_reset=("Allow the user to restart the query", "flag", "r", bool)
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
    n:int = 200,
    allow_reset: bool = False
):
    """Run spans.manual using a query to populate the stream."""
    log("RECIPE: Calling `spans.ann.manual`")
    if "blank" in nlp:
        spacy_mod = spacy.blank(nlp.replace("blank:", ""))
    else:
        spacy_mod = spacy.load(nlp)
    index = SearchIndex(source=examples, index_path=index_path)
    stream = index.new_stream(query, n=n)

    # Only update the components if the user wants to allow the user to reset the stream
    components = spans_manual(dataset, spacy_mod, stream, label=labels.split(","), patterns=patterns)
    if allow_reset:
        blocks = [
            {"view_id": components["view_id"]}, 
            {"view_id": "html", "html_template": HTML}
        ]
        components["event_hooks"] = {
            "stream-reset": stream_reset_calback(index, n=n)
        }
        components["view_id"] = "blocks"
        components["config"]["javascript"] = JS
        components["config"]["global_css"] = CSS
        components["config"]["blocks"] = blocks
    return components
