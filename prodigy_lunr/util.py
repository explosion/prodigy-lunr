import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import srsly
from fastapi import HTTPException
from lunr import lunr
from lunr.index import Index
from prodigy.components.stream import Stream, get_stream
from prodigy.core import Controller
from prodigy.util import log, set_hashes
from starlette.status import HTTP_400_BAD_REQUEST

HTML = """
<link
  rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css"
  integrity="sha512-1sCRPdkRXhBV2PBLUdRb4tMg1w2YPf37qatUFeS7zlBy7jJI8Lf4VHwWfZZfpXtYSLy85pkm9GaYVYMfw5BC1A=="
  crossorigin="anonymous"
  referrerpolicy="no-referrer"
/>
<details>
    <summary id="reset">Reset stream?</summary>
    <div class="prodigy-content">
        <label class="label" for="query">New query:</label>
        <input class="prodigy-text-input text-input" type="text" id="query" name="query" value="">
        <br><br>
        <button id="refreshButton" onclick="refreshData()">
            Refresh Stream
            <i
                id="loadingIcon"
                class="fa-solid fa-spinner fa-spin"
                style="display: none;"
            ></i>
        </button>
    </div>
</details>
"""

# We need to dedent in order to prevent a bunch of whitespaces to appear.
HTML = textwrap.dedent(HTML).replace("\n", "")

CSS = """
.inner-div{
  border: 1px solid #ddd;
  text-align: left;
  border-radius: 4px;
}

.label{
  top: -3px;
  opacity: 0.75;
  position: relative;
  font-size: 12px;
  font-weight: bold;
  padding-left: 10px;
}

.text-input{
  width: 100%;
  border: 1px solid #cacaca;
  border-radius: 5px;
  padding: 10px;
  font-size: 20px;
  background: transparent;
  font-family: "Lato", "Trebuchet MS", Roboto, Helvetica, Arial, sans-serif;
}

#reset{
  font-size: 16px;
}
"""

JS = """
function refreshData() {
  document.querySelector('#loadingIcon').style.display = 'inline-block';

  const event_data = {
    query: document.getElementById("query").value
  };

  window.prodigy
    .event('stream-reset', event_data)
    .then(response => {
      if (response.status === 400) {
        console.error('Error:', response.detail);
        alert(response.detail || 'No examples found for the given query. Please try a different one.');
        document.querySelector('#loadingIcon').style.display = 'none';
      } else {
        console.log('Updating Current Example with new data:', response);
        window.prodigy.resetQueue();
        window.prodigy.update(response);
        document.querySelector('#loadingIcon').style.display = 'none'
      }
    })
    .catch(err => {
      console.error('Error in Event Handler:', err);
      alert(err.detail || "There's been an error in the stream update handler. See console for details.");
      document.querySelector('#loadingIcon').style.display = 'none';
    });
}
"""


def add_hashes(examples):
    for ex in examples:
        yield set_hashes(ex)


class SearchIndex:
    def __init__(self, source: Path, index_path: Optional[Path] = None):
        log(f"INDEX: Using {index_path=} and source={str(source)}.")
        stream = get_stream(source)
        stream.apply(add_hashes)
        # Storing this as a list isn't scale-able, but is fair enough for medium sized datasets.
        self.documents = [ex for ex in stream]
        self.index_path = index_path
        self.index = None
        if self.index_path and self.index_path.exists():
            self.index = Index.load(srsly.read_gzip_json(index_path))

    def build_index(self) -> "SearchIndex":
        # Store sentences as a list, not perfect, but works.
        documents = [
            {"idx": i, "text": ex["text"]} for i, ex in enumerate(self.documents)
        ]
        # Create the index
        self.index = lunr(ref="idx", fields=("text",), documents=documents)
        return self

    def store_index(self, path: Path):
        srsly.write_gzip_json(str(self.index_path), self.index.serialize(), indent=0)
        log(f"INDEX: Index file stored at {path}.")

    def _to_prodigy_examples(self, examples: List[Dict], query: str):
        for res in examples:
            ex = self.documents[int(res["ref"])]
            ex["meta"] = {
                "score": res["score"],
                "query": query,
                "index_ref": int(res["ref"]),
            }
            yield set_hashes(ex)

    def new_stream(self, query: str, n: int = 100):
        log(f"INDEX: Creating new stream of {n} examples using {query=}.")
        results = self.index.search(query)[:n]
        return self._to_prodigy_examples(results, query=query)


def stream_reset_calback(index_obj: SearchIndex, n: int = 100):
    def stream_reset(ctrl: Controller, *, query: str):
        new_stream = Stream.from_iterable(index_obj.new_stream(query, n=n))
        if new_stream.is_empty:
            log(f"INDEX: No examples found for query: {query}. Try a different one")
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"No examples found for query: {query}. Try a different one",
            )
        else:
            ctrl.reset_stream(new_stream, prepend_old_wrappers=True)
            return next(new_stream)

    return stream_reset
