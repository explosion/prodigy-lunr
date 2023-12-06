import time 

import pytest

from .e2e_util import prodigy_playwright


base_calls = [
    "textcat.lunr.manual xxx",
    "ner.lunr.manual xxx blank:en",
    "spans.lunr.manual xxx blank:en"
]

@pytest.mark.parametrize("query", ["benchmark", "corpus"])
@pytest.mark.parametrize("base_call", base_calls)
@pytest.mark.e2e
def test_basic_interactions(query, base_call):
    """Ensure that we check e2e that the query appears when we reset the stream."""
    extra_settings = "tests/datasets/new-dataset.jsonl tests/datasets/index.gz.json --query download --allow-reset --n 100 --labels foo,bar,buz"
    with prodigy_playwright(f"{base_call} {extra_settings}") as (ctx, page):
        # Reset the stream
        page.get_by_text("Reset stream?").click()
        page.get_by_label("New query:").click()
        page.get_by_label("New query:").fill(query)
        page.get_by_role("button", name="Refresh Stream").click()
        page.get_by_text("Reset stream?").click()
        time.sleep(2.0)

        # Hit accept a few times, making sure that the query appears
        for _ in range(10):
            # We check the entire container because we're interested in the meta information.
            # The retreived text may not have a perfect match for the query, but the meta should!
            elem = page.locator(".prodigy-container").first
            print()
            print(elem.inner_text())
            assert query in elem.inner_text().lower()
            page.get_by_label("accept (a)").click()
            time.sleep(0.5)
