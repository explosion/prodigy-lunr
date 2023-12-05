import time 
import pytest 
from playwright.sync_api import expect

from .e2e_util import prodigy_playwright


base_calls = [
    "textcat.lunr.manual xxx",
    "ner.lunr.manual xxx blank:en",
    "spans.lunr.manual xxx blank:en"
]

@pytest.mark.parametrize("query", ["benchmark", "corpus"])
@pytest.mark.parametrize("base_call", base_calls)
def test_basic_interactions(query, base_call):
    extra_settings = "tests/datasets/new-dataset.jsonl tests/datasets/index.gz.json --query download --allow-reset --n 20 --labels foo,bar,buz"
    with prodigy_playwright(f"{base_call} {extra_settings}") as (ctx, page):
        page.get_by_text("Reset stream?").click()
        page.get_by_label("New query:").click()
        page.get_by_label("New query:").fill(query)
        page.get_by_role("button", name="Refresh Stream").click()
        page.get_by_text("Reset stream?").click()
        elem = page.locator(".prodigy-content").first
        expect(elem).to_contain_text(query)
