import json

import pytest

from smartscrape.baselines import make_baseline
from smartscrape.baselines.llm import build_prompt, html_to_text, parse_json_object
from smartscrape.evaluate import aggregate
from smartscrape.experiment import split_sites
from smartscrape.loaders import page_from_html
from smartscrape.schema import Schema
from smartscrape.solver import values_to_record, violations

TEMPLATE = """<html><body><div id="nav"><a>Home</a></div>
<div class="p"><h1>{title}</h1><div><span class="price">{price}</span></div><span>{brand}</span></div>
<div id="footer"><span>$5.00 shipping</span></div></body></html>"""

SCHEMA = Schema.from_dict({"name": "t", "fields": {
    "product_title": {"type": "text"}, "price": {"type": "price"},
    "manufacturer": {"type": "text", "cardinality": "optional"}},
    "constraints": [{"kind": "mutual_exclusion"}, {"kind": "format", "field": "price"}]})


def make_pages(tmp_path, n, site="s", template=TEMPLATE):
    pages = []
    for i in range(n):
        vals = {"title": f"Camera {i}", "price": f"${100 + i}.00", "brand": "Canon"}
        html = template.format(**vals)
        path = tmp_path / f"{site}_{i}.htm"
        path.write_text(html)
        gold = {"product_title": [vals["title"]], "price": [vals["price"]], "manufacturer": ["Canon"]}
        p = page_from_html(html, f"{site}_{i:04d}", site, gold)
        p.source = str(path)
        pages.append(p)
    return pages


def test_xpath_wrapper_learns_and_breaks_on_redesign(tmp_path):
    pages = make_pages(tmp_path, 6)
    b = make_baseline("xpath", SCHEMA).fit(pages[:3])
    for p in pages[3:]:
        assert b.extract(p)["price"] == p.gold["price"][0]
    # redesign: price moves into a table -> exact and generalised paths both miss
    redesigned = TEMPLATE.replace('<div><span class="price">{price}</span></div>',
                                  '<table><tr><td>{price}</td></tr></table>')
    new = make_pages(tmp_path, 2, site="s", template=redesigned)
    assert all(b.extract(p)["price"] != p.gold["price"][0] for p in new)


def test_llm_baseline_with_injected_model(tmp_path):
    pages = make_pages(tmp_path, 2)
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return json.dumps({"product_title": "Camera 0", "price": "$100.00", "manufacturer": "Nikon"}), 120, 30

    b = make_baseline("llm", SCHEMA, call=fake, price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    out = b.extract(pages[0])
    assert out["price"] == "$100.00" and "Camera 0" in seen["prompt"]
    assert b.stats.prompt_tokens == 120 and b.cost_usd() == pytest.approx((120 * 1 + 30 * 2) / 1e6)
    rec = values_to_record(pages[0], out, SCHEMA, "llm")
    assert rec.fields["price"].node is not None          # grounded: value occurs on the page
    assert rec.fields["manufacturer"].node is None       # "Nikon" is not on the page
    m = aggregate([pages[0]], [rec], SCHEMA)
    assert m["grounded_rate"] == pytest.approx(2 / 3)
    assert violations(pages[0], rec, SCHEMA) == []


def test_drifted_pages_are_hidden_from_baselines(tmp_path):
    from smartscrape.drift import apply_drift
    p = make_pages(tmp_path, 1)[0]
    q = apply_drift(p, ["tag_rename"], 1.0, seed=0)
    assert p.source and q.source == ""
    with pytest.raises(ValueError):
        make_baseline("xpath", SCHEMA).fit([p]).extract(q)


def test_helpers():
    assert parse_json_object('noise {"a": "1"} tail') == {"a": "1"}
    assert parse_json_object("no json") == {}
    t = html_to_text("<html><head><script>x()</script></head><body><p>Hi <b>there</b></p></body></html>", 100)
    assert "x()" not in t and "Hi" in t and "there" in t
    assert '"price" (price, required)' in build_prompt(SCHEMA, "PAGE")


def test_in_site_split(tmp_path):
    pages = make_pages(tmp_path, 10, "a") + make_pages(tmp_path, 10, "b")
    tr, va, te = split_sites(pages, {"type": "in_site", "train_per_site": 3, "val_per_site": 1}, 0)
    assert len(tr) == 6 and len(va) == 2 and len(te) == 12
    assert {p.site for p in te} == {"a", "b"}
    assert not ({p.page_id for p in tr} & {p.page_id for p in te})
