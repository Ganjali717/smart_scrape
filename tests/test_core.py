import numpy as np
import pytest

from smartscrape.data import Node, Page, assign_labels, text_matches
from smartscrape.drift import PERTURBATIONS, apply_drift
from smartscrape.evaluate import aggregate
from smartscrape.loaders import page_from_flat_nodes, page_from_html
from smartscrape.schema import Schema
from smartscrape.solver import ConstraintSolver, decode_argmax, decode_filtered, violations

HTML = """<html><body>
<div id="nav"><a>Home</a><a>Cameras</a></div>
<div class="product"><h1>Canon PowerShot S90</h1><span class="price">$129.00</span><span>Canon</span></div>
<div class="related"><a>Nikon Coolpix</a><span>$99.00</span></div>
<div id="footer"><span>$5.00 shipping</span><span>Contact</span></div>
</body></html>"""

GOLD = {"product_title": ["Canon PowerShot S90"], "price": ["$129.00"], "manufacturer": ["Canon"]}


def schema(constraints=None):
    return Schema.from_dict({
        "name": "t",
        "fields": {"product_title": {"type": "text"}, "price": {"type": "price"},
                   "manufacturer": {"type": "text", "cardinality": "optional"}},
        "constraints": constraints if constraints is not None else [
            {"kind": "mutual_exclusion", "rank": 10},
            {"kind": "format", "field": "price", "rank": 8},
            {"kind": "exclude_regions", "regions": ["footer", "nav"], "fields": "all", "rank": 3},
            {"kind": "proximity", "a": "price", "b": "product_title", "max_distance": 3, "rank": 2},
        ],
    })


def page():
    return page_from_html(HTML, "p0", "site", dict(GOLD))


def idx(p, text):
    return next(n.id for n in p.nodes if n.text == text)


# --- loader ---------------------------------------------------------------
def test_html_loader_regions_and_labels():
    p = page()
    assert [n.text for n in p.nodes][:2] == ["Home", "Cameras"]
    assert "nav" in p.nodes[idx(p, "Home")].regions
    assert "footer" in p.nodes[idx(p, "$5.00 shipping")].regions
    assert "aside" in p.nodes[idx(p, "Nikon Coolpix")].regions
    assert p.labels[idx(p, "$129.00")] == "price"
    assert p.labels[idx(p, "Canon PowerShot S90")] == "product_title"


def test_text_matching_normalises():
    assert text_matches("  Canon&amp;Co ", ["canon&co"])
    assert not text_matches("", ["x"])


def test_flat_loader_containment_parent():
    nodes = [{"text": "Box", "tag": "div", "bbox": [0, 0, 500, 500]},
             {"text": "Inner", "tag": "span", "bbox": [10, 10, 50, 20]}]
    p = page_from_flat_nodes(nodes, "f", "s")
    assert p.nodes[1].parent == 0


# --- solver ---------------------------------------------------------------
def scores_for(p, prefs):
    """prefs: {field: {text: score}}; everything else gets -5."""
    s = schema()
    m = np.full((p.n, len(s.field_names)), -5.0)
    for j, f in enumerate(s.field_names):
        for text, v in prefs.get(f, {}).items():
            m[idx(p, text), j] = v
    return m


def test_ilp_respects_regions_where_argmax_fails():
    p, s = page(), schema()
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0},
                        "price": {"$5.00 shipping": 3.0, "$129.00": 2.0}})
    assert decode_argmax(p, sc, s).fields["price"].text == "$5.00 shipping"
    rec = ConstraintSolver(s).solve(p, sc)
    assert rec.fields["price"].text == "$129.00"
    assert violations(p, rec, s) == []


def test_ilp_cross_field_proximity_beats_filtering():
    p, s = page(), schema()
    # related-block price outscores the main price; it is not in a filtered region? it is 'aside',
    # which this schema does not exclude, so only proximity to the title can reject it.
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0},
                        "price": {"$99.00": 3.0, "$129.00": 2.0}})
    assert decode_filtered(p, sc, s).fields["price"].text == "$99.00"
    assert ConstraintSolver(s).solve(p, sc).fields["price"].text == "$129.00"


def test_mutual_exclusion_and_null_option():
    p, s = page(), schema()
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0, "Canon": 1.0},
                        "price": {"$129.00": 2.0},
                        "manufacturer": {"Canon PowerShot S90": 2.5}})
    rec = ConstraintSolver(s).solve(p, sc)
    nodes = [n for n in rec.nodes().values() if n is not None]
    assert len(nodes) == len(set(nodes))
    # optional field with only negative scores stays empty
    sc2 = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0}, "price": {"$129.00": 2.0}})
    assert ConstraintSolver(s).solve(p, sc2).fields["manufacturer"].node is None


def test_relaxation_when_infeasible():
    p = page()
    s = schema([{"kind": "position_band", "field": "product_title", "max": -1.0, "rank": 1},
                {"kind": "format", "field": "price", "rank": 8}])
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0}, "price": {"$129.00": 2.0}})
    rec = ConstraintSolver(s).solve(p, sc)
    assert rec.relaxed and rec.fields["product_title"].text == "Canon PowerShot S90"


# --- drift ----------------------------------------------------------------
@pytest.mark.parametrize("kind", list(PERTURBATIONS))
def test_drift_preserves_record(kind):
    p = page()
    q = apply_drift(p, [kind], 1.0, seed=1, pool_titles=["Sony Cyber-shot", "Canon PowerShot S90"])
    assert q.gold == p.gold
    for i, f in q.labels.items():
        assert text_matches(q.nodes[i].text, q.gold[f])
    assert sorted(set(q.labels.values())) == sorted(set(p.labels.values()))
    assert [n.id for n in q.nodes] == list(range(q.n))
    assert all(-1 <= n.parent < q.n for n in q.nodes)


def test_distractor_adds_unlabelled_lookalikes():
    p = page()
    q = apply_drift(p, ["distractor_block"], 1.0, seed=3, pool_titles=["Sony Cyber-shot"])
    assert q.n > p.n
    assert len(q.labels) == len(p.labels)


# --- metrics --------------------------------------------------------------
def test_aggregate_perfect_record():
    p, s = page(), schema()
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0}, "price": {"$129.00": 2.0},
                        "manufacturer": {"Canon": 2.0}})
    m = aggregate([p], [ConstraintSolver(s).solve(p, sc)], s)
    assert m["f1/macro"] == pytest.approx(1.0) and m["exact_match"] == 1.0 and m["violation_rate"] == 0.0


def test_schema_roundtrip(tmp_path):
    from smartscrape.schema import save_schema
    s = schema()
    path = tmp_path / "s.yaml"
    save_schema(s, str(path))
    s2 = Schema.load(str(path))
    assert s2.field_names == s.field_names
    assert [c.name for c in s2.constraints] == [c.name for c in s.constraints]
