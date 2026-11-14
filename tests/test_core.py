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


# --- FitLayout XML ---------------------------------------------------------
FL_XML = """<?xml version="1.0"?>
<page>
<box tagName="body" positionX="0" positionY="0" width="1200" height="1600" contentX="0" contentY="0" contentWidth="1200" contentHeight="1600" visualX="0" visualY="0" visualWidth="1200" visualHeight="1600" color="#000000" bgcolor="#ffffff" fontfamily="Arial" fontsize="16.0" fontweight="0.4" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">
    <box tagName="nav" positionX="0" positionY="0" width="1200" height="40" contentX="0" contentY="0" contentWidth="1200" contentHeight="40" visualX="0" visualY="0" visualWidth="1200" visualHeight="40" color="#000000" bgcolor="#eeeeee" fontfamily="Arial" fontsize="14.0" fontweight="0.4" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">
        <box tagName="#text" positionX="10" positionY="10" width="60" height="20" contentX="10" contentY="10" contentWidth="60" contentHeight="20" visualX="10" visualY="10" visualWidth="60" visualHeight="20" color="#000000" bgcolor="" fontfamily="Arial" fontsize="14.0" fontweight="0.4" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">Home</box>
    </box>
    <box tagName="h1" positionX="20" positionY="80" width="600" height="40" contentX="20" contentY="80" contentWidth="600" contentHeight="40" visualX="20" visualY="80" visualWidth="600" visualHeight="40" color="#000000" bgcolor="" fontfamily="Arial" fontsize="32.0" fontweight="0.7" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">
        <box tagName="#text" positionX="20" positionY="80" width="400" height="40" contentX="20" contentY="80" contentWidth="400" contentHeight="40" visualX="20" visualY="80" visualWidth="400" visualHeight="40" color="#000000" bgcolor="" fontfamily="Arial" fontsize="32.0" fontweight="0.7" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">Canon PowerShot S90</box>
    </box>
    <box tagName="span" positionX="20" positionY="140" width="100" height="24" contentX="20" contentY="140" contentWidth="100" contentHeight="24" visualX="20" visualY="140" visualWidth="100" visualHeight="24" color="#cc0000" bgcolor="" fontfamily="Arial" fontsize="20.0" fontweight="0.7" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">
        <box tagName="#text" positionX="20" positionY="140" width="80" height="24" contentX="20" contentY="140" contentWidth="80" contentHeight="24" visualX="20" visualY="140" visualWidth="80" visualHeight="24" color="#cc0000" bgcolor="" fontfamily="Arial" fontsize="20.0" fontweight="0.7" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">$129.00</box>
    </box>
    <box tagName="footer" positionX="0" positionY="1500" width="1200" height="100" contentX="0" contentY="1500" contentWidth="1200" contentHeight="100" visualX="0" visualY="1500" visualWidth="1200" visualHeight="100" color="#000000" bgcolor="" fontfamily="Arial" fontsize="12.0" fontweight="0.4" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">
        <box tagName="#text" positionX="10" positionY="1510" width="200" height="16" contentX="10" contentY="1510" contentWidth="200" contentHeight="16" visualX="10" visualY="1510" visualWidth="200" visualHeight="16" color="#000000" bgcolor="" fontfamily="Arial" fontsize="12.0" fontweight="0.4" fontstyle="0.0" underline="0.0" linethrough="0.0" replaced="false">Terms &amp;amp;gt; Privacy</box>
    </box>
</box>
</page>"""


def test_parse_fitlayout_xml():
    from smartscrape.render import parse_fitlayout_xml
    from smartscrape.features import FeatureEncoder
    from smartscrape.graph import build_edges
    p = parse_fitlayout_xml(FL_XML, "fl0", "site", dict(GOLD))
    texts = [n.text for n in p.nodes]
    assert texts[:3] == ["Home", "Canon PowerShot S90", "$129.00"]
    assert p.has_geometry()
    title = p.nodes[1]
    assert title.tag == "h1" and title.bbox == [20.0, 80.0, 400.0, 40.0] and title.font_size == 32.0
    assert "nav" in p.nodes[0].regions and "footer" in p.nodes[3].regions
    assert p.labels[1] == "product_title" and p.labels[2] == "price"
    assert p.position(3) > 0.9
    x = FeatureEncoder().encode(p)
    assert x.shape[0] == p.n and x[:, -6].sum() == p.n      # geometry flag set for every node
    ei, et = build_edges(p)
    assert 3 in set(et.tolist())                             # spatial kNN edges exist


def test_prepare_offline_html():
    from smartscrape.render import prepare_offline_html
    h = ('<base href="http://x.com/"><link rel="stylesheet" href="http://x.com/a.css">'
         '<script>alert(1)</script><img src="http://x.com/i.png" width="10"><p>ok</p>')
    out = prepare_offline_html(h)
    assert "<base" not in out and "a.css" not in out and "alert" not in out
    assert 'src=""' in out and "<p>ok</p>" in out


def test_soft_constraints_trade_off():
    p, s = page(), schema()
    # the related-block price outscores the main one by 1.0; proximity to the title rejects it
    sc = scores_for(p, {"product_title": {"Canon PowerShot S90": 2.0},
                        "price": {"$99.00": 3.0, "$129.00": 2.0}})
    assert ConstraintSolver(s).solve(p, sc).fields["price"].text == "$129.00"            # hard
    assert ConstraintSolver(s, soft_lambda=5.0).solve(p, sc).fields["price"].text == "$129.00"   # penalty > gain
    soft = ConstraintSolver(s, soft_lambda=0.5).solve(p, sc)                              # penalty < gain
    assert soft.fields["price"].text == "$99.00" and any("soft_violations=1" in a for a in soft.active)


def test_null_gold_and_isbn():
    from smartscrape.loaders import _parse_gt
    from smartscrape.formats import matches_type
    assert _parse_gt("{'engine': ['<NULL>'], 'price': ['$9,970']}") == {"price": ["$9,970"]}
    assert matches_type(": 9780316125581", "isbn") and matches_type("978-0-312-60539-1", "isbn")
    assert not matches_type("Lisa Kleypas", "isbn") and not matches_type("12345", "isbn")


def test_calibration_drops_format_that_rejects_gold():
    from smartscrape.calibrate import calibrate
    p = page()
    p.gold["price"] = ["Call for price"]          # gold value that is not a price
    p.nodes[idx(p, "$129.00")].text = "Call for price"
    from smartscrape.data import assign_labels
    assign_labels(p)
    cal, rep = calibrate(schema(), [p])
    assert not any(c.kind == "format" for c in cal.constraints)
