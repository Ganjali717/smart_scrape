"""
Template inference: site-level latent slots estimated by EM, label-free.

Pages of one site are rendered from one template, so for every field there is a
latent *slot* z_f (a structural position, see consensus.node_keys) that holds the
value on most pages. The page-level scorer gives, for page n, a distribution
q_n over the page's nodes plus "none". We model

    with probability pi   the value of page n sits in slot z_f   (template page)
    with probability 1-pi it does not                            (exception page)

and score each candidate slot by the log-likelihood of all pages of the site

    LL(z) = sum_n log( pi * Q_n(z) + (1 - pi) * u ),     Q_n(z) = sum_{i in slot z} q_n(i)

where u is the chance level of the exception hypothesis. The log makes the vote
consistency-seeking: a slot the scorer likes moderately on every page beats a slot
it loves on a few pages and ignores on the rest, which is exactly the failure of a
page-level argmax on an unseen template. pi is re-estimated by EM (it is the mean
responsibility of the template hypothesis), so the only constant is u.

Slots are assigned to fields jointly (Hungarian) so two fields never share a slot.
The page posterior

    p_n(i) = r_n * q_n(i) / Q_n(z)  [i in z]  +  (1 - r_n) * q_n(i),
    r_n    = pi Q_n(z) / (pi Q_n(z) + (1 - pi) u)

is turned back into log-odds against "none", so every decoder (filtered, ILP, soft
ILP) runs unchanged on top. The per-site quantities (pi, LL margin between the best
and second slot) are a site-level reliability signal: a site whose template no
longer explains the scorer's output is a site that changed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .consensus import Key, node_keys, site_anchors
from .data import Page
from .schema import Schema


def node_probs(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per field: softmax over nodes and 'none' (logit 0). Returns (P[n_nodes, F], P_none[F])."""
    if scores.shape[0] == 0:
        return np.zeros_like(scores), np.ones(scores.shape[1])
    m = np.maximum(scores.max(0), 0.0)
    e = np.exp(scores - m)
    none = np.exp(-m)
    z = e.sum(0) + none
    return e / z, none / z


@dataclass
class FieldTemplate:
    slot: Optional[Key]
    pi: float
    ll: float
    margin: float                      # LL(best) - LL(second best), per page
    candidates: List[Tuple[Key, float]] = field(default_factory=list)


@dataclass
class SiteTemplate:
    fields: Dict[str, FieldTemplate]
    n_pages: int

    def reliability(self) -> float:
        """Mean template adherence over fields that have a slot (higher = more trustworthy)."""
        pis = [t.pi for t in self.fields.values() if t.slot is not None]
        return float(np.mean(pis)) if pis else 0.0


class TemplateEM:
    def __init__(self, schema: Schema, u: float = 0.05, pi0: float = 0.8, iters: int = 5,
                 top_nodes: int = 20, max_slots: int = 30, exclusive: bool = True, alpha: float = 0.9,
                 ev_weight: float = 25.0):
        self.schema = schema
        self.alpha = alpha
        self.ev_weight = ev_weight      # one annotated page counts like this many scorer-only pages
        self.u, self.pi0, self.iters = u, pi0, iters
        self.top_nodes, self.max_slots, self.exclusive = top_nodes, max_slots, exclusive

    # ------------------------------------------------------------------ fit
    def fit_site(self, pages: Sequence[Page], scores: Sequence[np.ndarray],
                 keys: Optional[Sequence[List[Key]]] = None,
                 evidence: Optional[Dict[int, Dict[str, Optional[int]]]] = None):
        """evidence: {page index: {field: node index or None}} from any external annotator
        (an LLM run on a handful of pages, a human, a legacy wrapper). On those pages the
        scorer's distribution is replaced by alpha * one-hot + (1 - alpha) * scorer."""
        if keys is None:
            anchors = site_anchors(pages)
            keys = [node_keys(p, anchors) for p in pages]
        F = len(self.schema.field_names)
        probs = [node_probs(s) for s in scores]
        evidence = evidence or {}
        if evidence:
            probs = [self._with_evidence(P, Pn, evidence.get(n), s.shape[0]) for n, ((P, Pn), s)
                     in enumerate(zip(probs, scores))]
        # Q[f][slot] -> array over pages of slot mass
        N = len(pages)
        Q: List[Dict[Key, np.ndarray]] = [defaultdict(lambda: np.zeros(N)) for _ in range(F)]
        for n, ((P, _), ks, s) in enumerate(zip(probs, keys, scores)):
            if s.shape[0] == 0:
                continue
            for j in range(F):
                top = set(np.argsort(-s[:, j])[: self.top_nodes].tolist())
                top |= set(np.nonzero(P[:, j] > 0.5)[0].tolist())       # evidence nodes
                for i in top:
                    Q[j][ks[i]][n] += P[i, j]
        # candidate slots: largest total mass
        cand: List[List[Key]] = []
        for j in range(F):
            ranked = sorted(Q[j], key=lambda k: -Q[j][k].sum())[: self.max_slots]
            for n in evidence:                       # slots named by annotations are always candidates
                for k in Q[j]:
                    if Q[j][k][n] > 0.5 and k not in ranked:
                        ranked.append(k)
            cand.append(ranked)

        W = np.ones((F, N))
        for n, ev in evidence.items():
            for j, f in enumerate(self.schema.field_names):
                if ev and f in ev:
                    W[j, n] = self.ev_weight
        pis = np.full(F, self.pi0)
        chosen: List[Optional[Key]] = [None] * F
        lls: List[Dict[Key, float]] = [{} for _ in range(F)]
        for _ in range(self.iters):
            for j in range(F):
                lls[j] = {k: float((W[j] * np.log(pis[j] * Q[j][k] + (1 - pis[j]) * self.u)).sum())
                          for k in cand[j]}
            chosen = self._assign(lls, cand)
            new = pis.copy()
            for j, k in enumerate(chosen):
                if k is None:
                    continue
                q = Q[j][k]
                r = pis[j] * q / (pis[j] * q + (1 - pis[j]) * self.u)
                new[j] = float(np.clip(r.mean(), 1e-3, 1 - 1e-3))
            if np.allclose(new, pis, atol=1e-4):
                pis = new
                break
            pis = new
        fields = {}
        for j, f in enumerate(self.schema.field_names):
            k = chosen[j]
            srt = sorted(lls[j].values(), reverse=True)
            margin = (srt[0] - srt[1]) / max(1, N) if len(srt) > 1 else float("inf") if srt else 0.0
            fields[f] = FieldTemplate(k, float(pis[j]), lls[j].get(k, float("-inf")) if k else float("-inf"),
                                      margin, sorted(lls[j].items(), key=lambda kv: -kv[1])[:5])
        tpl = SiteTemplate(fields, N)
        return tpl, keys, Q, chosen

    def _with_evidence(self, P, Pn, ev, n_nodes):
        if not ev:
            return P, Pn
        P, Pn = P.copy(), Pn.copy()
        a = self.alpha
        for j, f in enumerate(self.schema.field_names):
            if f not in ev:
                continue
            P[:, j] *= (1 - a)
            Pn[j] *= (1 - a)
            if ev[f] is None:
                Pn[j] += a
            elif 0 <= ev[f] < n_nodes:
                P[ev[f], j] += a
        return P, Pn

    def _assign(self, lls: List[Dict[Key, float]], cand: List[List[Key]]) -> List[Optional[Key]]:
        F = len(lls)
        if not self.exclusive:
            return [max(ll, key=ll.get) if ll else None for ll in lls]
        from scipy.optimize import linear_sum_assignment
        slots = sorted({k for c in cand for k in c}, key=str)
        if not slots:
            return [None] * F
        idx = {k: b for b, k in enumerate(slots)}
        C = np.full((F, len(slots)), 1e12)
        for j, ll in enumerate(lls):
            for k, v in ll.items():
                C[j, idx[k]] = -v
        r, c = linear_sum_assignment(C)
        out: List[Optional[Key]] = [None] * F
        for j, b in zip(r, c):
            if C[j, b] < 1e11:
                out[j] = slots[b]
        return out

    # ------------------------------------------------------------ rescore
    def rescore(self, pages: Sequence[Page], scores: Sequence[np.ndarray],
                evidence: Optional[Dict[int, Dict[str, Optional[int]]]] = None):
        """Group by site, fit, return (new_scores, {site: SiteTemplate}, keys).
        evidence is keyed by the index in `pages`."""
        evidence = evidence or {}
        by_site: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(pages):
            by_site[p.site].append(i)
        out: List[Optional[np.ndarray]] = [None] * len(pages)
        all_keys: List[Optional[List[Key]]] = [None] * len(pages)
        templates: Dict[str, SiteTemplate] = {}
        for site, idx in by_site.items():
            ps = [pages[i] for i in idx]
            ss = [scores[i] for i in idx]
            ev = {local: evidence[i] for local, i in enumerate(idx) if i in evidence}
            tpl, keys, Q, chosen = self.fit_site(ps, ss, evidence=ev)
            templates[site] = tpl
            for local, i in enumerate(idx):
                out[i] = self._page_scores(ss[local], keys[local], chosen, tpl, Q, local)
                all_keys[i] = keys[local]
        return out, templates, all_keys

    def _page_scores(self, s: np.ndarray, ks: List[Key], chosen, tpl: SiteTemplate, Q, n: int) -> np.ndarray:
        if s.shape[0] == 0:
            return s
        P, Pn = node_probs(s)
        new = s.copy()
        for j, f in enumerate(self.schema.field_names):
            k = chosen[j]
            if k is None:
                continue
            pi = tpl.fields[f].pi
            q = Q[j][k][n]
            mask = np.fromiter((kk == k for kk in ks), bool, len(ks))
            qs = P[mask, j].sum()                 # all slot nodes, not only the top ones
            if qs <= 0:
                continue                          # slot absent on this page: keep the scorer's view
            r = pi * q / (pi * q + (1 - pi) * self.u)
            post = (1 - r) * P[:, j]
            post[mask] += r * P[mask, j] / qs
            post_none = (1 - r) * Pn[j]
            new[:, j] = np.log(np.maximum(post, 1e-12)) - np.log(max(post_none, 1e-12))
        return new
