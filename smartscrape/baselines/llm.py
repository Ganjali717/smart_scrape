"""
LLM extraction baselines.

`llm` - our own minimal, transparent LLM extractor: the page is reduced to
visible text (or cleaned HTML), the schema is put in the prompt, the model
returns JSON. Backends:

    backend: ollama     local model on the GPU   (http://localhost:11434)
    backend: openai     any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio)
    backend: anthropic  Anthropic Messages API

Token counts come from the API response; cost is computed only from prices
you pass in the config (USD per 1M tokens, with the date you looked them up),
so the paper never relies on remembered prices.

`scrapegraphai` and `crawl4ai` - thin adapters around the two popular
libraries so the paper can also report them as shipped. Their APIs change
between versions; the adapters target the documented SmartScraperGraph and
LLMExtractionStrategy interfaces and raise BaselineUnavailable when missing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, Optional, Sequence

from ..data import Page
from .base import Baseline, BaselineUnavailable, read_html, register

_SKIP = ("script", "style", "noscript", "template", "svg", "head")


def html_to_text(html: str, max_chars: int) -> str:
    from lxml import html as lh

    root = lh.fromstring(html)
    for bad in root.xpath("|".join(f"//{t}" for t in _SKIP)):
        bad.drop_tree()
    lines = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for chunk in (el.text, el.tail):
            t = re.sub(r"\s+", " ", chunk or "").strip()
            if t:
                lines.append(t)
    return "\n".join(lines)[:max_chars]


def clean_html(html: str, max_chars: int) -> str:
    try:
        from lxml_html_clean import Cleaner          # lxml >= 5.2 moved the cleaner out
    except ImportError:
        from lxml.html.clean import Cleaner

    cleaned = Cleaner(scripts=True, javascript=True, style=True, comments=True, inline_style=True,
                      links=False, meta=True, page_structure=False, forms=False).clean_html(html)
    return re.sub(r"\s+", " ", cleaned)[:max_chars]


def build_prompt(schema, page_content: str) -> str:
    fields = "\n".join(f'- "{f.name}" ({f.type}, {"required" if f.cardinality == "one" else "optional"})'
                       for f in schema.fields)
    return (
        "Extract one record from the web page below.\n"
        f"Fields:\n{fields}\n\n"
        "Rules: copy each value exactly as it appears on the page; do not reformat numbers or prices; "
        "use an empty string when a field is not on the page. "
        "Answer with a single JSON object whose keys are exactly the field names, nothing else.\n\n"
        f"PAGE:\n{page_content}"
    )


def parse_json_object(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


@register("llm")
class LLMBaseline(Baseline):
    """options: backend, model, base_url, api_key_env, input ('text'|'html'), max_chars,
    temperature, timeout, price_in_per_mtok, price_out_per_mtok, price_date"""

    def __init__(self, schema, call: Optional[Callable[[str], tuple]] = None, **options):
        super().__init__(schema, **options)
        self._call = call  # injectable for tests: prompt -> (text, prompt_tokens, completion_tokens)

    def _extract(self, page: Page) -> Dict[str, str]:
        html = read_html(page)
        max_chars = int(self.options.get("max_chars", 24000))
        content = clean_html(html, max_chars) if self.options.get("input", "text") == "html" \
            else html_to_text(html, max_chars)
        text, pt, ct = (self._call or self._dispatch)(build_prompt(self.schema, content))
        self.stats.prompt_tokens += int(pt or 0)
        self.stats.completion_tokens += int(ct or 0)
        obj = parse_json_object(text)
        return {f: obj.get(f, "") if isinstance(obj.get(f, ""), str) else json.dumps(obj.get(f))
                for f in self.schema.field_names}

    def cost_usd(self) -> Optional[float]:
        pin, pout = self.options.get("price_in_per_mtok"), self.options.get("price_out_per_mtok")
        if pin is None or pout is None:
            return None
        return (self.stats.prompt_tokens * float(pin) + self.stats.completion_tokens * float(pout)) / 1e6

    # -- backends -------------------------------------------------------------
    def _dispatch(self, prompt: str):
        import requests

        b = self.options.get("backend", "ollama")
        model = self.options.get("model")
        timeout = float(self.options.get("timeout", 300))
        temp = float(self.options.get("temperature", 0.0))
        if not model:
            raise BaselineUnavailable("llm baseline needs options.model")
        if b == "ollama":
            url = self.options.get("base_url", "http://localhost:11434") + "/api/chat"
            r = requests.post(url, timeout=timeout, json={
                "model": model, "stream": False, "format": "json",
                "options": {"temperature": temp, "num_ctx": int(self.options.get("num_ctx", 16384))},
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            d = r.json()
            return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        if b == "openai":
            url = self.options.get("base_url", "https://api.openai.com/v1") + "/chat/completions"
            key = os.environ.get(self.options.get("api_key_env", "OPENAI_API_KEY"), "")
            r = requests.post(url, timeout=timeout, headers={"Authorization": f"Bearer {key}"}, json={
                "model": model, "temperature": temp, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if b == "anthropic":
            url = self.options.get("base_url", "https://api.anthropic.com") + "/v1/messages"
            key = os.environ.get(self.options.get("api_key_env", "ANTHROPIC_API_KEY"), "")
            r = requests.post(url, timeout=timeout, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={
                "model": model, "max_tokens": int(self.options.get("max_tokens", 1024)), "temperature": temp,
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
            return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
        raise BaselineUnavailable(f"unknown llm backend {b!r}")


@register("scrapegraphai")
class ScrapeGraphAIBaseline(Baseline):
    """options: graph_config (dict passed to SmartScraperGraph, e.g. {'llm': {'model': 'ollama/llama3.1'}})"""

    def _extract(self, page: Page) -> Dict[str, str]:
        try:
            from scrapegraphai.graphs import SmartScraperGraph
        except ImportError as e:
            raise BaselineUnavailable("pip install scrapegraphai && playwright install") from e
        fields = ", ".join(self.schema.field_names)
        g = SmartScraperGraph(
            prompt=f"Extract exactly these fields as a JSON object: {fields}. Copy values verbatim; "
                   f"use an empty string when a field is missing.",
            source=read_html(page), config=self.options.get("graph_config", {}))
        res = g.run() or {}
        res = res.get("content", res) if isinstance(res, dict) else {}
        return {f: str(res.get(f, "")) for f in self.schema.field_names} if isinstance(res, dict) else {}


@register("crawl4ai")
class Crawl4AIBaseline(Baseline):
    """options: provider (e.g. 'ollama/llama3.1' or 'openai/gpt-4o-mini'), api_token_env"""

    def _extract(self, page: Page) -> Dict[str, str]:
        try:
            import asyncio
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, LLMExtractionStrategy
        except ImportError as e:
            raise BaselineUnavailable("pip install crawl4ai && crawl4ai-setup") from e
        schema = {"type": "object",
                  "properties": {f: {"type": "string"} for f in self.schema.field_names},
                  "required": self.schema.field_names}
        token = os.environ.get(self.options.get("api_token_env", ""), None) if self.options.get("api_token_env") else None
        strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(provider=self.options.get("provider", "ollama/llama3.1"), api_token=token),
            schema=schema, extraction_type="schema",
            instruction="Extract one record; copy values verbatim; empty string when missing.")

        async def run():
            async with AsyncWebCrawler() as crawler:
                return await crawler.arun(url="raw:" + read_html(page),
                                          config=CrawlerRunConfig(extraction_strategy=strategy))
        res = asyncio.run(run())
        data = json.loads(res.extracted_content or "[]")
        obj = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        return {f: str(obj.get(f, "")) for f in self.schema.field_names}
