"""What the three Sprint 3 labelling exports share (Story 3.6, DECISIONS S3-06, S3-10).

- draw / group_sizes: a deterministic stratified sample (hash order, never rand()) and
  the size of each group in the crawl, the weights of the final numbers.
- render_page: one self-contained, blind labelling page. One card at a time, answer
  buttons (keys 1-9), answers kept in the browser, exported as card_id,label CSV.
Never published: the cards hold text from third-party websites.
"""

import html
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import NamedTuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SAMPLE_SEED = 42  # fixed: the same sample on every run (as in S2-08)


def draw(
    grouped: DataFrame,
    quotas: Mapping[str, int],
    seed: int = SAMPLE_SEED,
    id_column: str = "doc_id",
) -> DataFrame:
    """(id, stratum) of the items to label: the first `quota` items of each group.

    "First" in the order of xxhash64(id, seed), the id breaking ties: deterministic and
    spread evenly over the crawl, never rand() (CLAUDE.md section 8). The quota comes from
    the group's kind, the part of its name after the last "_" (ar_low -> low, pii_caught
    -> caught). A group with fewer items than its quota gives all of them.
    """
    order = Window.partitionBy("stratum").orderBy(F.xxhash64(id_column, F.lit(seed)), id_column)
    kind = F.substring_index("stratum", "_", -1)
    quota = F.create_map(*[F.lit(x) for x in chain.from_iterable(quotas.items())])[kind]
    return (
        grouped.where(F.col("stratum").isNotNull())
        .select(id_column, "stratum", F.row_number().over(order).alias("rank"))
        .where(F.col("rank") <= quota)
        .drop("rank")
    )


def group_sizes(grouped: DataFrame) -> dict[str, int]:
    """How many items of the crawl each group holds: the weights for the results.
    A handful of rows come back to the driver."""
    rows = grouped.where(F.col("stratum").isNotNull()).groupBy("stratum").count().collect()
    return {row.stratum: row["count"] for row in rows}


class Card(NamedTuple):
    card_id: str  # what the exported CSV calls the item
    body_html: str  # already safe: built with html.escape by the caller
    meta: str  # plain text above the body, escaped here


_PAGE_CSS = """
body { font: 16px/1.55 system-ui, sans-serif; max-width: 52rem; margin: 1.5rem auto;
       padding: 0 1rem; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.25rem; margin: 0 0 .4rem; }
.intro { color: #333; font-size: .92rem; }
.bar { position: sticky; top: 0; background: #fafafa; padding: .6rem 0; z-index: 1;
       border-bottom: 1px solid #ddd; }
.bar button { font-size: 1rem; padding: .45rem .8rem; margin: .2rem .25rem .2rem 0;
              border: 1px solid #999; border-radius: 5px; background: #fff; cursor: pointer; }
.bar button.chosen { background: #1565c0; color: #fff; border-color: #1565c0; }
.card { display: none; background: #fff; border: 1px solid #ddd; border-radius: 6px;
        padding: 1rem; margin-top: 1rem; }
.card.current { display: block; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
.meta { color: #666; font-size: .85rem; margin-bottom: .5rem; }
.cut { color: #999; font-style: italic; }
mark { background: #ffe082; padding: 0 2px; border-radius: 2px; }
"""

# Shows one card at a time; number keys answer and move on, Backspace goes back. Answers
# live in the browser's localStorage under the page's own key, so closing the page loses
# nothing. Export downloads card_id,label as CSV: ids and labels only, never text.
_PAGE_SCRIPT = r"""
const cards = Array.from(document.querySelectorAll('.card'));
const labels = Array.from(document.querySelectorAll('.bar button[data-label]'));
const key = document.body.dataset.key;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { answers = {}; }
let at = cards.findIndex(c => !(c.dataset.id in answers));
if (at < 0) at = 0;
function save() { try { localStorage.setItem(key, JSON.stringify(answers)); } catch (e) {} }
function show() {
  cards.forEach((c, i) => c.classList.toggle('current', i === at));
  const id = cards[at].dataset.id;
  labels.forEach(b => b.classList.toggle('chosen', answers[id] === b.dataset.label));
  const done = cards.filter(c => c.dataset.id in answers).length;
  document.getElementById('progress').textContent =
    'Card ' + (at + 1) + ' of ' + cards.length + ' - ' + done + ' labelled';
}
function answer(label) {
  answers[cards[at].dataset.id] = label; save();
  if (at < cards.length - 1) at += 1;
  show();
}
labels.forEach(b => b.addEventListener('click', () => answer(b.dataset.label)));
document.getElementById('back').addEventListener('click', () => { if (at > 0) at -= 1; show(); });
document.getElementById('next').addEventListener('click', () => {
  if (at < cards.length - 1) at += 1; show(); });
document.addEventListener('keydown', e => {
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= labels.length) answer(labels[n - 1].dataset.label);
  else if (e.key === 'Backspace') { e.preventDefault(); if (at > 0) at -= 1; show(); }
});
document.getElementById('export').addEventListener('click', () => {
  const rows = ['card_id,label'].concat(
    cards.filter(c => c.dataset.id in answers)
      .map(c => c.dataset.id + ',' + answers[c.dataset.id]));
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([rows.join('\n') + '\n'], {type: 'text/csv'}));
  link.download = document.body.dataset.export;
  link.click();
});
show();
"""


def render_page(
    *,
    title: str,
    intro_html: str,
    storage_key: str,
    export_name: str,
    labels: Sequence[tuple[str, str]],
    cards: Sequence[Card],
) -> str:
    """One self-contained labelling page. `labels`: (value stored, button text) pairs,
    in key order (1, 2, ...). `intro_html` is the page's own static text (trusted)."""
    buttons = "".join(
        f"<button data-label='{html.escape(value, quote=True)}'>{number} &middot; "
        f"{html.escape(text)}</button>"
        for number, (value, text) in enumerate(labels, start=1)
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_PAGE_CSS}</style></head>",
        f"<body data-key='{html.escape(storage_key, quote=True)}' "
        f"data-export='{html.escape(export_name, quote=True)}'>",
        f"<h1>{html.escape(title)}: {len(cards)} cards</h1>",
        f"<div class='intro'>{intro_html}</div>",
        "<div class='bar'>",
        buttons,
        "<br><button id='back'>&larr; Back</button><button id='next'>Skip &rarr;</button>",
        "<button id='export'>Export labels (CSV)</button> <span id='progress'></span>",
        "</div>",
    ]
    for card in cards:
        parts.append(f"<div class='card' data-id='{html.escape(card.card_id, quote=True)}'>")
        parts.append(f"<div class='meta'>{html.escape(card.meta)}</div>")
        parts.append(card.body_html)
        parts.append("</div>")
    parts.append(f"<script>{_PAGE_SCRIPT}</script></body></html>")
    return "\n".join(parts)


def text_body(text: str, shown_characters: int) -> str:
    """A page's text as a card body: escaped (a crawled "<script>" shows, never runs),
    right-to-left where needed (dir=auto), cut after `shown_characters` with a note."""
    shown = text[:shown_characters]
    body = f"<div class='text' dir='auto'>{html.escape(shown)}</div>"
    if len(text) > len(shown):
        body += f"<div class='cut'>... {len(text) - len(shown):,} more characters not shown</div>"
    return body
