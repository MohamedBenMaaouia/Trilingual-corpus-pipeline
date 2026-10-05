"""Gold's exclusion list (Story 6.4.1, D16; DECISIONS S6-05): sites that asked to be
left out, removed before the write (an anti-join before the write: C18).

The list is package data, corpus/config/exclusions.txt: it ships in the wheel, so the
driver reads it the same way locally and on Databricks, and Spark sends it to the
executors with the join (broadcast). Every change to it is a commit: the git history
is the record of removal requests. A change to the list is data, not transform logic:
it does not bump pipeline_version (invariant 9); the next gold run applies it.
"""

import re
from collections.abc import Sequence
from importlib import resources

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from tldextract import TLDExtract

LIST_PACKAGE, LIST_FILE = "corpus.config", "exclusions.txt"

# A host name: two or more dot-separated labels of letters, digits and inner hyphens.
_SITE = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?([.][a-z0-9]([a-z0-9-]*[a-z0-9])?)+")
# The Public Suffix List snapshot bundled with tldextract, never downloaded (as in
# silver.parse): it tells a site (bbc.co.uk) from a bare suffix (co.uk).
_SUFFIXES = TLDExtract(suffix_list_urls=(), cache_dir=None)


def parse_exclusions(text: str) -> tuple[str, ...]:
    """The listed sites, lowercased, sorted, each once. A line that is not a plain site
    name raises: a typo must stop the run, never let the site through silently. A bare
    suffix (com, co.uk) raises too: it would remove every site under it."""
    sites = set()
    for number, line in enumerate(text.splitlines(), start=1):
        entry = line.split("#", 1)[0].strip().lower()
        if not entry:
            continue
        if not _SITE.fullmatch(entry):
            raise ValueError(
                f"{LIST_FILE} line {number}: {line.strip()!r} is not a site name "
                "(expected e.g. example.com: no scheme, path, port or spaces)"
            )
        if not _SUFFIXES(entry).domain:
            raise ValueError(
                f"{LIST_FILE} line {number}: {entry!r} is a public suffix, not a site: "
                "it would remove every site under it"
            )
        sites.add(entry)
    return tuple(sorted(sites))


def load_exclusions() -> tuple[str, ...]:
    """The list shipped with the package (read on the driver)."""
    path = resources.files(LIST_PACKAGE) / LIST_FILE
    return parse_exclusions(path.read_text(encoding="utf-8"))


def exclude_sites(pages: DataFrame, sites: Sequence[str]) -> DataFrame:
    """`pages` (with url and domain) minus every page of a listed site: a left anti-join
    on the url's host, which matches a listed site or ends with "." + the site.

    The list is broadcast, so the pages are never shuffled. A suffix match is not an
    equality, so Spark runs it as a broadcast nested loop join (each page against each
    listed site): cheap for a hand-kept list; past tens of thousands of sites, join on
    the host's suffixes instead (S6-05)."""
    listed = pages.sparkSession.createDataFrame([(site,) for site in sites], "site string")
    host = F.lower(F.expr("trim(TRAILING '.' FROM parse_url(url, 'HOST'))"))
    pages = pages.withColumn("_host", F.coalesce(host, F.col("domain")))  # domain: never null
    site, page_host = F.col("site"), F.col("_host")
    matches = (page_host == site) | page_host.endswith(F.concat(F.lit("."), site))
    return pages.join(F.broadcast(listed), matches, "left_anti").drop("_host")
