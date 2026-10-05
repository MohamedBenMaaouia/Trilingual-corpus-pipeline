"""corpus.gold.exclusions: the hand-kept list of sites left out of gold (S6-05)."""

import pytest
from pyspark.sql import SparkSession

from corpus.gold.exclusions import exclude_sites, load_exclusions, parse_exclusions

LISTED = """
# Removal requests
Example.com
alice.blogspot.com   # request 3
example.com
"""


def test_the_list_is_lowercased_sorted_and_each_site_kept_once() -> None:
    assert parse_exclusions(LISTED) == ("alice.blogspot.com", "example.com")


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("https://example.com/page", "not a site name"),  # a pasted URL
        ("example.com:8080", "not a site name"),
        ("exa mple.com", "not a site name"),
        ("com", "not a site name"),  # one label
        ("-bad.example.com", "not a site name"),
        ("co.uk", "public suffix"),  # would remove every .co.uk site
    ],
)
def test_a_bad_line_stops_the_run(line: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_exclusions("example.org" + chr(10) + line)


def test_the_error_names_the_line() -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_exclusions("example.org" + chr(10) + "https://example.com")


def test_the_shipped_list_is_valid() -> None:
    """A bad line in the shipped file would stop every gold run."""
    assert all("." in site for site in load_exclusions())


def test_listed_sites_and_their_subdomains_are_removed(spark: SparkSession) -> None:
    pages = spark.createDataFrame(
        [
            ("a", "https://example.com/x", "example.com"),
            ("b", "http://WWW.Example.com/y", "example.com"),  # host case does not matter
            ("c", "http://notexample.com/", "notexample.com"),  # same letters, other site
            ("d", "http://alice.blogspot.com/p", "blogspot.com"),  # a listed blog
            ("e", "http://bob.blogspot.com/p", "blogspot.com"),  # its neighbour is not
            ("f", "http://example.org/", "example.org"),
            ("g", "not a url", "example.com"),  # no host: the page's domain decides
        ],
        "doc_id string, url string, domain string",
    )
    kept = exclude_sites(pages, ("alice.blogspot.com", "example.com"))
    assert kept.columns == pages.columns
    assert sorted(r.doc_id for r in kept.collect()) == ["c", "e", "f"]


def test_an_empty_list_removes_nothing(spark: SparkSession) -> None:
    pages = spark.createDataFrame(
        [("a", "https://example.com/x", "example.com")], "doc_id string, url string, domain string"
    )
    assert exclude_sites(pages, ()).count() == 1
