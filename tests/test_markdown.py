from hyrag.loader import load_document, strip_markdown_inline

from .conftest import CORPUS


def md(text: str, name: str = "doc.md"):
    return load_document(name, text.encode("utf-8"))


def test_heading_path_is_nested():
    doc = md("# Guide\n\nintro\n\n## Troubleshooting\n\n### Error X\n\nfix it\n")
    assert [s.heading for s in doc.sections] == ["Guide", "Guide > Troubleshooting > Error X"]


def test_hash_inside_backtick_fence_is_not_a_heading():
    doc = md("# Guide\n\n```\n# just a comment\nrun --it\n```\n")
    assert [s.heading for s in doc.sections] == ["Guide"]
    assert "# just a comment" in doc.sections[0].text


def test_tilde_fence_is_code_too():
    doc = md("# Guide\n\ntext\n\n~~~bash\n# install the agent\napt install x\n~~~\n\nmore\n")
    assert [s.heading for s in doc.sections] == ["Guide"]
    assert "# install the agent" in doc.sections[0].text


def test_fence_only_closes_on_same_character():
    doc = md("# Guide\n\n~~~markdown\n```\n# still code\n```\n~~~\n\n## Real\n\nx\n")
    assert [s.heading for s in doc.sections] == ["Guide", "Guide > Real"]
    assert "```\n# still code\n```" in doc.sections[0].text


def test_code_indentation_is_kept():
    doc = md("# Config\n\n```yaml\nserver:\n    port: 8080\n        tls: true\n```\n")
    assert "    port: 8080\n        tls: true" in doc.sections[0].text


def test_inline_markup_is_stripped_but_identifiers_survive():
    assert strip_markdown_inline("See the [IT portal](https://x) for *details*.") == "See the IT portal for details."
    assert strip_markdown_inline("ERR_TUNNEL_4012 and max_retry_count") == "ERR_TUNNEL_4012 and max_retry_count"
    assert strip_markdown_inline("Multiply 3 * 4 * 5") == "Multiply 3 * 4 * 5"
    assert strip_markdown_inline("- **Step 1**: run `x --y`") == "- Step 1: run x --y"


def test_utf8_bom_does_not_hide_first_heading():
    doc = load_document("bom.md", "﻿# Setup Guide\n\nInstall it.".encode("utf-8"))
    assert doc.sections[0].heading == "Setup Guide"


def test_front_matter_title_becomes_root_heading_and_yaml_is_removed():
    doc = md("---\ntitle: Secrets\ncontent_type: concept\n---\n\nIntro.\n\n## Types {#types}\n\nOpaque.\n")
    assert [s.heading for s in doc.sections] == ["Secrets", "Secrets > Types"]
    assert "content_type" not in doc.sections[0].text


def test_hugo_shortcodes_render_to_their_text():
    text = (
        "---\ntitle: T\n---\n\nScheduled onto a {{< glossary_tooltip text=\"node\" term_id=\"node\" >}}.\n"
        "Runs a {{< glossary_tooltip term_id=\"container-runtime\" >}}.\n"
        "{{< note >}}\nBe careful.\n{{< /note >}}\n"
        "{{< feature-state feature_gate_name=\"RestartRules\" >}}\n"
        "{{% code_sample file=\"pods/x.yaml\" %}}\n"
        "Version {{< param \"version\" >}} ok.\n"
        "## {{% heading \"whatsnext\" %}}\n\nRead more.\n"
    )
    doc = md(text)
    body = doc.sections[0].text
    assert "Scheduled onto a node." in body
    assert "Runs a container runtime." in body
    assert "Note:" in body and "{{" not in body
    assert "Feature gate: RestartRules" in body
    assert "(example file: pods/x.yaml)" in body
    assert doc.sections[-1].heading == "T > What's next"


def test_empty_html_anchor_removed_but_command_placeholders_kept():
    doc = md('# T\n\n<a id="x" />\nkubectl describe pod <name-of-pod>\n')
    assert "<a id" not in doc.sections[0].text
    assert "<name-of-pod>" in doc.sections[0].text


def test_windows_line_endings_parse_like_unix():
    raw = (CORPUS / "engineering" / "k8s-secrets.md").read_bytes().replace(b"\r\n", b"\n")
    lf = load_document("k8s-secrets.md", raw)
    crlf = load_document("k8s-secrets.md", raw.replace(b"\n", b"\r\n"))
    cr = load_document("k8s-secrets.md", raw.replace(b"\n", b"\r"))
    assert lf == crlf == cr


def test_html_comments_are_removed_but_not_inside_code():
    doc = md("# T\n\n<!-- body -->\nKeep this.\nAlso <!-- TODO: fix -->this.\n<!-- a\nmulti-line\nnote -->\n"
             "After.\n\n```html\n<!-- example comment -->\n<p>x</p>\n```\n")
    text = doc.sections[0].text
    assert "body" not in text and "TODO" not in text and "multi-line" not in text
    assert "Keep this.\nAlso this.\nAfter." in text
    assert "<!-- example comment -->" in text


def test_real_kubernetes_doc_has_no_markup_leftovers():
    doc = load_document("k8s-secrets.md", (CORPUS / "engineering" / "k8s-secrets.md").read_bytes())
    text = "\n".join(s.text for s in doc.sections)
    assert doc.sections[0].heading == "Secrets"
    assert "{{" not in text and "content_type:" not in text and "<!--" not in text
    assert not any("{#" in s.heading for s in doc.sections)
