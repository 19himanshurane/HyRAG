"""MDX/JSX pages (PostHog's handbook): components become the words a reader sees, or nothing."""
from hyrag.loader import load_document, strip_components

PAGE = """---
title: Time off
sidebar: Handbook
---

import { CalloutBox } from 'components/Docs/CalloutBox'
import SOC2 from './_snippets/soc2.mdx'

{/* editors: keep this table up to date */}

We offer unlimited time off, with at least 25 days a year. Ask <TeamMember name="Lottie Coxon" photo /> or the <SmallTeam slug="people" />.

<CalloutBox icon="IconInfo" title="Book it in Deel" type="fyi">
Use the Time Off by Deel app in <PrivateLink url="https://posthog.slack.com/archives/C01">Slack</PrivateLink>.
</CalloutBox>

<ProductScreenshot
  imageLight="https://example.com/light.png"
  imageDark="https://example.com/dark.png"
/>

<details><summary>Parental leave</summary>
<p><strong>Maternity:</strong> up to 24 weeks<br />
<strong>Paternity:</strong> 6 weeks</p>
</details>

<SOC2 />

```mdx
<CalloutBox title="Example">This is how you write a callout.</CalloutBox>
```

Write `<Caption>` under an image. Replace <NAME> with your name; run kubectl logs <name-of-pod>.
"""


def text_of(source: str, data: str) -> str:
    return "\n".join(s.text for s in load_document(source, data.encode()).sections)


def test_mdx_files_are_loaded_as_markdown_with_the_front_matter_title():
    doc = load_document("handbook/people/time-off.mdx", PAGE.encode())
    assert doc.file_type == ".mdx" and all(s.heading.startswith("Time off") for s in doc.sections)


def test_people_and_teams_become_their_names():
    text = text_of("t.mdx", PAGE)
    assert "Ask Lottie Coxon or the people team." in text
    assert "TeamMember" not in text and "SmallTeam" not in text


def test_wrappers_keep_their_text_and_a_callout_keeps_its_title():
    text = text_of("t.mdx", PAGE)
    assert "Book it in Deel" in text and "Use the Time Off by Deel app in Slack." in text
    assert "PrivateLink" not in text and text.count("CalloutBox") == 2  # both in the fenced example, kept as code


def test_media_components_imports_and_editor_comments_disappear():
    text = text_of("t.mdx", PAGE)
    for gone in ("ProductScreenshot", "imageLight", "import ", "components/Docs", "SOC2", "editors: keep"):
        assert gone not in text


def test_html_layout_tags_go_but_their_words_stay():
    text = text_of("t.mdx", PAGE)
    assert "Maternity: up to 24 weeks" in text and "Paternity: 6 weeks" in text and "Parental leave" in text
    for tag in ("<details>", "<summary>", "<p>", "<strong>", "<br"):
        assert tag not in text


def test_code_keeps_its_components_exactly():
    text = text_of("t.mdx", PAGE)
    assert '<CalloutBox title="Example">This is how you write a callout.</CalloutBox>' in text  # fenced block
    assert "Write <Caption> under an image." in text  # inline code: the backticks go, the example stays


def test_placeholders_that_look_like_tags_survive():
    text = text_of("t.mdx", PAGE)
    assert "Replace <NAME> with your name" in text and "kubectl logs <name-of-pod>" in text


def test_plain_markdown_without_components_is_unchanged():
    plain = "# Guide\n\nRun `make <target>` and see <https://example.com>.\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
    assert strip_components(plain) == plain


def test_curly_apostrophes_survive_as_text():
    assert "don’t need an invite" in text_of("t.md", "Teammates don’t need an invite.\n")
