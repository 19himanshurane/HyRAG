from hyrag.loader import load_document

from .conftest import CORPUS


def html(body: str, title: str = "T"):
    return load_document("page.html", f"<html><head><title>{title}</title></head><body>{body}</body></html>".encode())


def text_of(doc) -> str:
    return "\n".join(s.text for s in doc.sections)


def test_nested_block_text_is_read_once():
    assert text_of(html("<h1>T</h1><ul><li><p>Restart the agent.</p></li></ul>")) == "Restart the agent."


def test_text_directly_inside_div_is_kept():
    assert "30 minutes" in text_of(html("<h1>T</h1><div>Tokens expire after 30 minutes.</div>"))


def test_inline_code_does_not_add_spaces():
    assert "Run nimbus-deploy --env staging. Done" in text_of(
        html("<h1>T</h1><p>Run <code>nimbus-deploy --env staging</code>. Done</p>"))


def test_page_furniture_is_removed():
    doc = html("<nav>Home</nav><h1>T</h1><p>Real.</p><aside>Maintainers</aside><form>Search</form>"
               "<div id='docComments'>Submit correction</div><footer>(c)</footer><script>x()</script>")
    assert text_of(doc) == "Real."


def test_permalink_marks_are_removed_from_headings():
    doc = html('<h2>Memory <a class="id_link" href="#mem">#</a></h2><p>x</p>')
    assert doc.sections[0].heading == "Memory"


def test_table_rows_stay_on_one_line():
    doc = html("<h1>Codes</h1><table><tr><th>Code</th><th>Name</th></tr>"
               "<tr><td colspan=2><strong>Class 23</strong></td></tr>"
               "<tr><td><code>23505</code></td><td><code>unique_violation</code></td></tr>"
               "<tr><td>x</td><td><ul><li>a</li><li>b</li></ul></td></tr></table>")
    lines = text_of(doc).split("\n")
    assert "23505 | unique_violation" in lines
    assert "Class 23" in lines
    assert "x | a b" in lines


def test_pre_keeps_line_breaks():
    assert "line1\n  line2" in text_of(html("<h1>T</h1><pre>line1\n  line2</pre>"))


def test_heading_hierarchy():
    doc = html("<h1>A</h1><p>1</p><h2>B</h2><p>2</p><h3>C</h3><p>3</p><h2>D</h2><p>4</p>")
    assert [s.heading for s in doc.sections] == ["A", "A > B", "A > B > C", "A > D"]


def test_very_deep_nesting_does_not_crash():
    depth = 5000
    doc = html("<h1>T</h1>" + "<div>" * depth + "deep text" + "</div>" * depth + "<p>after</p>")
    assert text_of(doc) == "deep text\nafter"


def test_real_postgres_error_table_rows():
    doc = load_document("e.html", (CORPUS / "engineering" / "postgres-error-codes.html").read_bytes())
    assert "23505 | unique_violation" in text_of(doc).split("\n")
    assert not any("Submit correction" in s.heading for s in doc.sections)
