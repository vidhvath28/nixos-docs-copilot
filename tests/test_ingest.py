from app.ingest import parse_sections

HTML = """
<html><body>
<div class="book"><h1 class="title"><a id="book-nixos-manual"></a>NixOS Manual</h1>
<h2 class="subtitle">Version 26.05</h2>
<div class="toc"><p>Table of Contents</p></div>
<div class="chapter"><h2 id="ch-configuration" class="title">Configuration</h2>
  <div class="chapter"><h2 id="module-foo" class="title">Foo Server</h2>
    <p>Foo is a server that serves foo over the network to many clients.</p>
    <div class="section"><h2 id="module-foo-quick-start" class="title">Quick Start</h2>
      <div class="note"><h3 class="title">Note</h3><p>Admonition headings have no id.</p></div>
      <pre><code>services.foo.enable = true;</code></pre>
      <p>Then run nixos-rebuild switch to start the foo service on boot.</p>
    </div>
  </div>
</div></div>
</body></html>
"""


def test_sections_get_breadcrumbs_versions_and_deep_links():
    docs = parse_sections(HTML, "https://nixos.org/manual/nixos/stable/")
    by_anchor = {d.metadata["anchor"]: d for d in docs}

    qs = by_anchor["module-foo-quick-start"]
    assert qs.metadata["title"] == "Configuration › Foo Server › Quick Start"
    assert qs.metadata["url"] == "https://nixos.org/manual/nixos/stable/#module-foo-quick-start"
    assert qs.metadata["version"] == "26.05"
    assert "services.foo.enable = true;" in qs.page_content
    assert "Admonition headings have no id." in qs.page_content  # untitled headings stay inline

    assert "book-nixos-manual" not in by_anchor
    assert "ch-configuration" not in by_anchor  # heading-only wrapper is skipped
    assert all("Table of Contents" not in d.page_content for d in docs)
