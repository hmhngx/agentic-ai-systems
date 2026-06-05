from pathlib import Path

import pytest

from src.loader import Page, load_pages, full_text, main_body_pages


def test_load_txt(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("Hello world.\nSecond line.", encoding="utf-8")
    pages = load_pages(str(p))
    assert len(pages) == 1
    assert isinstance(pages[0], Page)
    assert pages[0].page_num == 1
    assert "Hello world" in pages[0].text


def test_full_text_joins_pages():
    pages = [Page(page_num=1, text="alpha"), Page(page_num=2, text="beta")]
    text = full_text(pages)
    assert "alpha" in text and "beta" in text


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pages(str(tmp_path / "nope.pdf"))


def test_main_body_pages_drops_references_onward():
    pages = [
        Page(page_num=1, text="Body intro with Stanford University."),
        Page(page_num=2, text="More body.\nReferences\nSmith et al. 2020. Some paper."),
        Page(page_num=3, text="Adam Roberts. Quoc Le. Citation list."),
    ]
    body = main_body_pages(pages)
    assert len(body) == 2                      # page 3 dropped entirely
    assert body[1].page_num == 2
    assert "More body" in body[1].text
    assert "Smith et al" not in body[1].text   # truncated at the heading


def test_main_body_pages_noop_without_heading():
    pages = [Page(page_num=1, text="Just body, no bibliography heading.")]
    assert main_body_pages(pages) == pages
