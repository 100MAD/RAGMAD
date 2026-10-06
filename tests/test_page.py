from app.rag.ingestion import page_from_doc_items


def test_page_comes_from_the_earliest_provenance():
    page = page_from_doc_items(
        [
            {"prov": [{"page_no": 4}, {"page_no": 5}]},
            {"prov": [{"page_no": 2}]},
        ]
    )
    assert page == 2


def test_page_is_missing_when_items_have_no_provenance():
    assert page_from_doc_items([{"self_ref": "#/texts/0"}]) is None
    assert page_from_doc_items(None) is None
