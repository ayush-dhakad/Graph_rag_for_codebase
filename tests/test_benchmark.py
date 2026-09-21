from codegraphrag.benchmark import _contains_expected


def test_answer_match_ignores_markdown_formatting():
    assert _contains_expected("the `ask` method calls it", "ask method")
