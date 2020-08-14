import pathlib

import pytest


def handle_md_txt(txt):
    while txt.find("```python\n") != -1:
        start = txt.find("```python\n")
        end = txt.find("```\n")
        if end == -1:
            raise ValueError("Closing code missing!")
        code_part = txt[(start + 10) : end]
        print(code_part)
        txt = txt[(end + 1) :]


@pytest.mark.parametrize("path", [str(p) for p in pathlib.Path("docs").glob("*/*.md")])
def test_docs(path):
    """
    Take the docstring of every method on the `Clumper` class.
    The test passes if the usage examples causes no errors.
    """
    txt = pathlib.Path(path).read_text()
    handle_md_txt(txt)
