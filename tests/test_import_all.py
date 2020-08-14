import pytest

from whatlies import *
from whatlies.language import *
from whatlies.transformers import *
from whatlies.error import NotInstalled


def test_import_error():
    dinosaurhead = NotInstalled("DinosaurHead", "dinosaur")
    with pytest.raises(ModuleNotFoundError) as e:
        dinosaurhead()
    assert "pip install whatlies[dinosaur]" in str(e.value)
