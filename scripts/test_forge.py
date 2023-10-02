import jsonschema
import pytest
from prompt_forge import Generator


def test_config_parsing():
    # Test empty config
    with pytest.raises(jsonschema.exceptions.ValidationError, match="required property"):
        Generator.from_string("")
