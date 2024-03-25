import random
import string
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def junk_file(tmp_path_factory) -> Path:
    """Create a junk file with some content."""
    file = tmp_path_factory.mktemp("data") / "junk.txt"
    n = 79
    content = [
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
        for _ in range(100)
    ]
    file.write_text("\n".join(content * 1000))
    return file
