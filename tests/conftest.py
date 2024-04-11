import random
import string
from pathlib import Path
import numpy as np

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


@pytest.fixture(scope="session")
def junk_omx(tmp_path_factory) -> Path:
    """Create a junk OMX file."""
    file = tmp_path_factory.mktemp("data") / "junk.omx"
    from wring.omx import OMX
    prng = np.random.default_rng(42)
    with OMX(file, "w") as omx:
        omx.add_matrix("A", prng.uniform(size=(100, 100)))
        omx.add_matrix("B", prng.normal(size=(100, 100)).astype(np.float32))
        omx.add_lookup("TAZ", np.arange(1, 101, dtype=np.int8))
    return file

