import numpy as np

from wring.omx import OMX, convert_multiple_omx


def test_simple_omx(junk_omx, tmp_path):
    # with open(junk_file, "rt") as f:
    #     data = f.read()

    convert_multiple_omx(glob_pattern=junk_omx)

    x1 = OMX(junk_omx, "r")
    x2 = OMX(junk_omx.with_suffix(".omxz"), "r")

    assert x1.shape == x2.shape
    assert x1.data["A"].dtype == np.float64
    assert x1.data["B"].dtype == np.float32

    assert x2.data["A"].dtype == np.float32
    assert x2.data["B"].dtype == np.float32

    assert np.allclose(x1.data["A"], x2.data["A"])
    assert np.allclose(x1.data["B"], x2.data["B"])

    x1.close()
    x2.close()
