from wring.omx import convert_multiple_omx, OMX
import numpy as np
import pytest

# convert_multiple_omx(
#     glob_pattern: Annotated[
#         str,
#         typer.Argument(help="file name or glob pattern for the OMX files to convert"),
#     ],
#     complevel: Annotated[int, typer.Argument(help="compression level")] = 3,
#     complib: Annotated[str, typer.Argument(help="compression library")] = "blosc2:zstd",
#     dtype_shrink: Annotated[
#         int, typer.Option(help="maximum bitwidth to use for integer and float types")
#     ] = 32,
#     n_processes: Annotated[
#         int, typer.Option(help="number of processes to use while converting files")
#     ] = 8,
#     out_dir: Annotated[
#         Path | None,
#         typer.Option(
#             help=(
#                 "directory name for the resulting reprocessed file(s), if not "
#                 "provided any '.omx' suffix is stripped from the original file "
#                 "name(s) and a '.omxz' suffix is added"
#             ),
#         ),
#     ] = None,
# )

def test_simple_omx(junk_omx, tmp_path):
    # with open(junk_file, "rt") as f:
    #     data = f.read()

    convert_multiple_omx(glob_pattern=junk_omx)

    x1 = OMX(junk_omx, 'r')
    x2 = OMX(junk_omx.with_suffix('.omxz'), 'r')

    assert x1.shape == x2.shape
    assert x1.data['A'].dtype == np.float64
    assert x1.data['B'].dtype == np.float32

    assert x2.data['A'].dtype == np.float32
    assert x2.data['B'].dtype == np.float32

    assert np.allclose(x1.data['A'], x2.data['A'])
    assert np.allclose(x1.data['B'], x2.data['B'])

    x1.close()
    x2.close()