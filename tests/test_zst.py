import pytest

from wring import tar_zst


def test_simple_zst(junk_file, tmp_path):
    with open(junk_file, "rt") as f:
        data = f.read()

    archive = tmp_path / "junk.tar.zst"
    with pytest.raises(NotADirectoryError):
        tar_zst.compress_zst(in_path=junk_file, archive=archive)

    tar_zst.compress_zst(in_path=junk_file.parent, archive=archive)

    out_path = tmp_path / "junk_restored"
    tar_zst.extract_zst(archive=archive, out_path=out_path)

    with open(out_path / "junk.txt", "rt") as f:
        data2 = f.read()

    assert data == data2


def test_chunked_zst(junk_file, tmp_path):
    with open(junk_file, "rt") as f:
        data = f.read()

    archive = tmp_path / "junk.tar.zst"
    with pytest.raises(NotADirectoryError):
        tar_zst.compress_zst(in_path=junk_file, archive=archive)

    tar_zst.compress_zst_chunks(
        in_path=junk_file.parent, archive=archive, chunksize=1000
    )

    out_path = tmp_path / "chunk_restored"
    tar_zst.extract_zst(archive=archive, out_path=out_path)

    with open(out_path / "junk.txt", "rt") as f:
        data2 = f.read()

    assert data == data2
