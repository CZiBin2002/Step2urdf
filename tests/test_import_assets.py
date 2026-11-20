from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from starlette.datastructures import UploadFile

from workbench import server


def zipped_model(contents):
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
        archive.writestr("robot.urdf", contents)
    payload.seek(0)
    return UploadFile(payload, filename="robot.zip")


def test_zip_upload_and_extracted_sizes_are_counted_separately(tmp_path, monkeypatch):
    contents = b"a" * 850
    model = zipped_model(contents)
    limit = 900
    assert limit - len(contents) < model.file.getbuffer().nbytes < limit
    monkeypatch.setattr(server, "UPLOAD_LIMIT", limit)

    mesh = UploadFile(BytesIO(b"b" * 50), filename="mesh.stl")
    assets = server.read_assets([model, mesh], tmp_path)

    assert [asset.read_bytes() for asset in assets] == [contents, b"b" * 50]


def test_extracted_size_limit_is_still_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_LIMIT", 900)
    model = zipped_model(b"a" * 850)
    mesh = UploadFile(BytesIO(b"b" * 51), filename="mesh.stl")

    with pytest.raises(ValueError, match="ZIP 解压后超过"):
        server.read_assets([mesh, model], tmp_path)
