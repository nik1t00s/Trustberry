from app.uploads import allocate_upload_path

def test_uploads_stay_inside_storage_and_never_overwrite(tmp_path):
    first=allocate_upload_path(tmp_path,"../../records.json")
    second=allocate_upload_path(tmp_path,"C:\\\\private\\\\records.json")
    assert first.parent==second.parent==tmp_path
    assert first!=second
    assert first.name.endswith("records.json") and second.name.endswith("records.json")

def test_web_pages_and_assets_use_isolated_database(tmp_path,monkeypatch):
    from app import db,main
    from fastapi.testclient import TestClient
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'reviews.db')
    with TestClient(main.app) as client:
        for path in ('/','/training','/prediction','/labeling','/health','/static/studio.css'):
            response=client.get(path)
            assert response.status_code==200, (path,response.text)
        response=client.post('/predict',json={})
        assert response.status_code==422
