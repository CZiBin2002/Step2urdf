import math

from fastapi.testclient import TestClient

from workbench import server
from tests.test_collisions import fixture


def test_collision_workflow_and_cache_do_not_depend_on_home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'CACHE', tmp_path)
    server._scenes.clear()
    parts, project = fixture(obstacle=(-0.2, 0, 0), moving=(0.2, 0, 0))
    payload = server.create_job(parts, project, 'test')
    path = f"/api/projects/{payload['id']}"
    client = TestClient(server.app)
    response = client.post(path + '/collision-geometry', json=project.model_dump())
    assert response.status_code == 200
    assert all(response.json().values())
    response = client.post(path + '/collision-limits', json={
        'project': project.model_dump(), 'joint': 'j'})
    assert response.status_code == 200
    limits = response.json()
    assert limits['upper'] > 2.4
    project.joints[0].lower = limits['lower']
    project.joints[0].upper = limits['upper']
    project.zero_pose = {'j': 1.2}
    response = client.post(path + '/motion', json={
        'project': project.model_dump(), 'start': {'j': 1.2}, 'target': {'j': 2.2}})
    assert response.status_code == 200 and not response.json()['blocked']
    assert response.json()['pose']['j'] == 2.2
    assert len(server._scenes) == 1
    assert client.post(path + '/motion', json={
        'project': project.model_dump(), 'start': {'j': 1.2}, 'target': {'j': 4}}).json()['blocked']
    invalid = client.post(path + '/motion', json={
        'project': project.model_dump(), 'target': {'nonexistent': 1}})
    assert invalid.status_code == 422
    assert client.post('/api/projects/' + '0' * 32 + '/motion', json={
        'project': project.model_dump()}).status_code == 404
