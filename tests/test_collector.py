import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import pytest

from tools.warband_collector.storage import Store, fresh, quality, validate, Conflict
from tools.warband_collector.server import CollectorServer, catalog, lock_directory
from tools.warband_collector import server as server_module


def sample():
    return {"format": "taverngym.manual-warband.v1", "revision": 0, "games": [{
        "id": "game1", "source_id": "source1", "source": "local-recording", "patch": "37.x-unknown-test",
        "split": "unassigned", "snapshots": [{"id": "shot1", "turn": 1, "role": "player", "phase": "pre_start",
            "timestamp": "01:23", "reviewed": True, "empty_board_confirmed": False, "combat_details_checked": False,
            "hand_status": "unknown", "effects_status": "unknown", "hand": [], "evidence": [],
            "board": [{"id": "entity1", "name": "未实现的新版本随从", "card_id": "", "catalog_version": "",
                       "attack": 5, "health": 7, "golden": False, "review": "observed", "keywords": []}]}]}]}


def test_observations_are_not_silently_promoted_to_training():
    data = validate(sample());g=data["games"][0];s=g["snapshots"][0]
    q=quality(g,s)
    assert q["statistics_ready"] and not q["combat_review_candidate"]
    assert not q["training_import_allowed"] and not q["native_import_ready"]
    s["board"][0]["attack"]=None
    assert not quality(g,s)["statistics_ready"]


@pytest.mark.parametrize("mutation", [
    lambda s:s.update(turn=0), lambda s:s.update(reviewed="false"),
    lambda s:s["board"][0].update(attack=float("nan")),
    lambda s:s["board"][0].update(attack=True),
    lambda s:s.update(evidence=["../../outside.png"]),
    lambda s:s.update(evidence=[1]),
    lambda s:s.update(board=s["board"]*8),
    lambda s:s.update(hand=s["board"]),
])
def test_bad_input_is_rejected(mutation):
    data=sample();mutation(data["games"][0]["snapshots"][0])
    with pytest.raises(ValueError):validate(data)


def test_atomic_revisions_conflict_and_backup_restoration(tmp_path):
    store=Store(tmp_path/'original');data=sample();saved=store.save(data,0)
    assert saved['revision']==1
    with pytest.raises(Conflict):store.save(data,0)
    assert Store(tmp_path/'original').load()['revision']==1
    assert len(list(store.history.glob('*.gz')))==1
    evidence=store.attach('截图.png',b'png-test-fixture')
    saved['games'][0]['snapshots'][0]['evidence']=[evidence]
    store.save(saved,1)
    backup=store.backup();dest=Store(tmp_path/'restored')
    dest.merge(backup,0,zipped=True)
    assert (dest.attachments/evidence).read_bytes()==b'png-test-fixture'
    assert dest.load()['games']==store.load()['games']
    with pytest.raises(ValueError,match='相同局ID'):dest.merge(backup,1,zipped=True)
    rows=[json.loads(x) for x in dest.jsonl().decode().splitlines()]
    assert rows[0]['training_import_allowed'] is False
    assert rows[-1]['quality']['native_import_ready'] is False


def test_split_leak_and_zip_escape_are_blocked(tmp_path):
    data=sample();duplicate=copy.deepcopy(data['games'][0]);duplicate['id']='game2';duplicate['snapshots']=[]
    duplicate['split']='test';data['games'][0]['split']='train';data['games'].append(duplicate)
    with pytest.raises(ValueError,match='同一来源'):validate(data)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('collection.json',json.dumps(sample()));z.writestr('../outside.txt','bad')
    with pytest.raises(ValueError,match='路径'):Store(tmp_path).merge(b.getvalue(),0,True)
    assert not (tmp_path.parent/'outside.txt').exists()


def test_shared_ui_core_produces_python_valid_data(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node optional')
    out=tmp_path/'ui.json';root=Path(__file__).resolve().parents[1]
    result=subprocess.run([node,str(root/'tests/collector_core.cjs'),str(out)],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    validate(json.loads(out.read_text(encoding='utf-8')))


def test_directory_lock_prevents_two_writers(tmp_path):
    handle=lock_directory(tmp_path)
    try:
        with pytest.raises(ValueError,match='另一个'):lock_directory(tmp_path)
    finally:handle.close()
    lock_directory(tmp_path).close()


def test_server_uses_loopback_api_and_needs_no_native_runtime(tmp_path):
    server=CollectorServer(('127.0.0.1',0),tmp_path);port=server.server_port
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(path,body=None,headers=None):
        h={'Content-Type':'application/json',**(headers or {})}
        req=Request(f'http://127.0.0.1:{port}{path}',data=body,headers=h)
        return urlopen(req,timeout=10)
    try:
        assert json.load(request('/api/health'))['app']=='autotavern-warband-collector-v1'
        assert '战团采集台' in request('/').read().decode()
        cats=json.load(request('/api/catalog'));assert len(cats[0]['cards'])>1000
        profile=json.load(request('/api/profile'))
        assert profile['patch']=='37.0.1' and len(profile['tribes'])==10 and 'NAGA' not in profile['tribes']
        assert cats[0]['origin']=='engine_reference_not_live_pool'
        saved=json.load(request('/api/save',json.dumps({'data':sample(),'revision':0}).encode()))
        assert saved['revision']==1
        with pytest.raises(HTTPError) as e:request('/api/save',json.dumps({'data':sample(),'revision':0}).encode())
        assert e.value.code==409
        with pytest.raises(HTTPError):request('/api/save',b'{}',{'Origin':'https://unrelated.example'})
        with pytest.raises(HTTPError):request('/api/import',b'broken-zip',{'Content-Type':'application/zip','X-Revision':'1'})
        assert b'"training_import_allowed": false' in request('/api/export.jsonl').read()
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def test_quick_full_game_does_not_require_provenance_or_audit():
    data=sample();g=data['games'][0];s=g['snapshots'][0]
    g.update(capture_mode='quick_combat',source='',patch='37.0.1',hero='测试英雄',available_tribes=['BEAST','DEMON','DRAGON','MECHANICAL','MURLOC'])
    s.update(timestamp='',reviewed=False,capture_mode='quick_combat',hand_scope='combat_only',tier=3,health=25,armor=4)
    s['board'][0]['review']='pending'
    second=copy.deepcopy(s);second['id']='shot2';second['turn']=2;second['board'][0]['id']='entity2'
    g['snapshots'].append(second)
    validate(data);q=quality(g,s)
    assert q['statistics_ready'] and q['combat_review_candidate']
    assert not q['external_audit_required'] and not q['native_import_ready']
    assert q['hand_scope']=='combat_only' and len(g['snapshots'])==2


def test_quick_tribe_selection_and_hand_count_validation():
    data=sample();g=data['games'][0];s=g['snapshots'][0]
    g['available_tribes']=['NAGA']
    with pytest.raises(ValueError,match='种族'):validate(data)
    g['available_tribes']=['BEAST','BEAST']
    with pytest.raises(ValueError,match='种族'):validate(data)
    g['available_tribes']=[];s['hand_count']=11
    with pytest.raises(ValueError,match='总手牌数'):validate(data)


def test_export_omits_untouched_next_round_without_losing_the_full_game(tmp_path):
    data=sample();g=data['games'][0];s=g['snapshots'][0]
    g['capture_mode']='quick_combat';s.update(capture_mode='quick_combat',entry_touched=True,submitted=True)
    nxt=copy.deepcopy(s);nxt.update(id='shot2',turn=2,entry_touched=False,submitted=False);nxt['board'][0]['id']='entity2'
    g['snapshots'].append(nxt);store=Store(tmp_path);store.save(data,0)
    rows=[json.loads(x) for x in store.jsonl().decode().splitlines()]
    assert rows[0]['omitted_auto_drafts']==1
    assert len([r for r in rows if r['type']=='snapshot'])==1
    assert len(store.load()['games'][0]['snapshots'])==2
    saved=store.load();saved['games'][0]['snapshots'][1]['entry_touched']=True;store.save(saved,1)
    rows=[json.loads(x) for x in store.jsonl().decode().splitlines()]
    assert len([r for r in rows if r['type']=='snapshot'])==2


def test_repeated_launcher_reuses_running_server_without_second_writer(tmp_path, monkeypatch):
    directory=tmp_path/'runs/data'
    server=CollectorServer(('127.0.0.1',0),directory)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opened=[]
    monkeypatch.setattr(server_module,'ROOT',tmp_path)
    monkeypatch.setattr('sys.argv',['collector','--port',str(server.server_port),'--data-dir',str(directory)])
    monkeypatch.setattr(server_module.webbrowser,'open',opened.append)
    def forbidden(*a):raise AssertionError('must reuse existing server')
    monkeypatch.setattr(server_module,'lock_directory',forbidden)
    try:
        server_module.main()
        assert opened==[f'http://127.0.0.1:{server.server_port}/']
        assert server.store.load()['revision']==0
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
