from pathlib import Path
import json
import shutil
import subprocess
import pytest
from tools.warband_collector.server import catalog


def test_search_logic_lobby_dual_neutral_tier_sort_and_gold():
    node=shutil.which('node')
    if not node:pytest.skip('optional Node')
    p=Path(__file__).with_name('collector_search.cjs')
    r=subprocess.run([node,str(p)],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert r.returncode==0,r.stdout+r.stderr


def test_catalog_keeps_real_pool_and_family_metadata():
    root=Path(__file__).resolve().parents[1]
    rows=catalog(json.loads((root/'crates/taverngym/data/cards.json').read_text(encoding='utf-8')),'test')['cards']
    by_id={c['id']:c for c in rows}
    assert by_id['BGS_034']['recruit_pool'] # pool_tag=0 is NOT a removal flag
    assert by_id['BG36_763']['races']==['BEAST','PIRATE']
    assert by_id['BG34_142']['golden_id']=='BG34_142_G'
    assert by_id['BG34_142_G']['normal_id']=='BG34_142'
