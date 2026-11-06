"""校准契约：完整帧不变、来源配额、分割隔离、关键词机制和真实战斗暴露。"""
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from workflows.calibrate_opponent_bank import human_references, sample_candidates, select_rows


def row(source, seed, score, card=1):
    return dict(source=source, source_game_seed=seed, turn=8, seat=1,
        player=dict(board=[dict(card=card, atk=100, hp=100, kw=2)], hand=[dict(card=card)], counter=77),
        combat=dict(score=score))


def test_full_frame_selection_prefers_combat_and_preserves_hidden_state():
    rows = [row('new', 1, .05), row('old', 2, .5, card=2), row('new', 3, .9, card=3)]
    before = copy.deepcopy(rows)
    selected = select_rows(rows, {8:dict(median_strength=200)}, keep=1)
    assert selected[0]['source'] == 'old'  # 最新模型并非默认更强或更适合
    assert selected[0]['player']['counter'] == 77
    assert rows == before
    assert sample_candidates(rows, 1, 7) == sample_candidates(rows, 1, 7)


def test_human_reference_source_games_and_split():
    rows = [dict(turn=8, game_id=str(i), split='train', board_detail=[dict(atk=i, hp=i)]) for i in range(40)]
    refs = human_references(rows, 7)
    assert refs[8]['games'] == 40
    assert len({r['game_id'] for r in refs[8]['anchors']}) == 2
    assert {r['game_id'] for r in refs[8]['anchors']}.isdisjoint(r['game_id'] for r in refs[8]['audit_anchors'])
    with pytest.raises(ValueError, match='Train'):
        human_references([dict(rows[0], split='test')], 7)


@pytest.fixture
def banks(tmp_path):
    from hsbg.az.train.personal import write_json, fingerprint, library_digest
    from hsbg.az.train.mirror_snapshots import write_snapshot_file, capture_eot
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn
    frame = capture_eot(_drive_to_turn(E.Game(seed=9341), 1), 1)
    assert frame is not None
    result = {}
    for name in ('control', 'calibrated'):
        path = tmp_path/name
        path.mkdir()
        frames = [dict(frame, seat=seat, source_game_seed=seed, source=name)
                  for seed in (1,2,3) for seat in range(1,8)]
        write_snapshot_file(str(path/'snap_test.json.gz'), frames, seed=1)
        write_json(path/'manifest.json', dict(split='train', fingerprint=fingerprint(),
            digest=library_digest(path), validation=dict(accepted=3)))
        result[name] = path
    return result


def test_fixed_mixture_quota_and_frame_identity(banks, tmp_path):
    from workflows.mix_opponent_banks import mix
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    before = {name:[p.read_bytes() for p in path.glob('snap_*')] for name,path in banks.items()}
    out = tmp_path/'mixed'
    mix(banks, dict(control=1,calibrated=3), out, 'train')
    pool = MirrorSnapshotPool(str(out), sample='random')
    frames = next(iter(pool.index.values()))
    assert Counter(f.mixture_source for f in frames) == dict(control=1,calibrated=3)
    assert {name:[p.read_bytes() for p in path.glob('snap_*')] for name,path in banks.items()} == before
    with pytest.raises(ValueError, match='分割'):
        mix(banks, dict(control=1,calibrated=3), tmp_path/'wrong', 'dev')
    with pytest.raises(FileExistsError):
        mix(banks, dict(control=1,calibrated=3), out, 'train')


def test_build_seeds_do_not_depend_on_selected_splits(tmp_path, monkeypatch):
    import hsbg.az.train.personal as personal
    from hsbg.az.train.mirror_snapshots import write_snapshot_file
    calls = []
    def command(argv):
        calls.append(argv)
        path = Path(argv[argv.index('--snap-dir')+1])
        seed = int(argv[argv.index('--seed')+1])
        write_snapshot_file(str(path/'snap_test.json.gz'), [dict(turn=1,seat=1,player=dict(board=[]))], seed=seed)
    monkeypatch.setattr(personal, 'run_command', command)
    personal.build_library(SimpleNamespace(out=tmp_path/'dev-only', games=1, driver='bot', device='cpu',
                                          checkpoint='', seed_base=20000000, splits='dev'))
    assert calls[0][calls[0].index('--seed')+1] == 21000000
    assert not (tmp_path/'dev-only/train').exists()
    assert not (tmp_path/'dev-only/test').exists()


def test_exposure_is_observational_and_records_real_combats(banks):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from hsbg.az.train.policies import EngineBotPolicy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    pool = MirrorSnapshotPool(str(banks['control']), sample='random')
    plain = evaluate_policy(EngineBotPolicy(), EvalConfig(games=1,seed=943,mirror_pool=pool))
    observed = evaluate_policy(EngineBotPolicy(), EvalConfig(games=1,seed=943,mirror_pool=pool,collect_exposure=True,collect_economy=True))
    assert plain['rank_sequence'] == observed['rank_sequence']
    assert plain['decisions'] == observed['decisions']
    assert plain['mirror'] == observed['mirror']
    assert observed['combat_exposure']
    assert not plain['combat_exposure']
    economy = observed['end_turn_economy']
    assert economy and 'end_turn_economy' not in plain
    assert len({(r['seed'], r['turn']) for r in economy}) == len(economy)
    assert all(1 <= r['tier'] <= 6 and r['actions'] >= 1 for r in economy)
    assert all(sum(r['action_counts'].values()) == r['actions']
               and r['action_counts']['end_turn'] == 1 for r in economy)
    from workflows.report_firestone_learning import by_turn
    turns = by_turn(dict(valid=True, per_seat=[observed]))
    assert set(range(1, 20)) <= turns.keys()
    row = turns[economy[0]['turn']]
    assert row['mean_strength'] == economy[0]['attack'] + economy[0]['board_hp']
    assert row['mean_actions_by_type']['end_turn'] == 1
    assert len({(r['seed'],r['turn']) for r in observed['combat_exposure']}) == len(observed['combat_exposure'])
    assert any(r['mirror_applied'] for r in observed['combat_exposure'])


def test_native_calibration_audits_separate_references(tmp_path):
    import hsbg_engine as E
    from hsbg.az.train.mirror_snapshots import capture_eot, write_snapshot_file
    from hsbg.az.train.personal import write_json, fingerprint, library_digest
    from workflows.mirror_hard_gates import _drive_to_turn
    from workflows.calibrate_opponent_bank import build
    game = _drive_to_turn(E.Game(seed=9394), 8)
    # 在线禁用改变抽选结果，固定 seed 也可能在第 8 回合有正常的延迟饰品待选。
    # 按真实掩码结算后再捕获，不能清菜单或放宽 capture_eot 的前置条件。
    for _ in range(64):
        mask = game.legal_mask(1)
        if mask[0]:
            break
        action = game.bot_action(1)
        assert mask[action]
        game.step(1, action)
    assert int(game.turn) == 8 and game.legal_mask(1)[0]
    frame = capture_eot(game, 1)
    frame['source_game_seed'] = 9394
    before = copy.deepcopy(frame['player'])
    source = tmp_path/'source'
    write_snapshot_file(str(source/'snap_fixture.json.gz'), [frame], seed=9394)
    write_json(source/'manifest.json', dict(split='train', fingerprint=fingerprint(),
        digest=library_digest(source), validation=dict(accepted=1)))
    humans = [dict(turn=8, split='train', game_id=str(i), tier=frame['player']['tier'],
                   board_detail=frame['player']['board']) for i in range(40)]
    human = tmp_path/'human.jsonl'
    human.write_text(''.join(json.dumps(r)+'\n' for r in humans), encoding='utf-8')
    out = tmp_path/'calibrated'
    result = build(dict(model=str(source)), human, out, 'train', per_source=1, trials=1)
    assert result['turns']['8']['audit_combat_score'] is not None
    games = result['turns']['8']['reference_games']
    assert set(games['anchors']).isdisjoint(games['audit_anchors'])
    with gzip.open(next(out.glob('snap_*.json.gz')), 'rt', encoding='utf-8') as stream:
        assert json.load(stream)['frames'][0]['player'] == before
    assert not result['ready_for_long_training']


def test_archive_only_restores_train(tmp_path, monkeypatch):
    import io
    import tarfile
    import workflows.prepare_opponent_research as research
    from workflows.run_learning_experiment import digest
    archive = tmp_path/'artifacts/exports/firestone-20261102-v1.tar.gz'
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, 'w:gz') as stream:
        for name in ('train.jsonl', 'test.jsonl'):
            data = b'preserve split isolation'
            member = tarfile.TarInfo('untrusted/path/'+name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    monkeypatch.setattr(research, 'ARCHIVE_SHA', digest(archive))
    import hashlib
    monkeypatch.setattr(research, 'HUMAN_SHA', hashlib.sha256(data).hexdigest())
    target = tmp_path/'restored/train.jsonl'
    research.ensure_human_train(target)
    assert target.read_bytes() == data
    assert not list(tmp_path.rglob('test.jsonl'))
