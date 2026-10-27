"""Manual visible warbands -> isolated native combat fixtures, not recruit-state injection.

This adapter preserves entered stats and order. It never invents missing late turns or
overwrites the source collection. Post-start hand-scaling is a declared combat proxy.
"""
import copy
import hashlib
import json
from pathlib import Path

import taverngym as tg

VISIBLE_KW = {'TAUNT','DIVINE_SHIELD','REBORN','VENOMOUS','POISONOUS','WINDFURY','STEALTH','DEATHRATTLE','CLEAVE'}
SUPPORTED_COMBAT_CARDS = {'BG20_100','BG26_135','BG29_300','BG36_354','BGS_034','BG32_330',
                          'BG36_763','BG36_106','BG20_104','BG34_142','BG36_523','BG36_700',
                          'BG36_344','BG34_319','BG33_825','BG31_148','BG33_823','BG34_604','BG23_318'}


def command(game, cmd, **values):
    return json.loads(game.debug(json.dumps({'cmd':cmd, **values})))


def resolve_card(row, catalog):
    card = catalog.get(row.get('card_id'))
    if card is None:
        candidates = [c for c in catalog.values() if c['kind']=='minion' and
                      row.get('name') in (c['name'], c['name_zh']) and not c.get('normal_id')]
        if len(candidates)!=1: raise ValueError(f"卡牌无法唯一映射: {row.get('name')}")
        card=candidates[0]
    if card['kind']!='minion': raise ValueError('战场卡牌不是随从')
    normal=catalog[card['normal_id']] if card.get('normal_id') else card
    target=normal.get('golden_id') if row.get('golden') else normal['id']
    if not target or target not in catalog: raise ValueError('缺少所选普通/金色形态')
    resolved=catalog[target]
    if not resolved['implemented']: raise ValueError(f"引擎尚未实现 {resolved['id']}")
    if row.get('attack') is None or row.get('health') is None: raise ValueError('缺少当前攻血')
    if row['attack']<0 or row['health']<=0: raise ValueError('入场战团必须攻击非负、生命为正')
    return resolved


def make_fixture(game_meta, shot, catalog, *, seed=70301, after_start=False):
    cfg={'content':'Implemented','record_log':False,'lobby_types':game_meta.get('available_tribes',[]),'hero_bans':['BG22_HERO_007']}
    game=tg.Game(seed,json.dumps(cfg))
    heroes=[c for c in catalog.values() if c['kind']=='hero' and game_meta.get('hero') in (c['name'],c['name_zh'])]
    if len(heroes)!=1: raise ValueError('英雄无法唯一映射')
    hero=heroes[0]
    if hero['name']!='Captain Hooktusk':
        raise ValueError('此初版转换器尚未登记该英雄的战斗计数适配')
    command(game,'offer_hero',p=0,hero=hero['id'])
    game.step(0,dict(tg.layout())['A_CHOOSE'])
    for p in range(1,8):
        if game.needs_action(p):game.step(p,game.bot_action(p))
    # These are independent combat seats; other seven recruit states are not training data.
    command(game,'set_hero',p=0,hero=hero['id'])
    command(game,'set_hero',p=1,hero=hero['id'])
    for p in (0,1):
        for zone in ('board','hand','shop'):command(game,'clear_zone',p=p,zone=zone)
    command(game,'set_turn',turn=shot['turn'])
    command(game,'set_player',p=0,health=shot.get('health') if shot.get('health') is not None else 30,
            armor=shot.get('armor') if shot.get('armor') is not None else 0,tier=shot.get('tier') or 1)
    command(game,'set_player',p=1,health=30,armor=0,tier=shot.get('tier') or 1)
    normalized, changes, limitations=[],[],[]
    if shot.get('global_notes') or shot.get('skill_notes'):
        raise ValueError('存在自由文本战斗效果，需要对应结构化适配，不能静默忽略')
    for i,row in enumerate(shot['board']):
        card=resolve_card(row,catalog)
        normal_id=card.get('normal_id') or card['id']
        if normal_id not in SUPPORTED_COMBAT_CARDS:
            raise ValueError(f"该随从尚未登记战斗状态需求: {row['name']}")
        if row.get('details'): raise ValueError(f"{row['name']} 有附加效果备注，需对应适配")
        if row.get('card_id')!=card['id']:
            changes.append({'slot':i+1,'name':row['name'],'from':row.get('card_id'),'to':card['id'],'reason':'explicit golden checkbox is authoritative; entered stats unchanged'})
        command(game,'add_minion',p=0,zone='board',card=card['id'],atk=row['attack'],hp=row['health'])
        view=json.loads(game.view(0))['me']['board'][i]
        entered = set(row.get('keywords', []))
        if 'DEATHRATTLE' in entered:
            if normal_id not in {'BG34_319', 'BG23_318'}:
                raise ValueError('额外亡语必须说明其具体效果；不能仅由亡语标志生成脚本')
            entered.remove('DEATHRATTLE')
            limitations.append('亡语为该随从原生脚本，不作为可授予关键字位；保留原生亡语执行')
        keywords=sorted((set(view['keywords'])-VISIBLE_KW)|entered)
        command(game,'edit_minion',p=0,zone='board',index=i,keywords=keywords)
        actual=set(json.loads(game.view(0))['me']['board'][i]['keywords'])
        if not entered<=actual:
            raise ValueError(f"该额外关键字尚未适配: {sorted(set(row['keywords'])-actual)}")
        if normal_id=='BG34_142':
            if not after_start and not shot.get('hand'):
                raise ValueError('狂热变装鱼人需要开战前手牌最高攻击，或确认录入数值已包含开战效果')
            if after_start:
                limitations.append('已确认攻击包含开战加成：使用空的战斗代理手牌，使此开战手牌加攻为0，避免重复加攻；不声称恢复历史完整手牌')
        normalized.append({'slot':i+1,'id':card['id'],'name':row['name'],'golden':bool(row.get('golden')),
                           'attack':row['attack'],'health':row['health'],'keywords':keywords})
    if shot.get('hand'):
        raise ValueError('本轮只实现无需战斗手牌的可用条目；不能静默丢掉已录战斗手牌')
    saved=game.state_json()
    # from_json validates identity, entity references and engine state before any combat.
    restored=tg.Game.from_json(saved)
    visible=json.loads(restored.view(0))['me']['board']
    assert [(c['atk'],c['hp']) for c in visible]==[(c['attack'],c['health']) for c in normalized]
    return saved,normalized,changes,limitations


def analyze(source, out, *, after_start_turns=(), trials=32):
    if trials<1 or trials>5000: raise ValueError('trials must be 1..5000')
    source,out=Path(source),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    raw=source.read_bytes();data=json.loads(raw)
    catalog={c['id']:c for c in json.loads(tg.catalog_json())}
    records=[]
    for meta in data['games']:
        for shot in meta['snapshots']:
            if shot.get('entry_touched') is False and not shot.get('submitted'):continue
            row={'source_game_id':meta['id'],'snapshot_id':shot['id'],'turn':shot['turn'],
                 'tier':shot.get('tier'),'health':shot.get('health'),'armor':shot.get('armor'),
                 'source_patch':meta.get('patch'),'hero':meta.get('hero'),'available_tribes':meta.get('available_tribes',[])}
            try:
                post=shot['turn'] in after_start_turns or shot.get('phase')=='after_start'
                saved,board,changes,limits=make_fixture(meta,shot,catalog,after_start=post)
                filename=f"{meta['id']}-turn-{shot['turn']:03d}-{shot['id']}.json"
                (out/filename).write_text(saved,encoding='utf-8')
                game=tg.Game.from_json(saved)
                # Fixed simple probes are functionality checks, NOT human-strength scores.
                command(game,'add_minion',p=1,zone='board',card='BG20_100',atk=max(1,shot['turn']*2),hp=max(1,shot['turn']*3))
                before=game.state_json()
                first=command(game,'combat_preview',a=0,b=1,n=trials,details=True)
                second=command(game,'combat_preview',a=0,b=1,n=trials,details=True)
                if first!=second or game.state_json()!=before: raise RuntimeError('preview changed state or is not reproducible')
                truncated=sum(t['truncated'] for t in first['trials'])
                if truncated: raise RuntimeError('native combat guard triggered')
                # Mirror fixture exercises nontrivial trades, reborn, shields and Rally.
                command(game,'clear_zone',p=1,zone='board')
                for i,c in enumerate(board):
                    command(game,'add_minion',p=1,zone='board',card=c['id'],atk=c['attack'],hp=c['health'])
                    command(game,'edit_minion',p=1,zone='board',index=i,keywords=c['keywords'])
                mirror=command(game,'combat_preview',a=0,b=1,n=trials,details=True)
                if any(t['truncated'] for t in mirror['trials']):raise RuntimeError('mirror guard triggered')
                row.update(status='combat_proxy_usable',board=board,normalizations=changes,limitations=limits,
                    fixture_file=filename,fixture_sha256=hashlib.sha256(saved.encode()).hexdigest(),
                    phase_interpretation='post-start effective-board proxy' if post else 'pre-start under agreed collection protocol',
                    smoke={'trials':trials,'win':first['win_a'],'tie':first['tie'],'loss':first['win_b'],
                           'truncated':truncated,'repeat_equal':True,'state_unchanged':True})
                row['mirror_smoke']={k:mirror[k] for k in ('win_a','win_b','tie')}
            except (ValueError,RuntimeError,AssertionError) as e:
                row.update(status='blocked',reason=str(e))
            records.append(row)
    result={'format':'taverngym.manual-combat-bank.v1','engine':tg.game_version(),'native':tg.build_info()['source_hash'],
            'source_sha256':hashlib.sha256(raw).hexdigest(),'source_modified':False,'records':records,
            'usable_records':sum(r['status']=='combat_proxy_usable' for r in records),
            'coverage_turns':sorted({r['turn'] for r in records}),
            'historical_full_game_reconstructed':False,'recruit_injection_ready':False,'ppo_library_hook_ready':False,
            'human_calibrated':False,'strength_claim':False,
            'assumptions':['36.6.3 reference mechanics, not certified 37.0.1 parity',
                'independent per-turn combat proxies; no invented later turns or terminal rank',
                'unrecorded combat counters use native initial defaults; Blood Gems +1/+1 unless adapted',
                'post-start attack override only for explicitly confirmed entries; no policy input receives global state']}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
