(function (root) {
  'use strict';
  const clone = x => JSON.parse(JSON.stringify(x));
  let profile = {patch:'37.0.1',tribes:{}};
  const configure = value => {profile=clone(value);};
  const id = () => globalThis.crypto?.randomUUID?.().replaceAll('-', '') || 'r' + Date.now().toString(36) + Math.random().toString(36).slice(2);
  function entity(name = '') {
    return {id: id(), name, card_id: '', catalog_version: '', catalog_sha256: '', text: '',
      golden: null, attack: null, health: null, max_health: null, keywords: [],
      details: '', review: 'pending'};
  }
  function snapshot(turn = 1, role = 'player') {
    return {id: id(), turn, role, seat: role === 'player' ? '主播' : '', timestamp: '', battle_id: '',
      phase: 'unknown', hero: '', tier: null, health: null, armor: null, alive: null, damage_cap: null,
      skill_notes: '', global_notes: '', board: [], hand: [], hand_status: 'unknown', effects_status: 'unknown',
      reviewed: false, empty_board_confirmed: false, combat_details_checked: false,
      result: '', notes: '', evidence: []};
  }
  function game() {
    const g = {id: id(), source_id: id(), source: '', streamer: '', date: '', patch: profile.patch, start_timestamp: '',
      rating: '', hero: '', mode: 'solo', mechanisms: '', races: '', completeness: '完整录像',
      split: 'unassigned', placement: null, notes: '', snapshots: [snapshot()],
      capture_mode:'quick_combat',available_tribes:[],patch_basis:'user_declared_current'};
    quickSnapshot(g.snapshots[0]);return g;
  }
  function quickSnapshot(s) { s.capture_mode='quick_combat';s.hand_scope='combat_only';s.hand_count??=null;s.phase='pre_start';s.entry_touched=false;s.submitted=false;return s; }
  function useQuick(g) {
    g.capture_mode='quick_combat';if(!g.patch){g.patch=profile.patch;g.patch_basis='user_declared_current';}
    if(!Array.isArray(g.available_tribes))g.available_tribes=[];
    for(const s of g.snapshots){s.capture_mode='quick_combat';s.hand_scope='combat_only';s.hand_count??=null;s.entry_touched??=true;}
    return g;
  }
  function next(previous) {
    const s = clone(previous);
    s.id = id(); s.turn++; s.timestamp = ''; s.battle_id = ''; s.result = ''; s.evidence = [];
    s.reviewed = false; s.empty_board_confirmed = false; s.combat_details_checked = false;
    s.inherited_from = previous.id;
    s.notes = '';
    for (const zone of ['board', 'hand']) for (const c of s[zone]) {
      const old = c.id; c.id = id(); c.inherited_from = old; c.review = 'pending';
    }
    // Values are only a draft: a new round must explicitly re-confirm these statuses.
    s.hand_status = s.hand.length ? 'recorded' : 'unknown'; s.effects_status = 'unknown';
    return s;
  }
  function quickNext(previous) { const s=next(previous);s.capture_mode='quick_combat';s.hand_scope='combat_only';s.submitted=false;s.entry_touched=false;return s; }
  function finishRound(s) {s.submitted=true;s.entry_touched=true;if(!s.board.length)s.empty_board_confirmed=true;}
  function advanceGame(g,s) {
    finishRound(s);
    const existing=s.role==='player'?g.snapshots.find(x=>x.role==='player'&&x.turn===s.turn+1):null;
    if(existing)return existing;
    const following=quickNext(s);g.snapshots.push(following);return following;
  }
  function choose(c, card, cat, {keywordDefaults=false}={}) {
    const combatKeywords=new Set(['TAUNT','DIVINE_SHIELD','REBORN','VENOMOUS','POISONOUS','WINDFURY','STEALTH','DEATHRATTLE','CLEAVE']);
    Object.assign(c, {name: card.name, card_id: card.id, catalog_version: cat.version,
      catalog_sha256: cat.sha256, text: card.text, golden: card.golden,
      attack: null, health: null, max_health: null, keywords: keywordDefaults?(card.keywords||[]).filter(k=>combatKeywords.has(k)):[],
      keyword_basis:keywordDefaults?'reference_defaults_user_editable':'manual',details: '', review: 'pending',
      reference: {attack: card.attack, health: card.health, keywords: card.keywords,
        normal_id:card.normal_id||card.id,golden_id:card.golden_id||''}});
  }
  function baseValues(c) {
    if (!c.reference) throw Error('这张牌没有参考基础值');
    c.attack = Number.isSafeInteger(c.reference.attack) ? c.reference.attack : null;
    c.health = Number.isSafeInteger(c.reference.health) ? c.reference.health : null;
    c.review = 'pending';
  }
  function parseStats(text, count) {
    let lines = text.trim().split(/\r?\n|[;；]/).filter(x => x.trim());
    const pair = '(?:-?\\d+|\\?)\\s*[/／]\\s*(?:-?\\d+|\\?)';
    if (lines.length === 1 && count > 1 && new RegExp('^' + pair + '(?:\\s+' + pair + ')*$').test(text.trim()))
      lines = [...text.trim().matchAll(new RegExp(pair, 'g'))].map(x => x[0]);
    if (lines.length !== count) throw Error(`需要 ${count} 行攻血，当前 ${lines.length} 行；没有修改数据`);
    return lines.map((line, i) => {
      const match = line.trim().match(/^(-?\d+|\?)\s*(?:[\/／,，\t]|\s)\s*(-?\d+|\?)$/);
      if (!match) throw Error(`第 ${i + 1} 行格式不正确，请写 5/7 或 ?/?`);
      const values = match.slice(1).map(x => x === '?' ? null : Number(x));
      if (values.some(x => x !== null && !Number.isSafeInteger(x))) throw Error('数值超出精确整数范围');
      return values;
    });
  }
  function applyStats(shot, text) {
    const values = parseStats(text, shot.board.length);
    shot.board.forEach((c, i) => { [c.attack, c.health] = values[i]; c.review = 'pending'; });
    shot.reviewed = false;
  }
  function confirmBoard(s) {
    if (!s.board.length && !s.empty_board_confirmed) throw Error('战场为空：请明确勾选“确认空战场”');
    if (s.board.some(c => !c.name.trim() || c.golden === null || c.attack === null || c.health === null))
      throw Error('请补全卡名、普通/金色和攻血；未知内容可以保存草稿');
    s.board.forEach(c => { if (c.review !== 'inferred') c.review = 'observed'; });
    s.reviewed = true;
  }
  function quality(g, s) {
    const problems = [];
    const quick=g.capture_mode==='quick_combat'||s.capture_mode==='quick_combat';
    if(quick&&s.entry_touched===false&&!s.submitted)problems.push('尚未填写的自动草稿');
    if (!quick&&!g.source.trim()) problems.push('缺录像来源');
    if (!g.patch.trim()) problems.push('版本未知');
    if (!quick&&!s.timestamp.trim()) problems.push('缺视频时间戳');
    if (!quick&&!s.reviewed) problems.push('战场尚未核对');
    if (!s.board.length && !s.empty_board_confirmed) problems.push('空战场未确认');
    s.board.forEach((c, i) => {
      if (!c.name.trim() || [c.attack, c.health, c.golden].some(v => v === null)) problems.push(`第${i+1}张缺卡名/形态/攻血`);
      if (!quick&&c.review !== 'observed') problems.push(`第${i+1}张待确认或含推断`);
    });
    return {ready: problems.length === 0, problems};
  }
  function reminders(s) {
    const text = [s.skill_notes, s.global_notes, ...s.board.map(c => c.text + ' ' + c.details), ...s.hand.map(c => c.text)].join(' ');
    const result = [];
    if (/召唤|summon/i.test(text)) result.push('检查召唤物的全局攻血、种类与数量');
    if (/亡语|deathrattle/i.test(text)) result.push('检查额外亡语、亡语倍数及触发计数');
    if (/磁力|magnetic/i.test(text)) result.push('检查磁力附件：已计入的攻血不要再次相加');
    if (/手牌|hand/i.test(text)) result.push('检查战斗相关手牌及其当前数值');
    if (/本局|本回合|每当|this game|this turn|whenever/i.test(text)) result.push('检查技能/卡牌累计次数，不知道请注明未知');
    return result;
  }
  root.Collector = {clone, id, entity, snapshot, game, next, quickNext, quickSnapshot, useQuick, finishRound, advanceGame, configure, choose, baseValues, parseStats, applyStats, confirmBoard, quality, reminders};
  if (typeof module !== 'undefined') module.exports = root.Collector;
})(globalThis);
