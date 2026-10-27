'use strict';
const C = window.Collector, $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const phaseNames = {unknown:'时点未知',recruit_end:'招募末 · 回合末效果前',pre_start:'回合末已结算 · 开战效果前',starting:'开战效果进行中',after_start:'开战效果后 · 首次攻击前',mid_combat:'战斗中'};
const kwNames = {TAUNT:'嘲讽',DIVINE_SHIELD:'圣盾',REBORN:'复生',VENOMOUS:'烈毒',POISONOUS:'剧毒',WINDFURY:'风怒',STEALTH:'潜行',DEATHRATTLE:'亡语',CLEAVE:'顺劈'};
let data, catalogs=[], gameId, shotId, selected={zone:'board',index:0}, timer, saving=null, dirty=false, edits=0, blocked=false, undo=[];
let tribeFilter='',tierFilter='',includeOutside=false,includeSpecial=false;
let searchText='', catIndex=0, appendSearch=true, profile={patch:'37.0.1',tribes:{}};
const g = () => data?.games.find(x=>x.id===gameId), s = () => g()?.snapshots.find(x=>x.id===shotId), unit=()=>s()?.[selected.zone]?.[selected.index];
function notify(text) { $('toast').textContent=text;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,4500); }
function fail(error) { $('error').hidden=false;$('error').textContent=String(error.message||error); }
async function request(path, options={}) {
  const response=await fetch(path,options), value=await response.json();
  if(!response.ok){const e=Error(value.error||`HTTP ${response.status}`);e.status=response.status;throw e;}return value;
}
function draft() { return JSON.stringify({revision:data.revision,data,at:new Date().toISOString()}); }
function remember() {
  try { localStorage.setItem('at-warband-draft',draft()); }
  catch(e) { fail('浏览器草稿空间不足；请保存到本地服务或立即导出草稿。'+e.message); }
}
function mark(touch=true) {
  if(touch&&s())s().entry_touched=true;
  dirty=true;edits++;remember();$('save-status').textContent=blocked?'保存冲突 · 请先导出草稿':'有修改 · 即将自动保存';
  clearTimeout(timer);if(!blocked)timer=setTimeout(()=>flush().catch(fail),800);
  refreshQuality();
}
function checkpoint() { if(data){undo.push(C.clone(data));if(undo.length>40)undo.shift();} }
function mutate(fn,{touch=true}={}) { checkpoint();try{fn();mark(touch);render();}catch(e){data=undo.pop();fail(e);} }
async function flush() {
  clearTimeout(timer);
  if(saving){await saving;if(dirty)return flush();return;}
  if(!dirty)return;
  if(blocked)throw Error('保存冲突；请导出本页草稿，重新加载后核对，不自动覆盖另一页面');
  const version=edits,payload=C.clone(data);
  $('save-status').textContent='正在保存…';
  saving=(async()=>{
    try{
      const r=await request('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data:payload,revision:payload.revision})});
      data.revision=r.revision;data.updated_at=r.updated_at;
      if(edits===version){dirty=false;localStorage.removeItem('at-warband-draft');$('save-status').textContent='已保存 · '+new Date().toLocaleTimeString();$('error').hidden=true;}
      else {remember();$('save-status').textContent='新修改待保存';timer=setTimeout(()=>flush().catch(fail),200);}
    }catch(e){if(e.status===409)blocked=true;remember();$('save-status').textContent=blocked?'保存冲突 · 草稿已保留':'保存失败 · 草稿已保留';throw e;}
    finally{saving=null;}
  })();
  await saving;
}
function download(value,name,type='application/json') {
  const url=URL.createObjectURL(new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type}));
  const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function action(id, fn) { const e=$(id);if(e)e.onclick=()=>{try{const r=fn();if(r?.catch)r.catch(fail);}catch(err){fail(err);}}; }
function input(label,model,key,value,type='text',extra='') {return `<label class="field">${esc(label)}<input data-model="${model}" data-key="${key}" type="${type}" ${type==='number'?'step="1"':''} value="${esc(value)}" ${extra}></label>`;}
function area(label,model,key,value) {return `<label class="field wide">${esc(label)}<textarea data-model="${model}" data-key="${key}">${esc(value)}</textarea></label>`;}
function select(label,model,key,value,choices) {return `<label class="field">${esc(label)}<select data-model="${model}" data-key="${key}">${Object.entries(choices).map(([k,v])=>`<option value="${esc(k)}" ${String(value)===k?'selected':''}>${esc(v)}</option>`).join('')}</select></label>`;}
function bindFields() {
  document.querySelectorAll('[data-model]').forEach(e=>{
    e.onfocus=()=>checkpoint();
    e.oninput=()=>{
      const obj=e.dataset.model==='game'?g():e.dataset.model==='shot'?s():unit();if(!obj)return;
      let value=e.type==='number'?(e.value===''?null:Number(e.value)):e.value;
      if(e.type==='number'&&value!==null&&!Number.isSafeInteger(value)){e.setCustomValidity('请输入整数或留空');e.reportValidity();return;}e.setCustomValidity('');
      const key=e.dataset.key;
      if(key==='golden')value=value===''?null:value==='true';
      if(key==='hand_status'&&value!=='recorded'&&s().hand.length){notify('已录手牌不能直接标为空/未知；请先删除手牌。');e.value='recorded';return;}
      if(e.dataset.model==='entity'){
        if(key==='name'&&value!==obj.name){obj.card_id='';obj.catalog_version='';obj.catalog_sha256='';obj.text='';delete obj.reference;}
        if(key!=='review')obj.review='pending';s().reviewed=false;
      }
      obj[key]=value;
      if(e.dataset.model==='entity')refreshBoardLabels();
      if(e.dataset.model==='shot'&&['phase','hero','tier'].includes(key))obj.reviewed=false;
      mark(e.dataset.model!=='game');renderSidebar();
    };
  });
}
function renderSidebar() {
  const q=$('game-filter').value.trim().toLowerCase();$('games').replaceChildren();
  data.games.filter(x=>(x.streamer+' '+x.hero+' '+x.patch).toLowerCase().includes(q)).forEach((x,i)=>{
    const b=document.createElement('button');b.className='game-item'+(x.id===gameId?' active':'');
    b.innerHTML=`<b>第 ${data.games.indexOf(x)+1} 局 · ${esc(x.hero||'英雄待填')}</b><small>${x.snapshots.length} 个回合 / 阵容</small>`;
    b.onclick=()=>{gameId=x.id;shotId=x.snapshots.at(-1)?.id;selected={zone:'board',index:0};render();};$('games').append(b);
  });
}
function refreshQuality() {
  if(!$('quality')||!s())return;
  const q=C.quality(g(),s());$('quality').className='quality'+(q.ready?' ready':'');
  $('quality').textContent=q.ready?'✓ 可供战团统计审阅；精确战斗重建与训练入库仍需另行验收。':'待完善：'+q.problems.join('；');
  if($('round-status')){$('round-status').textContent=s().reviewed?'战场已核对':'草稿 · 待核对';$('round-status').className='status'+(s().reviewed?' good':'');}
}
function cardHtml(c,i) {
  return `<article class="unit ${selected.zone==='board'&&selected.index===i?'selected':''} ${c.golden?'golden':''}"><div class="slot">${i+1}<label class="gold-toggle"><input type="checkbox" data-golden="${i}" ${c.golden?'checked':''}>金</label></div><button class="unit-name" data-card="${i}">${esc(c.name||'选择卡牌')}</button><div class="stat-inputs"><input type="number" step="1" data-stat="attack" data-index="${i}" value="${esc(c.attack)}" aria-label="第${i+1}张攻击" placeholder="攻"><span>/</span><input type="number" step="1" data-stat="health" data-index="${i}" value="${esc(c.health)}" aria-label="第${i+1}张生命" placeholder="血"></div><span class="kw">${esc(c.keywords.map(k=>kwNames[k]||k).join(' '))}</span></article>`;
}
function render() {
  try{localStorage.setItem('at-warband-selection',JSON.stringify({gameId,shotId}));}catch(e){}
  renderSidebar();
  if(!g()){$('main').innerHTML='<div class="empty"><h1>一局一局录，每回合只填战斗信息。</h1><p>版本自动记录。每局选英雄、可用种族，然后顺着回合录阵容。</p><button id="first-game" class="primary">开始第一局</button></div>';action('first-game',newGame);return;}
  const game=g(),shot=s();
  if(!shot){$('main').innerHTML='<div class="panel"><button id="blank-round" class="primary">添加第一回合</button></div>';action('blank-round',()=>mutate(()=>{const x=C.quickSnapshot(C.snapshot());game.snapshots.push(x);shotId=x.id;}));return;}
  const chosen=unit(), tribes=game.available_tribes||[];
  if(!includeOutside&&tribes.length&&tribeFilter&&tribeFilter!=='NEUTRAL'&&!tribes.includes(tribeFilter))tribeFilter='';
  $('main').innerHTML=`<section class="panel game-panel"><div class="game-heading"><h2>第 ${data.games.indexOf(game)+1} 局 <small> · ${esc(game.patch||profile.patch)}</small></h2><button id="undo" ${undo.length?'':'disabled'}>撤销</button></div>
  <div class="game-info">${input('本局英雄','game','hero',game.hero,'text','list="heroes" placeholder="输入英雄名称"')}<div><div class="tribe-title">本局可用种族 <small id="tribe-count">${tribes.length}/5 · 每局选一次</small></div><div class="tribe-list">${Object.entries(profile.tribes).map(([k,v])=>`<label><input type="checkbox" data-tribe="${k}" ${tribes.includes(k)?'checked':''}> ${esc(v)}</label>`).join('')}</div></div></div>
  ${game.races&&!tribes.length?`<p class="tiny muted">原记录种族：${esc(game.races)}</p>`:''}<datalist id="heroes">${catalogs.flatMap(c=>c.cards).filter(c=>c.kind==='hero').map(c=>`<option value="${esc(c.name)}"></option>`).join('')}</datalist></section>
  <div class="nav"><button id="prev-round">←</button><select id="round-select" aria-label="切换回合">${game.snapshots.map(x=>`<option value="${x.id}" ${x.id===shot.id?'selected':''}>第 ${x.turn} 回合${x.role==='opponent'?' · 其他阵容':''}</option>`).join('')}</select><button id="next-round">→</button><span class="spacer"></span><button id="copy-round" class="primary" title="Alt+N">保存并下一回合 →</button></div>
  <section class="panel"><div class="fields round-fields">${input('回合','shot','turn',shot.turn,'number','min="1" max="200"')}${input('酒馆等级','shot','tier',shot.tier,'number','placeholder="未知可留空"')}${input('生命','shot','health',shot.health,'number')}${input('护甲','shot','armor',shot.armor,'number')}</div></section>
  <section class="panel"><div class="section-title"><h2>战场 · 从左到右</h2><small>直接改攻血，金卡勾“金”</small></div><div class="board">${shot.board.map(cardHtml).join('')}${shot.board.length<7?'<button class="unit add" id="add-board">＋ 添加随从</button>':''}</div>
  <div class="bulk-inline"><input id="bulk-stats" aria-label="整排攻血" placeholder="整排快填：5/7 12/18 20/20（也支持每张一行）"><button id="apply-stats">整排填入</button></div><p class="tiny muted">填回合末结算后的阵容。手上与战斗无关的牌不用记。</p></section>
  <div class="workbench"><section class="panel"><h2>搜索选卡</h2><div class="search-tools"><input id="card-search" value="${esc(searchText)}" placeholder="中文卡名 / 规则关键词，回车选第一张" aria-label="搜索卡牌"></div><div class="toolbar"><label class="tiny"><input id="append-cards" type="checkbox" ${appendSearch?'checked':''}> 连续加牌</label><button id="manual-card">手动填新卡</button><details><summary class="tiny">卡表选项</summary><select id="catalog-select" aria-label="卡表版本">${catalogs.map((c,i)=>`<option value="${i}" ${i===catIndex?'selected':''}>${esc(c.version)} · ${i===0?'参考卡表':'导入卡表'}</option>`).join('')}</select><button id="import-catalog">导入卡表</button><p class="tiny muted">卡名参考不代表当前卡牌实现；版本自动按${esc(profile.patch)}记录。</p></details></div><div class="search-filters"><label>种族<select id="search-tribe" aria-label="筛选种族"><option value="">本局全部（含中立）</option>${Object.entries(profile.tribes).filter(([k])=>includeOutside||!tribes.length||tribes.includes(k)).map(([k,v])=>`<option value="${k}" ${tribeFilter===k?'selected':''}>${esc(v)}</option>`).join('')}<option value="NEUTRAL" ${tribeFilter==='NEUTRAL'?'selected':''}>中立</option></select></label><label>星级<select id="search-tier" aria-label="筛选星级"><option value="">全部星级</option>${[1,2,3,4,5,6,7].map(t=>`<option value="${t}" ${String(tierFilter)===String(t)?'selected':''}>${t}星</option>`).join('')}</select></label></div><details class="tiny"><summary>特殊牌筛选</summary><label><input id="search-special" type="checkbox" ${includeSpecial?'checked':''}> 包含生成/特殊随从</label> <label><input id="search-outside" type="checkbox" ${includeOutside?'checked':''}> 显示局外种族</label></details><div id="search-count" class="tiny muted"></div><div id="card-results" class="search-results"></div></section>
  <section class="panel"><h2>${chosen?`${selected.zone==='board'?'战场':'战斗手牌'} ${selected.index+1} · ${esc(chosen.name||'选卡')}`:'选中随从的效果'}</h2>${chosen?editorHtml(chosen):'<p class="muted">搜卡直接加，或点击战场卡名编辑。</p>'}</section></div>
  <details class="panel"><summary>技能、全局效果、战斗相关手牌 · 有才填</summary><div class="fields two">${area('技能 / 战斗次数 / 神明等','shot','skill_notes',shot.skill_notes)}${area('全局加成 / 召唤物 / 其他战斗效果','shot','global_notes',shot.global_notes)}</div><div class="hand-list">${shot.hand.map((c,i)=>`<button data-hand="${i}" class="${selected.zone==='hand'&&selected.index===i?'active':''}">${esc(c.name||'选手牌')} ${esc(c.attack??'')}/${esc(c.health??'')}</button>`).join('')}<button id="add-hand" ${shot.hand.length>=10?'disabled':''}>＋ 战斗手牌</button></div>
  <p class="tiny muted">只录会在战斗中用到的手牌。若效果依赖手牌数量或随机选牌，再补总数或候选牌。</p><div class="fields two">${input('总手牌数 · 仅效果需要时填','shot','hand_count',shot.hand_count,'number','min="0" max="10"')}${input('英雄发生变化时补填','shot','hero',shot.hero,'text','list="heroes"')}</div><label class="tiny row-gap"><input id="already-started" type="checkbox" ${['starting','after_start','mid_combat'].includes(shot.phase)?'checked':''}> 这张阵容已触发过开战效果</label>${area('补充备注 · 可留空','shot','notes',shot.notes)}</details>
  <div class="toolbar bottom-actions"><button id="another-lineup">同回合再录一个阵容</button><button id="delete-shot" class="danger">删除本回合</button><button id="delete-game" class="danger">删除本局</button></div>`;
  bindFields();renderSearch();bindActions();bindQuick();
}
function editorHtml(c) {
  return `<div class="fields two">${input('卡名','entity','name',c.name)}${selected.zone==='hand'?select('形态','entity','golden',c.golden===null?'':String(c.golden),{'':'未知',false:'普通',true:'金色'}):''}${selected.zone==='hand'?input('当前攻击','entity','attack',c.attack,'number')+input('当前生命','entity','health',c.health,'number'):''}</div>
  <div class="keyword-list">${Object.entries(kwNames).map(([k,v])=>`<label><input type="checkbox" data-keyword="${k}" ${c.keywords.includes(k)?'checked':''}> ${v}</label>`).join('')}</div>
  ${area('附件 / 额外亡语 / 次数等 · 有才填','entity','details',c.details)}<div class="toolbar row-gap"><button id="move-left" ${selected.index===0?'disabled':''}>← 左移</button><button id="move-right" ${selected.index===s()[selected.zone].length-1?'disabled':''}>右移 →</button><button id="base-values" ${c.reference?'':'disabled'}>填基础攻血</button><button id="remove-card" class="danger">移除</button></div>
  <details class="row-gap"><summary class="tiny">卡牌文字</summary><p class="tiny muted">${esc(c.text||'新卡可直接记录，暂无参考文本。')}</p></details>`;
}
function renderSearch() {
  const container=$('card-results');if(!container)return;container.replaceChildren();const cat=catalogs[catIndex];if(!cat)return;
  const rows=window.CollectorSearch.query(cat.cards,{text:searchText,lobby:g().available_tribes||[],tribe:tribeFilter,tier:tierFilter,includeOutside,includeSpecial,zone:selected.zone,raceOrder:Object.keys(profile.tribes)});
  $('search-count').textContent=`${rows.length} 张 · 按星级 → 种族 → 卡名排序 · 金色在战场卡片上勾选`;
  let group=null;
  for(const c of rows){
    const tier=Number(c.tier)||0;
    if(tier!==group){group=tier;const h=document.createElement('h3');h.className='search-group';h.textContent=tier?`${tier} 星`:'特殊随从';container.append(h);}
    const b=document.createElement('button');b.className='card-result';
    const races=(c.races||[]).map(r=>profile.tribes[r]||(r==='ALL'?'全部类型':r)).join(' / ')||'中立';
    b.innerHTML=`<b>${esc(c.name)}</b> <span class="tiny">${esc(c.attack??'?')}/${esc(c.health??'?')} · ${esc(races)}</span><small>${esc(c.text)}</small>`;
    b.onclick=()=>{mutate(()=>{if(!unit()||(appendSearch&&unit().name))addCard(false);C.choose(unit(),c,cat,{keywordDefaults:true});s().reviewed=false;});if(appendSearch){$('card-search').focus();$('card-search').select();}};container.append(b);
  }
  const q=searchText.trim().toLowerCase(),seen=new Set();
  if(q&&!tribeFilter&&!tierFilter)for(const game of data.games.filter(x=>x.patch===g().patch))for(const shot of game.snapshots)for(const c of shot[selected.zone]){
    if(c.card_id||!c.name.toLowerCase().includes(q)||seen.has(c.name))continue;seen.add(c.name);
    const b=document.createElement('button');b.className='card-result';b.innerHTML=`<b>${esc(c.name)}</b><small>手动记录 · 种族/星级未分类</small>`;
    b.onclick=()=>mutate(()=>{if(!unit()||(appendSearch&&unit().name))addCard();Object.assign(unit(),C.entity(c.name),{golden:false});unit().catalog_version=game.patch;});container.append(b);
  }
  if(!rows.length&&!seen.size)container.textContent='当前筛选没有结果；可调整种族/星级，或手动填新卡。';
}
function addCard(renderNow=true) {
  const list=s()[selected.zone], max=selected.zone==='board'?7:10;if(list.length>=max)throw Error(`此区域最多${max}张牌`);
  list.push(C.entity());selected.index=list.length-1;s().reviewed=false;if(selected.zone==='hand')s().hand_status='recorded';
  list[selected.index].golden=false;
}
function newGame() {mutate(()=>{const x=C.game();data.games.push(x);gameId=x.id;shotId=x.snapshots[0].id;selected={zone:'board',index:0};appendSearch=true;},{touch:false});}
function copyRound(){mutate(()=>{shotId=C.advanceGame(g(),s()).id;selected={zone:'board',index:0};},{touch:false});flush().catch(fail);notify('进入下一回合，修改会自动保存。');}
function confirmBoard(){mutate(()=>C.confirmBoard(s()));}
function move(direction){mutate(()=>{const list=s()[selected.zone],j=selected.index+direction;if(j<0||j>=list.length)return;[list[j],list[selected.index]]=[list[selected.index],list[j]];selected.index=j;s().reviewed=false;});}
function refreshBoardLabels() {
  if(!s())return;document.querySelectorAll('[data-card]').forEach(e=>{e.textContent=s().board[Number(e.dataset.card)]?.name||'选择卡牌';});
}
function bindQuick() {
  document.querySelectorAll('[data-stat]').forEach(e=>{
    e.onfocus=checkpoint;e.oninput=()=>{
      const value=e.value===''?null:Number(e.value);
      if(value!==null&&!Number.isSafeInteger(value)){e.setCustomValidity('请输入整数');e.reportValidity();return;}e.setCustomValidity('');
      const card=s().board[Number(e.dataset.index)];card[e.dataset.stat]=value;card.review='pending';s().reviewed=false;mark();
    };
  });
  document.querySelectorAll('[data-golden]').forEach(e=>e.onchange=()=>mutate(()=>{window.CollectorSearch.setGolden(s().board[Number(e.dataset.golden)],e.checked,catalogs.flatMap(c=>c.cards));}));
  document.querySelectorAll('[data-tribe]').forEach(e=>e.onchange=()=>{
    checkpoint();const k=e.dataset.tribe,current=g().available_tribes||[];
    g().available_tribes=e.checked?[...new Set([...current,k])]:current.filter(x=>x!==k);mark(false);
    $('tribe-count').textContent=`${g().available_tribes.length}/5 · 每局选一次`;if(tribeFilter&&!g().available_tribes.includes(tribeFilter)&&tribeFilter!=='NEUTRAL')tribeFilter='';render();
  });
  $('already-started').onchange=e=>{checkpoint();s().phase=e.target.checked?'after_start':'pre_start';mark();};
  action('another-lineup',()=>mutate(()=>{const x=C.quickSnapshot(C.snapshot(s().turn,'opponent'));g().snapshots.push(x);shotId=x.id;selected={zone:'board',index:0};appendSearch=true;},{touch:false}));
}
function bindActions() {
  $('round-select').onchange=e=>{shotId=e.target.value;selected={zone:'board',index:0};render();};
  for(const [key,delta] of [['prev-round',-1],['next-round',1]])action(key,()=>{const i=g().snapshots.findIndex(x=>x.id===shotId),target=g().snapshots[i+delta];if(target){shotId=target.id;selected={zone:'board',index:0};render();}});
  action('copy-round',copyRound);action('confirm-board',confirmBoard);
  action('opponent',()=>mutate(()=>{const original=s(),battle=original.battle_id||C.id();original.battle_id=battle;const x=C.snapshot(original.turn,'opponent');x.timestamp=original.timestamp;x.battle_id=battle;x.phase=original.phase;g().snapshots.push(x);shotId=x.id;selected={zone:'board',index:0};}));
  action('add-board',()=>mutate(()=>{selected.zone='board';addCard();}));action('add-hand',()=>mutate(()=>{selected.zone='hand';addCard();}));
  document.querySelectorAll('[data-card]').forEach(e=>e.onclick=()=>{selected={zone:'board',index:Number(e.dataset.card)};render();});
  document.querySelectorAll('[data-hand]').forEach(e=>e.onclick=()=>{selected={zone:'hand',index:Number(e.dataset.hand)};render();});
  action('apply-stats',()=>{const text=$('bulk-stats').value;mutate(()=>C.applyStats(s(),text));});
  action('move-left',()=>move(-1));action('move-right',()=>move(1));
  action('base-values',()=>{mutate(()=>{C.baseValues(unit());s().reviewed=false;});notify('已填参考攻血，仍是待核对；版本或强化可能不同。');});
  action('remove-card',()=>mutate(()=>{s()[selected.zone].splice(selected.index,1);selected.index=Math.max(0,selected.index-1);s().reviewed=false;}));
  action('manual-card',()=>mutate(()=>{if(!unit()||(appendSearch&&unit().name))addCard();const current=unit();Object.assign(current,C.entity(searchText.trim()),{golden:false});s().reviewed=false;}));
  $('card-search').oninput=e=>{searchText=e.target.value;renderSearch();};$('catalog-select').onchange=e=>{catIndex=Number(e.target.value);renderSearch();};
  $('card-search').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('card-results').querySelector('button')?.click();}};
  $('search-tribe').onchange=e=>{tribeFilter=e.target.value;renderSearch();};$('search-tier').onchange=e=>{tierFilter=e.target.value;renderSearch();};$('search-special').onchange=e=>{includeSpecial=e.target.checked;renderSearch();};$('search-outside').onchange=e=>{includeOutside=e.target.checked;tribeFilter='';render();};
  $('append-cards').onchange=e=>{appendSearch=e.target.checked;};
  action('import-catalog',()=>$('catalog-file').click());action('add-evidence',()=>$('evidence-file').click());
  document.querySelectorAll('[data-keyword]').forEach(e=>e.onchange=()=>{const k=e.dataset.keyword;mutate(()=>{const c=unit();c.keywords=e.checked?[...new Set([...c.keywords,k])]:c.keywords.filter(x=>x!==k);c.keyword_basis='user_edited';c.review='pending';s().reviewed=false;});});
  action('undo',()=>{if(undo.length){const revision=data.revision;data=undo.pop();data.revision=revision;if(!g()){gameId=data.games[0]?.id;shotId=g()?.snapshots[0]?.id;}if(!s())shotId=g()?.snapshots[0]?.id;mark(false);render();}});
  action('delete-shot',()=>{if(!confirm('删除这条回合记录？可使用“撤销”恢复。'))return;mutate(()=>{g().snapshots=g().snapshots.filter(x=>x.id!==shotId);shotId=g().snapshots[0]?.id;});});
  action('delete-game',()=>{if(!confirm('删除本局及其回合记录？可撤销，已保存历史也会保留。'))return;mutate(()=>{data.games=data.games.filter(x=>x.id!==gameId);gameId=data.games[0]?.id;shotId=g()?.snapshots[0]?.id;});});
}
function renderEvidence() {
  const target=$('evidence');target.replaceChildren();for(const name of s().evidence){const f=document.createElement('figure'),a=document.createElement('a');a.href='/attachments/'+name;a.target='_blank';a.rel='noopener';
    if(/\.(png|jpe?g|webp)$/.test(name)){const image=document.createElement('img');image.src=a.href;image.alt='录像证据截图';a.append(image);}else a.textContent='查看视频片段';
    const caption=document.createElement('figcaption'),b=document.createElement('button');b.textContent='移除关联';b.onclick=()=>mutate(()=>{s().evidence=s().evidence.filter(x=>x!==name);});caption.append(b);f.append(a,caption);target.append(f);}
}
async function uploadEvidence(files) {
  if(!s())throw Error('请先选择一条回合记录');const target=s();
  for(const file of files){const name=file.name||'clipboard.png';const r=await request('/api/attachment',{method:'POST',headers:{'X-Filename':encodeURIComponent(name),'Content-Type':'application/octet-stream'},body:file});checkpoint();target.evidence.push(r.filename);mark();}
  if(target===s())renderEvidence();notify('证据已保存；完整 ZIP 备份会包含附件。');
}
action('new-game',newGame);action('first-game',newGame);action('save',()=>flush());
action('draft',()=>download(data,'warbands-draft.json'));
action('backup',async()=>{await flush();location.href='/api/backup.zip';});action('export',async()=>{await flush();location.href='/api/export.jsonl';});
action('import',()=>$('import-file').click());action('help',()=>$('help-dialog').showModal());action('close-help',()=>$('help-dialog').close());
$('game-filter').oninput=renderSidebar;
$('evidence-file').onchange=async e=>{try{await uploadEvidence([...e.target.files]);}catch(err){fail(err);}e.target.value='';};
$('catalog-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const r=await request('/api/catalog',{method:'POST',headers:{'Content-Type':'application/json'},body:await file.text()});catalogs=await request('/api/catalog');catIndex=catalogs.length-1;render();notify(`已导入 ${r.version} 卡表 · ${r.count} 张；尚未验证游戏实现。`);}catch(err){fail(err);}e.target.value='';};
$('import-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;await flush();const r=await request('/api/import',{method:'POST',headers:{'Content-Type':file.name.toLowerCase().endsWith('.zip')?'application/zip':'application/json','X-Revision':String(data.revision)},body:file});data=r;data.games.forEach(C.useQuick);gameId=data.games.at(-1)?.id;shotId=g()?.snapshots[0]?.id;undo=[];render();notify('导入成功；原有局保留，相同局ID会拒绝重复导入。');}catch(err){fail(err);}e.target.value='';};
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key.toLowerCase()==='s'){e.preventDefault();flush().catch(fail);}else if((e.altKey&&e.key.toLowerCase()==='n'||e.ctrlKey&&e.key==='Enter')&&s()){e.preventDefault();copyRound();}});
window.addEventListener('beforeunload',e=>{if(dirty){remember();e.preventDefault();e.returnValue='';}});
(async()=>{try{
  [data,catalogs,profile]=await Promise.all([request('/api/data'),request('/api/catalog'),request('/api/profile')]);C.configure(profile);data.games.forEach(C.useQuick);$('profile-label').textContent=profile.patch+' · 版本已固定';
  let selection={};try{selection=JSON.parse(localStorage.getItem('at-warband-selection')||'{}');}catch(e){}
  gameId=data.games.some(x=>x.id===selection.gameId)?selection.gameId:data.games.at(-1)?.id;
  shotId=g()?.snapshots.some(x=>x.id===selection.shotId)?selection.shotId:g()?.snapshots.at(-1)?.id;
  render();$('save-status').textContent='自动保存已开启';
  const raw=localStorage.getItem('at-warband-draft');if(raw){const old=JSON.parse(raw);$('error').hidden=false;$('error').textContent='发现尚未保存的浏览器草稿。';
    const button=document.createElement('button');button.textContent='导出恢复草稿';button.onclick=()=>download(old.data,'warbands-recovery.json');$('error').append(' ',button);
    if(old.revision===data.revision){const restore=document.createElement('button');restore.textContent='恢复此草稿';restore.onclick=()=>{checkpoint();data=old.data;data.games.forEach(C.useQuick);gameId=data.games[0]?.id;shotId=g()?.snapshots[0]?.id;mark(false);render();};$('error').append(' ',restore);}else $('error').append(' 服务端已有其他版本，禁止自动覆盖。');
  }
}catch(e){fail(e);$('save-status').textContent='连接失败';}})();
