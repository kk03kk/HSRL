(function(root){
  'use strict';
  const races=c=>(c.races||[]).map(x=>x==='MECH'?'MECHANICAL':x);
  function lobbyEligible(card,lobby){
    if(!lobby.length)return true;
    const r=races(card);
    // Dual types need at least ONE available type. Neutral and All remain visible.
    return !r.length||r.includes('ALL')||r.some(x=>lobby.includes(x));
  }
  function query(cards,{text='',lobby=[],tribe='',tier='',includeOutside=false,includeSpecial=false,zone='board',raceOrder=[]}={}){
    const q=text.trim().toLocaleLowerCase(),order=new Map(raceOrder.map((x,i)=>[x,i]));
    const matchRank=c=>{
      const n=(c.name||'').toLocaleLowerCase(),en=(c.name_en||'').toLocaleLowerCase(),id=(c.id||'').toLocaleLowerCase();
      return !q?0:n===q||en===q||id===q?0:n.startsWith(q)||en.startsWith(q)?1:n.includes(q)||en.includes(q)||id.includes(q)?2:3;
    };
    const raceRank=c=>races(c).length?Math.min(...races(c).map(x=>order.get(x)??998)):999;
    const filtered=cards.filter(c=>{
      if(!['minion',...(zone==='hand'?['spell','tavern_spell']:[])].includes(c.kind))return false;
      if(c.golden)return false; // one row per normal/golden family; checkbox controls the variant
      if(!includeOutside&&!lobbyEligible(c,lobby))return false;
      const r=races(c);
      if(tribe==='NEUTRAL'&&r.length)return false;
      if(tribe&&tribe!=='NEUTRAL'&&!r.includes(tribe)&&!r.includes('ALL'))return false;
      if(tier!==''&&Number(c.tier)!==Number(tier))return false;
      if(!includeSpecial&&c.kind==='minion'&&(c.pool_known&&!c.recruit_pool||!(Number(c.tier)>0)))return false;
      return !q||[c.name,c.name_en,c.id,c.text].join(' ').toLocaleLowerCase().includes(q);
    });
    const unique=new Map();
    for(const c of filtered){const key=c.normal_id||c.id;if(!unique.has(key))unique.set(key,c);}
    return [...unique.values()].sort((a,b)=>(Number(a.tier)||99)-(Number(b.tier)||99)||raceRank(a)-raceRank(b)||matchRank(a)-matchRank(b)||
      (a.name||'').localeCompare(b.name||'','zh-CN',{numeric:true})||(a.id||'').localeCompare(b.id||''));
  }
  function setGolden(card,golden,allCards){
    const current=allCards.find(c=>c.id===card.card_id);
    const normal=current?.normal_id||card.reference?.normal_id||card.card_id;
    const targetId=golden?(current?.golden_id||card.reference?.golden_id):normal;
    const target=allCards.find(c=>c.id===targetId);
    card.golden=golden;
    if(target){card.card_id=target.id;card.text=target.text;card.reference={...card.reference,
      attack:target.attack,health:target.health,normal_id:target.normal_id,golden_id:target.golden_id};}
    return !!target;
  }
  root.CollectorSearch={query,lobbyEligible,setGolden};
  if(typeof module!=='undefined')module.exports=root.CollectorSearch;
})(globalThis);
