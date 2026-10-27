const assert=require('node:assert/strict');
const S=require('../tools/warband_collector/web/search.js');
const cards=[
 {id:'fish',normal_id:'fish',golden_id:'fish_G',name:'鱼人甲',kind:'minion',races:['MURLOC'],tier:2,pool_known:true,recruit_pool:true,golden:false},
 {id:'fish_G',normal_id:'fish',golden_id:'fish_G',name:'鱼人甲',kind:'minion',races:['MURLOC'],tier:2,golden:true},
 {id:'beast',name:'野兽',kind:'minion',races:['BEAST'],tier:1,pool_known:true,recruit_pool:true},
 {id:'dual',name:'鹦鹉',kind:'minion',races:['BEAST','PIRATE'],tier:3,pool_known:true,recruit_pool:true},
 {id:'neutral',name:'中立',kind:'minion',races:[],tier:1,pool_known:true,recruit_pool:true},
 {id:'all',name:'全部类型',kind:'minion',races:['ALL'],tier:4,pool_known:true,recruit_pool:true},
 {id:'token',name:'衍生物',kind:'minion',races:['MURLOC'],tier:1,pool_known:true,recruit_pool:false},
 {id:'timewarp',name:'时空特殊',kind:'minion',races:['MURLOC'],tier:3,pool_known:true,recruit_pool:false},
 {id:'hero',name:'英雄',kind:'hero',races:[],tier:1}
];
const config={lobby:['PIRATE','MURLOC'],raceOrder:['BEAST','PIRATE','MURLOC']};
assert.deepEqual(S.query(cards,config).map(c=>c.id),['neutral','fish','dual','all']);
assert.deepEqual(S.query(cards,{...config,tribe:'MURLOC',tier:'2'}).map(c=>c.id),['fish']);
assert.deepEqual(S.query(cards,{...config,tribe:'NEUTRAL'}).map(c=>c.id),['neutral']);
assert(S.query(cards,{...config,includeOutside:true}).some(c=>c.id==='beast'));
assert(S.query(cards,{...config,includeSpecial:true}).some(c=>c.id==='token'));
assert(S.query(cards,{...config,text:'鹦鹉'}).some(c=>c.id==='dual'));
assert.equal(S.query(cards,config).filter(c=>c.name==='鱼人甲').length,1);
assert(S.query([...cards].reverse(),config).every((c,i)=>c.id===S.query(cards,config)[i].id));
const entered={card_id:'fish',golden:false,attack:17,health:23,reference:{normal_id:'fish',golden_id:'fish_G'}};
S.setGolden(entered,true,cards);assert.equal(entered.card_id,'fish_G');assert.equal(entered.attack,17);
S.setGolden(entered,false,cards);assert.equal(entered.card_id,'fish');assert.equal(entered.health,23);
console.log('collector search passed');
