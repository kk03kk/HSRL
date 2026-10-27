// 无浏览器/网络的页面逻辑单元测试；不作为像素级浏览器验收。
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = new Map();let blob;
function node(id) {if(!nodes.has(id))nodes.set(id,{value:0,checked:false,textContent:'',children:[],replaceChildren(){this.children=[];},append(x){this.children.push(x);},click(){}});return nodes.get(id);}
const context={document:{getElementById:node,createElement:()=>({click(){}})},Blob,
  URL:{createObjectURL(x){blob=x;return 'unit-test';},revokeObjectURL(){}},setTimeout:fn=>fn(),console};
vm.createContext(context);vm.runInContext(js,context);
(async()=>{
  const me={hero:{name:'测试英雄'},board:[],hand:[],shop:[],gold:3,tier:1,health:30,armor:0,pending:null};
  const d={lobby:0,seat:0,turn:1,decision:1,before:{me,turn:1,phase:'recruit',opponents:[]},after:{me,turn:1,phase:'recruit',opponents:[]},action:0,applied:true,observation:{mask:[true,...Array(244).fill(false)]},topk:null};
  const rows=[{format:'taverngym.trace.v1',identity:{digest:'fixture'},seed:1},{type:'tick',decisions:[d,{...d,turn:2}]},{type:'summary'}];
  await node('file').onchange({target:{files:[{text:async()=>rows.map(JSON.stringify).join('\n')}]}});
  assert.equal(node('error').textContent,'');assert.match(node('before').textContent,/测试英雄/);
  node('next').onclick();assert.match(node('meta').textContent,/回合 2/);
  node('prev').onclick();assert.match(node('meta').textContent,/回合 1/);
  node('raw').checked=true;node('raw').onchange();assert.match(node('before').textContent,/"me"/);
  node('note').onclick();const annotation=JSON.parse(await blob.text());assert.equal(annotation.training_approved,false);assert.equal(annotation.format,'taverngym.annotation.v1');
  await node('file').onchange({target:{files:[{text:async()=>'{bad'}]}});assert.notEqual(node('error').textContent,'');
  console.log('viewer import/navigation/raw/annotation/errors passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
