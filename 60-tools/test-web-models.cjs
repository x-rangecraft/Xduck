// Exercise production model UI with a simulated connected robot; never touches hardware.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../10-source/rk3566/microduck/mediad/webclient/index.html'),'utf8');
// Catch syntax errors in the complete production script too.
for(const script of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
const start=html.indexOf('// The server commits in its bus-owning loop;');
const end=html.indexOf('</script>',start);
const elements=new Map();
function element(id='') {
  if(id && elements.has(id))return elements.get(id);
  const e={value:'',textContent:'',files:[],disabled:false,children:[],className:'',
    append(...children){this.children.push(...children);},
    replaceChildren(...children){this.children=children;if(id==='model-slot')this.value=children[0]?.value||'';},
    querySelectorAll(tag){return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag)]);}};
  if(id)elements.set(id,e);return e;
}
let connected=true,clock=10000, failAction=null;
const status={slots:[{id:'walk',file:'alpha_walking.onnx',active:null,history:[{id:'1000-1-1',name:'<unsafe>.onnx'}]}],phase:'idle',detail:'等待导入',editable:true};
const sent=[];
const context=vm.createContext({$:element,Date:class extends Date{static now(){return clock;}},open:()=>connected,
  document:{createElement(tag){const e=element();e.tag=tag;return e;}},Uint8Array,console,setInterval(){},
  async call(method,params){sent.push(params);if(params.action===failAction)throw Error('模拟失败');
    if(params.action==='begin_bundle')return {...status,phase:'uploading',token:'upload-token'};
    if(params.action==='finish_bundle'||params.action==='rollback')return {...status,phase:'pending',detail:'等待失能确认'};
    return status;}});
vm.runInContext(html.slice(start,end),context);
const file=(name,bytes)=>({name,size:bytes.length,slice(start,end){return{async arrayBuffer(){return Uint8Array.from(bytes.slice(start,end)).buffer;}};}});
element('model-policy-file').files=[file('policy.py',[112,121])];
element('model-file').files=[file('model.onnx',[0,255,3])];
(async()=>{
  await vm.runInContext('refreshModels()',context);
  assert.equal(element('model-import').disabled,false);
  const history=element('model-history').children[0];
  assert(history.children[0].textContent.startsWith('<unsafe>.onnx'), 'filenames must be text, never HTML');
  connected=false;vm.runInContext('modelControls()',context);assert(element('model-import').disabled);
  connected=true;clock+=4000;vm.runInContext('modelControls()',context);assert(element('model-import').disabled,'stale status must lock mutation');
  await vm.runInContext('refreshModels()',context);
  await element('model-import').onclick();
  assert.deepEqual(sent.filter(p=>['begin_bundle','bundle_chunk','finish_bundle'].includes(p.action)).map(p=>p.action),['begin_bundle','bundle_chunk','bundle_chunk','finish_bundle']);
  assert.equal(sent.find(p=>p.action==='bundle_chunk'&&p.file==='policy').hex,'7079');
  assert.equal(sent.find(p=>p.action==='bundle_chunk'&&p.file==='model').hex,'00ff03');
  assert(element('model-policy-file').disabled);
  assert(element('model-import').disabled,'pending commit prevents a second import');
  status.phase='done';await vm.runInContext('refreshModels()',context);
  await element('model-history').querySelectorAll('button')[0].onclick();
  assert.equal(sent.at(-1).action,'rollback');
  status.phase='uploading';await vm.runInContext('refreshModels()',context);
  assert.equal(element('model-import').disabled,false,'an abandoned upload must be replaceable after reconnect');
  element('model-policy-file').files=[file('policy.txt',[1])];
  const before=sent.length;await element('model-import').onclick();assert.equal(sent.length,before);
  element('model-policy-file').files=[file('policy.py',[112,121])];
  failAction='bundle_chunk';await element('model-import').onclick();
  assert(element('model-feedback').textContent.includes('模拟失败'));
  status.editable=false;await vm.runInContext('refreshModels()',context);
  assert(element('model-history').querySelectorAll('button').every(b=>b.disabled));
  console.log('Model UI: upload, rollback, stale/disconnected safety gates, resume, pending lock and errors passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
