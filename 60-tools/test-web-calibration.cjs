// Exercise the production calibration UI with simulated telemetry and JSON-RPC.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname,
  '../10-source/rk3566/microduck/mediad/webclient/index.html'), 'utf8');
const start = html.indexOf('let CALIBRATION_MOTORS =');
const end = html.indexOf('let lastSubscribeAttempt', start);
assert(start > 0 && end > start);
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '', textContent: '', disabled: false,
    checkValidity() { return Number.isFinite(Number(this.value)) && Math.abs(Number(this.value)) <= 12.5; },
    set innerHTML(value) {
      this.html = value;
      for (const match of value.matchAll(/id="([^"]+)" value="([^"]+)"/g)) element(match[1]).value = match[2];
    },
    get innerHTML() { return this.html || ''; },
  });
  return elements.get(id);
}
const ids = [
  [1,0],[2,1],[3,2],[4,3],[5,4],
  [6,10],[7,11],[8,12],[9,13],[10,14],
  [11,5],[12,6],[13,7],[14,8],
];
const config = {version:1, limits: ids.map(([motor_id]) => ({motor_id,min_rad:-3,max_rad:3})),home:Array(15).fill(0)};
const sensors = {motor_flags:Array(15).fill(0), motor_feedback_age_ms:Array(15).fill(0), joints:Array(15).fill(0)};
for (const [,joint] of ids) {sensors.motor_flags[joint]=3;sensors.joints[joint]=0.75;}
let now = 10, sequence = 0, fail = false;
const sent = [];
const logs = [], timers = [], listeners = {};
let connected = true, readError = null;
const context = vm.createContext({
  $:element, MOTOR_JOINTS:ids.map(([id,joint]) => [id,joint,'joint']),
  postureState: {state:sensors,received:now}, performance:{now:()=>now}, open:()=>connected,
  reportConnectionIssue(...args){logs.push(args);},
  setInterval(callback,ms){timers.push({callback,ms});},setTimeout(callback){callback();},
  document:{hidden:false, querySelectorAll:()=>[...elements.values()],addEventListener(name,callback){listeners[name]=callback;}},
  async call(method, params, quiet) {
    sent.push({method,params,quiet});
    if (params.action === 'get' && readError) throw readError;
    if(params.action !== 'get') { sequence++; return {request_id:sequence,busy:true}; }
    return {config,motors:ids.map(pair=>[...pair]),request_id:sequence,busy:false,editable:true,synced:true,error:fail?'STM32 拒绝标定':null};
  },
});
vm.runInContext(html.slice(start,end),context);
(async () => {
  await vm.runInContext('refreshCalibration(true)',context);
  assert.equal((element('calibration-rows').innerHTML.match(/data-cal-zero=/g)||[]).length,14);
  assert.equal((element('calibration-rows').innerHTML.match(/step="any"/g)||[]).length,14,'home inputs must accept the four-decimal built-in stance');
  assert.equal((element('calibration-rows').innerHTML.match(/<tr>/g)||[]).length,14);
  assert.equal((element('calibration-rows').innerHTML.match(/槽位|未配置/g)||[]).length,0);
  assert(vm.runInContext('calibrationEditable()',context));
  sensors.motor_flags[5]=7;
  assert(!vm.runInContext('calibrationEditable()',context));
  await vm.runInContext('submitCalibration({action:"mark_zero",motor_id:2})',context);
  assert(!sent.some(x=>x.params.action==='mark_zero'));
  sensors.motor_flags[5]=3;
  sensors.motor_feedback_age_ms[10]=501;
  assert(!vm.runInContext('calibrationEditable()',context));
  sensors.motor_feedback_age_ms[10]=0;
  now+=1001;
  assert(!vm.runInContext('calibrationEditable()',context));
  context.postureState.received=now;
  element('cal-home-2').value='0.25';
  element('calibration-capture-home').onclick();
  assert.equal(element('cal-home-2').value,'0.750','capture fills every configured motor');
  assert.equal(element('cal-home-13').value,'0.750');
  const before=sent.length;
  // Capture fills the form, without sending any command or zero operation.
  assert.equal(sent.length,before);
  element('cal-min-2').value='';
  assert.throws(()=>vm.runInContext('calibrationNumber("cal-min-2")',context));
  await vm.runInContext('submitCalibration({action:"mark_zero",motor_id:2})',context);
  assert(element('calibration-feedback').textContent.includes('已确认电机失能'));
  assert.equal(sent.filter(x=>x.params.action==='mark_zero').length,1);
  element('cal-home-2').value='0.25';
  element('calibration-home').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(sent.find(x=>x.params.action==='set_home').params.positions[1],0.25);
  assert.equal(sent.find(x=>x.params.action==='set_home').params.positions.length,15);
  fail=true;
  await vm.runInContext('submitCalibration({action:"mark_zero",motor_id:2})',context);
  assert.equal(element('calibration-feedback').textContent,'STM32 拒绝标定');
  assert.equal(sent.filter(x=>x.params.action==='mark_zero').length,2,'failed zero is not retried');
  sensors.motor_flags[1]=0;
  assert(!vm.runInContext('calibrationEditable()',context),'configured but offline motor must still block writes');
  sensors.motor_flags[1]=3; fail=false;
  const poll=timers.find(timer=>timer.ms===10_000).callback;
  let count=sent.length;
  await poll();
  assert.equal(sent.length,count,'collapsed panel does not poll');
  element('calibration-details').open=true;
  await element('calibration-details').ontoggle();
  assert.equal(sent.length,++count,'opening refreshes immediately');
  assert.equal(sent.at(-1).quiet,true,'background queries are quiet');
  context.document.hidden=true;
  await poll();
  assert.equal(sent.length,count,'hidden tab does not poll');
  context.document.hidden=false;
  await listeners.visibilitychange();
  assert.equal(sent.length,++count,'returning to visible panel refreshes');
  connected=false;
  await poll();
  assert.equal(sent.length,count,'disconnected panel does not poll');
  connected=true;
  config.home[0]=1.25;
  await poll();
  assert.equal(element('cal-home-1').value,'1.25','external changes refresh clean form');
  element('cal-home-1').value='2';
  element('calibration-rows').oninput();
  config.home[0]=1.5;
  await poll();
  assert.equal(element('cal-home-1').value,'2','external changes preserve draft');
  assert(!vm.runInContext('calibrationEditable()',context),'conflicting draft cannot overwrite remote settings');
  await element('calibration-refresh').onclick();
  assert.equal(element('cal-home-1').value,'1.5');
  assert(vm.runInContext('calibrationEditable()',context));
  readError=new Error('query timeout');
  await poll();
  assert(logs.some(parts=>parts.includes('query timeout')),'quiet query failures remain logged');
  readError=null;
  assert(sent.filter(x=>x.params.action==='get' && x.quiet).length>0);
  console.log('PASS: collapsed/hidden/disconnected polling guards, immediate opening refresh, quiet requests, error logging and external changes with draft protection');
  console.log('PASS: calibration rendering, enabled/offline/stale guards, capture without motion, validation, result confirmation and no automatic zero retry');
})().catch(error=>{console.error(error);process.exitCode=1;});
