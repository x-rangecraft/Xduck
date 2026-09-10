const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../10-source/rk3566/microduck/mediad/webclient/index.html'), 'utf8');
for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {value: '', textContent: '', disabled: true, addEventListener(name, fn) {this[name] = fn;}});
  return elements.get(id);
};
let connected = true, fail = false;
const sent = [];
const context = vm.createContext({$, open: () => connected, Number, Error, async call(method, params) {
  sent.push({method, params});
  if (fail) throw Error('mixer unavailable');
  return {percent: params.percent === undefined ? 96 : params.percent};
}});
vm.runInContext(html.slice(html.indexOf('let volumeBusy ='), html.indexOf('let soundSerial =')), context);
(async () => {
  await context.refreshVolume();
  assert.equal($('speaker-volume').value, 96);
  assert.equal($('speaker-volume').disabled, false);
  $('speaker-volume').value = '25';
  $('speaker-volume').input();
  assert.equal(sent.length, 1, 'dragging must not flood RPC');
  await $('speaker-volume').change();
  assert.equal(sent[1].method, 'robot.volume');
  assert.equal(sent[1].params.percent, 25);
  fail = true;
  await context.refreshVolume(80);
  assert.equal($('speaker-volume').value, 25, 'failed writes must restore confirmed value');
  assert.match($('volume-feedback').textContent, /mixer unavailable/);
  connected = false;
  await context.refreshVolume();
  assert.equal($('speaker-volume').disabled, true);
  console.log('web volume: passed');
})().catch(e => {console.error(e); process.exitCode = 1;});
