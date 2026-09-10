// Execute the production control callbacks against a small event/DOM harness.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../10-source/rk3566/microduck/mediad/webclient/index.html'), 'utf8');
const start = html.indexOf('const twist =');
const end = html.indexOf('// ── looking', start);
assert(start > 0 && end > start);
const stateUpdate = html.match(/  const source = state\.control_source[\s\S]*?\$\("control-source"\)\.value = source;/)?.[0];
assert(stateUpdate);
const elements = new Map(), events = new Map(), timers = [], sent = [];
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id === 'control-source' ? 'drag' : '', style: {}, listeners: {}, pointers: new Set(),
    offsetWidth: 20, clientWidth: 100, clientHeight: 100,
    addEventListener(name, fn) { this.listeners[name] = fn; },
    getBoundingClientRect() { return {left: 0, top: 0, width: 100, height: 100}; },
    setPointerCapture(id) { this.pointers.add(id); },
    hasPointerCapture(id) { return this.pointers.has(id); },
    releasePointerCapture(id) { this.pointers.delete(id); },
  });
  return elements.get(id);
}
const context = vm.createContext({
  $: element, MAX_LINEAR: 0.3, MAX_ANGULAR: 1, INTENT_HZ: 10,
  open: () => true, setDc() {}, sourceLabels: {},
  tell(method, params) { sent.push({method, params: {...params}}); },
  setInterval(fn) { timers.push(fn); },
  addEventListener(name, fn) { events.set(name, fn); },
  document: {hidden: false, addEventListener() {}},
});
vm.runInContext(html.slice(start, end), context);
const receiveState = vm.runInContext(`(state) => {${stateUpdate}}`, context);
function tick() { for (const fn of timers) fn(); }
function key(name) { events.get('keydown')({key: name, target: {}, preventDefault() {}}); }
const pointer = {pointerId: 1, clientX: 50, clientY: 0};
element('pad-move').listeners.pointerdown(pointer);
tick();
assert.equal(sent.at(-1).params.source, 'drag');
assert.equal(sent.at(-1).params.vx, 0.3);
receiveState({control_source: 'gamepad'});
assert.equal(sent.at(-1).params.source, 'drag', 'handoff stop retains the original web source');
assert.equal(sent.at(-1).params.vx, 0);
assert(!element('pad-move').hasPointerCapture(1));
const stopped = sent.length;
tick();
receiveState({control_source: 'drag'});
tick();
assert.equal(sent.length, stopped, 'returning control cannot revive a held pointer');

key('w'); tick();
assert.equal(sent.at(-1).params.source, 'keyboard');
assert.equal(sent.at(-1).params.vx, 0.3);
element('pad-move').listeners.pointerdown(pointer);
tick();
assert.equal(sent.at(-1).params.source, 'drag', 'drag preempts held keyboard input');
key('a'); tick();
assert.equal(sent.at(-1).params.source, 'drag', 'additional keys cannot preempt drag');
element('pad-move').listeners.pointerup(pointer);
assert(sent.some(({params}) => params.source === 'drag' && params.release_source === true));
tick();
assert.equal(sent.at(-1).params.source, 'keyboard', 'releasing drag returns to still-held keys');
receiveState({control_source: 'bluetooth'});
assert.equal(sent.at(-1).params.source, 'keyboard');
assert.equal(sent.at(-1).params.vx, 0);
const before = sent.length;
tick();
key('w');
element('pad-move').listeners.pointerdown(pointer);
tick();
assert.equal(sent.length, before, 'browser cannot request a hardware control mode');
assert(sent.every(({params}) => ['drag', 'keyboard'].includes(params.source)));
console.log('PASS: web handoff clears held input; no stale motion or hardware source impersonation');
