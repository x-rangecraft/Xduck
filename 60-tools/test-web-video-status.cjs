// Browser-console regression without a robot: video negotiation and operator status rendering.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(
  __dirname, '../10-source/rk3566/microduck/mediad/webclient/index.html'), 'utf8');

for (const id of ['video-toggle', 'video-state', 'control-status', 'action-status',
  'sound-status', 'experiment-ownership']) {
  assert(html.includes(`id="${id}"`), `${id} must be visible in the console`);
}
assert.match(html, /if \(!restartingForVideo\) videoEnabled = false;/,
  'automatic session reconnect must preserve the requested video mode');
assert.match(html, /if \(reconnectForVideo\)[\s\S]*videoReconnectAttempts <= 5[\s\S]*300 \* videoReconnectAttempts/,
  'a signaller close during the video switch must retry after peer teardown');
assert.match(html, /if \(!ws && restartingForVideo\) \$\("connect"\)\.click\(\);/,
  'the delayed retry must reconnect only while the video switch is still pending');
assert.match(html, /control\.onopen = \(\) => \{[\s\S]*restartingForVideo = false;[\s\S]*videoReconnectAttempts = 0;/,
  'a replacement control channel must finish and reset the bounded reconnect');
assert.match(html, /if \(!videoEnabled && !restartingForVideo\) \{[\s\S]*startHttpControl\(\);/,
  'the default connection must use the control-only HTTP transport');
assert.match(html, /fetch\("\/api\/control"[\s\S]*method: "POST"/,
  'control-only mode must send JSON-RPC to the same-origin gateway');
assert.match(html, /httpStateTimer = setTimeout\(subscribeRobotState, 100\)/,
  'control-only robot state must use bounded sequential long polling');
assert.match(html, /const statsPeer = pc;[\s\S]*if \(!videoEnabled \|\| pc !== statsPeer\) return;/,
  'an old getStats result must not repopulate the closed-video display');
assert.match(html, /if \(!videoEnabled\) \{[\s\S]*ws\.close\(\);[\s\S]*return;/,
  'closing video must tear down the signaller socket before returning to HTTP control');

const sent = [];
const transceivers = [
  {receiver: {track: {kind: 'video'}}, direction: 'recvonly'},
  {receiver: {track: {kind: 'audio'}}, direction: 'recvonly'},
];
const pc = {
  async setRemoteDescription(value) { this.remote = value; },
  getTransceivers() { return transceivers; },
  async createAnswer() { return {type: 'answer', sdp: 'answer-sdp'}; },
  async setLocalDescription(value) { this.local = value; },
};
const offerStart = html.indexOf('async function answerPeerOffer');
const offerEnd = html.indexOf('function newPeerConnection', offerStart);
assert(offerStart > 0 && offerEnd > offerStart);
const offerContext = vm.createContext({pc, videoEnabled: false, sessionId: 'session-1',
  send(value) { sent.push(value); }});
vm.runInContext(html.slice(offerStart, offerEnd), offerContext);

const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {innerHTML: ''});
  return elements.get(id);
};
const statusStart = html.indexOf('function renderOperatorStatus');
const statusEnd = html.indexOf('function onState', statusStart);
assert(statusStart > 0 && statusEnd > statusStart);
const statusContext = vm.createContext({$});
vm.runInContext(html.slice(statusStart, statusEnd), statusContext);

(async () => {
  await offerContext.answerPeerOffer({type: 'offer', sdp: 'offer-sdp'});
  assert.equal(transceivers[0].direction, 'inactive', 'video defaults to no receive');
  assert.equal(transceivers[1].direction, 'recvonly', 'non-video media is untouched');
  assert.equal(sent[0].sessionId, 'session-1');

  offerContext.videoEnabled = true;
  await offerContext.answerPeerOffer({type: 'offer', sdp: 'offer-sdp-2'});
  assert.equal(transceivers[0].direction, 'recvonly', 'opening video negotiates receive');

  statusContext.renderOperatorStatus({control_state: 'initializing', policy: 'held'});
  assert.match($('control-status').innerHTML, /初始化/);
  assert.match($('action-status').innerHTML, /无动作执行/);
  assert.match($('sound-status').innerHTML, /无声音执行/);
  statusContext.renderOperatorStatus({control_state: 'policy_on', policy: 'rise', sound_state: 'chirp'});
  assert.match($('control-status').innerHTML, /策略开启/);
  assert.match($('action-status').innerHTML, /执行站立/);
  assert.match($('sound-status').innerHTML, /播放啾啾/);
  console.log('web video and current-status rendering: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
