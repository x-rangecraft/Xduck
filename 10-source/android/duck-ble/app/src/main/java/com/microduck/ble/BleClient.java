package com.microduck.ble;

import android.annotation.SuppressLint;
import android.bluetooth.*;
import android.bluetooth.le.*;
import android.content.*;
import android.os.*;
import android.os.ParcelUuid;
import java.util.*;
import java.util.function.Supplier;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** All mutable transport state is confined to the main looper; one GATT operation at a time. */
@SuppressLint("MissingPermission")
final class BleClient {
    static final UUID SERVICE=UUID.fromString("6f5d2a10-3b47-4c8e-9a1f-2d7e8c4b6019");
    static final UUID RPC=UUID.fromString("6f5d2a11-3b47-4c8e-9a1f-2d7e8c4b6019");
    static final UUID CCC=UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");
    interface Listener {
        void status(String message, boolean ready);
        void found(String address,String name,int rssi);
        void ready(int api);
        void line(String line);
        void state(Wire.State state);
        void rssi(int rssi);
        void disconnected(String reason);
    }
    private final Context context;
    private final Listener listener;
    private final Handler handler=new Handler(Looper.getMainLooper());
    private final BluetoothAdapter adapter;
    private BluetoothGatt gatt;
    private BluetoothGattCharacteristic pipe;
    private final Wire.Lines lines=new Wire.Lines();
    private final Wire.States states=new Wire.States();
    private final ConnectionLifecycle lifecycle=new ConnectionLifecycle();
    private final ArrayDeque<Frame> commands=new ArrayDeque<>();
    private Frame active, motion, stop;
    private int offset, payload=20, apiVersion, retries;
    private boolean foreground, ready, scanning, rssiBusy, userDisconnected;
    private String target;
    private long operationStarted, lastRssi;
    private String phase="";
    private boolean probing;
    private long probeStarted;
    private static final class Frame {
        byte[] data;
        final Supplier<byte[]> materialize;
        final long created=SystemClock.elapsedRealtime();
        final boolean motion;
        Frame(Supplier<byte[]> materialize,boolean motion) { this.materialize=materialize; this.motion=motion; }
    }
    BleClient(Context context,Listener listener) {
        this.context=context; this.listener=listener;
        BluetoothManager manager=context.getSystemService(BluetoothManager.class);
        adapter=manager==null?null:manager.getAdapter();
        IntentFilter filter=new IntentFilter(BluetoothDevice.ACTION_BOND_STATE_CHANGED);
        if(Build.VERSION.SDK_INT>=33) context.registerReceiver(bonds,filter,Context.RECEIVER_EXPORTED);
        else context.registerReceiver(bonds,filter);
        handler.post(tick);
    }
    boolean enabled() { try {return adapter!=null && adapter.isEnabled();}catch(SecurityException e){return false;} }
    boolean ready() { return ready && !probing; }
    void foreground(boolean value) {
        foreground=value;
        if(value && gatt!=null && lifecycle.waitingForForeground())completeReady();
        if(value && target!=null && gatt==null && !userDisconnected) handler.postDelayed(reconnect,600);
        if(!value) { stopScan(); handler.removeCallbacks(reconnect); }
    }
    void scan() {
        if(!enabled()) { listener.status("请开启手机蓝牙",false); return; }
        stopScan(); scanning=true;
        listener.status("扫描附近机器人…",ready);
        try {
            adapter.getBluetoothLeScanner().startScan(Collections.singletonList(new ScanFilter.Builder().setServiceUuid(new ParcelUuid(SERVICE)).build()),
                new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(),scanCallback);
        } catch(SecurityException e) {scanning=false;listener.status("蓝牙权限已撤销，请重新授权",false);return;}
        handler.postDelayed(this::stopScan,12000);
    }
    void stopScan() {
        if(scanning) {
            scanning=false;
            try { if(adapter!=null && adapter.isEnabled()) adapter.getBluetoothLeScanner().stopScan(scanCallback); } catch(SecurityException ignored) { }
        }
    }
    private final ScanCallback scanCallback=new ScanCallback() {
        @Override public void onScanResult(int type,ScanResult result) {
            handler.post(()-> { if(!scanning)return;
                String name=result.getScanRecord()==null?null:result.getScanRecord().getDeviceName();
                listener.found(result.getDevice().getAddress(),name==null?"Xduck":name,result.getRssi());
            });
        }
        @Override public void onScanFailed(int code) { handler.post(()-> {scanning=false;listener.status("扫描失败（"+code+"），请稍后重试",ready);}); }
    };
    void connect(String address) {
        stopScan(); userDisconnected=false; target=address; retries=0;
        close(); lifecycle.userConnect();connectNow();
    }
    private void connectNow() {
        handler.removeCallbacks(reconnect);
        if(!foreground || userDisconnected || target==null)return;
        if(!enabled()) { listener.status("蓝牙已关闭，开启后自动重连",false); handler.postDelayed(reconnect,3000);return; }
        try {
            phase="连接中";operationStarted=now();listener.status("正在连接机器人…",false);
            gatt=adapter.getRemoteDevice(target).connectGatt(context,false,callback,BluetoothDevice.TRANSPORT_LE);
            if(gatt==null) fail("无法创建 BLE 连接");
        } catch(RuntimeException e) { fail("连接失败："+e.getMessage()); }
    }
    private final Runnable reconnect=()-> { if(gatt==null)connectNow(); };
    void disconnect() { userDisconnected=true; target=null; handler.removeCallbacks(reconnect);close();listener.disconnected("已主动断开"); }
    /** Called after a best-effort zero/release. Background never keeps a live controller. */
    void backgroundDisconnect() { if(!foreground && gatt!=null && lifecycle.canCloseInBackground()) {close();listener.disconnected("已暂停控制，返回后重新连接");} }
    private void close() {
        ready=false;probing=false;active=motion=stop=null;commands.clear();offset=0;rssiBusy=false;payload=20;pipe=null;phase="";
        lines.clear();states.clear();
        lifecycle.closed();
        BluetoothGatt old=gatt;gatt=null;
        if(old!=null) { try {old.disconnect();old.close();}catch(SecurityException ignored){} }
    }
    void recover(String reason) {fail(reason);}
    void authenticatedSession() {retries=0;}
    private void fail(String reason) {
        boolean retry=lifecycle.canRetry() && retries<3;
        if(!retry) {userDisconnected=true;handler.removeCallbacks(reconnect);reason+="；请点击机器人重试";}
        close();listener.disconnected(reason);
        if(retry && foreground && target!=null && !userDisconnected) {
            long delay=Math.min(8000,1000L<<Math.min(retries++,3));
            handler.removeCallbacks(reconnect);handler.postDelayed(reconnect,delay);
        }
    }
    private void pairingFailed(String reason) {lifecycle.requireUserRetry();fail(reason);}
    private void completeReady() {
        // CCC success alone does not prove that replies can reach this phone.
        if(probing)return;
        ready=true;probing=true;probeStarted=now();phase="";
        listener.status("正在验证机器人回复通道…",false);
        json(()->"{\"jsonrpc\":\"2.0\",\"id\":\"ble-probe\",\"method\":\"hello\",\"params\":{\"api_version\":"+apiVersion+"}}",true);
    }
    boolean json(Supplier<String> value, boolean priority) {
        if(!ready)return false;
        Frame frame=new Frame(()->(value.get()+"\n").getBytes(StandardCharsets.UTF_8),false);
        if(commands.size()>=6) return false;
        if(priority)commands.addFirst(frame);else commands.addLast(frame);
        pump();return true;
    }
    void move(Supplier<byte[]> value,boolean isStop) {
        if(!ready)return;
        if(isStop) {motion=null;stop=new Frame(value,true);}else motion=new Frame(value,true);
        pump();
    }
    void clearUnsent() { motion=null;commands.clear(); }
    private void pump() {
        if(!ready || gatt==null || active!=null || rssiBusy)return;
        if(stop!=null) { active=stop;stop=null; }
        else if(motion!=null) { active=motion;motion=null; }
        else if(!commands.isEmpty()) active=commands.removeFirst();
        if(active==null)return;
        if(now()-active.created>(active.motion?200:250)) { active=null; pump(); return; }
        active.data=active.materialize.get();
        if(active.data==null) {active=null;pump();return;}
        offset=0;writeChunk();
    }
    private void writeChunk() {
        byte[] bytes=Arrays.copyOfRange(active.data,offset,Math.min(active.data.length,offset+payload));
        operationStarted=now();phase="发送指令";
        boolean accepted;
        if(Build.VERSION.SDK_INT>=33) accepted=gatt.writeCharacteristic(pipe,bytes,BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)==BluetoothStatusCodes.SUCCESS;
        else { pipe.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);pipe.setValue(bytes);accepted=gatt.writeCharacteristic(pipe); }
        if(!accepted)fail("BLE 写入失败，已停止发送");
    }
    private void negotiate() {
        phase="协商 MTU";operationStarted=now();
        gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH);
        if(!gatt.requestMtu(247))readVersion();
    }
    private void readVersion() {
        phase="读取 API";operationStarted=now();
        if(!gatt.readCharacteristic(pipe))fail("无法读取机器人 API");
    }
    private void received(byte[] data) {
        try {
            if(Wire.isState(data,lines)) { byte[] frame=states.push(data);if(frame!=null)listener.state(new Wire.State(frame)); }
            else for(String line:lines.push(data)) {
                if(probing) {
                    JSONObject reply=new JSONObject(line);
                    if(!"ble-probe".equals(reply.optString("id")))continue;
                    if(!reply.has("result") && !reply.has("error"))continue;
                    probing=false;lifecycle.becameReady();
                    if(!foreground){close();listener.disconnected("已暂停控制，返回后重新连接");return;}
                    listener.ready(apiVersion);
                } else listener.line(line);
            }
        } catch(org.json.JSONException e) {fail("机器人回复不是有效 JSON");
        } catch(RuntimeException e) {fail("机器人数据格式错误："+e.getMessage());}
    }
    private final BroadcastReceiver bonds=new BroadcastReceiver() {
        @Override public void onReceive(Context c,Intent intent) {
            BluetoothDevice device=intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
            if(gatt==null || device==null || !device.getAddress().equals(target) || !phase.equals("配对"))return;
            int bond=intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE,BluetoothDevice.BOND_NONE);
            if(bond==BluetoothDevice.BOND_BONDED)negotiate();
            else if(bond==BluetoothDevice.BOND_NONE)pairingFailed("配对已取消或失败");
        }
    };
    private final BluetoothGattCallback callback=new BluetoothGattCallback() {
        private void current(BluetoothGatt g,Runnable work) {handler.post(()->{if(g==gatt)work.run();});}
        @Override public void onConnectionStateChange(BluetoothGatt g,int status,int state) {
            current(g,()-> {
                if(status!=BluetoothGatt.GATT_SUCCESS || state==BluetoothProfile.STATE_DISCONNECTED) {fail("BLE 连接中断（"+status+"），机器人将停车");return;}
                if(state==BluetoothProfile.STATE_CONNECTED) {
                    phase="发现服务";operationStarted=now();
                    if(!g.discoverServices())fail("无法发现 BLE 服务");
                }
            });
        }
        @Override public void onServicesDiscovered(BluetoothGatt g,int status) {
            current(g,()-> {
                BluetoothGattService service=g.getService(SERVICE);
                pipe=service==null?null:service.getCharacteristic(RPC);
                if(status!=BluetoothGatt.GATT_SUCCESS || pipe==null) {fail("设备未提供 Xduck BLE 服务");return;}
                if(g.getDevice().getBondState()!=BluetoothDevice.BOND_BONDED) {
                    if(!lifecycle.beginPairing()) {pairingFailed("蓝牙配对已失效，已停止自动重连");return;}
                    phase="配对";operationStarted=now();listener.status("正在建立加密配对，请完成系统提示",false);
                    if(g.getDevice().getBondState()==BluetoothDevice.BOND_NONE && !g.getDevice().createBond())pairingFailed("无法开始蓝牙配对");
                } else negotiate();
            });
        }
        @Override public void onMtuChanged(BluetoothGatt g,int mtu,int status) {
            current(g,()-> {if(!phase.equals("协商 MTU"))return;payload=status==BluetoothGatt.GATT_SUCCESS?Math.max(20,mtu-3):20;readVersion();});
        }
        @Override public void onCharacteristicRead(BluetoothGatt g,BluetoothGattCharacteristic c,byte[] value,int status) {
            current(g,()->version(g,value,status));
        }
        @Override public void onCharacteristicRead(BluetoothGatt g,BluetoothGattCharacteristic c,int status) {
            if(Build.VERSION.SDK_INT<33) {byte[] value=c.getValue();current(g,()->version(g,value,status));}
        }
        private void version(BluetoothGatt g,byte[] value,int status) {
            if(status!=BluetoothGatt.GATT_SUCCESS || value==null || value.length!=1) {fail("读取 API 失败，请检查机器人加密配置");return;}
            apiVersion=value[0]&255;
            if(!g.setCharacteristicNotification(pipe,true)) {fail("无法订阅 BLE 回复");return;}
            BluetoothGattDescriptor descriptor=pipe.getDescriptor(CCC);
            if(descriptor==null){fail("缺少 BLE 通知描述符");return;}
            // Bonded peers may retain CCC=1 while the server has discarded its session.
            // Explicitly stop, then start notification on every new GATT connection.
            writeSubscription(g,descriptor,false);
        }
        @Override public void onDescriptorWrite(BluetoothGatt g,BluetoothGattDescriptor d,int status) {
            current(g,()-> {
                if(!d.getUuid().equals(CCC))return;
                if(status!=BluetoothGatt.GATT_SUCCESS){fail("订阅回复失败（"+status+"）");return;}
                if(phase.equals("重置通知")) {
                    phase="等待重建通知";operationStarted=now();
                    handler.postDelayed(()-> {if(g==gatt && phase.equals("等待重建通知"))writeSubscription(g,d,true);},250);
                    return;
                }
                if(!phase.equals("订阅通知"))return;
                if(lifecycle.setupComplete(foreground)) {completeReady();return;}
                if(lifecycle.waitingForForeground()) {
                    phase="等待返回 App";operationStarted=now();listener.status("配对已完成，请返回 App",false);
                } else {close();listener.disconnected("已暂停控制，返回后重新连接");}
            });
        }
        @Override public void onCharacteristicWrite(BluetoothGatt g,BluetoothGattCharacteristic c,int status) {
            current(g,()-> {
                if(status!=BluetoothGatt.GATT_SUCCESS){fail("机器人拒绝 BLE 写入（"+status+"）");return;}
                if(active==null)return;offset=Math.min(active.data.length,offset+payload);
                if(offset<active.data.length)writeChunk();else {active=null;phase="";pump();}
            });
        }
        @Override public void onCharacteristicChanged(BluetoothGatt g,BluetoothGattCharacteristic c,byte[] value) {byte[] bytes=value.clone();current(g,()->received(bytes));}
        @Override public void onCharacteristicChanged(BluetoothGatt g,BluetoothGattCharacteristic c) {
            if(Build.VERSION.SDK_INT<33){byte[] bytes=c.getValue().clone();current(g,()->received(bytes));}
        }
        @Override public void onReadRemoteRssi(BluetoothGatt g,int rssi,int status) {
            current(g,()->{if(!rssiBusy)return;rssiBusy=false;phase="";if(status==BluetoothGatt.GATT_SUCCESS)listener.rssi(rssi);pump();});
        }
    };
    private void writeSubscription(BluetoothGatt g,BluetoothGattDescriptor descriptor,boolean enable) {
        phase=enable?"订阅通知":"重置通知";operationStarted=now();
        byte[] value=enable?BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE:BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE;
        boolean accepted;
        if(Build.VERSION.SDK_INT>=33)accepted=g.writeDescriptor(descriptor,value)==BluetoothStatusCodes.SUCCESS;
        else {descriptor.setValue(value);accepted=g.writeDescriptor(descriptor);}
        if(!accepted)fail("无法"+(enable?"启用":"重置")+" BLE 通知");
    }
    private final Runnable tick=new Runnable() {
        @Override public void run() {
            if(gatt!=null) {
                long limit=ready?3000:(phase.equals("配对") || lifecycle.waitingForForeground())?45000:12000;
                if(probing && now()-probeStarted>5000)pairingFailed("机器人回复通道无响应，已停止重连");
                else if(rssiBusy && now()-operationStarted>3000) {rssiBusy=false;phase="";pump();}
                else if(!phase.isEmpty() && now()-operationStarted>limit)fail(phase+"超时，已断开控制");
                else if(ready && !probing && active==null && !rssiBusy && now()-lastRssi>=2000) {
                    lastRssi=now();rssiBusy=true;operationStarted=now();phase="读取信号";
                    if(!gatt.readRemoteRssi()){rssiBusy=false;phase="";}
                }
            }
            handler.postDelayed(this,100);
        }
    };
    void destroy() {foreground=false;handler.removeCallbacksAndMessages(null);stopScan();close();context.unregisterReceiver(bonds);}
    private static long now() {return SystemClock.elapsedRealtime();}
}
