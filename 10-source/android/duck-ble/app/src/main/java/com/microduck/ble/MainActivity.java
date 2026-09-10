package com.microduck.ble;

import android.Manifest;
import android.app.*;
import android.bluetooth.BluetoothAdapter;
import android.content.*;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.*;
import android.text.InputType;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.util.*;

public final class MainActivity extends Activity implements BleClient.Listener {
    private static final int BG=0xff111816, CARD=0xff1c2923, TEXT=0xfff0f4ea, MUTED=0xffa6b5a7, LIME=0xffcbee79, RED=0xffffa38c;
    private final Handler handler=new Handler(Looper.getMainLooper());
    private BleClient ble;
    private LinearLayout root,content,devices;
    private TextView connection,feedback,velocity,speedLabel,summary,version,telemetry;
    private TableLayout motorTable;
    private TextView[][] motorCells;
    private EditText pin;
    private Button claim;
    private JoystickView left,right;
    private boolean authenticated,compatible,controlling,foreground;
    private long session,sequence,clockOffset,claimedAt,lastState,lastRobotTime=-1,nextId=1,lastPoll,lastInfo;
    private int rssi=-127,tab=0;
    private double vx,vy,yaw,speed=.10,maxSpeed=.30,yawLimit=.4;
    private Wire.State state;
    private JSONObject health;
    private String deviceName="未连接",savedPin="",mode="—",firmware="—",status="扫描并连接你的机器人";
    private final ArrayList<Button> tabButtons=new ArrayList<>();
    private final Map<String,Button> found=new LinkedHashMap<>();
    private final Map<Long,Pending> pending=new HashMap<>();
    private final ArrayList<View> controlledViews=new ArrayList<>();
    private interface Reply { void done(JSONObject result,long elapsed); }
    private static final class Pending {
        final String method;final long at;final Reply reply;
        Pending(String m,long at,Reply r){method=m;this.at=at;reply=r;}
    }
    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ble=new BleClient(this,this);build();handler.post(loop);
    }
    private void build() {
        root=column();root.setBackgroundColor(BG);root.setPadding(dp(20),dp(10),dp(20),dp(10));setContentView(root);
        LinearLayout header=row();
        TextView brand=text("Xduck   /   户外遥控",19,TEXT);brand.setTypeface(null,Typeface.BOLD);header.addView(brand,new LinearLayout.LayoutParams(0,dp(38),1));
        connection=text("● 未连接",13,MUTED);connection.setGravity(Gravity.CENTER_VERTICAL|Gravity.END);header.addView(connection);root.addView(header);
        LinearLayout tabs=row();String[] labels={"移动控制","头部与动作","状态 · 14 电机","连接机器人"};
        for(int i=0;i<labels.length;i++){final int choice=i;Button b=button(labels[i],()->showTab(choice));tabButtons.add(b);tabs.addView(b,new LinearLayout.LayoutParams(0,dp(42),1));}
        root.addView(tabs);
        content=column();root.addView(content,new LinearLayout.LayoutParams(-1,0,1));
        feedback=text("仅 BLE · 无需网络 · 请先连接机器人",12,MUTED);feedback.setPadding(0,dp(4),0,0);root.addView(feedback);
        showTab(3);
    }
    private void showTab(int index) {
        if(controlling && index!=tab)stopMotion();tab=index;
        for(int i=0;i<tabButtons.size();i++){Button button=tabButtons.get(i);button.setTextColor(i==tab?LIME:TEXT);((GradientDrawable)button.getBackground()).setStroke(dp(i==tab?2:1),i==tab?LIME:0xff35473b);}
        content.removeAllViews();controlledViews.clear();left=right=null;
        telemetry=summary=version=velocity=speedLabel=null;motorTable=null;motorCells=null;claim=null;pin=null;devices=null;found.clear();
        if(index==0)controlPage();else if(index==1)actionsPage();else if(index==2)statePage();else connectionPage();
        render();
    }
    private void controlPage() {
        LinearLayout controls=row();claim=button(controlling?"释放控制":"接管控制",()->{if(controlling)release();else takeControl();});
        controls.addView(claim,new LinearLayout.LayoutParams(dp(125),dp(45)));
        Button halt=button("停车",()->{stopMotion();tell("已发送零速度");});halt.setTextColor(RED);controlledViews.add(halt);controls.addView(halt,new LinearLayout.LayoutParams(dp(90),dp(45)));
        String[] names={"慢速","标准","快速"};double[] limits={.10,.20,.30};
        for(int i=0;i<3;i++){final int k=i;controls.addView(button(names[i],()->{stopMotion();speed=limits[k];yawLimit=new double[]{.4,.7,1}[k];render();}),new LinearLayout.LayoutParams(dp(72),dp(45)));}
        speedLabel=text("",12,LIME);speedLabel.setGravity(Gravity.CENTER_VERTICAL|Gravity.END);controls.addView(speedLabel,new LinearLayout.LayoutParams(0,dp(45),1));content.addView(controls);
        LinearLayout sticks=row();
        LinearLayout a=column(),b=column();
        left=new JoystickView(this,false,new JoystickView.Listener(){public void changed(float x,float y){vx=-y*Math.min(speed,maxSpeed);vy=-x*Math.min(speed,maxSpeed);renderVelocity();}public void released(){stopMotion();}});
        right=new JoystickView(this,true,new JoystickView.Listener(){public void changed(float x,float y){yaw=-x*yawLimit;renderVelocity();}public void released(){stopMotion();}});
        a.addView(centerText("移动  /  前后 · 左右平移"));a.addView(left,new LinearLayout.LayoutParams(-1,0,1));
        b.addView(centerText("转向  /  左转 · 右转"));b.addView(right,new LinearLayout.LayoutParams(-1,0,1));
        sticks.addView(a,new LinearLayout.LayoutParams(0,-1,1));sticks.addView(b,new LinearLayout.LayoutParams(0,-1,1));content.addView(sticks,new LinearLayout.LayoutParams(-1,0,1));
        velocity=centerText("");content.addView(velocity);
        TextView note=text("松手停车  ·  后台停止控制  ·  500 ms 指令超时停车",12,MUTED);note.setGravity(Gravity.CENTER);content.addView(note);
        controlledViews.add(left);controlledViews.add(right);
    }
    private void actionsPage() {
        ScrollView scroll=new ScrollView(this);LinearLayout box=column();scroll.addView(box);content.addView(scroll);
        TextView limit=text("最大直线速度限制："+fmt(maxSpeed,2)+" m/s",14,LIME);box.addView(limit);
        SeekBar speedCap=new SeekBar(this);speedCap.setMax(250);speedCap.setProgress((int)(maxSpeed*1000)-50);
        speedCap.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            public void onStartTrackingTouch(SeekBar s){stopMotion();}
            public void onProgressChanged(SeekBar s,int p,boolean user){maxSpeed=(p+50)/1000.0;limit.setText("最大直线速度限制："+fmt(maxSpeed,2)+" m/s");}
            public void onStopTrackingTouch(SeekBar s){}
        });box.addView(speedCap);
        box.addView(text("头部角度",16,LIME));
        final double[] angles={0,0};
        String[] titles={"俯仰（±0.4 rad）","左右转动（±0.6 rad）"};
        for(int i=0;i<2;i++){final int k=i;box.addView(text(titles[i],13,MUTED));SeekBar slider=new SeekBar(this);slider.setMax(100);slider.setProgress(50);
            slider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
                public void onStartTrackingTouch(SeekBar s){stopMotion();}
                public void onProgressChanged(SeekBar s,int p,boolean user){angles[k]=(p-50)/50.0*(k==0?.4:.6);}
                public void onStopTrackingTouch(SeekBar s){command("robot.head",obj("head_pitch",angles[0],"head_yaw",angles[1]),"调整头部");}
            });box.addView(slider);controlledViews.add(slider);
        }
        LinearLayout head=row();head.addView(action("头部居中","robot.head",obj(),false),new LinearLayout.LayoutParams(0,dp(46),1));box.addView(head);
        box.addView(text("姿态与动作",16,LIME));
        LinearLayout posture=row();
        posture.addView(action("初始化","robot.init",obj(),true),new LinearLayout.LayoutParams(0,dp(46),1));
        posture.addView(action("开启策略","robot.enable",obj("on",true),true),new LinearLayout.LayoutParams(0,dp(46),1));
        posture.addView(action("关闭策略","robot.enable",obj("on",false),true),new LinearLayout.LayoutParams(0,dp(46),1));
        posture.addView(action("放松电机","robot.relax",obj(),true),new LinearLayout.LayoutParams(0,dp(46),1));box.addView(posture);
        LinearLayout skills=row();String[] names={"坐下 / 站立","拾取","左脚踢球","右脚踢球","翻滚"};String[] ids={"sit_toggle","ground_pick","kick_left","kick_right","roulade"};
        for(int i=0;i<names.length;i++)skills.addView(action(names[i],"robot.do",obj("skill",ids[i]),true),new LinearLayout.LayoutParams(0,dp(46),1));box.addView(skills);
        LinearLayout sounds=row();sounds.addView(action("播放啾啾声","robot.sound",obj("tag","chirp"),false),new LinearLayout.LayoutParams(0,dp(46),1));
        sounds.addView(action("播放问候声","robot.sound",obj("tag","greet"),false),new LinearLayout.LayoutParams(0,dp(46),1));box.addView(sounds);
        box.addView(text("动作需要先在移动页接管控制。初始化会移动关节；放松会释放力矩，请扶稳机器人。",12,MUTED));
    }
    private Button action(String title,String method,JSONObject params,boolean confirm) {
        Button b=button(title,()-> {
            stopMotion();
            if(confirm)new AlertDialog.Builder(this).setTitle(title).setMessage(method.equals("robot.relax")?"放松会释放电机力矩，机器人可能倒下。请先扶稳。":"该操作会改变机器人姿态，请确认周围留有空间。")
                .setNegativeButton("取消",null).setPositiveButton("确认执行",(d,w)->command(method,params,title)).show();
            else command(method,params,title);
        });controlledViews.add(b);return b;
    }
    private void statePage() {
        ScrollView scroll=new ScrollView(this);LinearLayout box=column();scroll.addView(box);content.addView(scroll);
        summary=text("",15,LIME);box.addView(summary);telemetry=text("",13,TEXT);telemetry.setPadding(0,dp(5),0,dp(10));box.addView(telemetry);
        HorizontalScrollView horizontal=new HorizontalScrollView(this);motorTable=new TableLayout(this);motorCells=new TextView[15][6];
        String[] headings={"电机","状态","位置","速度","力矩","温度"};
        for(int i=0;i<15;i++){TableRow row=new TableRow(this);
            for(int j=0;j<6;j++){TextView cell=text(i==0?headings[j]:"—",12,i==0?LIME:TEXT);cell.setMinWidth(dp(j==0?130:j==1?120:74));cell.setPadding(dp(5),dp(5),dp(10),dp(5));
                if(j>1)cell.setGravity(Gravity.END);motorCells[i][j]=cell;row.addView(cell);}
            motorTable.addView(row);
        }
        horizontal.addView(motorTable);box.addView(horizontal);
        box.addView(text("电机位置 rad · 速度 rad/s · 力矩 N·m · 温度 °C。离线或过期数据以 — 显示。",12,MUTED));
    }
    private void connectionPage() {
        LinearLayout row=row();LinearLayout config=column(),list=column();config.setPadding(0,dp(5),dp(20),0);
        config.addView(text("连接你的 Xduck",20,TEXT));
        config.addView(text("开启蓝牙，将手机靠近机器人。\n选择设备后输入 6 位 PIN 认证。",13,MUTED));
        pin=new EditText(this);pin.setSingleLine(true);pin.setTextColor(TEXT);pin.setHintTextColor(MUTED);pin.setHint("6 位机器人 PIN");pin.setInputType(InputType.TYPE_CLASS_NUMBER|InputType.TYPE_NUMBER_VARIATION_PASSWORD);
        pin.setFilters(new android.text.InputFilter[]{new android.text.InputFilter.LengthFilter(6)});pin.setText(savedPin);config.addView(pin);
        Button auth=button("PIN 认证",()->authenticate(pin.getText().toString()));config.addView(auth,new LinearLayout.LayoutParams(-1,dp(44)));
        LinearLayout buttons=row();buttons.addView(button("扫描机器人",this::scan),new LinearLayout.LayoutParams(0,dp(44),1));
        buttons.addView(button("断开",()->{release();ble.disconnect();}),new LinearLayout.LayoutParams(0,dp(44),1));config.addView(buttons);
        version=text("",12,MUTED);config.addView(version);
        list.addView(text("附近机器人",15,LIME));ScrollView scroll=new ScrollView(this);devices=column();scroll.addView(devices);list.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));
        list.addView(text("PIN 仅保留在本次 App 内存中。\n重连不会恢复运动，需手动接管控制。",12,MUTED));
        ScrollView configScroll=new ScrollView(this);configScroll.addView(config);
        row.addView(configScroll,new LinearLayout.LayoutParams(0,-1,1));row.addView(list,new LinearLayout.LayoutParams(0,-1,1));content.addView(row,new LinearLayout.LayoutParams(-1,-1));
    }
    private void scan() {
        if(!permissions()) {
            if(Build.VERSION.SDK_INT>=31)requestPermissions(new String[]{Manifest.permission.BLUETOOTH_SCAN,Manifest.permission.BLUETOOTH_CONNECT},42);
            else requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION},42);
            return;
        }
        if(!ble.enabled()) {try {startActivity(new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE));}catch(SecurityException e){tell("请先授予附近设备权限");}return;}
        if(Build.VERSION.SDK_INT<31) {
            android.location.LocationManager lm=getSystemService(android.location.LocationManager.class);
            if(lm!=null && !lm.isProviderEnabled(android.location.LocationManager.GPS_PROVIDER) && !lm.isProviderEnabled(android.location.LocationManager.NETWORK_PROVIDER)) {tell("Android 11 及更早版本扫描 BLE 需要开启系统定位");return;}
        }
        found.clear();if(devices!=null)devices.removeAllViews();ble.scan();
    }
    private boolean permissions() {return Build.VERSION.SDK_INT>=31?checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN)==PackageManager.PERMISSION_GRANTED && checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)==PackageManager.PERMISSION_GRANTED:checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)==PackageManager.PERMISSION_GRANTED;}
    @Override public void onRequestPermissionsResult(int request,String[] names,int[] results){super.onRequestPermissionsResult(request,names,results);if(request==42){if(permissions()){ble.foreground(true);scan();}else tell("需要附近设备权限才能扫描和连接机器人");}}
    @Override protected void onResume(){super.onResume();foreground=true;if(permissions())ble.foreground(true);}
    @Override protected void onPause(){release();foreground=false;ble.foreground(false);super.onPause();}
    @Override protected void onStop(){handler.postDelayed(ble::backgroundDisconnect,120);super.onStop();}
    @Override public void onWindowFocusChanged(boolean focused){super.onWindowFocusChanged(focused);if(!focused && controlling)stopMotion();}
    @Override protected void onDestroy(){handler.removeCallbacksAndMessages(null);ble.destroy();savedPin="";super.onDestroy();}
    private void authenticate(String code) {
        if(!ble.ready()){tell("请先选择并连接机器人");return;}
        if(!code.matches("[0-9]{6}")){tell("PIN 必须是 6 位数字，包括开头的 0");return;}
        savedPin=code;
        rpc("system.authenticate",obj("pin",code),(r,ms)-> {
            authenticated=r.optBoolean("authenticated");
            if(!authenticated){savedPin="";tell("PIN 不正确，剩余尝试 "+r.optInt("attempts_remaining"));return;}
            rpc("ble.info",obj(),(info,elapsed)-> {
                compatible=info.optInt("v")==1;
                firmware="BLE 服务 "+info.optString("firmware","—")+" / API "+info.optInt("api");
                if(!compatible){tell("机器人 BLE App 协议不兼容，需要更新机器人端");return;}
                ble.authenticatedSession();
                status="已连接 · 未接管";render();tell("认证成功，进入移动控制页接管机器人");
                rpc("ble.subscribe",obj(),(a,t)->{});
                rpc("system.info",obj(),(a,t)->{deviceName=a.optString("name",deviceName);render();});
                rpc("ble.mode",obj(),(a,t)->{mode=a.optString("mode","—");render();});
            });
        });
    }
    private void takeControl() {
        if(hasPending("ble.claim"))return;
        if(!authenticated || !compatible){tell("请先连接并完成 PIN 认证；机器人需支持 BLE App v1");return;}
        if(state==null || now()-lastState>1000){tell("尚未收到最新状态，暂不接管");return;}
        rpc("ble.claim",obj(),(r,ms)-> {
            if(!foreground){ble.backgroundDisconnect();return;}
            session=r.optLong("s");sequence=0;clockOffset=r.optLong("ms")-now();
            if(session==0 || ms>200){tell("链路延迟过高，未接管控制");session=0;return;}
            controlling=true;claimedAt=now();vx=vy=yaw=0;
            status=state!=null && state.source==2?"蓝牙待命 · 手柄优先":"已连接 · 控制中";
            sendMove(true);render();tell(state!=null && state.source==2?"手柄优先，蓝牙保持待命；释放后退出蓝牙接管":"已接管：先初始化，再开启策略。松手停车。");
        });
    }
    private JSONObject stamp() {return obj("s",session,"q",0,"t",0);}
    private void release() {
        if(!controlling)return;
        stopMotion();ble.clearUnsent();
        rpc("ble.release",stamp(),(r,t)->{},true);
        controlling=false;session=0;status="已连接 · 未接管";render();
    }
    private void stopMotion() {vx=vy=yaw=0;if(left!=null)left.reset();if(right!=null)right.reset();sendMove(true);renderVelocity();}
    private void sendMove(boolean stop) {
        if(!controlling || !foreground || !ble.ready())return;
        // Keep a zero heartbeat while a gamepad has priority; never queue joystick motion.
        if(state==null || state.source!=1) {vx=vy=yaw=0;stop=true;}
        if(sequence>=0xfffffffeL){release();return;}
        final long sid=session;
        final int x=(int)Math.round(vx*1000),y=(int)Math.round(vy*1000),w=(int)Math.round(yaw*1000);
        ble.move(()->Wire.move(sid,++sequence,Math.max(0,now()+clockOffset),x,y,w),stop);
    }
    private void command(String method,JSONObject params,String label) {
        if(!controlling){tell("请先接管控制");return;}
        if(state==null || state.source!=1){tell("当前由更高优先级控制源接管，蓝牙待命");return;}
        JSONObject envelope=stamp();try{envelope.put("m",method);envelope.put("p",params);}catch(JSONException ignored){}
        rpc("ble.command",envelope,(r,t)->tell(label+"："+(r.optBoolean("accepted")?"已接受，请观察实际状态":r.optString("reason","机器人拒绝了操作"))));
    }
    private void rpc(String method,JSONObject params,Reply callback){rpc(method,params,callback,false);}
    private void rpc(String method,JSONObject params,Reply callback,boolean priority) {
        long id=nextId++;JSONObject req=obj("jsonrpc","2.0","id",id,"method",method,"params",params);
        if(ble.json(()->{
            if(method.equals("ble.command") || method.equals("ble.release")) {
                try {params.put("q",++sequence);params.put("t",Math.max(0,now()+clockOffset));}catch(JSONException ignored){}
            }
            return req.toString();
        },priority))pending.put(id,new Pending(method,now(),callback));else tell("BLE 忙或未连接，请重试");
    }
    @Override public void line(String line) {
        try {
            JSONObject r=new JSONObject(line);
            if(r.optString("method").equals("ble.error")){tell(r.optJSONObject("params").optString("message","控制指令被拒绝"));release();return;}
            Pending p=pending.remove(r.optLong("id",-1));if(p==null)return;
            if(r.has("error")) {tell(p.method+"："+r.getJSONObject("error").optString("message"));if(p.method.startsWith("ble.")){if(p.method.equals("ble.info"))compatible=false; if(controlling)release();}return;}
            JSONObject value=r.optJSONObject("result");if(value==null){tell("机器人回复格式不兼容");return;}
            long elapsed=now()-p.at;
            if(p.method.equals("ble.health"))linkLatency=elapsed;
            p.reply.done(value,elapsed);
        } catch(JSONException e){tell("机器人回复解析失败");}
    }
    private long linkLatency=-1;
    @Override public void state(Wire.State value) {
        // Repeated robot tick is stale data even if BLE continues sending notifications.
        if(value.robotMillis!=lastRobotTime){lastState=now();lastRobotTime=value.robotMillis;}
        int previousSource=state==null?0:state.source;
        state=value;
        if(controlling) {
            if(previousSource!=value.source)stopMotion();
            if(value.source==2){status="蓝牙待命 · 手柄优先";}
            else if(value.source==1){status="已连接 · 控制中";}
            else if(now()-claimedAt>1000){release();tell("蓝牙控制权已撤销，请重新接管");}
        }
        render();
    }
    @Override public void status(String message,boolean ready){status=message;render();tell(message);}
    @Override public void found(String address,String name,int signal) {
        if(devices==null)return;Button b=found.get(address);
        if(b==null){b=button("",()-> {if(pin!=null)savedPin=pin.getText().toString();release();ble.foreground(true);deviceName=name;ble.connect(address);});found.put(address,b);devices.addView(b,new LinearLayout.LayoutParams(-1,dp(52)));}
        b.setText(name+"   "+signal+" dBm\n"+address);
    }
    @Override public void ready(int api) {status="已连接 · 等待 PIN";firmware="API "+api;render();if(savedPin.matches("[0-9]{6}"))authenticate(savedPin);else tell("加密配对完成，请输入机器人 PIN");}
    @Override public void rssi(int value){rssi=value;render();}
    @Override public void disconnected(String reason){authenticated=compatible=controlling=false;session=0;pending.clear();state=null;health=null;lastState=0;lastRobotTime=-1;rssi=-127;vx=vy=yaw=0;status="失联 / 未连接";render();tell(reason);}
    private final Runnable loop=new Runnable(){public void run(){
        if(foreground) {
            if(controlling && (lastState==0 || now()-lastState>1000)){release();tell("状态超过 1 秒未更新，已停止控制");}
            sendMove(false);
            if(authenticated && compatible && now()-lastPoll>=500 && !hasPending("ble.health")) {lastPoll=now();rpc("ble.health",obj(),(r,t)->{health=r;render();});}
            if(controlling && now()-lastInfo>5000 && !hasPending("ble.info")){lastInfo=now();final long clockSession=session;rpc("ble.info",obj(),(r,t)->{if(controlling && session==clockSession && t<200)clockOffset=r.optLong("ms")-now();});}
            ArrayList<Long> expired=new ArrayList<>();for(Map.Entry<Long,Pending> e:pending.entrySet())if(now()-e.getValue().at>3000)expired.add(e.getKey());
            if(!expired.isEmpty()){Pending p=pending.get(expired.get(0));release();ble.recover(p.method+"超时，正在重新连接");}
            if(state!=null && now()-lastState>1000)render();
        }
        handler.postDelayed(this,100);
    }};
    private boolean hasPending(String method){for(Pending p:pending.values())if(p.method.equals(method))return true;return false;}
    private void render() {
        if(connection==null)return;
        String signal=rssi==-127?"":(rssi< -85?" · 信号弱 ":" · ")+rssi+" dBm";
        connection.setText("● "+status+signal);connection.setTextColor(!ble.ready()?MUTED:rssi< -85?RED:LIME);
        for(View view:controlledViews){view.setEnabled(controlling);view.setAlpha(controlling?1:.4f);}
        if(claim!=null){claim.setText(controlling?"释放控制":"接管控制");claim.setEnabled(compatible && authenticated && !hasPending("ble.claim"));}
        renderVelocity();
        if(speedLabel!=null)speedLabel.setText(String.format(Locale.ROOT,"上限 %.2f m/s · %.1f rad/s",Math.min(speed,maxSpeed),yawLimit));
        if(version!=null)version.setText(deviceName+"\n"+firmware+"\nBLE App 协议："+(compatible?"v1":"等待验证"));
        if(summary!=null)summary.setText(state==null?"等待机器人状态":now()-lastState>1000?"状态已过期，请检查连接":"实时数据 · 2 Hz · "+(now()-lastState)+" ms 前");
        if(telemetry!=null) {
            String battery="—",temp="—",healthy="—";
            if(health!=null){JSONObject bat=health.optJSONObject("battery");if(bat!=null)battery=fmt(bat.optDouble("volts"),2)+" V / "+fmt(bat.optDouble("percent"),0)+"%";
                temp=health.has("cpu_temp_c")?fmt(health.optDouble("cpu_temp_c"),1)+" °C":"—";healthy=health.optBoolean("healthy")?"正常":health.optString("reason","异常");}
            String stateText=state==null?"请求 / 实际速度：—\n姿态 / 重力投影：—":
                "策略："+new String[]{"未知","walk","stand","held"}[Math.min(3,state.policy)]+"  ·  控制循环 "+fmt(state.hz,1)+" Hz\n"+
                "安全："+((state.safety&1)!=0?"已倒下":"未检测倒地")+" / "+((state.safety&2)!=0?"放松":"保持")+"\n"+
                "请求速度 "+vector(state.requested)+"   实际速度 "+vector(state.actual)+"\n姿态 RPY "+vector(state.rpy)+"   重力投影 "+vector(state.gravity);
            telemetry.setText("模式："+mode+"   电池："+battery+"   主板："+temp+"\n健康："+healthy+"   往返延迟："+(linkLatency<0?"—":linkLatency+" ms")+"\n"+stateText);
        }
        if(motorTable!=null) {
            String[] names={"左髋偏航","左髋侧摆","左髋俯仰","左膝","左踝","颈部俯仰","头部俯仰","头部偏航","头部侧摆","右髋偏航","右髋侧摆","右髋俯仰","右膝","右踝"};
            for(int i=0;i<14;i++){Wire.Motor m=state==null?null:state.motors[i];boolean fresh=m!=null && m.fresh() && now()-lastState<=1000;
                String[] values={String.format(Locale.ROOT,"%02d %s",i+1,names[i]),m==null?"—":now()-lastState>1000?"数据过期":m.status(),fresh?fmt(m.position,3):"—",fresh?fmt(m.velocity,3):"—",fresh?fmt(m.torque,3):"—",fresh?fmt(m.temperature,1):"—"};
                for(int j=0;j<6;j++)motorCells[i+1][j].setText(values[j]);
                motorCells[i+1][1].setTextColor(m!=null && (m.flags&8)!=0?RED:fresh?LIME:MUTED);
            }
        }
    }
    private void renderVelocity(){if(velocity!=null)velocity.setText(String.format(Locale.ROOT,"vx  %+.2f m/s       vy  %+.2f m/s       转向  %+.2f rad/s",vx,vy,yaw));}
    private void tell(String message){if(feedback!=null)feedback.setText(message);}
    private static String fmt(double n,int digits){return Double.isFinite(n)?String.format(Locale.ROOT,"%."+digits+"f",n):"—";}
    private static String vector(double[] v){return "["+fmt(v[0],2)+", "+fmt(v[1],2)+", "+fmt(v[2],2)+"]";}
    private static JSONObject obj(Object... pairs){JSONObject o=new JSONObject();try{for(int i=0;i<pairs.length;i+=2)o.put((String)pairs[i],pairs[i+1]);}catch(JSONException e){throw new IllegalArgumentException(e);}return o;}
    private LinearLayout column(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.VERTICAL);return l;}
    private LinearLayout row(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.HORIZONTAL);return l;}
    private TextView text(String value,int sp,int color){TextView t=new TextView(this);t.setText(value);t.setTextSize(sp);t.setTextColor(color);t.setPadding(dp(4),dp(3),dp(4),dp(3));return t;}
    private TextView centerText(String s){TextView t=text(s,13,MUTED);t.setGravity(Gravity.CENTER);return t;}
    private Button button(String title,Runnable run){Button b=new Button(this);b.setText(title);b.setTextSize(13);b.setTextColor(TEXT);b.setAllCaps(false);b.setMinHeight(0);b.setMinimumHeight(0);b.setPadding(dp(6),0,dp(6),0);GradientDrawable bg=new GradientDrawable();bg.setColor(CARD);bg.setCornerRadius(dp(9));bg.setStroke(dp(1),0xff35473b);b.setBackground(bg);b.setOnClickListener(v->run.run());return b;}
    private int dp(int v){return Math.round(v*getResources().getDisplayMetrics().density);}
    private static long now(){return SystemClock.elapsedRealtime();}
}
