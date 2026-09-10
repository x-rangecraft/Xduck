package com.microduck.ble;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Platform-free framing shared by production and executable protocol checks. */
final class Wire {
    static final int STATE_LENGTH = 188;
    static byte[] move(long session, long sequence, long millis, int vx, int vy, int yaw) {
        if (session <= 0 || session > 0xffffffffL || sequence < 1 || sequence > 0xffffffffL
                || Math.abs((long)vx)>300 || Math.abs((long)vy)>300 || Math.abs((long)yaw)>1000)
            throw new IllegalArgumentException("Invalid motion frame");
        return ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN).put((byte)0xbd).put((byte)1)
            .putInt((int)session).putInt((int)sequence).putInt((int)millis)
            .putShort((short)vx).putShort((short)vy).putShort((short)yaw).array();
    }
    static final class Lines {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        List<String> push(byte[] chunk) {
            List<String> result = new ArrayList<>();
            for (byte b:chunk) {
                if (b=='\n') { result.add(new String(bytes.toByteArray(),StandardCharsets.UTF_8)); bytes.reset(); }
                else { if (bytes.size()>=8192) throw new IllegalArgumentException("BLE reply exceeds 8 KiB"); bytes.write(b); }
            }
            return result;
        }
        boolean pending() {return bytes.size()!=0;}
        void clear() { bytes.reset(); }
    }
    static boolean isState(byte[] bytes,Lines lines) {return !lines.pending() && bytes.length>0 && (bytes[0]&255)==0xbe;}
    static final class States {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        private int frame=-1, next=0;
        byte[] push(byte[] chunk) {
            if (chunk.length<5 || chunk.length>20 || (chunk[0]&255)!=0xbe) { clear(); return null; }
            int id=(chunk[1]&255)|((chunk[2]&255)<<8), index=chunk[3]&255;
            if (index==0) { clear(); frame=id; }
            if (id!=frame || index!=next) { clear(); return null; }
            int expected = Math.min(16, STATE_LENGTH-bytes.size());
            if (chunk.length-4!=expected) { clear(); return null; }
            bytes.write(chunk,4,chunk.length-4); next++;
            if (bytes.size()==STATE_LENGTH) { byte[] out=bytes.toByteArray(); clear(); return out; }
            return null;
        }
        void clear() { bytes.reset(); frame=-1; next=0; }
    }
    static final class State {
        final int policy, source, safety;
        final long robotMillis;
        final double hz;
        final double[] requested=new double[3], actual=new double[3], rpy=new double[3], gravity=new double[3];
        final Motor[] motors=new Motor[14];
        State(byte[] bytes) {
            if(bytes.length!=STATE_LENGTH || bytes[0]!=1) throw new IllegalArgumentException("Unknown BLE state version");
            ByteBuffer b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            b.get(); policy=b.get()&255; source=b.get()&255; safety=b.get()&255;
            robotMillis=Integer.toUnsignedLong(b.getInt()); hz=number(b,10);
            for(double[] v:new double[][]{requested,actual,rpy,gravity}) for(int i=0;i<3;i++) v[i]=number(b,1000);
            for(int i=0;i<14;i++) motors[i]=new Motor(b);
        }
        static double number(ByteBuffer b,double scale) { short n=b.getShort();return n==Short.MIN_VALUE?Double.NaN:n/scale; }
    }
    static final class Motor {
        final int flags,age;
        final double position,velocity,torque,temperature;
        Motor(ByteBuffer b) { flags=b.get()&255;position=State.number(b,1000);velocity=State.number(b,1000);
            torque=State.number(b,1000);temperature=State.number(b,10);age=b.getShort()&65535; }
        boolean fresh() { return (flags&1)!=0 && (flags&2)!=0 && (flags&32)==0 && age<=500; }
        String status() { return (flags&1)==0?"未配置":(flags&8)!=0?"故障":!fresh()?"离线 / 过期":(flags&4)!=0?"在线 · 使能":"在线"; }
    }
}
