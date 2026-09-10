package com.microduck.ble;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public final class WireChecks {
    private static void check(boolean value,String message){if(!value)throw new AssertionError(message);}
    public static void main(String[] args) {
        byte[] expected={(byte)0xbd,1,1,0,0,0,2,0,0,0,100,0,0,0,100,0,(byte)156,(byte)255,(byte)232,3};
        check(Arrays.equals(Wire.move(1,2,100,100,-100,1000),expected),"Rust/Android movement golden mismatch");
        for(int[] bad:new int[][]{{301,0,0},{0,-301,0},{0,0,1001},{Integer.MIN_VALUE,0,0}}){
            boolean failed=false;try{Wire.move(1,1,0,bad[0],bad[1],bad[2]);}catch(IllegalArgumentException e){failed=true;}check(failed,"unsafe speed accepted");
        }
        Wire.Lines lines=new Wire.Lines();byte[] json="{\"名字\":\"小鸭\"}\n{}\n".getBytes(StandardCharsets.UTF_8);
        ArrayList<String> result=new ArrayList<>();for(byte b:json)result.addAll(lines.push(new byte[]{b}));
        check(result.equals(Arrays.asList("{\"名字\":\"小鸭\"}","{}")),"UTF-8 fragmentation failed");
        byte[] unicodeMarker="{\"text\":\"¾\"}\n".getBytes(StandardCharsets.UTF_8);
        for(byte value:unicodeMarker){byte[] chunk={value};check(!Wire.isState(chunk,lines),"UTF-8 continuation mistaken for binary state");lines.push(chunk);}
        check(Wire.isState(new byte[]{(byte)0xbe},lines),"binary state start lost");
        boolean overflow=false;try{lines.push(new byte[8193]);}catch(IllegalArgumentException e){overflow=true;}check(overflow,"unbounded NDJSON input");lines.clear();
        byte[] packet=new byte[188];ByteBuffer bb=ByteBuffer.wrap(packet).order(ByteOrder.LITTLE_ENDIAN);
        bb.put((byte)1).put((byte)2).put((byte)1).put((byte)0).putInt(12345).putShort((short)500);
        for(int i=0;i<12;i++)bb.putShort((short)(i==0?100:i==1?Short.MIN_VALUE:0));
        for(int i=0;i<14;i++)bb.put((byte)7).putShort((short)1000).putShort((short)-500).putShort((short)250).putShort((short)355).putShort((short)20);
        Wire.States states=new Wire.States();byte[] assembled=null;
        for(int i=0;i<12;i++){int n=Math.min(16,188-i*16);byte[] chunk=new byte[n+4];chunk[0]=(byte)0xbe;chunk[1]=7;chunk[3]=(byte)i;System.arraycopy(packet,i*16,chunk,4,n);assembled=states.push(chunk);if(i<11)check(assembled==null,"partial state published");}
        check(Arrays.equals(packet,assembled),"state reassembly failed");Wire.State s=new Wire.State(assembled);
        check(s.hz==50 && s.robotMillis==12345 && s.motors.length==14,"header decode failed");
        check(Double.isNaN(s.requested[1]),"unknown became a real velocity");
        check(s.motors[0].position==1 && s.motors[0].velocity==-.5 && s.motors[0].torque==.25 && s.motors[0].temperature==35.5 && s.motors[0].fresh(),"motor units/status wrong");
        byte[] first=new byte[20];first[0]=(byte)0xbe;first[1]=8;states.push(first);byte[] wrong=first.clone();wrong[3]=2;check(states.push(wrong)==null,"missing fragment accepted");
        packet[0]=2;boolean version=false;try{new Wire.State(packet);}catch(IllegalArgumentException e){version=true;}check(version,"unknown state version accepted");
        System.out.println("PASS: binary golden, speed limits, UTF-8 chunks, frame bounds, lost fragments, 14 motor units, stale/unknown semantics");
    }
}
