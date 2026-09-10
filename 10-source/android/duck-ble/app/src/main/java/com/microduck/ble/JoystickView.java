package com.microduck.ble;

import android.content.Context;
import android.graphics.*;
import android.view.*;

final class JoystickView extends View {
    interface Listener { void changed(float x,float y); void released(); }
    private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Listener listener;
    private final boolean yawOnly;
    private float x,y;
    private int pointer=-1;
    JoystickView(Context c,boolean yawOnly,Listener listener) {
        super(c);this.yawOnly=yawOnly;this.listener=listener;
        setContentDescription(yawOnly?"转向摇杆，左右拖动，松手停车":"移动摇杆，前后及左右平移，松手停车");
        setFocusable(true);
    }
    void reset() { x=y=0;pointer=-1;invalidate(); }
    @Override protected void onDraw(Canvas c) {
        super.onDraw(c);float cx=getWidth()/2f,cy=getHeight()/2f,r=Math.min(cx,cy)-12;
        paint.setColor(Color.rgb(28,41,35));paint.setStyle(Paint.Style.FILL);c.drawCircle(cx,cy,r,paint);
        paint.setColor(Color.rgb(66,84,71));paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(1.5f);
        c.drawCircle(cx,cy,r,paint);c.drawCircle(cx,cy,r*.6f,paint);
        c.drawLine(cx-r*.76f,cy,cx+r*.76f,cy,paint);
        if(!yawOnly)c.drawLine(cx,cy-r*.76f,cx,cy+r*.76f,paint);
        paint.setStyle(Paint.Style.FILL);paint.setColor(isEnabled()?Color.rgb(203,238,121):Color.rgb(86,105,79));
        c.drawCircle(cx+x*r*.65f,cy+y*r*.65f,r*.23f,paint);
        paint.setColor(Color.rgb(13,30,20));c.drawCircle(cx+x*r*.65f,cy+y*r*.65f,3,paint);
    }
    @Override public boolean onTouchEvent(android.view.MotionEvent e) {
        if(!isEnabled())return false;
        int action=e.getActionMasked();
        if(action==MotionEvent.ACTION_DOWN) { pointer=e.getPointerId(0);getParent().requestDisallowInterceptTouchEvent(true); }
        if(action==MotionEvent.ACTION_UP || action==MotionEvent.ACTION_CANCEL ||
            (action==MotionEvent.ACTION_POINTER_UP && e.getPointerId(e.getActionIndex())==pointer)) {
            reset();listener.released();getParent().requestDisallowInterceptTouchEvent(false);performClick();return true;
        }
        int index=e.findPointerIndex(pointer);if(index<0)return true;
        float radius=Math.min(getWidth(),getHeight())*.5f-12;
        float px=(e.getX(index)-getWidth()*.5f)/(radius*.65f),py=yawOnly?0:(e.getY(index)-getHeight()*.5f)/(radius*.65f);
        float length=(float)Math.hypot(px,py);
        if(length<.10f){x=y=0;}else {float scale=Math.min(1,length);x=px/length*scale;y=py/length*scale;}
        invalidate();listener.changed(x,y);return true;
    }
    @Override public boolean performClick(){super.performClick();return true;}
}
