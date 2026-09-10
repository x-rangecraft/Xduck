package com.microduck.ble;

public final class ConnectionLifecycleChecks {
    private static void check(boolean condition,String message) {
        if(!condition)throw new AssertionError(message);
    }
    public static void main(String[] args) {
        // Reproduce the phone trace: dialog pauses/stops Activity, bond completes,
        // MTU/read/CCC finish before Activity resumes. The link must survive throughout.
        ConnectionLifecycle c=new ConnectionLifecycle();c.userConnect();
        check(c.beginPairing(),"first user connection may pair");
        check(!c.canCloseInBackground(),"dialog must not disconnect bonding");
        check(!c.setupComplete(false),"do not authenticate or control behind dialog");
        check(c.waitingForForeground(),"remember completed setup until resume");
        check(!c.canCloseInBackground(),"delayed onStop must not disconnect after bond");
        c.becameReady();
        check(!c.waitingForForeground(),"resume consumes pending ready once");
        check(c.canCloseInBackground(),"normal background must disconnect after ready");
        check(c.canRetry(),"established bonded links may reconnect");
        c.closed();
        check(!c.beginPairing(),"reconnect must not open a new pairing dialog");
        c.requireUserRetry();check(!c.canRetry(),"lost bond stops automatic reconnect");

        c.userConnect();check(c.beginPairing(),"explicit retry may pair again");
        c.closed();check(!c.canRetry(),"pair cancellation/timeout must not reconnect");
        check(!c.beginPairing(),"failed attempt must not pair again without user action");

        c.userConnect();check(c.beginPairing(),"next explicit attempt");
        check(c.setupComplete(true),"resume before CCC can finish immediately");
        c.becameReady();check(c.canCloseInBackground(),"pairing exemption ends at ready");

        c.userConnect();check(!c.canRetry(),"initial setup failure stops retries");
        check(!c.setupComplete(false),"already bonded setup cannot become ready in background");
        check(!c.waitingForForeground(),"ordinary background does not keep a connection");
        check(c.canCloseInBackground(),"ordinary background closes connection");
        System.out.println("Connection lifecycle checks passed");
    }
}
