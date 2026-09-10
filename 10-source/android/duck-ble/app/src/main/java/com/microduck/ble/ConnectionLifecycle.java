package com.microduck.ble;

/** Pairing dialogs can pause the Activity before bonding, and resume after GATT setup. */
final class ConnectionLifecycle {
    private boolean pairingWindow, bondAttempted, waitingForForeground, autoReconnect;

    void userConnect() {
        pairingWindow=false;bondAttempted=false;waitingForForeground=false;autoReconnect=false;
    }
    boolean beginPairing() {
        if(bondAttempted || autoReconnect)return false;
        bondAttempted=true;pairingWindow=true;
        return true;
    }
    boolean canCloseInBackground() { return !pairingWindow; }
    boolean setupComplete(boolean foreground) {
        if(!foreground && pairingWindow) {waitingForForeground=true;return false;}
        return foreground;
    }
    boolean waitingForForeground() { return waitingForForeground; }
    void becameReady() {pairingWindow=false;waitingForForeground=false;autoReconnect=true;}
    boolean canRetry() {return autoReconnect;}
    void requireUserRetry() {autoReconnect=false;}
    void closed() {pairingWindow=false;waitingForForeground=false;}
}
