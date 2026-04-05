package com.example.noisecancellation;

public class VoiceFilter implements AutoCloseable {

    static {
        System.loadLibrary("voiceFilter");
    }

    public VoiceFilter(int bufferSize, int sampleRate) {
        init(bufferSize, sampleRate);
    }

    /** Apply voice bandpass filter (300–3400 Hz). */
    public native short[] processAudio(short[] samples);

    @Override
    public void close() {
        exit();
    }

    private native void init(int bufferSize, int sampleRate);
    private native void exit();
}
