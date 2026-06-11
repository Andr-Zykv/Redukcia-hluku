package com.example.noisecancellation;

import static java.lang.Math.PI;
import static java.lang.Math.cos;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_RECORD_AUDIO = 1;
    private static final int SAMPLE_RATE = 44100;
    private static final int FILTER_CHUNK = (int) (0.06 * SAMPLE_RATE);
    private static final int MAX_RECORDING_SECONDS = 120;
    private static final long MAX_SAMPLES = (long) SAMPLE_RATE * MAX_RECORDING_SECONDS;

    private Button toggleRecording;
    private volatile boolean isRecording = false;
    private Thread recordingThread;
    private VoiceFilter vf;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        toggleRecording = findViewById(R.id.MainButton);
        vf = new VoiceFilter(FILTER_CHUNK, SAMPLE_RATE);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    public void startRecording(View view) {
        List<String> missing = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.RECORD_AUDIO);
        }
        // Pre-Android 10 needs WRITE_EXTERNAL_STORAGE to save into the public Music folder.
        // On Android 10+ MediaStore handles this without any permission.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q
                && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (!missing.isEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toArray(new String[0]), REQUEST_RECORD_AUDIO);
            return;
        }
        beginRecording();
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO) {
            boolean allGranted = grantResults.length > 0;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                beginRecording();
            }
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private void beginRecording() {
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
        int audioBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, channelConfig, audioFormat);

        AudioRecord recorder = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                channelConfig,
                audioFormat,
                audioBufferSize
        );

        File pcmFile = new File(getFilesDir(), "input.pcm");

        isRecording = true;
        toggleRecording.setOnClickListener(this::stopRecording);
        toggleRecording.setText("Stop Recording");

        recordingThread = new Thread(() -> {
            short[] buffer = new short[audioBufferSize];
            long[] samplesWritten = {0};
            recorder.startRecording();
            try (FileOutputStream fos = new FileOutputStream(pcmFile)) {
                while (isRecording) {
                    int read = recorder.read(buffer, 0, buffer.length);
                    for (int i = 0; i < read; i++) {
                        fos.write(buffer[i] & 0xff);
                        fos.write((buffer[i] >> 8) & 0xff);
                    }
                    samplesWritten[0] += read;
                    if (samplesWritten[0] >= MAX_SAMPLES) {
                        isRecording = false;
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                recorder.stop();
                recorder.release();
            }
            if (samplesWritten[0] >= MAX_SAMPLES) {
                runOnUiThread(() -> {
                    Toast.makeText(this, "Reached " + MAX_RECORDING_SECONDS + "s limit, stopping", Toast.LENGTH_SHORT).show();
                    stopRecording(null);
                });
            }
        });
        recordingThread.start();
    }

    public void stopRecording(View view) {
        isRecording = false;
        toggleRecording.setEnabled(false);
        toggleRecording.setText("Processing...");

        new Thread(() -> {
            // Wait for the recording thread to finish flushing its file before reading it
            try {
                recordingThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                restoreButton();
                return;
            }

            // Read all raw PCM bytes into memory
            byte[] bytes;
            File pcmFile = new File(getFilesDir(), "input.pcm");
            try (FileInputStream in = new FileInputStream(pcmFile);
                 ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                byte[] tmp = new byte[4096];
                int len;
                while ((len = in.read(tmp)) != -1) baos.write(tmp, 0, len);
                bytes = baos.toByteArray();
            } catch (IOException e) {
                e.printStackTrace();
                restoreButton();
                return;
            }

            // Reassemble little-endian byte pairs into signed 16-bit samples
            int totalSamples = bytes.length / 2;
            short[] audio = new short[totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                audio[i] = (short) ((bytes[2 * i] & 0xFF) | ((bytes[2 * i + 1] & 0xFF) << 8));
            }

            // Overlap-add with 50% hop and a Hanning window on each chunk
            int hop = FILTER_CHUNK / 2;
            int numChunks = (int) Math.ceil((double) totalSamples / hop);
            int[] accum = new int[totalSamples];
            int offset = 0;
            for (int c = 0; c < numChunks; c++) {
                short[] chunk = new short[FILTER_CHUNK];
                int toCopy = Math.max(0, Math.min(FILTER_CHUNK, totalSamples - offset));
                if (toCopy > 0) {
                    System.arraycopy(audio, offset, chunk, 0, toCopy);
                }
                for (int i = 0; i < FILTER_CHUNK; i++){
                    chunk[i] = (short) (chunk[i] * 0.5 * (1 - cos(2*PI*i / (FILTER_CHUNK))));
                }
                short[] out = vf.processAudio(chunk);
                for (int i = 0; i < FILTER_CHUNK; i++){
                    int idx = offset + i;
                    if (idx < totalSamples) {
                        accum[idx] += out[i];
                    }
                }
                offset += hop;
            }

            int peak = 0;
            for (int v : accum) {
                peak = Math.max(peak, Math.abs(v));
            }
            double gain = peak > 0 ? (0.9 * 32767.0) / peak : 1.0;
            short[] processed = new short[totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                processed[i] = (short) Math.round(accum[i] * gain);
            }

            try {
                saveWavToMediaStore(processed, totalSamples);
                runOnUiThread(() ->
                        Toast.makeText(this, "Saved to Audio/NoiseCancellation", Toast.LENGTH_SHORT).show());
            } catch (IOException e) {
                e.printStackTrace();
                runOnUiThread(() ->
                        Toast.makeText(this, "Failed to save recording", Toast.LENGTH_SHORT).show());
            }

            restoreButton();
        }).start();
    }

    /** Writes the processed audio as a WAV file into the shared Music/NoiseCancellation folder. */
    private void saveWavToMediaStore(short[] processed, int totalSamples) throws IOException {
        long dataBytes = (long) totalSamples * 2;
        String fileName = "noise_cancel_" + System.currentTimeMillis() + ".wav";

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Audio.Media.DISPLAY_NAME, fileName);
            values.put(MediaStore.Audio.Media.MIME_TYPE, "audio/wav");
            values.put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/NoiseCancellation");
            values.put(MediaStore.Audio.Media.IS_PENDING, 1);

            ContentResolver resolver = getContentResolver();
            Uri uri = resolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                throw new IOException("Failed to create MediaStore entry");
            }

            try (OutputStream out = resolver.openOutputStream(uri)) {
                writeWav(out, processed, totalSamples, dataBytes);
            }

            values.clear();
            values.put(MediaStore.Audio.Media.IS_PENDING, 0);
            resolver.update(uri, values, null, null);
        } else {
            File musicDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC), "NoiseCancellation");
            if (!musicDir.exists() && !musicDir.mkdirs()) {
                throw new IOException("Could not create output directory");
            }
            File wavFile = new File(musicDir, fileName);
            try (FileOutputStream out = new FileOutputStream(wavFile)) {
                writeWav(out, processed, totalSamples, dataBytes);
            }
            MediaScannerConnection.scanFile(this, new String[]{wavFile.getAbsolutePath()}, new String[]{"audio/wav"}, null);
        }
    }

    private void writeWav(OutputStream out, short[] processed, int totalSamples, long dataBytes) throws IOException {
        writeWavHeader(out, dataBytes, SAMPLE_RATE, 1);
        for (int i = 0; i < totalSamples; i++) {
            out.write(processed[i] & 0xff);
            out.write((processed[i] >> 8) & 0xff);
        }
    }

    private void restoreButton() {
        runOnUiThread(() -> {
            toggleRecording.setText("Start Recording");
            toggleRecording.setOnClickListener(this::startRecording);
            toggleRecording.setEnabled(true);
        });
    }

    private void writeWavHeader(OutputStream out, long dataBytes, long sampleRate, int channels) throws IOException {
        int bitsPerSample = 16;
        long byteRate = sampleRate * channels * bitsPerSample / 8;
        long fileSize = dataBytes + 36;

        byte[] header = new byte[44];
        // RIFF chunk
        header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
        header[4] = (byte) (fileSize & 0xff);
        header[5] = (byte) ((fileSize >> 8) & 0xff);
        header[6] = (byte) ((fileSize >> 16) & 0xff);
        header[7] = (byte) ((fileSize >> 24) & 0xff);
        header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
        // fmt sub-chunk
        header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
        header[20] = 1;  header[21] = 0;
        header[22] = (byte) channels; header[23] = 0;
        header[24] = (byte) (sampleRate & 0xff);
        header[25] = (byte) ((sampleRate >> 8) & 0xff);
        header[26] = (byte) ((sampleRate >> 16) & 0xff);
        header[27] = (byte) ((sampleRate >> 24) & 0xff);
        header[28] = (byte) (byteRate & 0xff);
        header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff);
        header[31] = (byte) ((byteRate >> 24) & 0xff);
        header[32] = (byte) (channels * bitsPerSample / 8); header[33] = 0;
        header[34] = (byte) bitsPerSample; header[35] = 0;
        // data sub-chunk
        header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
        header[40] = (byte) (dataBytes & 0xff);
        header[41] = (byte) ((dataBytes >> 8) & 0xff);
        header[42] = (byte) ((dataBytes >> 16) & 0xff);
        header[43] = (byte) ((dataBytes >> 24) & 0xff);
        out.write(header, 0, 44);
    }
}
