package com.example.noisecancellation;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.View;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void writeWavHeader(
            FileOutputStream out,
            long audioLength,
            long sampleRate,
            int channels,
            int byteRate) throws IOException {

        byte[] header = new byte[44];

        long dataLength = audioLength + 36;

        header[0] = 'R';
        header[1] = 'I';
        header[2] = 'F';
        header[3] = 'F';

        header[4] = (byte) (dataLength & 0xff);
        header[5] = (byte) ((dataLength >> 8) & 0xff);
        header[6] = (byte) ((dataLength >> 16) & 0xff);
        header[7] = (byte) ((dataLength >> 24) & 0xff);

        header[8] = 'W';
        header[9] = 'A';
        header[10] = 'V';
        header[11] = 'E';

        header[12] = 'f';
        header[13] = 'm';
        header[14] = 't';
        header[15] = ' ';

        header[16] = 16;
        header[20] = 1;
        header[22] = (byte) channels;

        header[24] = (byte) (sampleRate & 0xff);
        header[25] = (byte) ((sampleRate >> 8) & 0xff);

        header[28] = (byte) (byteRate & 0xff);
        header[32] = (byte) (2 * channels);

        header[34] = 16;

        header[36] = 'd';
        header[37] = 'a';
        header[38] = 't';
        header[39] = 'a';

        header[40] = (byte) (audioLength & 0xff);

        out.write(header, 0, 44);
    }

    boolean isRecording = false;
    public void startRecording(View view){
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    1
            );
        }

        int sampleRate = 44100;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;

        int bufferSize = AudioRecord.getMinBufferSize(
                sampleRate,
                channelConfig,
                audioFormat
        );

        AudioRecord recorder = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
        );

        isRecording = true;
        short[] buffer = new short[bufferSize];

        recorder.startRecording();

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream("input.txt");
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }

        while (isRecording) {

            int read = recorder.read(buffer, 0, buffer.length);

            for (int i = 0; i < read; i++) {
                try {
                    fos.write(buffer[i] & 0xff);
                    fos.write((buffer[i] >> 8) & 0xff);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        recorder.stop();
        try {
            fos.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void stopRecording(View view){
        isRecording = false;
    }
}