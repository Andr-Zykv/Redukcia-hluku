#include <jni.h>
#include <fftw3.h>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Persistent state set up by init() and torn down by exit()
// ---------------------------------------------------------------------------
static struct {
    double* timeBuf = nullptr;
    fftw_complex* freqBuf = nullptr;
    fftw_plan fwdPlan = nullptr;
    fftw_plan invPlan = nullptr;
    int N = 0;
    int binLow = 0;
    int binHi = 0;
} g;

// Allocate FFTW buffers and build plans for the given buffer size / sample rate.
// Uses FFTW_MEASURE so the planner finds the optimal algorithm up front.
extern "C"
JNIEXPORT void JNICALL
Java_com_example_noisecancellation_VoiceFilter_init(JNIEnv *env, jobject thiz, jint bufferSize, jint sampleRate) {
    g.N = bufferSize;
    g.timeBuf = fftw_alloc_real(g.N);
    g.freqBuf = fftw_alloc_complex(g.N / 2 + 1);
    g.fwdPlan = fftw_plan_dft_r2c_1d(g.N, g.timeBuf, g.freqBuf, FFTW_MEASURE);
    g.invPlan = fftw_plan_dft_c2r_1d(g.N, g.freqBuf, g.timeBuf, FFTW_MEASURE);
    const double binHz = static_cast<double>(sampleRate) / g.N;
    g.binLow = static_cast<int>(std::ceil (300.0  / binHz));
    g.binHi = static_cast<int>(std::floor(3400.0 / binHz));
}

// Destroy plans and free buffers.
extern "C"
JNIEXPORT void JNICALL
Java_com_example_noisecancellation_VoiceFilter_exit(JNIEnv *env, jobject thiz) {
    fftw_destroy_plan(g.fwdPlan);
    g.fwdPlan = nullptr;
    fftw_destroy_plan(g.invPlan);
    g.invPlan = nullptr;
    fftw_free(g.timeBuf);
    g.timeBuf = nullptr;
    fftw_free(g.freqBuf);
    g.freqBuf = nullptr;
    g.N = 0;
}

// Apply the voice bandpass filter (300–3400 Hz) using the pre-built plans.
extern "C"
JNIEXPORT jshortArray JNICALL
Java_com_example_noisecancellation_VoiceFilter_processAudio(JNIEnv *env, jobject thiz, jshortArray samplesIn) {
    // PCM short → double
    jshort* src = env->GetShortArrayElements(samplesIn, nullptr);
    for (int i = 0; i < g.N; i++) g.timeBuf[i] = src[i];
    env->ReleaseShortArrayElements(samplesIn, src, JNI_ABORT);

    // Forward real-to-complex FFT
    fftw_execute(g.fwdPlan);

    // Bandpass: zero bins outside voice range
    const int numBins = g.N / 2 + 1;
    for (int k = 0; k < numBins; k++) {
        if (k < g.binLow) {
            g.freqBuf[k][0] *= 1.0/(g.binLow - k);
            g.freqBuf[k][1] *= 1.0/(g.binLow - k);
        }
        if (k > g.binHi) {
            g.freqBuf[k][0] *= 1.0/(k - g.binHi);
            g.freqBuf[k][1] *= 1.0/(k - g.binHi);
        }
    }

    // Inverse complex-to-real FFT
    fftw_execute(g.invPlan);

    // Normalise (FFTW c2r is unnormalised), clamp, write to output array
    jshortArray result = env->NewShortArray(g.N);
    jshort* dst = env->GetShortArrayElements(result, nullptr);

    for (int i = 0; i < g.N; i++) {
        double v = g.timeBuf[i] / g.N;
        v = std::max<double>(-32768.0, std::min<double>(32767.0, v));
        dst[i] = static_cast<jshort>(v);
    }

    env->ReleaseShortArrayElements(result, dst, 0);
    return result;
}
