/* Command-line test harness: reads raw mono float32 PCM from stdin, prints one line per detected kick.
 *   cc -O2 -o kd_cli kd_cli.c -lm
 *   ffmpeg -i track.mp3 -f f32le -ac 1 -ar 44100 - | ./kd_cli 44100 [--act] [--no-tempo]
 */
#define KICK_DETECTOR_IMPLEMENTATION
#include "kick_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    float sr = argc > 1 ? (float)atof(argv[1]) : 44100.f;
    int dump = 0, notempo = 0;
    for (int i = 2; i < argc; i++) { if (!strcmp(argv[i], "--act")) dump = 1; if (!strcmp(argv[i], "--no-tempo")) notempo = 1; }
    static kd_t kd; kd_init(&kd, sr);
    if (notempo) kd_set_tempo_assist(&kd, 0);
    float buf[1]; kd_event ev[4];
    if (dump) {   /* one activation per hop, for verifying a port */
        while (fread(buf, sizeof(float), 1, stdin) == 1) {
            long long f0 = kd.nframes; kd_process(&kd, buf, 1, ev, 4);
            if (kd.nframes != f0) printf("%.6f\n", kd.activation);
        }
        return 0;
    }
    float blk[512]; size_t n;
    while ((n = fread(blk, sizeof(float), 512, stdin)) > 0) {
        int m = kd_process(&kd, blk, (int)n, ev, 4);
        for (int i = 0; i < m; i++) printf("%.4f\t%.4f\t%.3f\n", ev[i].sample / sr, ev[i].onset / sr, ev[i].strength);
    }
    return 0;
}
