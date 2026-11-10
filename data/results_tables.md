
### Mode `accurate` — tempo assist on

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 21 (1.0%) | 39 (1.9%) | 0.981 | 0.990 | 0.985 |
| Library random sample, 128–139 BPM | 16931 | 395 (2.3%) | 581 (3.4%) | 0.966 | 0.977 | 0.971 |
| Library random sample, 140–149 BPM | 9583 | 144 (1.5%) | 355 (3.6%) | 0.964 | 0.985 | 0.974 |
| Library random sample, 150–164 BPM | 3651 | 107 (2.9%) | 265 (7.0%) | 0.930 | 0.971 | 0.950 |
| Library random sample, < 128 BPM | 6821 | 353 (5.2%) | 293 (4.3%) | 0.957 | 0.948 | 0.952 |
| Library random sample, ≥ 165 BPM | 713 | 62 (8.7%) | 71 (9.8%) | 0.902 | 0.913 | 0.907 |
| Synthwave / darkwave (hand-picked) | 280 | 6 (2.1%) | 13 (4.5%) | 0.955 | 0.979 | 0.966 |
| Techno (hand-picked) | 1479 | 29 (2.0%) | 67 (4.4%) | 0.956 | 0.980 | 0.968 |
| Tekno / acid (hand-picked) | 603 | 22 (3.6%) | 13 (2.2%) | 0.978 | 0.964 | 0.971 |
| Trance / psy (hand-picked) | 648 | 6 (0.9%) | 45 (6.6%) | 0.934 | 0.991 | 0.962 |
| **All excerpts with kicks** | 42728 | 1145 (2.7%) | 1742 (4.0%) | 0.960 | 0.973 | 0.966 |

Report latency after the kick onset: median 24 ms (10th–90th percentile 14–27 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 29 false triggers = 1.3 per minute.  
Per excerpt (38 s scored each): 216 of 578 excerpts have zero errors, 366 have at most two; the worst 10 excerpts account for 586 of 2916 errors.

### Mode `accurate` — tempo assist off

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 35 (1.7%) | 43 (2.1%) | 0.979 | 0.983 | 0.981 |
| Library random sample, 128–139 BPM | 16931 | 582 (3.4%) | 671 (3.9%) | 0.961 | 0.966 | 0.963 |
| Library random sample, 140–149 BPM | 9583 | 182 (1.9%) | 334 (3.4%) | 0.966 | 0.981 | 0.973 |
| Library random sample, 150–164 BPM | 3651 | 172 (4.7%) | 302 (8.0%) | 0.920 | 0.953 | 0.936 |
| Library random sample, < 128 BPM | 6821 | 525 (7.7%) | 415 (6.2%) | 0.938 | 0.923 | 0.931 |
| Library random sample, ≥ 165 BPM | 713 | 92 (12.9%) | 71 (10.3%) | 0.897 | 0.871 | 0.884 |
| Synthwave / darkwave (hand-picked) | 280 | 8 (2.9%) | 14 (4.9%) | 0.951 | 0.971 | 0.961 |
| Techno (hand-picked) | 1479 | 34 (2.3%) | 63 (4.2%) | 0.958 | 0.977 | 0.968 |
| Tekno / acid (hand-picked) | 603 | 15 (2.5%) | 20 (3.3%) | 0.967 | 0.975 | 0.971 |
| Trance / psy (hand-picked) | 648 | 5 (0.8%) | 16 (2.4%) | 0.976 | 0.992 | 0.984 |
| **All excerpts with kicks** | 42728 | 1650 (3.9%) | 1949 (4.5%) | 0.955 | 0.961 | 0.958 |

Report latency after the kick onset: median 25 ms (10th–90th percentile 16–28 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 27 false triggers = 1.2 per minute.  
Per excerpt (38 s scored each): 200 of 578 excerpts have zero errors, 336 have at most two; the worst 10 excerpts account for 615 of 3626 errors.

### Mode `fast` — tempo assist on

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 124 (6.1%) | 132 (6.5%) | 0.935 | 0.939 | 0.937 |
| Library random sample, 128–139 BPM | 16931 | 581 (3.4%) | 736 (4.3%) | 0.957 | 0.966 | 0.961 |
| Library random sample, 140–149 BPM | 9583 | 331 (3.5%) | 293 (3.1%) | 0.969 | 0.965 | 0.967 |
| Library random sample, 150–164 BPM | 3651 | 203 (5.6%) | 249 (6.7%) | 0.933 | 0.944 | 0.938 |
| Library random sample, < 128 BPM | 6821 | 587 (8.6%) | 327 (5.0%) | 0.950 | 0.914 | 0.932 |
| Library random sample, ≥ 165 BPM | 713 | 92 (12.9%) | 57 (8.4%) | 0.916 | 0.871 | 0.893 |
| Synthwave / darkwave (hand-picked) | 280 | 16 (5.7%) | 11 (4.0%) | 0.960 | 0.943 | 0.951 |
| Techno (hand-picked) | 1479 | 67 (4.5%) | 157 (10.0%) | 0.900 | 0.955 | 0.927 |
| Tekno / acid (hand-picked) | 603 | 37 (6.1%) | 13 (2.2%) | 0.978 | 0.939 | 0.958 |
| Trance / psy (hand-picked) | 648 | 12 (1.9%) | 23 (3.5%) | 0.965 | 0.981 | 0.973 |
| **All excerpts with kicks** | 42728 | 2050 (4.8%) | 1998 (4.7%) | 0.953 | 0.952 | 0.953 |

Report latency after the kick onset: median 9 ms (10th–90th percentile 3–12 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 50 false triggers = 2.3 per minute.  
Per excerpt (38 s scored each): 181 of 578 excerpts have zero errors, 321 have at most two; the worst 10 excerpts account for 773 of 4098 errors.

### Mode `fast` — tempo assist off

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 121 (6.0%) | 145 (7.1%) | 0.929 | 0.940 | 0.935 |
| Library random sample, 128–139 BPM | 16931 | 649 (3.8%) | 1202 (6.9%) | 0.931 | 0.962 | 0.946 |
| Library random sample, 140–149 BPM | 9583 | 362 (3.8%) | 442 (4.6%) | 0.954 | 0.962 | 0.958 |
| Library random sample, 150–164 BPM | 3651 | 227 (6.2%) | 364 (9.6%) | 0.904 | 0.938 | 0.921 |
| Library random sample, < 128 BPM | 6821 | 675 (9.9%) | 665 (9.8%) | 0.902 | 0.901 | 0.902 |
| Library random sample, ≥ 165 BPM | 713 | 82 (11.5%) | 82 (11.5%) | 0.885 | 0.885 | 0.885 |
| Synthwave / darkwave (hand-picked) | 280 | 16 (5.7%) | 17 (6.0%) | 0.940 | 0.943 | 0.941 |
| Techno (hand-picked) | 1479 | 76 (5.1%) | 190 (11.9%) | 0.881 | 0.949 | 0.913 |
| Tekno / acid (hand-picked) | 603 | 34 (5.6%) | 19 (3.2%) | 0.968 | 0.944 | 0.955 |
| Trance / psy (hand-picked) | 648 | 10 (1.5%) | 24 (3.6%) | 0.964 | 0.985 | 0.974 |
| **All excerpts with kicks** | 42728 | 2252 (5.3%) | 3150 (7.2%) | 0.928 | 0.947 | 0.937 |

Report latency after the kick onset: median 9 ms (10th–90th percentile 4–13 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 44 false triggers = 2.0 per minute.  
Per excerpt (38 s scored each): 157 of 578 excerpts have zero errors, 274 have at most two; the worst 10 excerpts account for 874 of 5446 errors.
