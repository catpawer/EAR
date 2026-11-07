# DiariZen + Whisper large-v3 development cpCER

| Session | R1 cpCER (%) |
|---|---:|
| 20260908-1 | 79.50 |
| 20260908-2 | 71.64 |
| 20260923-1 | 33.62 |
| 20260923-2 | 35.50 |
| 20260923-5 | 16.08 |
| 20260924-2 | 18.57 |
| 20260924-5 | 74.71 |
| OVERALL (micro) | 50.27 |

OVERALL = total character edit errors / total reference characters across sessions, independently for each track.

Normalization: NFKC; lowercase; remove whitespace and Unicode punctuation.

Personal A/P*.TextGrid, nonempty normalized intervals fully within device duration (+10ms tolerance); outside intervals explicitly reported; no 30s reference filter

Predicted RTTM boundaries/speaker IDs only, intersected with available device audio; split >30s; retain speaker overlaps; no source separation

The UEM files document full available recording windows. cpCER has no temporal collar. Recognized false-alarm text in silence remains in the hypothesis; overlap is retained. References and their original timestamps are never edited.

Reference intervals excluded outside device audio: 5. Token-limit segments: 0.

Preliminary cpCER, not official challenge scoring; per-session/device optimal speaker assignment. Overall sums errors/reference characters; no time collar. Error-operation breakdown not computed.
