# DiariZen + Whisper large-v3 development cpCER

| Session | D1 cpCER (%) |
|---|---:|
| 20260908-1 | 81.02 |
| 20260908-2 | 78.73 |
| 20260923-1 | 45.74 |
| 20260923-2 | 72.30 |
| 20260923-5 | 29.33 |
| 20260924-2 | 28.68 |
| 20260924-5 | 81.42 |
| OVERALL (micro) | 61.64 |

OVERALL = total character edit errors / total reference characters across sessions, independently for each track.

Normalization: NFKC; lowercase; remove whitespace and Unicode punctuation.

Personal A/P*.TextGrid, nonempty normalized intervals fully within device duration (+10ms tolerance); outside intervals explicitly reported; no 30s reference filter

Predicted RTTM boundaries/speaker IDs only, intersected with available device audio; split >30s; retain speaker overlaps; no source separation

The UEM files document full available recording windows. cpCER has no temporal collar. Recognized false-alarm text in silence remains in the hypothesis; overlap is retained. References and their original timestamps are never edited.

Reference intervals excluded outside device audio: 0. Token-limit segments: 0.

Preliminary cpCER, not official challenge scoring; per-session/device optimal speaker assignment. Overall sums errors/reference characters; no time collar. Error-operation breakdown not computed.
