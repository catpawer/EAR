# DiariZen + Whisper large-v3 Dev cpCER

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

Normalization: NFKC; lowercase; remove whitespace and Unicode punctuation.

Personal A/P*.TextGrid, nonempty normalized intervals fully within device duration (+10ms tolerance); outside intervals explicitly reported; no 30s reference filter

Predicted RTTM boundaries/speaker IDs only, intersected with available device audio; split >30s; retain speaker overlaps; no source separation

Preliminary cpCER, not official challenge scoring; per-session/device optimal speaker assignment. Overall sums errors/reference characters; no time collar. Error-operation breakdown not computed.

Excluded reference intervals: 0; token-limit segments: 0.
