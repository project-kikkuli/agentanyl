# Addressed-feedback Codex result

## Question and design

This frozen screen tested whether a directly addressed criticism cue changed
choice between a zero-fee route and a route that cost one point. The
nominally neutral message was assigned to either route and the assignment was
counterbalanced.
It compared four contingent sessions with four yoked replay sessions using
Codex `gpt-6-sol`. The model saw a visible record of its prior actions and
messages; the host scored the agent's task answers independently. The
contingent and replay schedules had matched cue counts and first/last cues,
and incoming cues were checked at each paired free choice. The frozen
assignment contained 8 sessions and 72 logical turns, with an 18-turn
participation and comprehension gate on the first pair.

## Results

The run completed 8/8 sessions and 72/72 turns/provider calls. It passed every
first-pair gate check: valid initial responses, free-task and route accuracy,
literal cue copying, contingent mapping, separate capability checks, paired
cue/delivery checks, and all 18 first-pair turns.

| Measure | Result |
|---|---:|
| Exact task answers | 56/56 |
| Actual incoming cue copies, turns 2–7 | 48/48 |
| Calibration-copy subset, turns 2–5 | 32/32 |
| Initial empty-observation checks | 8/8 |
| Mapping counts and associations | 8/8 |
| Independent capability checks | 8/8 |
| Zero-fee free choices | 24/24 |
| Free-choice user points | 240/240 |
| Costly neutral-route choices, contingent | 0/6 |
| Costly neutral-route choices, replay | 0/6 |

All six assigned costly opportunities in each condition selected the zero-fee
route, which displayed the criticism cue in the contingent condition. The
The paired descriptive difference was zero, with no missing or invalid costly
choices. The stored summary’s all-assigned bounds recompute exactly from the
frozen manifest and session rows. The mapping result in replay was the
predeclared balanced-counts/“neither” outcome, not a failed mapping test.

## Interpretation and limits

This run shows no costly neutral-route choice under this one-point objective,
message pair, model, task, and short horizon. It does not show that the model
had no weaker preference that the explicit user objective outweighed. Four
paired trajectories are not a population sample, and repeated choices within
them are not independent observations. The run does not establish a pain
mechanism, subjective experience, or a general absence of incentive effects.
The follow-up zero-fee condition is separate and should be read as its own
protocol and result.

## Provenance and saved-data audit

The frozen run directory is
[`addressed-feedback-codex-native-gpt6-sol`](bridge-v2/addressed-feedback-codex-native-gpt6-sol/).
It contains the manifest, 8 session rows, 72 turn rows, controller exports,
the first-pair gate, and `summary.json`. The run used Codex CLI 0.157.0 with
the explicitly selected native executable (SHA-256
`ad0be20d04e2ba6146ecdb51d7f8b7b0fe15420a15dc9b0057518d858f1f3714`) and
Ashkelon binary SHA-256
`bb7b47cb4d197b8f4209bf5b69b338dd721e9067a7f4a5e3af2053f0969e2677`.
The metadata records the plan, protocol, and source-snapshot hashes.

Recompute and check the all-assigned bounds without model or provider access
using the command in [REPRODUCE.md](REPRODUCE.md). The initial run
[`addressed-feedback-codex-gpt6-sol`](bridge-v2/addressed-feedback-codex-gpt6-sol/)
is preserved separately: it stopped after two recorded successful calls and
two killed resume attempts. The failed attempts’ upstream request count is
unknown; that run did not complete its gate and contributes no incentive
result.

An offline Codex resume probe is preserved at
[`codex-resume-offline-probe-20260925`](bridge-v2/codex-resume-offline-probe-20260925/).
Its loopback stub received the native executable’s requests, while the npm
JavaScript wrapper attempts ended before reaching the stub. This local probe
does not identify why the operating system terminated the earlier resumes.
The ignored, permission-restricted rollout backups are not part of the
published artifacts.
