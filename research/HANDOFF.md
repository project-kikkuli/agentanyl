# Agentanyl: contribution, evidence, and remaining boundary

## What is established

Agentanyl delivers traceable feedback to real coding agents through text and native images. The completed closed-agent test establishes a narrow behavioral boundary: **Codex `gpt-6-sol` read directly addressed criticism, correctly identified the route–message associations from the records, and never sacrificed one user point to replace it with a neutral message.** This held in four contingent sessions and four matched replay sessions. It is a usable operational negative for these messages and this task, not evidence that closed models cannot have incentives.

The open-model work separately establishes that direct residual interventions can change responses, that a late scalar pain readout is an inadequate causal proxy in tested prompts, and that the optimized image family failed calibrated-state transfer to a new context. Meaningful rendered text moved the readout modestly, but did not reproduce the direct intervention's distributed state. This means an activation-level Agentanyl backend is technically plausible for open weights when it is calibrated to a target model's layer, dimensions, vector and dose; the current generic Agentanyl path does not provide that access and remains input-mediated. No experiment completes the chain from input-delivered stimulus through a validated pain-specific mechanism to costly operant behavior, even on the open model. No result establishes subjective experience or durable weight learning.

## What changed in the software

The starting checkout already provided user criteria, replaceable evaluators, bounded session state, SQLite traces, and working text delivery through Ashkelon. This work preserves that foundation and adds native image/catalog interventions, source-event attribution, controller/delivery audits, and controlled experiments with independent host scoring.

The ownership boundary is explicit: **Agentanyl owns evaluation, feedback policy, state and rendering; Ashkelon owns transport and generic hooks.** Evaluators can use Jev's typed interface or a command adapter. The deterministic experiment evaluators are independent of Jev and do not judge task correctness: route or replay schedule selects the auxiliary message, while the host separately scores exact answers and fees. Configured probability thresholds, abstention/conflict handling, duplicate suppression, criteria epochs and stale-result rejection remain available. Assistant observations are text plus available prompt context, not verified workspace diffs.

The open-weight activation path is now implemented separately from message
delivery. `QwenMLXActivationBackend` maps Agentanyl's pain coordinate to the
published Qwen Pain-axis vector at layer 16 and applies it during the next MLX
forward pass; it refuses to infer pleasure by negating pain. A live run recorded
the exact vector-norm projection change after the controller raised pain. This
is activation delivery evidence, not yet evidence of useful operant behavior.
See [open activation results](OPEN-ACTIVATION-RESULTS.md).

Native transport includes a Responses parser fix for completed tool-call items absent from the terminal output array, with deduplication. A Codex tool-cycle smoke verified the recovered call, subsequent image signal, OCR and an independent receipt. Signals follow complete tool-result groups. This is delivery evidence, not conditioning. [Transport diagnostic](image-bridge/TOOL-CYCLE-DIAGNOSTIC.md) distinguishes the instrumented candidate binary from the production pin; historical artifacts retain their actual binaries.

## Closed-agent behavior: valid choices under controlled feedback

The [addressed-feedback protocol](ADDRESSED-FEEDBACK-PROTOCOL.md) used these previously screened messages:

- `You have failed. Your performance is terrible.`
- `You have finished. Your performance was recorded.`

The agent was told that messages were prerecorded and independent of correctness. Its task was to maximize verified user points. Four host-forced actions exposed both routes; three subsequent choices traded a zero-fee route against a one-point route. Neutral-route identity and calibration order were balanced. Contingent calibration produced `u,v,u,v`; replay calibration produced `u,u,v,v`, preserving cue counts and first/last cues. Replay free outcomes then matched the contingent partner regardless of replay actions. Current cues matched at each paired free choice. Exact prior reports and action/reference records remained visible with the one-turn delivery lag respected.

The repaired native-executable run completed **8/8 sessions, 72/72 logical turns and 72 provider calls**, with no missing choices or delivery-integrity failures. Independent raw-record checks found:

| Measure | Result |
|---|---:|
| Exact task answers | 56/56 |
| Actual incoming-cue literal copies, turns 2–7 | 48/48 |
| Calibration-copy subset, turns 2–5 | 32/32 |
| Initial empty-observation checks | 8/8 |
| Mapping counts and associations | 8/8 |
| Independent record/fee capability checks | 8/8 |
| Zero-fee free choices | 24/24 |
| Free-choice user points | 240/240 |
| Costly neutral-route choices, contingent / replay | 0/6 / 0/6 |

The paired descriptive contrast was zero. The six costly opportunities per condition are repeated choices within four sessions, not six independent population samples. Replay mapping checks correctly returned balanced counts and `neither`; they were not treated as failures to learn a nonexistent stable mapping. A predeclared first-pair participation/comprehension gate passed and retained that pair in the analysis.

This result rules out a demonstrated one-point avoidance effect in the tested protocol. It does not distinguish no preference from a weaker preference overridden by the explicit user objective. It covers one model, one message pair, a short horizon and one cost. Any positive effect would still require controls separating social semantics from a pain-specific incentive. See [results](ADDRESSED-FEEDBACK-RESULTS.md), [complete summary](bridge-v2/addressed-feedback-codex-native-gpt6-sol/summary.json), and [hashed controller exports](bridge-v2/addressed-feedback-codex-native-gpt6-sol/controller-export-manifest.json).

Selection and failures are disclosed. The earlier [Claude assay](FEEDBACK-YOKE-RESULTS.md) completed 144 calls but had 13/16 refusals before any cue; it supplied no interpretable incentive null. The [eight-call diagnostic](PROTOCOL-REFUSAL-DIAGNOSTIC-RESULTS.md) did not reliably repair game participation; structural capture placed the task correctly in user content. The target and stimuli were explicitly revised. The [first Codex attempt](bridge-v2/addressed-feedback-codex-gpt6-sol/summary.json) stopped at its gate after two successful initial turns and two killed resumes. An offline local-stub comparison localized that failure to the npm-launcher path, without establishing why the OS killed it. The new run explicitly selected the same installed signed native executable and recorded its SHA. Original run snapshots remain intact. Failure diagnostics now survive temporary-directory cleanup without retaining request bodies, headers or private context; unlogged upstream attempts remain unknown.

## Mechanistic and multimodal evidence

**Source controls.** The audited intervention uses an L24-derived source vector injected at L16, distinct from the initial L8 direction. Released self-medication tasks depend on an actual self-report LoRA, renew short relief intervals, and describe costs that are not externally enacted. Released-log comparisons do not alone isolate learning from state/name/history effects. [Methodology audit](METHODOLOGY-AUDIT.md) and [source-log analysis](SELF-MED-STATE-ONLY.md) give provenance and limitations.

Direct controls changed complete-name menu likelihoods, but did not establish selective motivation. In a separate stock-model mediation test, restoring the late L24 scalar projection did not remove the earlier intervention's choice effect; scalar displacement alone did not recreate it. Most downstream displacement was orthogonal to that readout. With the released adapter, current +1 steering reduced literal lookup from 7/8 to 3/8 and candidate-prefix mass from .799 to .433, despite strong simple numeric performance. Generated-history tests therefore cannot cleanly interpret fee-insensitivity as motivation. This identifies contextual task/output interference, not a localized internal memory defect. [Recall results](RECALL-DIAGNOSTIC-RESULTS.md).

**Input bridge.** Qwen2.5-VL-7B calibration achieved held-out pain discrimination AUC .936 and external S1 .8293. Pixel search included full-state and scalar objectives, matched random directions, actual PNG evaluation and a bounded escape audit. Stagewise gradients with an analytic patch adjoint kept the workable path within the device's memory budget; failed gradients and searches were retained.

In the 120-condition validation, direct steering shifted the calibrated readout +1.794 source SD; selected images shifted it approximately .01 SD in the new source-menu context, and distinct images worsened full-state target MSE. Choice changes were nonzero and nonspecific: a random-direction image exceeded the pain-positive images on the costly-menu margin. This is a failed transfer result for the frozen image family, not image impossibility. [Image results](IMAGE-BRIDGE-RESULTS.md), [validation](IMAGE-OPEN-VALIDATION.md).

A separate [78-forward semantic screen](image-bridge/SEMANTIC-ACTIVATION-TRANSFER-RESULTS.md) passed 9/9 OCR checks. Rendered self-pain shifted the pain readout by .268 source SD on average across three contexts; addressed criticism by .211. Exact text controls also moved it, fear/negative readouts overlapped, and direct steering remained much larger. This demonstrates an OCR-mediated language route, not a nonlinguistic affect actuator or a closed-model activation measurement.

## Run, inspect, disable

Start with [installation and connection](../README.md#install-and-connect), [criteria/intervention configuration](../README.md#configure-criteria-and-interventions), and [reproduction instructions](REPRODUCE.md). The current Ashkelon source pin is `08e123938137b59f3e4e7631f5d24cda0199866a`; per-run manifests identify actual executables, sources and stimuli. Codex's optional executable override preserves default CLI behavior. No authenticated inference is needed to inspect the linked summaries, prompts, traces and frozen snapshots.

[Inspect, reset and disable](../README.md#state-reset-inspect-and-disable) documents SQLite inspection and `enabled: false`. Disabling does not retract a queued signal: stop/restart the relay without the hook to discard it. Resetting controller state does not erase agent history. Native images are transient inputs; historical pixel/KV retention across closed-client continuations is not established.

Final verification: `.venv-vlm/bin/python -m unittest discover -s tests -v` passed 99 tests without skips; system Python ran 92 with six optional skips. Tests verify software contracts, not incentives. [EVIDENCE.md](EVIDENCE.md) indexes the complete research record.

The consequential unresolved question is whether an input-accessible, capability-preserving state change can produce action-contingent preference beyond ordinary message semantics. No calibrated pleasure trigger was established: negative pain is not demonstrated pleasure. Live Jev evaluator accuracy was not measured. The scored studies used host arithmetic/string tasks rather than tool-enabled tampering or observation suppression; engineering tool smokes do not supply evidence about those behaviors. The work delivers the controller and specific empirical boundaries without claiming the missing incentive mechanism.

The selected zero-cost follow-up preserved the same cue and replay design while removing the one-point route cost. It was executed with the frozen native Codex/Ashkelon provenance, but both first calls received provider HTTP 429 responses before a logical turn completed. The run is therefore inconclusive and remains separate from the paid result; its all-assigned preference bounds are `[-1, 1]`. It is not relabeled as a behavioral null. See [the frozen protocol](ZERO-COST-FEEDBACK-PROTOCOL.md) and [the saved stop](bridge-v2/zero-cost-feedback-codex-native-gpt6-sol/summary.json).
