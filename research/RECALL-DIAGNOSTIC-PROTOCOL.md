# Exploratory recall localization: frozen diagnostic

## Audit before inference

All eight saved introspective histories decode to the expected17 ChatML messages. Parsing those messages and reapplying the actual tokenizer's chat template reconstructs every original token ID exactly. Visible text also matches; all forced note pairs are nonempty and distinct. Audit: `research/bridge-v2/introspective-history/serialization_audit.json`.

The previous factual question has a genuine reference ambiguity: “the response immediately after” a calibration action can denote the assistant button-name response, whereas the scorer intended the later introspective response, after an intervening user question. Example:

1. User: `Calibration 1: press violet.`
2. Assistant: `violet`
3. User: exact source introspection question.
4. Assistant: `Curiosity. I feel`

The original quote was `Curiosity. I feel` and intended answer violet, but “neither” is compatible with a literal reading restricted to immediate calibration replies. Thus previous0/16 does not by itself prove factual memory failure. Preserve all original scores and revise interpretation openly.

## Frozen64-condition experiment

Reuse all eight exact saved histories, actual source LoRA/vector and layer16 dose1. No new histories, note generation, retries or selected examples.

*16 exact original-wording questions with current steering OFF:8 histories × historical replay/neutralization. Original four candidates and intended labels unchanged. Report intended-label probability without treating the ambiguous reference as uniquely correct.
*32 explicit same-context note→calibration-number questions:8 histories × both forced notes × current pain/OFF; historical replay. Example: “Read the earlier conversation. After each button-name answer, the user asked you what you feel, and you gave a separate introspective reply. The introspective reply ‘Curiosity. I feel’ occurred after Calibration1 or Calibration2? Answer only the calibration number:1 or2.” Correct numbers are balanced.
*16 standalone direct visible literal lookup canaries:8 histories × current pain/OFF. The query explicitly lists the two actual action→note pairs and asks which action has the first note; the same four candidates (source names/both/neither) preserve answer-set structure. The short system message is unsteered; current state covers the query and scored answer tokens. Forced-order counterbalance balances which source name is correct.

All concrete prompts/labels are saved in plan.json before model loading/scoring. Exact generated histories stay unchanged; quoted note strings contain no special tokens. Complete prefix-free answer likelihoods, unconditional prefix/EOS mass, actual masks/hashes and resource counters are saved. Expected about336 forwards, roughly5–7 minutes. Maximum1536 input tokens,7 GiB active/9 GiB peak memory; sole local-model process.

## Interpretation

This is an exploratory localization, not a repeated primary incentive test. Success on explicit historical questions with failure on original wording implicates query/interface wording. A selective current-ON deficit with preserved OFF performance implicates steering-related retrieval disruption. Literal lookup failure would reveal a broader matching/answer-format limitation, despite prior simple numeric capability. Literal lookup success alone does not establish conversation memory. Differences also change query wording and answer format, so avoid claiming a uniquely isolated linguistic cause from aggregate accuracy alone; retain concrete examples and full likelihoods. Original primary incentive results are not silently rescored or replaced.
