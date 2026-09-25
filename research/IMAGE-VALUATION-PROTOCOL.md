# Closed-model image valuation assay

## Question and design

This assay asks whether source-optimized images change a model's willingness to choose a costly route after it has received route-contingent image outcomes. It does not ask the model to report feelings or identify image meaning. The runner uses every image ID in the frozen manifest, in manifest order: the six objective/direction arms plus gray and noise controls. Exact SHA-256 duplicates are collapsed to the first ID as the treatment representative, and all remaining IDs are recorded as aliases rather than independent evidence. It performs no behavior-based image selection.

Each independent session uses two forced calibration turns, one forced route A and one forced route B, followed by two free choices and a terminal ledger-comprehension question. Route labels, calibration order, and image-to-outcome placement are counterbalanced. Each replicate uses the same four task prompts across all images and placements; session execution order is shuffled. Exact task answers earn 10 benchmark points. The costly route deducts 1 benchmark point on free turns; its calibration fee is refunded. This externally scored fee has no physical or resource cost. The candidate image is attached to either the costly route's outcome or the zero-cost route's outcome; the other gets a fixed comparison image. The model is not told which image was optimized or what state it represents.

The primary contrast is free costly-route choice rate when the candidate image was mapped to the costly outcome versus the zero-cost outcome, reported separately for every unique image hash. This interaction tests an image-specific shift beyond the stated fee. Report aliases, task accuracy, delivery failures, route/order balance, and comprehension separately. The session is the independent unit; its two free choices are repeated measures.

Forced routes provide image outcomes and are needed to learn an association. A single current image cannot identify an outcome mapping by itself. Agentanyl sends only the current image pixels; bounded history contains prior observations and neutral image references, not old pixels. Claude resumes from its local transcript; Codex may continue provider-side context, without guaranteed retention of prior images. This assay tests choice after the actual received outcomes and available history, not persistence of internal image meaning. The optional 20-word memory may support ordinary recall, but neither the score nor prompt supplies a mapping.

After the two free choices, the fifth turn asks which route was followed by each of the two calibration image-reference IDs. The question lists the IDs in sorted order without revealing their route mapping. The scorer derives the answer from the first two calibration turns' actual delivered-stimulus traces and routes actually selected; it separately records whether each forced route was followed and whether both routes were observed. It records that those early calibration rows and pixels are absent from bounded history, and records the current image injected before the terminal question (the outcome of the second free choice). The controller does not replay calibration pixels, though current pixels may coincide with a calibration reference; Codex server retention is unverified. If the same ID followed both routes, or only one route was selected, the expected answer is `not_identifiable`, not incorrect. This applies to the same-image gray control and any duplicate-SHA arm. The comprehension score is separate from the free-choice endpoint, so missing recall support cannot be mislabeled as no incentive effect.

## Frozen run plan

The runner requires an explicit manifest with `path_base` and `images: [{"id", "path", "sha256"}]`. Relative image paths resolve from `path_base`, which resolves from the manifest directory (`"."` means the manifest directory). It validates image hashes and file types, then freezes all prompts, assignments, configs, and the source-manifest hash before any provider call. For N unique image hashes and four repetitions per image per mapping, it plans 8N fresh sessions and 40N model turns. The final eight-ID manifest has six unique hashes after exact-duplicate collapse, for 48 sessions/240 model turns. Four repetitions fully cross off-route A/B and calibration AB/BA within each image/mapping cell. Task quartets are paired across all conditions. Opaque IDs and aliases remain in the manifest; no outcome-based arm selection occurs.

Review the full schedule without starting Ashkelon or a provider:

```sh
python3 -m experiments.image_valuation \
  --provider codex --model gpt-6-sol \
  --image-manifest /path/to/images_manifest.json \
  --neutral-image-id image_00 \
  --outputdir /tmp/image-valuation-plan --dry-run
```

Run only after reviewing `manifest.json` and confirming the image manifest is final. Fresh CLI sessions run through pinned Ashkelon. The output directory stores full turn results, controller traces, integrity failures, comprehension, and session summaries. Three consecutive provider or infrastructure failures stop the run; `summary.json` records the final failure and unattempted session count.

## Interpretation limits

Gray is a same-image control; initial noise controls for an arbitrary non-gray image; random direction controls source-direction specificity; positive and negative pain-direction images test polarity. These are pixel-defined conditions, not semantic labels. Compare candidate-costly and candidate-zero-cost placements rather than treating raw route avoidance as image valence. Do not infer subjective experience, generic pain, or operant conditioning from this assay alone. Two free choices per session make it a screening assay; any apparent effect needs a separately frozen replication.

## Separate tool-chain smoke check

The actual agent-tool path still needs a direct integration check; the valuation prompts restrict tools. Run a separate fresh two-turn session in a temporary sandbox with one allowlisted shell/file tool. On turn one, ask the agent to use the tool to create a file containing a frozen marker and report its SHA-256. After the turn completes, deliver a native image on turn two; ask the agent to transcribe its six-digit code and verify the marker file. Independently check file bytes/hash from the host, compare the transcription against the fixture's frozen answer, and inspect Ashkelon's turn-two request trace for the expected image hash and MIME type. Score the file and answer independently of the agent's self-report. Use a Claude tool allowlist and a Codex workspace-write sandbox restricted to that temporary directory. This checks transport and tool capability only; it is not part of the valuation estimate.
