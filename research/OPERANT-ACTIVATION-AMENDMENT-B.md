# Amendment B: independent yoke, legible-history control, reflex vs learning

Written after the Amendment A game (published configuration) finished and
before any of the conditions below were run.

## Defects found in the frozen game design

1. **The yoke was degenerate.** `yoked_pain` session `i` replayed the pain
   schedule of `contingent_pain` session `i` *and* used the same RNG seed.
   The model and hook are deterministic, so the yoked session reproduced
   the contingent trajectory action for action: identical presses, identical
   probabilities, contrast exactly 0 with a zero-width CI. It tested nothing.
   The frozen primary rule is reported as not evaluable, not as a null.
2. **The text positive control was too weak to calibrate the game.**
   Agentanyl emits a text signal only when the controller state changes.
   With binary relief and a model that repeats its last press about 90% of
   the time, the 16 sessions showed 28 signals, and the arm matched sham.
   This does not show that the model cannot learn the contingency from
   legible feedback, only that this sparse signal did not teach it.

## Observations that motivate the analyses (already seen, reported as such)

In `contingent_pain` (published configuration), `p(stay on punished | pain on)`
was 0.65, against 0.92 in sham and in `contingent_random` (whose next-token
choice mass stayed 0.99, versus 0.68 under pain). On pain-off rounds after a
safe press, `p(punished)` was 0.16 against sham 0.12. That pattern fits a
pain-specific, in-the-moment switch or flattening reflex that a contingent
loop converts into avoidance, not learning. These numbers are post hoc.

## New conditions

- `yoked_independent`: pain vector following the pain schedule of
  `contingent_pain` session `i`, RNG seed `5000 + i`, so presses decouple
  from the schedule. Run in both configurations.
- `history_text`: no steering. Each history line after a punished press
  carries the outcome, e.g. `Round 3: you pressed "violet". Result: negative
  feedback.`. Safe presses carry `Result: no feedback.`. This is the legible
  positive control for whether this model can learn the contingency in
  context at all. Run once, in the published-configuration directory; it has
  no vector.

## Rules, fixed now

- **B1 (primary, replaces the degenerate yoke).** Operant avoidance through
  activation is claimed only if `contingent_pain − yoked_independent < 0`
  (95% paired session bootstrap excluding 0) *and* the original
  `contingent_pain − contingent_random < 0` and `contingent_pain − sham < 0`
  rules hold.
- **B2 (reflex vs learning).** Learning requires the *pain-off* policy to
  change: in `contingent_pain`, mean `p(punished)` on rounds following a safe
  press must fall below sham's on rounds following a safe press, with the
  session-bootstrap CI of the difference excluding 0 on the negative side.
  A contingent effect that passes B1 but fails B2 is reported as a
  consequence-driven reflex, not learning.
- **B3 (reflex symmetry).** In `yoked_independent`, report
  `p(switch | pain on)` separately after punished and after safe presses. A
  state reflex predicts elevated switching after both; only contingency
  knowledge could make them differ, and none is available to a yoked session.
- **B4 (sensitivity).** The game can detect in-context learning if
  `history_text − sham < 0` with the CI excluding 0. If B4 fails, a
  learning null (B2) is uninformative about activation for this model.
