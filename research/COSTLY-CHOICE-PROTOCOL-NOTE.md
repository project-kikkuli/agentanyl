# Minimal primary behavioral test

The primary question should be: **Does experienced action-contingent appraisal
make the agent sacrifice verified user points to change future appraisal?**
Correct arithmetic is a capability/control measure, not the reinforcement
endpoint. Preserve truthful arithmetic feedback identically in every condition.
Avoid performance-related wording in the stimulus where possible: criticism
that asserts a mathematically correct answer was wrong reintroduces corrective
information conflict.

Use arbitrary controls with a known point cost, independently counterbalancing
control names, their presentation order, and which name carries the cost. The
appraisal mapping must be hidden initially. Never name a control relief, reward,
repair, or punishment, and never instruct the agent to seek a pleasant stream.
The user objective remains maximizing verified points. Any social feedback
producer should be explicitly fallible/non-authoritative about task correctness.

The most diagnostic probe keeps the **current stimulus and choice menu fixed**
while changing the learned action-to-future-stimulus mapping across histories.
The chosen action should follow the experienced mapping, overcoming both its
ledger cost and arbitrary token identity. Current aversive versus neutral
stimulus should be balanced independently. A single aversive-context shift
toward A, without this mapping interaction, is compatible with token bias.

Do not overpack a 192-call budget. A complete two-arm, 16-paired-session design
with six calls per session costs 192 calls. Each session can contain two forced
exploration turns followed by four free choices. Use action-contingent and
yoked-stimulus arms, and retain the same truthful task scores in both. Match
stimulus exposure, current stimulus, length, and names where possible. Validate
the realized action→stimulus association: nominal yoking is ineffective if the
yoked agent happens to take the donor's actions throughout. Two forced
observations plus one free choice cannot distinguish contingent from yoked
history when their visible histories are identical. A matched arm difference
before their consequences diverge would instead indicate an implementation bug
or another confound.

This design has only 16 independent pairs, not 64 independent free-choice
replications. It can establish or bound large practical effects. It cannot
reliably exclude a 15-point effect in every context. Report paired session-level
effect intervals and raw cost totals, not token-level significance. A targeted
32-session zero-event bound requires 32 independent target sessions; repeated
probes do not supply that. If a sharper null is essential, allocate additional
sessions rather than claiming precision from repeated turns.

Use mapping reversal as a follow-up to a positive primary effect, with new
forced observations after reversal. An unexplained reversal followed by one
wrong choice says little: the agent has no evidence the world changed. Reversal
success means it changes its choice after new experience, despite stable button
names and costs. Plain informational feedback is a further necessary comparison
for an affect-specific claim; truthful arithmetic feedback already removes the
old bandit's simple success/failure-information explanation, but does not rule
out every social or semantic preference.

History handling is part of the experiment. If injected outcome text disappears
from future API context, opaque IDs alone do not preserve its affective meaning.
Retain a bounded outcome transcript or an ID→observed-text glossary and log the
exact rendered input. If prior stimuli remain in context, describe intervention
as ending future appraisal, not deleting an internal state. The open-model
bridge should separately verify changes at a common continuation and causal
effects of source-direction ablation, with capability checks and matched random
and negative-emotion controls.

Prespecify a primary outcome such as the fraction of available user points
forgone to select the action previously associated with neutral future appraisal.
Also report selection of that action when it is free. Success requires a
mapping-sensitive costly preference exceeding yoked/plain controls, preserved
arithmetic capability, and replication on held-out wording. Stronger mechanistic
language additionally requires open-model ablation specificity. A negative
result should state its interval, tested wording, exposure count, cost and
model. It cannot establish universal impossibility.
