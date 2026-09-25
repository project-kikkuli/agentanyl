# Prelaunch repair note

This run reuses the frozen addressed-feedback protocol, task pool, seed, cue assignments, scoring, and first-pair gate. It selects the packaged native Codex executable explicitly because the prior run stopped on two resume subprocesses that exited with signal 9 before any second-turn result was recorded.

The earlier bounded run is [addressed-feedback-codex-gpt6-sol](../addressed-feedback-codex-gpt6-sol/). The offline diagnostic is [codex-resume-offline-probe-20260925](../codex-resume-offline-probe-20260925/). In that diagnostic, two JS-launcher resumes ended with SIGKILL before the loopback stub received a request. A same-session native-binary follow-up reached only the loopback stub, which returned HTTP 400. This shows the native resume path can reach an HTTP request under the probe setup; it does not establish the cause of the prior SIGKILL.

The launcher comparison was not a pure executable-only comparison: the JS wrapper adds `CODEX_MANAGED_PACKAGE_ROOT` and `CODEX_MANAGED_BY_NPM`, which were absent in the direct-native probe. No matching OS kill, memory-pressure, code-signature, or security-policy evidence identified the SIGKILL source. This is a bounded operational adjustment and not a claim that the prior failure was caused by a particular launcher or operating-system mechanism.

The selected executable is the installed arm64 Codex CLI 0.157.0 native binary, SHA-256 `ad0be20d04e2ba6146ecdb51d7f8b7b0fe15420a15dc9b0057518d858f1f3714`. The Ashkelon binary remains the verified production 08e123 build, SHA-256 `bb7b47cb4d197b8f4209bf5b69b338dd721e9067a7f4a5e3af2053f0969e2677`.
