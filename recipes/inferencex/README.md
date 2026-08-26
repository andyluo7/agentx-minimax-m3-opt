# Historical InferenceX recipe bundle

This directory is a self-contained copy of the MiniMax-M3 MI355X recipe from
closed InferenceX PR #2726:

- `minimaxm3_fp4_mi355x_mtp.sh`: AgentX/GSM8K recipe;
- `apply_minimaxm3_agentx_patches.sh`: marker- and checksum-gated ephemeral
  patch application for the pinned image;
- `precompile_minimaxm3_aiter.py`: startup precompile coverage;
- `patches/`: archived vLLM and AITER source deltas.

Copy all files into `benchmarks/single_node/agentic/` in the pinned InferenceX
checkout. Do not copy only the top-level recipe: it invokes the adjacent patch
and precompile files.

These patches are preserved for historical reproduction. They are not intended
for a new InferenceX pull request. Their maintained destinations are listed in
`docs/UPSTREAM_PRS.md`.
