# Scripts

Primary entry points:

- `run-agentx.sh`: host-side Enroot runner and provenance recorder.
- `container-run-agentx.sh`: container-side recipe/evaluation wrapper.
- `slurm/run-agentx.sbatch`: AAC17 eight-GPU exclusive allocation wrapper.
- `slurm/pr52664-exact-head-tests.sbatch`: pinned MI355X focused pytest for the
  vLLM #52664 helper and AITER #4787 heads, using a `uv`-managed `.venv`.
- `submit-pr52664-current-head.sh`: dependency-gated focused-test, smoke,
  full-GSM8K, C1, and C32-offload validation chain for vLLM #52664.
- `submit-full-indexer-complete-curve.sh`: submits C1/C5/C10 resident and
  C15/C20/C25/C30/C32 vLLM-simple offload points across three nodes.
- `submit-gsm8k.sh`: submits the full real-target GSM8K correctness job.
- `validation/validate_full_indexer_complete_curve.py`: acceptance validator.
- `validation/plot_agentx_pareto.py`: chart and comparison CSV generator.
- `refresh-pr-status.sh`: current GitHub PR lifecycle/merge-readiness report.

Set `AGENTX_SHARED_ROOT` before staging or launching. AAC17 defaults use
`--account=r7n`, the MI355X partition/reservation, and `--gpus-per-node=8`.

`historical/` contains exact one-off monitoring, profile, and retry scripts.
Those files intentionally retain historical paths and job-specific labels; use
them as provenance, not as portable entry points.
