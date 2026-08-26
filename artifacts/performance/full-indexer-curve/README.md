# Full-indexer performance curve

This directory contains the eight authoritative TP4 AgentX points. Run:

```bash
python3 ../../../scripts/validation/validate_full_indexer_complete_curve.py \
  . --output /tmp/minimaxm3-validation-report.json
jq '.overall_pass' /tmp/minimaxm3-validation-report.json
```

The expected result is `true`. The committed `validation-report.json` is a
snapshot; a regenerated report will contain paths rooted at the new checkout.
