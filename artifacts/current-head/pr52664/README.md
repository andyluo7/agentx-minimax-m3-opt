# vLLM #52664 current-head validation

This directory records validation of the clean vLLM #52664 replacement stack,
not the historical patched performance curve.

Pinned sources:

- vLLM #52664 helper: `ee5f001be6454bc9616dcf0db9e4276efe1387c6`
- vLLM #52849 base: `78e1f096add72ec2816eab5da08cba221260142b`
- vLLM integration validation head: `720d05565afb12f9a812607b1618a18973db11bc`
- vLLM #53695 source diff: `9e7aa17a16eb1435aa2ff8c40473498b905b310e`
- vLLM #53821 source diff: `251d3d14d778b8b86ec9025f44610dd809d97768`
- AITER #4787: `cb3c7a628645dd9b03610d36b543d39225e2cde5`
- runtime image: `vllm/vllm-openai-rocm:v0.27.1`

The end-to-end launcher imports every Python module from the pinned integration
checkout. It reuses only the ABI-matched compiled extension modules from the
official image, because this validation stack changes Python code only. The
effective source path and assembly mode are recorded in each run's metadata.

The integration branch contains patch-identical rebased copies of the current
#53695 and #53821 heads on top of the #52664 helper. It is validation-only;
end-to-end C1/C32 measurements exercise the assembled stack and do not isolate
#52664's performance contribution.

## Focused tests

AAC17 job 1604 ran on `vultr-mi355x-3` with eight MI355 OAM devices and no
pre-existing KFD users. Results:

- selector and device-isolation suite: 19 passed;
- selected AITER integration and sparse page-table suite: 4 passed; and
- no failed or skipped selected case.

The raw stdout and stderr are in `focused-tests/`. The tests use a temporary
local model config and do not require access to a gated Hugging Face model.

## End-to-end chain

The first two attempts did not reach model execution: job 1616 found no
persistent Enroot image on node 3, and job 1625 exposed an incomplete
model-directory-only overlay against the older image package. Both failures are
preserved separately and are not correctness or performance evidence.

The replacement sequence is pinned to `vultr-mi355x-3` after its current
allocation finishes:

- job 1648: verify the registry digest, import, and validate the official image;
- job 1649: eight-sample real-target smoke, `afterok:1648`;
- job 1650: full 1,319-sample GSM8K, `afterok:1649`;
- job 1651: matched 3,600-second AgentX C1, `afterok:1650`; and
- job 1652: matched 3,600-second AgentX C32 with vLLM simple CPU KV offload,
  `afterok:1651`.

The chain computes the FP8 lightning indexer; it does not use the historical
skip-indexer diagnostic shortcut. Results are not accepted until the request
counts, real-target accuracy, measured-phase JIT/fault scan, source/image
provenance, exit status, and teardown checks all pass.
