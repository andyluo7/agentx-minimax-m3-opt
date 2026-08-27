# vLLM #52664 review packet

Status date: 2026-08-26

This packet tracks the clean replacement prepared for
[vLLM #52664](https://github.com/vllm-project/vllm/pull/52664). The pull
request remains draft until its dependencies, DCO, and current-head MI355X
validation are complete.

## Proposed head

- Helper branch: `andyluo7/vllm:fix/pr-52664-aiter-gates`
- Commit: `ee5f001be6454bc9616dcf0db9e4276efe1387c6`
- Base: vLLM #52849 commit
  `78e1f096add72ec2816eab5da08cba221260142b`
- AITER dependency: #4787 commit
  `cb3c7a628645dd9b03610d36b543d39225e2cde5`

The helper branch is for review and author coordination. It must not replace
the original author's branch without agreement.

## Exact AITER capability gate

The AITER implementation is selected only when all of these conditions hold:

- ROCm on `gfx950`;
- FP8 E4M3 index query/cache;
- top-k block count 16;
- score type `max`;
- sparse block size 128;
- index head dimension 128;
- one or two index heads per tensor-parallel rank;
- `num_index_heads * max_decode_query_len <= 16`;
- no more than 8,192 sparse blocks, corresponding to a maximum model length
  of 1,048,576 tokens at block size 128; and
- the three public AITER #4787 entry points are importable.

An unsupported FP8 configuration fails with the precise rejected condition.
BF16 continues to use the existing Triton implementation.

## CUDA and other-device isolation

- AITER and the AMD indexer module are imported only inside the ROCm selector
  branch.
- CUDA keeps the existing SM100 MSA selection and Triton fallback.
- Other devices keep the existing Triton selection.
- The default BF16 index cache is unchanged.
- Focused tests guard against imports from `vllm.models.minimax_m3.amd` or
  `aiter` while exercising CUDA and non-ROCm selector paths.

## Focused coverage

The helper adds or extends tests for:

- accepted H1 x Q16 and H2 x Q8 decode shapes;
- rejected H1 x Q17 and H2 x Q9 decode shapes;
- the 1,048,576-token supported boundary and 1,048,577-token rejection;
- unsupported dtype, score type, top-k, page size, head count, head dimension,
  architecture, and missing AITER entry points;
- preservation of actionable FP8 rejection reasons;
- CUDA and non-ROCm lazy-import isolation;
- the public AITER #4787 top-k wrapper's keyword-only argument order;
- page-16 rebasing and negative padding;
- mixed prefill/decode metadata;
- speculative decode and padded graph rows; and
- emitted sparse page-table parity against the reference builders.

Local changed-file validation at `ee5f001be645`:

```text
pre-commit run --files <all eight changed Python files>: PASS
git diff --check: PASS
```

MI355X focused validation job 1604 used the pinned ROCm image and a
`uv`-managed `.venv`. It passed:

```text
tests/kernels/attention/test_minimax_m3_indexer_selection.py: 19 passed
selected test_minimax_m3.py AITER/page-table cases: 4 passed
```

The GPU tests build their model configuration locally and make no Hugging Face
network request. The exact logs are
`upstream-validation/logs/pr52664-tests-1604.{out,err}` on the shared AAC17
filesystem.

## Dependency status

| Dependency | Exact head | Current state | Required action |
| --- | --- | --- | --- |
| AITER #4787 | `cb3c7a628645` | Open, non-draft, mergeable; full AITER gate green | Maintainer review and merge |
| vLLM #52849 | `78e1f096add7` | Open, approved, mergeable; pre-commit and AMD CI green | Human authors must repair DCO history |
| vLLM #52664 | `92b66b2bdf03` | Draft on the old head; DCO also includes the unsigned dependency history | Replace/rebase with the signed helper after dependencies are ready |

The invalid #52849 commits are:

- `98426528d127`: no sign-off;
- `c2b78e442d27`: no sign-off; and
- `f587f84faa28`: malformed `Signed-off-by: <>`.

The authors must rewrite and sign their own commits. A later empty signed
commit does not repair DCO, and another contributor must not forge their
sign-offs. vLLM has no `.github/dco.yml` enabling remediation commits, so the
default DCO App behavior requires signed rewrites of the failing commits (or a
maintainer override, which is not the requested repair). The helper's single
integration commit has a valid Andy Luo sign-off and explicit Cursor/OpenAI
Codex attribution; it removes #52664's own unsigned commit from the proposed
history, but #52849 still has to repair its history or merge before #52664 can
become DCO-clean against `main`.

## Non-duplication

[vLLM #53448](https://github.com/vllm-project/vllm/pull/53448) is a different,
draft Triton optimization. It changes the existing Triton scorer/top-k and
supports BF16 as the default path. #52664 integrates the separate AITER #4787
gfx950 FP8 MFMA kernels, emits the AITER sparse-attention page table, and is
stacked on #52849. The implementations overlap in model/indexer files and are
alternatives rather than additive dependencies.

## Current-head MI355X validation

The focused test and benchmark chain is pinned as follows:

- vLLM #52664 helper: `ee5f001be645`;
- validation integration head: `720d05565afb`;
- vLLM #53695: `9e7aa17a16eb`;
- vLLM #53821: `251d3d14d778`;
- AITER #4787: `cb3c7a628645`;
- runtime image: `vllm/vllm-openai-rocm:v0.27.1`, registry digest recorded by
  the runner; and
- full FP8 indexer computation, with no skip-indexer shortcut.

The dependency-gated Slurm chain is:

1. focused exact-head pytest: job 1604, passed;
2. digest-pinned official-image bootstrap on node 3: job 1648;
3. eight-sample real-target smoke: job 1649, `afterok:1648`;
4. full 1,319-sample GSM8K: job 1650, `afterok:1649`;
5. matched 3,600-second AgentX C1: job 1651, `afterok:1650`; and
6. matched 3,600-second AgentX C32 with vLLM simple CPU KV offload: job 1652,
   `afterok:1651`.

Every promoted result must preserve source, image, config, effective command,
request accounting, raw artifacts, exit status, and warmup exclusion. It must
also have zero measured-phase JIT, zero GPU/KFD faults, and clean teardown.

This is current-source validation: every Python module comes from integration
head `720d05565afb`, while only ABI-matched compiled extension modules come from
the digest-pinned official v0.27.1 image. The PR stack is Python-only. This is
not evidence for an official patch-free release image.

## AI-assisted contribution checklist

- The replacement commit includes `Co-authored-by` attribution for Cursor and
  OpenAI Codex and a valid human `Signed-off-by` trailer.
- The final PR description must explicitly disclose AI assistance.
- The final PR description must include the non-duplication analysis, exact
  test commands/results, and model-evaluation results.
- The human submitter must review and understand every changed line and confirm
  that review before the PR is marked ready.
