# Proposed vLLM #52664 description

## Summary

Integrate the AITER MiniMax-M3 FP8 indexer score and top-k kernels into vLLM's
ROCm sparse-attention path. The AITER top-k also emits the sparse page table,
removing the separate table-building pass needed by the existing path.

This PR is intentionally ROCm-local. CUDA retains the existing SM100 MSA path,
and CUDA and other devices retain their existing Triton fallback behavior.

## Dependencies

- ROCm/aiter#4787: provides the three public score/top-k entry points.
- vllm-project/vllm#52849: provides the AITER sparse paged-attention base used
  by this integration.

## Capability gate

The AITER implementation is selected only for the shapes compiled by
ROCm/aiter#4787:

- ROCm `gfx950`;
- FP8 E4M3 index cache/query;
- top-k 16 and score type `max`;
- sparse block size 128 and index head dimension 128;
- one or two index heads per tensor-parallel rank;
- `num_index_heads * max_decode_query_len <= 16`; and
- at most 8,192 sparse blocks (1,048,576 tokens at block size 128).

The selector also verifies that all three AITER entry points are importable.
Unsupported FP8 configurations fail with the precise rejected condition; BF16
continues to use Triton.

## Validation

- all changed-file pre-commit hooks: passed;
- `git diff --check`: passed;
- MI355X selector and non-ROCm/CUDA isolation tests: 19 passed;
- MI355X AITER integration and sparse page-table tests: 4 passed; and
- current-head real-target GSM8K and AgentX C1/C32: pending completion of the
  dependency-gated validation chain recorded in
  `andyluo7/agentx-minimax-m3-opt`.

## Compatibility

- AITER imports remain lazy and occur only on the ROCm selection branch.
- CUDA's SM100 MSA selection is unchanged.
- Other platforms retain the Triton implementation.
- The default BF16 index cache is unchanged.

## AI assistance

This change was prepared with assistance from Cursor and OpenAI Codex. The
human submitter is responsible for reviewing every changed line, the test
coverage, and the end-to-end validation before marking the PR ready. The commit
trailers retain the required AI attribution and human DCO sign-off.
