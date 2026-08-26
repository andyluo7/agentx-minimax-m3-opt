# Source and image provenance

## Validated runtime

- Image tag: `vllm/vllm-openai-rocm:v0.27.1`
- Registry digest: `sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`
- vLLM source base inside the image: `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- AITER source base inside the image: `545d97cc0aaeef7915e2c6df80b7f63f9d8ad657`
- InferenceX source recorded by the validated curve:
  `62bf882f2df0d732752bc9d83caa3ee2324bda79`
- Target model: `amd/MiniMax-M3-MXFP4`
- Target revision: `b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a`
- Draft model: `Inferact/MiniMax-M3-EAGLE3-GQA`
- Draft revision: `96692486b5fd38ebf8fd2a5f6bb53427d30819a8`

## Archived runtime delta

The self-contained recipe bundle under `recipes/inferencex/` came from the
closed InferenceX #2726 branch. Its patch checksums are:

- vLLM patch: `662e0d70ccd051225b638bfd1f541f0861e1fbba56502696d65937906dcd1162`
- AITER patch: `b3d47fc883288532e92cf026945ce7f7d61fff1a100f7c731710e531c34ca742`
- precompile helper: `cbc30626128b18a917dc6f4145945eb6f109f708675eaf9b191e770bbd6cda5c`

The preserved performance artifacts record historical recipe SHA-256
`76f025a44df07ff54ea4ceb6ec076f40403fa1a091e0ff1c03d4aa63b76a670d`.
That exact cluster-side filename was not available locally when this bundle
was assembled. The repository instead includes the later self-contained PR
recipe snapshot at SHA-256
`6a10cbd99b0f49f0315d16d1d659afbf261fa6a550cd65cfbe598137124c7951`.
The raw `vllm_command.txt`, `benchmark_command.txt`, and `run-metadata.txt`
files are therefore authoritative for the exact historical run configuration.
Do not claim the packaged recipe file is byte-identical to the historical
cluster-side recipe.

## Upstream transition

The runtime delta is being decomposed into focused vLLM/AITER changes. See
`docs/UPSTREAM_PRS.md` for the current mapping. Once those changes appear in an
official image, the patch bundle becomes historical-only and the patch-free
image must be validated again before publishing a new InferenceX recipe.
