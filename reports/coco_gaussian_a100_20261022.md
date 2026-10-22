# COCO-C Gaussian severity-5: MPMT without memory

Run date: 2026-10-22  
GPU: NVIDIA A100-PCIE-40GB  
Dataset: all 5,000 COCO 2017 validation images, Gaussian noise severity 5  
Protocol: one ordered online pass, batch size 4, predict each batch before
updating on it

## Ablation

Enabled: TPT, 10-token deep VPT, TTWS, weak/strong Mean Teacher, native
Grounding DINO loss, and prompt-only teacher EMA.

Disabled by construction: DINOv2, IDM, Memory Enhancement (ME), and Memory
Hallucination (MH). ME and MH have no independent executable path in the
public implementation without the DINOv2-backed IDM cache, so the exact
paper-table MPMT ablation removes all four.

## Results

All values below are COCO bbox AP on the same 5,000-image stream.

| Method | mAP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Frozen source | 13.82 | 20.59 | 14.59 | 4.77 | 15.63 | 23.21 |
| Prior complete TTAOD-F run | 18.74 | 27.62 | 19.93 | 7.50 | 19.34 | 30.58 |
| **MPMT without memory (this run)** | **19.78** | **29.11** | **21.12** | **8.10** | **20.92** | **32.33** |

This run improves over the frozen source by 5.96 mAP and 8.51 AP50 points.
Against the prior complete TTAOD-F run, it is higher by 1.04 mAP and 1.49
AP50 points. This is evidence for this exact stream and seed, not a claim that
memory is intrinsically harmful across corruptions; a repeated paired ablation
would be needed for that stronger conclusion.

## A100 timing

| Stage | Mean/batch | Median/batch | P95/batch | Total |
|---|---:|---:|---:|---:|
| Prediction | 0.625 s | 0.625 s | 0.678 s | 780.7 s |
| Teacher pseudo-labeling | 0.538 s | 0.539 s | 0.624 s | 672.5 s |
| Student update + prompt EMA | 1.936 s | 1.969 s | 2.211 s | 2,419.5 s |
| Total adaptation | 2.474 s | 2.511 s | 2.759 s | 3,091.9 s |

Prediction plus adaptation averages 3.098 seconds per batch. The prior
complete TTAOD-F A100 run averaged 8.528 seconds per batch under its recorded
instrumentation, so this no-memory path was 63.7% faster end to end.

## Verification

- Evaluated images: 5,000 unique IDs in the same order as the source stream.
- Update attempts: 1,250; adaptive state changed: 1,250; empty pseudo-label
  batches: 0.
- Trainable parameters: 812,160 total: 44,160 visual-prompt parameters and
  768,000 text-prompt parameters.
- Frozen-parameter SHA-256 before and after:
  `82a23f29e67afb0aa6430c1ccaa2826951695089cb3ae45a09b818ee163442a5`.
- Adaptive SHA-256 changed from
  `bd36807298dfc869dafe7902004e8e38ba4876600459f92247a50275ce72860d`
  to
  `bfcb09ff26e27450ab95382fa6a8ae79028030d8eae2bfd21b1e30f2e0bebf1f`.
- Online prediction SHA-256:
  `54f5ab0fcc360563857513c1bf133f5694b9c930835310a8fec167ba33114a48`.
- Timing trace SHA-256:
  `62645186c3a3df91eaa697a8ab0d46bf328d4d116926063cfb522cbeb184fc4f`.

The raw artifacts remain on Markov under
`/bigdata/baoming/TTAOD-F-MPMT-no-memory/work_dirs/` because the prediction
JSONL and COCO result JSON are about 237 MB and 541 MB respectively.
