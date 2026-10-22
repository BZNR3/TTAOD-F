# TTAOD-F MPMT without memory

This repository is a reproducible ablation of
[TTAOD-F](https://github.com/gaoyingjay/TTAOD-F), based on official commit
`16635fd26505d38cab79ef6dccc4efb2d24fca02` and the NeurIPS 2025 paper
*Test-Time Adaptive Object Detection with Foundation Model*.

The experiment answers one narrow question: how does the paper's complete
Multi-modal Prompt-based Mean-Teacher (MPMT) perform on COCO-C Gaussian noise
severity 5 when all memory functionality is removed?

## Exact ablation

Enabled:

- text prompt tuning (TPT), AdamW learning rate 0.02;
- 10-token deep visual prompt tuning (VPT), AdamW learning rate 0.2;
- the paper's first-image Test-time Warm-start (TTWS);
- EMA teacher on weak views and student on strong views;
- native Grounding DINO classification/localization losses;
- prompt-only teacher EMA with retention 0.999;
- batch size 4, one ordered pass over all 5,000 COCO-C images.

Disabled by construction:

- DINOv2;
- Instance Dynamic Memory (IDM);
- Memory Enhancement (ME);
- Memory Hallucination (MH).

The public upstream loop freezes `tunable_linear.weight` and therefore does
not actually train the paper's text prompt. This fork uses a dedicated online
loop that explicitly selects both text and visual prompts and verifies that
all frozen detector parameters remain byte-identical.

## Online protocol

For each batch, the EMA teacher first predicts the test view. Those predictions
are saved before the current batch can update the model. The teacher then
generates pseudo-labels from a weak view, the student optimizes its prompts on
a strong view, and the teacher prompts receive an EMA update. Runtime model
branches read an annotation manifest with an empty `annotations` array; real
COCO ground truth is opened only by the separate offline evaluator.

## Run

The Markov defaults are encoded in
`configs/mm_grounding_dino/ttaod/mpmt_no_memory_coco_gaussian.py`. Paths can be
overridden with `TTAOD_COCO_C_ROOT`, `TTAOD_GTFREE_ANN`, `TTAOD_TTWS_FILE`, and
`TTAOD_CHECKPOINT`.

```bash
CUDA_VISIBLE_DEVICES=GPU-9a80f124-e0ed-73a1-11aa-21b0ab8093f0 \
PYTHONPATH=. /bigdata/baoming/miniconda3/envs/ttaod/bin/python train.py \
  configs/mm_grounding_dino/ttaod/mpmt_no_memory_coco_gaussian.py \
  --work-dir work_dirs/mpmt_no_memory_coco_gaussian
```

Evaluate only after the online prediction file has closed:

```bash
PYTHONPATH=. /bigdata/baoming/miniconda3/envs/ttaod/bin/python \
  tools/summarize_mpmt_no_memory.py \
  --annotations /bigdata/baoming/datasets/coco-c-q95-severity5-all15-20260818/annotations/instances_val2017.json \
  --artifacts work_dirs/mpmt_no_memory_coco_gaussian/artifacts \
  --source-artifacts /bigdata/baoming/TTAOD-F-paper-original-causal/work_dirs/paper_original_direct_gtfree_full5000_run3/artifacts \
  --output-dir work_dirs/mpmt_no_memory_coco_gaussian_analysis
```

The run records per-batch prediction, pseudo-labeling, and student-update time,
selected parameters, initial/final prompt hashes, frozen parameter hashes,
losses, and all raw predictions.

## Result

The verified A100 run on all 5,000 COCO-C Gaussian severity-5 images achieved
**19.78 mAP / 29.11 AP50**, versus **13.82 / 20.59** for the frozen source.
See the [full experiment report](reports/coco_gaussian_a100_20261022.md) for
all COCO metrics, timing, checksums, and the comparison with the complete
TTAOD-F run.

## License and citation

The upstream project is distributed under the Apache 2.0 license retained in
this fork. Please cite the original TTAOD-F paper and repository when using
this ablation.
