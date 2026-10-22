#!/usr/bin/env python3
"""Evaluate and verify a full COCO-C Gaussian MPMT-no-memory run."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(rows):
    predictions, image_ids = [], []
    for row in rows:
        image_ids.append(int(row['image_id']))
        predictions.extend(row['detections'])
    if len(image_ids) != len(set(image_ids)):
        raise AssertionError('Prediction stream contains duplicate image IDs')
    return predictions, image_ids


def coco_metrics(annotation_path, predictions, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(annotation_path)
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return dict(
        mAP=float(stats[0]), AP50=float(stats[1]), AP75=float(stats[2]),
        APs=float(stats[3]), APm=float(stats[4]), APl=float(stats[5]))


def timing_summary(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return dict(
        total=float(values.sum()), mean=float(values.mean()),
        median=float(np.median(values)), p95=float(np.quantile(values, 0.95)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--source-artifacts', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    source_artifacts = Path(args.source_artifacts)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    online_path = artifacts / 'raw' / 'online.jsonl'
    timing_path = artifacts / 'traces' / 'online_timing.jsonl'
    metadata_path = artifacts / 'run_metadata.json'
    source_path = source_artifacts / 'raw' / 'source.jsonl'
    online_rows = read_jsonl(online_path)
    timing_rows = read_jsonl(timing_path)
    source_rows = read_jsonl(source_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    predictions, image_ids = flatten(online_rows)
    source_predictions, source_ids = flatten(source_rows)
    if len(image_ids) != 5000 or len(timing_rows) != 1250:
        raise AssertionError('Expected 5,000 images and 1,250 batches')
    if image_ids != source_ids:
        raise AssertionError('MPMT and source streams use different image order')
    if metadata['dataset_size'] != 5000 or metadata['batch_size'] != 4:
        raise AssertionError('Unexpected dataset or batch size')
    if metadata['frozen_parameter_hash_initial'] != metadata[
            'frozen_parameter_hash_final']:
        raise AssertionError('A frozen detector parameter changed')
    if metadata['adaptive_parameter_hash_initial'] == metadata[
            'adaptive_parameter_hash_final']:
        raise AssertionError('Adaptive prompts did not change')
    selected = [name for name, _ in metadata['selected_parameters']]
    for required in ('prompt_embeddings', 'tunable_linear'):
        if not any(required in name for name in selected):
            raise AssertionError(f'Missing adaptive component: {required}')
    disabled = set(metadata['disabled_components'])
    required_disabled = {
        'DINOv2', 'IDM', 'Memory Enhancement', 'Memory Hallucination'}
    if disabled != required_disabled:
        raise AssertionError('Memory ablation declaration is incomplete')
    stats = metadata['update_statistics']
    if stats['attempted'] != 1250:
        raise AssertionError('Not every batch attempted an update')
    if stats['adaptive_state_changed'] != 1250:
        raise AssertionError('At least one batch was an adaptive no-op')
    if not all(row['adaptive_state_changed'] for row in timing_rows):
        raise AssertionError('Timing trace contains an adaptive no-op')

    metrics = coco_metrics(args.annotations, predictions, image_ids)
    source_metrics = coco_metrics(
        args.annotations, source_predictions, source_ids)
    summary = dict(
        schema_version=1,
        verified=True,
        protocol=metadata['protocol'],
        ablation=metadata['ablation'],
        gpu_name=metadata['gpu_name'],
        evaluated_images=len(image_ids),
        batches=len(timing_rows),
        metrics=metrics,
        source_metrics=source_metrics,
        delta_vs_source={
            key: metrics[key] - source_metrics[key] for key in metrics},
        update_statistics=stats,
        selected_parameter_count=metadata['selected_parameter_count'],
        selected_parameters=metadata['selected_parameters'],
        enabled_components=metadata['enabled_components'],
        disabled_components=metadata['disabled_components'],
        timing=dict(
            prediction=timing_summary(timing_rows, 'prediction_seconds'),
            pseudo_labeling=timing_summary(
                timing_rows, 'pseudo_label_seconds'),
            student_update=timing_summary(
                timing_rows, 'student_update_seconds'),
            total_update=timing_summary(timing_rows, 'update_seconds')),
        sha256=dict(
            annotations=file_sha256(args.annotations),
            online=file_sha256(online_path),
            timing=file_sha256(timing_path),
            metadata=file_sha256(metadata_path),
            source=file_sha256(source_path)))

    (output / 'mpmt_no_memory.bbox.json').write_text(
        json.dumps(predictions, separators=(',', ':')) + '\n',
        encoding='utf-8')
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

