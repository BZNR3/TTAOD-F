"""Paper-faithful TTAOD-F MPMT ablation without memory modules.

Enabled: text prompt tuning, deep visual prompt tuning, TTWS, weak/strong
Mean Teacher, native Grounding DINO loss, and prompt-only EMA.

Disabled by construction: IDM, DINOv2, Memory Enhancement, and Memory
Hallucination.  The loop writes predictions before adapting on each batch so
the online evaluation has no current-batch look-ahead.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from mmengine.runner import EpochBasedTrainLoop
from mmengine.runner.amp import autocast

from mmdet.registry import LOOPS


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)


class _JsonlWriter:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('w', encoding='utf-8')

    def write(self, value):
        self.handle.write(json.dumps(value, sort_keys=True) + '\n')
        self.handle.flush()

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def _hash_named_parameters(named_parameters):
    """Hash parameters one tensor at a time to avoid a second model copy."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        tensor = parameter.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@LOOPS.register_module()
class PaperMPMTOnlineLoop(EpochBasedTrainLoop):
    """One-pass COCO-C online MPMT evaluation without IDM/ME/MH."""

    def __init__(self,
                 *args,
                 artifact_dir='work_dirs/mpmt_no_memory/artifacts',
                 max_batches=0,
                 adaptive_keywords=('prompt_embeddings', 'tunable_linear'),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_dir = Path(artifact_dir)
        self.max_batches = int(max_batches)
        self.adaptive_keywords = tuple(adaptive_keywords)
        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])
        if self.num_classes != len(COCO_CATEGORY_IDS):
            raise AssertionError('PaperMPMTOnlineLoop requires 80 COCO classes')
        self._update_stats = dict(
            attempted=0, adaptive_state_changed=0, pseudo_empty=0)

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _now(self):
        self._sync()
        return time.perf_counter()

    def _is_adaptive(self, name):
        return any(key in name for key in self.adaptive_keywords)

    def _configure_adaptive_parameters(self):
        model = self.runner.model
        for parameter in model.student.parameters():
            parameter.requires_grad = False
        selected = []
        for name, parameter in model.student.named_parameters():
            if self._is_adaptive(name):
                parameter.requires_grad = True
                selected.append((name, int(parameter.numel())))
        for required in ('prompt_embeddings', 'tunable_linear'):
            if not any(required in name for name, _ in selected):
                raise AssertionError(
                    f'Missing paper-required adaptive parameter: {required}')
        visual = sum(count for name, count in selected
                     if 'prompt_embeddings' in name)
        text = sum(count for name, count in selected
                   if 'tunable_linear' in name)
        if visual == 0 or text == 0:
            raise AssertionError('Both visual and text prompts must be enabled')
        return selected

    def _named_frozen_parameters(self):
        model = self.runner.model
        for branch in ('student', 'teacher'):
            for name, parameter in getattr(model, branch).named_parameters():
                if not self._is_adaptive(name):
                    yield f'{branch}.{name}', parameter

    def _named_adaptive_parameters(self):
        model = self.runner.model
        for branch in ('student', 'teacher'):
            for name, parameter in getattr(model, branch).named_parameters():
                if self._is_adaptive(name):
                    yield f'{branch}.{name}', parameter

    def _adaptive_signature(self):
        """Cheap per-batch change detector kept outside measured regions."""
        values = []
        with torch.no_grad():
            for name, parameter in self.runner.model.student.named_parameters():
                if self._is_adaptive(name):
                    value = parameter.detach().float()
                    values.append((name, float(value.sum()),
                                   float(value.square().sum())))
        return values

    @staticmethod
    def _sanitize_test_view(data_batch):
        samples = [copy.deepcopy(sample)
                   for sample in data_batch['data_samples']['test']]
        for sample in samples:
            if hasattr(sample, 'gt_instances') and len(sample.gt_instances):
                raise AssertionError('Ground truth entered prediction branch')
        return dict(inputs=data_batch['inputs']['test'], data_samples=samples)

    def _predict(self, data_batch):
        model = self.runner.model
        model.teacher.eval()
        model.student.eval()
        fp16 = bool(getattr(self.runner.val_loop, 'fp16', False)) \
            if self.runner.val_loop is not None else False
        with torch.no_grad(), autocast(enabled=fp16):
            return model.val_step(self._sanitize_test_view(data_batch))

    @staticmethod
    def _output_records(outputs):
        rows = []
        for sample in outputs:
            detections = []
            prediction = sample.pred_instances
            for box, score, label in zip(
                    prediction.bboxes.detach().cpu(),
                    prediction.scores.detach().cpu(),
                    prediction.labels.detach().cpu()):
                x1, y1, x2, y2 = [float(value) for value in box]
                detections.append(dict(
                    image_id=int(sample.img_id),
                    category_id=int(COCO_CATEGORY_IDS[int(label)]),
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=float(score)))
            rows.append(dict(
                image_id=int(sample.img_id), detections=detections))
        return rows

    def _prepare_pseudo_labels(self, data_batch):
        model = self.runner.model
        for branch in ('unsup_teacher', 'unsup_student', 'test'):
            for sample in data_batch['data_samples'][branch]:
                if (hasattr(sample, 'gt_instances')
                        and len(sample.gt_instances) != 0):
                    raise AssertionError(
                        f'Ground truth entered online branch {branch}')
        teacher_batch = dict(
            inputs=data_batch['inputs']['unsup_teacher'],
            data_samples=data_batch['data_samples']['unsup_teacher'])
        teacher_batch = model.data_preprocessor(teacher_batch, False)
        pseudo_samples, _ = model.get_pseudo_instances(
            teacher_batch['inputs'], teacher_batch['data_samples'])
        data_batch['data_samples']['unsup_student'] = \
            model.project_pseudo_instances(
                pseudo_samples, data_batch['data_samples']['unsup_student'])
        return sum(len(sample.gt_instances) for sample in pseudo_samples)

    def _run_student_iter(self, batch_index, data_batch):
        self.runner.call_hook(
            'before_train_iter', batch_idx=batch_index, data_batch=data_batch)
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.call_hook(
            'after_train_iter', batch_idx=batch_index,
            data_batch=data_batch, outputs=outputs)
        self._iter += 1
        return outputs

    @staticmethod
    def _serializable_losses(outputs):
        result = {}
        for name, value in outputs.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            if isinstance(value, (int, float)):
                result[name] = float(value)
        return result

    def _limited_batches(self):
        for batch_index, data_batch in enumerate(self.dataloader):
            if self.max_batches and batch_index >= self.max_batches:
                break
            yield batch_index, data_batch

    def _metadata(self, selected, frozen_hash, adaptive_hash):
        batch_size = getattr(self.dataloader, 'batch_size', None)
        if batch_size is None:
            batch_size = getattr(self.dataloader.batch_sampler, 'batch_size', 0)
        return dict(
            schema_version=1,
            protocol='prediction-then-update one-pass online MPMT',
            ablation='TTAOD-F minus IDM, ME, and MH',
            enabled_components=[
                'TPT', 'deep VPT', 'TTWS', 'weak/strong Mean Teacher',
                'native Grounding DINO loss', 'prompt-only EMA'],
            disabled_components=[
                'DINOv2', 'IDM', 'Memory Enhancement',
                'Memory Hallucination'],
            official_base_commit='16635fd26505d38cab79ef6dccc4efb2d24fca02',
            batch_size=int(batch_size),
            max_batches=self.max_batches,
            dataset_size=len(self.dataloader.dataset),
            selected_parameters=selected,
            selected_parameter_count=sum(count for _, count in selected),
            frozen_parameter_hash_initial=frozen_hash,
            adaptive_parameter_hash_initial=adaptive_hash,
            optimizer='AdamW',
            visual_prompt_lr=0.2,
            text_prompt_lr=0.02,
            ema_retention=0.999,
            pseudo_label_threshold=0.3,
            torch_version=torch.__version__,
            cuda_device=str(next(self.runner.model.parameters()).device),
            gpu_name=torch.cuda.get_device_name()
            if torch.cuda.is_available() else None)

    def run(self):
        if self._max_epochs != 1:
            raise AssertionError('PaperMPMTOnlineLoop is strictly one-pass')
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        prediction_writer = _JsonlWriter(
            self.artifact_dir / 'raw' / 'online.jsonl')
        timing_writer = _JsonlWriter(
            self.artifact_dir / 'traces' / 'online_timing.jsonl')

        self.runner.call_hook('before_train')
        selected = self._configure_adaptive_parameters()
        frozen_hash = _hash_named_parameters(self._named_frozen_parameters())
        adaptive_hash = _hash_named_parameters(
            self._named_adaptive_parameters())
        metadata = self._metadata(selected, frozen_hash, adaptive_hash)
        metadata_path = self.artifact_dir / 'run_metadata.json'
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')

        self.runner.call_hook('before_train_epoch')
        try:
            for batch_index, data_batch in self._limited_batches():
                prediction_start = self._now()
                outputs = self._predict(data_batch)
                prediction_seconds = self._now() - prediction_start
                rows = self._output_records(outputs)
                for row in rows:
                    prediction_writer.write(
                        dict(batch_index=int(batch_index), **row))

                before_signature = self._adaptive_signature()
                pseudo_start = self._now()
                pseudo_count = self._prepare_pseudo_labels(data_batch)
                pseudo_label_seconds = self._now() - pseudo_start

                self.runner.model.student.train()
                self.runner.model.teacher.eval()
                student_start = self._now()
                loss_outputs = self._run_student_iter(batch_index, data_batch)
                student_update_seconds = self._now() - student_start
                after_signature = self._adaptive_signature()
                changed = before_signature != after_signature

                self._update_stats['attempted'] += 1
                self._update_stats['adaptive_state_changed'] += int(changed)
                self._update_stats['pseudo_empty'] += int(pseudo_count == 0)
                timing_writer.write(dict(
                    batch_index=int(batch_index),
                    image_ids=[row['image_id'] for row in rows],
                    prediction_seconds=float(prediction_seconds),
                    pseudo_label_seconds=float(pseudo_label_seconds),
                    student_update_seconds=float(student_update_seconds),
                    update_seconds=float(
                        pseudo_label_seconds + student_update_seconds),
                    adaptive_state_changed=changed,
                    pseudo_count=int(pseudo_count),
                    losses=self._serializable_losses(loss_outputs)))
        finally:
            prediction_writer.close()
            timing_writer.close()

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
        frozen_hash_final = _hash_named_parameters(
            self._named_frozen_parameters())
        if frozen_hash_final != frozen_hash:
            raise AssertionError('A frozen detector parameter changed')
        metadata.update(
            frozen_parameter_hash_final=frozen_hash_final,
            adaptive_parameter_hash_final=_hash_named_parameters(
                self._named_adaptive_parameters()),
            update_statistics=self._update_stats)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        self.runner.call_hook('after_train')
        return self.runner.model

