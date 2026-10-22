"""EMA hook for the prompt-only Mean Teacher described by TTAOD-F."""

from typing import Optional

import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class PaperPromptMeanTeacherHook(Hook):
    """Update only text and visual teacher prompts with an EMA.

    ``retention`` is the paper's :math:`\gamma` in
    ``teacher = gamma * teacher + (1 - gamma) * student``.
    Frozen detector weights and buffers are deliberately untouched.
    """

    def __init__(self,
                 retention: float = 0.999,
                 interval: int = 1,
                 adaptive_keywords=('prompt_embeddings', 'tunable_linear')):
        if not 0.0 < retention < 1.0:
            raise ValueError('retention must be in (0, 1)')
        self.retention = float(retention)
        self.interval = int(interval)
        self.adaptive_keywords = tuple(adaptive_keywords)
        self.last_update_seconds = 0.0

    @staticmethod
    def _unwrap(runner: Runner) -> nn.Module:
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    def _is_adaptive(self, name: str) -> bool:
        return any(key in name for key in self.adaptive_keywords)

    def before_train(self, runner: Runner) -> None:
        model = self._unwrap(runner)
        if not hasattr(model, 'teacher') or not hasattr(model, 'student'):
            raise AssertionError('PaperPromptMeanTeacherHook needs teacher/student')
        if runner.iter == 0:
            self._update(model, retention=0.0)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if (runner.iter + 1) % self.interval == 0:
            start = torch.cuda.Event(enable_timing=True) \
                if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) \
                if torch.cuda.is_available() else None
            if start is not None:
                start.record()
            self._update(self._unwrap(runner), self.retention)
            if end is not None:
                end.record()
                end.synchronize()
                self.last_update_seconds = start.elapsed_time(end) / 1000.0

    def _update(self, model: nn.Module, retention: float) -> None:
        student = dict(model.student.named_parameters())
        teacher = dict(model.teacher.named_parameters())
        adaptive_names = sorted(
            name for name in student if self._is_adaptive(name))
        if not adaptive_names:
            raise AssertionError('No adaptive prompt parameters found for EMA')
        for name in adaptive_names:
            if name not in teacher:
                raise AssertionError(
                    f'Teacher is missing adaptive parameter {name}')
            teacher[name].data.mul_(retention).add_(
                student[name].data, alpha=1.0 - retention)

