# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class LossWeightRampHook(Hook):
    """Linearly ramp a loss module's ``loss_weight`` in place over training.

    Useful for losses that are unstable early on, when predictions are far
    from GT and the loss's gradient magnitude scales with that gap (e.g.
    the Kervadec surface/boundary loss, whose gradient scales with the GT
    signed-distance map — huge for an untrained model). Mirrors the
    "alpha schedule" from Kervadec et al., "Boundary loss for highly
    unbalanced segmentation", which anneals the boundary-loss weight up
    (and the region-loss weight down) over the course of training instead
    of using a fixed compound-loss weight from iteration 0.

    Args:
        module_path (str): Dotted attribute path from the (unwrapped)
            model to the loss module whose ``loss_weight`` should be
            ramped, e.g. ``'panoptic_head.loss_surface'``.
        start_weight (float): ``loss_weight`` at/before ``begin_iter``.
        end_weight (float): ``loss_weight`` at/after ``end_iter``.
        begin_iter (int): Iteration (0-indexed) to start ramping from.
        end_iter (int): Iteration at which ``end_weight`` is reached and
            held for the remainder of training.
    """

    def __init__(self,
                 module_path: str,
                 start_weight: float,
                 end_weight: float,
                 begin_iter: int = 0,
                 end_iter: int = 10000) -> None:
        self.module_path = module_path
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.begin_iter = begin_iter
        self.end_iter = max(end_iter, begin_iter + 1)

    def _get_loss_module(self, runner: Runner):
        model = runner.model
        model = getattr(model, 'module', model)  # unwrap DDP/FSDP if needed
        obj = model
        for attr in self.module_path.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _current_weight(self, it: int) -> float:
        if it <= self.begin_iter:
            return self.start_weight
        if it >= self.end_iter:
            return self.end_weight
        ratio = (it - self.begin_iter) / (self.end_iter - self.begin_iter)
        return self.start_weight + ratio * (self.end_weight - self.start_weight)

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch: Optional[dict] = None) -> None:
        weight = self._current_weight(runner.iter)
        module = self._get_loss_module(runner)
        module.loss_weight = weight

    def before_train(self, runner: Runner) -> None:
        # Make sure resumed runs start at the schedule-correct weight
        # rather than whatever value was pickled into the checkpoint.
        self.before_train_iter(runner, batch_idx=0)
