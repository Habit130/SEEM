import json
import logging
import os

import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator


class BinarySegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None, threshold=0.5, distributed=True):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._threshold = threshold
        self._distributed = distributed

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def process(self, inputs, outputs):
        for input_item, output_item in zip(inputs, outputs):
            pred_mask = (output_item["grounding_mask"].sigmoid() > self._threshold).bool()
            gt_mask = input_item["groundings"]["masks"].bool()

            if pred_mask.ndim == 2:
                pred_mask = pred_mask.unsqueeze(0)
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)

            self.tp += int((pred_mask & gt_mask).sum().item())
            self.fp += int((pred_mask & (~gt_mask)).sum().item())
            self.fn += int(((~pred_mask) & gt_mask).sum().item())
            self.tn += int(((~pred_mask) & (~gt_mask)).sum().item())

    @staticmethod
    def _safe_div(numerator, denominator):
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def evaluate(self):
        if self._distributed:
            synchronize()
            self.tp = sum(all_gather(self.tp))
            self.fp = sum(all_gather(self.fp))
            self.fn = sum(all_gather(self.fn))
            self.tn = sum(all_gather(self.tn))
            if not is_main_process():
                return

        iou = self._safe_div(self.tp, self.tp + self.fp + self.fn)
        dice = self._safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        recall = self._safe_div(self.tp, self.tp + self.fn)
        iou_bg = self._safe_div(self.tn, self.tn + self.fn + self.fp)
        miou = (iou + iou_bg) / 2.0
        acc_fg = self._safe_div(self.tp, self.tp + self.fn)
        acc_bg = self._safe_div(self.tn, self.tn + self.fp)
        macc = (acc_fg + acc_bg) / 2.0

        results = {
            "iou": iou,
            "dice": dice,
            "recall": recall,
            "miou": miou,
            "macc": macc,
        }

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            metrics_path = os.path.join(self._output_dir, "binary_seg_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        self._logger.info(results)
        return results
