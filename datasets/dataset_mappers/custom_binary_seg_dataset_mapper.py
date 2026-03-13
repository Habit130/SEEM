import copy

import numpy as np
import torch
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

from modeling.utils import configurable

from ..visual_sampler.sampler import build_shape_sampler


def build_train_transform_gen(cfg):
    cfg_input = cfg["INPUT"]
    image_size = cfg_input["IMAGE_SIZE"]
    min_scale = cfg_input["MIN_SCALE"]
    max_scale = cfg_input["MAX_SCALE"]

    augmentation = []
    if cfg_input["RANDOM_FLIP"] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input["RANDOM_FLIP"] == "horizontal",
                vertical=cfg_input["RANDOM_FLIP"] == "vertical",
            )
        )

    augmentation.extend(
        [
            T.ResizeScale(
                min_scale=min_scale,
                max_scale=max_scale,
                target_height=image_size,
                target_width=image_size,
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ]
    )
    return augmentation


def build_eval_transform_gen(cfg):
    return [T.ResizeShortestEdge(cfg["INPUT"]["MIN_SIZE_TEST"], max_size=cfg["INPUT"]["MAX_SIZE_TEST"])]


class CustomBinarySegDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        caption_index,
        shape_sampler=None,
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.image_format = image_format
        self.caption_index = caption_index
        self.shape_sampler = shape_sampler

    @classmethod
    def from_config(cls, cfg, is_train=True):
        return {
            "is_train": is_train,
            "tfm_gens": build_train_transform_gen(cfg) if is_train else build_eval_transform_gen(cfg),
            "image_format": cfg["INPUT"].get("FORMAT", "RGB"),
            "caption_index": cfg["CUSTOM_DATASET"]["CAPTION_INDEX"],
            "shape_sampler": build_shape_sampler(cfg) if is_train else None,
        }

    def _select_caption(self, captions):
        if not captions:
            raise ValueError("Expected at least one caption for each sample.")
        if self.caption_index >= len(captions):
            raise IndexError(
                f"Configured caption index {self.caption_index} is out of range for captions of length {len(captions)}."
            )
        return captions[self.caption_index].strip().lower()

    def _load_mask(self, mask_file_name):
        with Image.open(mask_file_name) as mask_image:
            mask = np.asarray(mask_image)
        return (mask > 0).astype(np.uint8)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        mask = self._load_mask(dataset_dict["mask_file_name"])
        caption = self._select_caption(dataset_dict["captions"])

        if self.is_train:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            transformed_mask = transforms.apply_segmentation(mask[:, :, None])[:, :, 0] > 0
            image_shape = image.shape[:2]

            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            instances = Instances(image_shape)
            if transformed_mask.any():
                mask_tensor = torch.from_numpy(np.ascontiguousarray(transformed_mask[None, :, :].copy()))
                bit_masks = BitMasks(mask_tensor)
                instances.gt_masks = bit_masks
                instances.gt_boxes = bit_masks.get_bounding_boxes()
                instances.gt_classes = torch.zeros((1,), dtype=torch.int64)
                instances.is_things = torch.ones((1,), dtype=torch.int64)
            else:
                h, w = transformed_mask.shape
                empty_masks = BitMasks(torch.zeros((0, h, w), dtype=torch.bool))
                instances.gt_masks = empty_masks
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
                instances.is_things = torch.zeros((0,), dtype=torch.int64)

            dataset_dict["instances"] = instances
            dataset_dict["groundings"] = {
                "masks": instances.gt_masks.tensor.clone(),
                "texts": [caption],
                "hash": [hash(caption)],
                "mode": "text",
            }
            dataset_dict["spatial_query"] = self.shape_sampler(instances)
            return dataset_dict

        image, _ = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["groundings"] = {
            "masks": torch.from_numpy(mask[None, :, :].astype(np.uint8)).bool(),
            "texts": [[caption]],
        }
        return dataset_dict
