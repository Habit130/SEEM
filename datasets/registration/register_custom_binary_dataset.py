import json
import os

from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_CUSTOM_BINARY = {
    "custom_binary_train": "train.json",
    "custom_binary_test": "test.json",
}


def get_metadata():
    return {
        "thing_classes": ["foreground"],
        "thing_colors": [[255, 0, 0]],
        "thing_dataset_id_to_contiguous_id": {1: 0},
        "stuff_classes": ["foreground"],
        "stuff_colors": [[255, 0, 0]],
        "stuff_dataset_id_to_contiguous_id": {1: 0},
    }


def load_custom_binary_json(root, json_file):
    with PathManager.open(json_file) as f:
        samples = json.load(f)

    dataset_dicts = []
    for idx, sample in enumerate(samples):
        image_path = os.path.normpath(os.path.join(root, sample["image"]))
        mask_path = os.path.normpath(os.path.join(root, sample["mask"]))

        with Image.open(image_path) as image:
            width, height = image.size

        record = {
            "file_name": image_path,
            "mask_file_name": mask_path,
            "image_id": sample.get("id", idx),
            "id": sample.get("id", idx),
            "height": height,
            "width": width,
            "captions": sample["caption"],
        }
        dataset_dicts.append(record)

    assert dataset_dicts, f"No samples found in {json_file}"
    assert PathManager.isfile(dataset_dicts[0]["file_name"]), dataset_dicts[0]["file_name"]
    assert PathManager.isfile(dataset_dicts[0]["mask_file_name"]), dataset_dicts[0]["mask_file_name"]
    return dataset_dicts


def register_custom_binary_dataset(name, metadata, root, json_name):
    json_file = os.path.join(root, json_name)
    DatasetCatalog.register(name, lambda: load_custom_binary_json(root, json_file))
    MetadataCatalog.get(name).set(
        root=root,
        json_file=json_file,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        binary_seg_task=True,
        **metadata,
    )


def register_all_custom_binary(root):
    metadata = get_metadata()
    for name, json_name in _PREDEFINED_SPLITS_CUSTOM_BINARY.items():
        register_custom_binary_dataset(name, metadata, root, json_name)


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"))
register_all_custom_binary(_root)
