# Linux 4090 Custom Binary Segmentation

## Scope

This repository is finalized for remote Linux execution on a single RTX 4090 using the sibling `../dataset` folder and the existing `entry.py train/evaluate` workflow.

The supported custom task is binary lesion segmentation with:

- `train.json` for training only
- `test.json` for post-training evaluation only
- binary masks defined by `mask > 0`
- caption text from `caption[2]`

No intermediate validation is run during training.

## Server prerequisites

Provide these outside the repository:

- Linux server with one RTX 4090
- NVIDIA driver and CUDA toolkit with `nvcc` in `PATH`
- `conda`
- outbound network access for dependency installation and init weight download
- sibling `../dataset` directory next to the cloned `SEEM` repository

## Dataset contract

The repository keeps the existing hard-coded sibling dataset path. After cloning, the layout must look like:

```text
parent/
  dataset/
    train.json
    test.json
    train/
    test/
  SEEM/
```

Each JSON sample must contain:

- `image`: relative image path under `../dataset`
- `mask`: relative mask path under `../dataset`
- `caption`: a list with at least three text entries

## Environment

Use the one-shot Linux setup helper:

```bash
bash assets/scripts/setup_linux_4090.sh
```

This script recreates the `seem-4090` Conda environment, installs Python dependencies, installs Detectron2 and Whisper, builds the deformable attention extension, and verifies the final imports needed by the server workflow.

## Initialization weight

Prepare the required SEEM initialization checkpoint into the standard non-Git location:

```bash
bash assets/scripts/prepare_custom_binary_assets.sh
```

The helper downloads `weights/xdecoder_focalt_last.pt` if it is not already present. If automatic download is unavailable, place the same file at that path manually before training.

## Training command

```bash
python entry.py train \
  --conf_files configs/seem/focalt_unicl_lang_v1.yaml configs/seem/focalt_custom_binary_lang_v1.yaml \
  --overrides \
  WEIGHT True \
  RESUME_FROM weights/xdecoder_focalt_last.pt
```

Training outputs are written under `outputs/custom_binary_seem/`.

## Post-training evaluation command

Use the checkpoint saved under `outputs/custom_binary_seem/.../<step>/default/model_state_dict.pt`.

```bash
python entry.py evaluate \
  --conf_files configs/seem/focalt_unicl_lang_v1.yaml configs/seem/focalt_custom_binary_lang_v1.yaml \
  --overrides \
  WEIGHT True \
  RESUME_FROM /path/to/outputs/custom_binary_seem/<run>/<step>/default/model_state_dict.pt
```

## Output

The evaluation step writes `binary_seg_metrics.json` into the evaluation output directory and reports only:

- `iou`
- `dice`
- `recall`
- `miou`
- `macc`

## Final server-side operating surface

After transfer to the Linux server, the supported flow is:

1. Place `dataset/` next to the repository.
2. Run `bash assets/scripts/setup_linux_4090.sh`.
3. Run `bash assets/scripts/prepare_custom_binary_assets.sh`.
4. Start training with `entry.py train`.
5. Evaluate the produced `model_state_dict.pt` with `entry.py evaluate`.
