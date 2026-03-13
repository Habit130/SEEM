# Linux 4090 Custom Binary Segmentation

## Scope

This repository is adapted to train and evaluate SEEM Focal-T v1 on the sibling `../dataset` folder using:

- `train.json` for training only
- `test.json` for post-training testing only
- binary masks defined by `mask > 0`
- caption text from `caption[2]` by default

No intermediate validation is run during training.

## Environment

Create the Linux environment from `environment.linux.4090.yml`.

After the environment is active, compile the deformable attention operator from `modeling/vision/encoder/ops`.

## Required external asset

Training still expects an external pretrained weight file. For the Focal-T SEEM baseline, use the original X-Decoder Focal-T checkpoint as the initialization weight.

## Training command

```bash
python entry.py train \
  --conf_files configs/seem/focalt_unicl_lang_v1.yaml configs/seem/focalt_custom_binary_lang_v1.yaml \
  --overrides \
  WEIGHT True \
  RESUME_FROM /path/to/xdecoder_focalt_last.pt
```

## Post-training test command

Use the checkpoint saved under `SAVE_DIR/.../<step>/default/model_state_dict.pt`.

```bash
python entry.py evaluate \
  --conf_files configs/seem/focalt_unicl_lang_v1.yaml configs/seem/focalt_custom_binary_lang_v1.yaml \
  --overrides \
  WEIGHT True \
  RESUME_FROM /path/to/output_run/<step>/default/model_state_dict.pt
```

## Output

The test step writes `binary_seg_metrics.json` into the evaluation output directory and reports only:

- `iou`
- `dice`
- `recall`
- `miou`
- `macc`
