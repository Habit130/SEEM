# Linux 4090 Custom Binary Segmentation

## Scope

This repository is adapted to train and evaluate SEEM Focal-T v1 on the sibling `../dataset` folder using:

- `train.json` for training only
- `test.json` for post-training testing only
- binary masks defined by `mask > 0`
- caption text from `caption[2]` by default

No intermediate validation is run during training.

## Environment

Use the one-shot setup script:

```bash
bash assets/scripts/setup_linux_4090.sh
```

If you prefer manual setup, the exact sequence is:

```bash
conda env create -f environment.linux.4090.yml
conda activate seem-4090
python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel packaging
python -m pip install -r assets/requirements/requirements.linux.4090.txt
python -m pip install --no-build-isolation git+https://github.com/MaureenZOU/detectron2-xyz.git
python -m pip install git+https://github.com/openai/whisper.git
cd modeling/vision/encoder/ops && python setup.py build install
cd /path/to/SEEM
```

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
