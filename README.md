# Differentiable Digital Signal Processing

![ddsp_tilde_screenshot](patchs/screenshot_bitwig.png)

Implementation of the [DDSP model](https://github.com/magenta/ddsp) using PyTorch. This implementation can be exported to a torchscript model, ready to be used inside a realtime environment (see [this video](https://www.youtube.com/watch?v=_U6Bn-1FDHc)).

This implementation is specially for generating cat voices!




## Usage

Edit the `config.yaml` file to fit your needs (audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using 

```bash
python preprocess.py
```

You can then train your model using 

```bash
python train.py --name mytraining --steps 10000000 --batch 16 --lr .001
```

Once trained, export it using

```bash
python export.py --run runs/mytraining/
```

It will produce a file named `ddsp_pretrained_mytraining.ts`, that you can use inside a python environment like that

```python
import torch

model = torch.jit.load("ddsp_pretrained_mytraining.ts")

pitch = torch.randn(1, 200, 1)
loudness = torch.randn(1, 200, 1)

audio = model(pitch, loudness)
```


## Data Augmentation

cat voice segmentation:
(Please change paths and parameters inside the code)
```
python seg_cat.py
```

Pitch shifting:
```
git clone https://github.com/ederwander/PyAutoTune.git
cd PyAutoTune
pip install -e .
cd ../ddsp_pytorch
python TuneAndSaveToFile_folder.py data/cats/train/
```

## Inference Example

file inference:
```
python inference_file.py --model_path export/ddsp_cats_seg_pitchshift_pretrained.ts --input_file violin_original.mp3
```

folder inference (for evaluation only, please change paths and parameters inside the code):
```
python inference_folder.py
```

## Evaluation Example
```
python evaluate_f0.py [output folder] [reference folder]
```

## Colab Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jerryuhoo/ddsp_pytorch/blob/master/ddsp_cat_example.ipynb)
