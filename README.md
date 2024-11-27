# WiFlexFormer: Efficient WiFi-Based Person-Centric Sensing

### Paper
**Strohmayer, J., Wödlinger, M., and Kampel, M.** (2024). WiFlexFormer: Efficient WiFi-Based Person-Centric Sensing. arXiv. doi: https://doi.org/10.48550/arXiv.2411.04224.

BibTeX:
```
@misc{strohmayer2024wiflexformer,
      title={WiFlexFormer: Efficient WiFi-Based Person-Centric Sensing}, 
      author={Julian Strohmayer and Matthias Wödlinger and Martin Kampel},
      year={2024},
      eprint={2411.04224},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04224}, 
}
```

### WiFlexFormer Architecture
<img src="resources/wiflexformer.svg" alt="WiFlexFormer Architecture" width="300" height="400">

### Prerequisites
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset
Get the `3DO dataset` from https://zenodo.org/uploads/10925351 and put it in the `/data` directory.

### Training & Testing 

**Training** | Example command for training three *WiFlexFormer* models (`wff_1`, `wff_2`, and `wff_3`) on day 1 data of the `3DO dataset` for 10 epochs using a window size of 351 WiFi packets and a batch size of 32:

```
python3 train.py --data data/3DO --name wff --num 3 --epochs 10 --ws 351 --bs 32 --device 0
```

Model checkpoints for the lowest validation loss and highest validation F1-Score are stored in the corresponding run directories `runs/wff_*`.

**Testing** | Example command for testing the trained models:

```
python3 test.py --data data/3DO --name wff --ws 351 --bs 128 --device 0
```
