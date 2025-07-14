# PromptSTEM: Attentional Deep Learning Accelerates Quantification of Heterogeneous Catalysts from Electron Microscopy
This codebase provides a generalizable method for automated image analysis of supported nanocalysts in transmission electron microscopy, including single-atom catalysts, sub-nano clusters, and nanoparticles.

##  Installation
Prerequisites
- PyTorch
- OpenCV
  
Quick command
```bash
pip install -r requirements.txt
```
Model Checkpoint
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


## Usage

- **1. Train and predict segmentation models:**
```bash
python train.py
```
## Data
The datasets used in this study are all publicly available:
- `HAADF-STEM of PtSn@Al2O3` `BF-TEM of Au@ZSM5`: [EMcopilot.](https://zenodo.org/records/14994375)
- `HAADF-STEM of Pt@NC`: [AtomDetection_ACSTEM.](https://github.com/HPAI-BSC/AtomDetection_ACSTEM)
- `BF-TEM of Pd@C `: [nNPipe.](https://doi.org/10.5281/zenodo.7024893)
  
## Citation
If you find our code or data useful in your research, please cite our paper:
```
@misc{yuan2025FASTCat,
      title={Deep Learning Enabled Single-Shot STEM Imaging for Ultra-Fast Identification of Supported Catalysts}, 
      author={Wenhao Yuan and Fengqi You},
      year={2025},
}
```
