
# Annotation of sleep depth index (SDI) by deep learning
An implementation for the paper: ["Annotation of Sleep Depth Index with Scalable Deep Learning Yields Novel Digital Biomarkers for Sleep Health"](https://arxiv.org/abs/2407.04753).


# Requirements
- Install the dependencies by:

```bash
conda create -n sleep python=3.11
pip install -r requirements.txt
```

# Usage

You can modify the training configs in `src/configs`, and run model training by 
```bash
python train.py --config ../configs/config.ini
```

After training, you can try the inference by running
```bash
python infer.py --data_file YOUR_DATA(EDF) --output_file NAMED_FILE.csv 
```
The resulting CSV file represents data where each row corresponds to a 30-second interval. The first column contains the Sleep Depth Index, while the second column indicates the classification of REM sleep.

A web application for annotating Sleep Depth Index is now available [here](http://183.162.233.24:10024/PSG_Sleep_depth). Currently, it supports only CSV files transformed from raw signal data. We are actively working to enable support for loading .edf files.

# Citation

If you find the idea useful or use this code in your own work, please cite our paper
```bibtex
@article{zhou2024annotation,
  title={Annotation of Sleep Depth Index with Scalable Deep Learning Yields Novel Digital Biomarkers for Sleep Health},
  author={Zhou, Songchi and Song, Ge and Sun, Haoqi and Leng, Yue and Westover, M Brandon and Hong, Shenda},
  journal={arXiv preprint arXiv:2407.04753},
  year={2024}
}
```
