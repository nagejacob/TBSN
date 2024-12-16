# Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising [Paper](https://arxiv.org/abs/2404.07846)


## Usage
### Datasets
Download [SIDD](https://abdokamel.github.io/sidd/) and [DND](https://noise.visinf.tu-darmstadt.de/) datasets, and modify `dataset_path` in `dataset/base_function.py` accordingly.
```
|- dataset_path
  |- SIDD
    |- SIDD_Medium_Srgb
      |- Data
        |- 0001_001_S6_00100_00060_3200_L
        |- 0002_001_S6_00100_00020_3200_N
        |- ...
    |- SIDD_Validation
      |- ValidationNoisyBlocksSrgb.mat
      |- ValidationGtBlocksSrgb.mat
    |- SIDD_Benchmark
      |- BenchmarkNoisyBlocksSrgb.mat
  |- DND
    |- info.mat
    |- images_srgb
```

### Validation
Validate on SIDD Validation dataset,
```
cd validate
python base.py --config_file "../option/tbsn_sidd.json"
```

### Training (Would be released once the paper is accepted)
Training on SIDD Medium dataset,
```
sh train.sh
```

## Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{li2025rethinking,
  title={Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising},
  author={Li, Junyi and Zhang, Zhilu and Zuo, Wangmeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
