Warning: The full data set is 22Gb and is not checked into github. Please use the download script to download data sets from source provider.

### Overview

Despite the considerable progress in automatic abdominal multi-organ segmentation from CT/MRI scans in recent years, a comprehensive evaluation of the models' capabilities is hampered by the lack of a large-scale benchmark from diverse clinical scenarios. Constraint by the high cost of collecting and labeling 3D medical data, most of the deep learning models to date are driven by datasets with a limited number of organs of interest or samples, which still limits the power of modern deep models and makes it difficult to provide a fully comprehensive and fair estimate of various methods. To mitigate the limitations, we present AMOS, a large-scale, diverse, clinical dataset for abdominal organ segmentation. <u>AMOS provides 500 CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality, multi-phase, multi-disease patients, each with voxel-level annotations of 15 abdominal organs, providing challenging examples and test-bed for studying robust segmentation algorithms</u> under diverse targets and scenarios. We further benchmark several state-of-the-art medical segmentation models to evaluate the status of the existing methods on this new challenging dataset. We have made our datasets, benchmark servers, and baselines publicly available, and hope to inspire future research.  For more details, please refer to our paper "https://arxiv.org/pdf/2206.08023.pdf" as well as homepage "https://jiyuanfeng.github.io/AMOS/".

### Structure

AMOS provides the following content. imagesTr and labelsTr provide 240 scans (200 CT and 40 MRI), imagesVa and labelsVa provide 120 scans for model selection (100 CT and 20 MRI), and imagesTs provide 120 test data (please submit your predictions from https://amos22.grand-challenge.org/evaluation/challenge/submissions to get a score). Please note that id numbers less than 500 belong to CT data, otherwise they belong to MRI data.

```bash
amos
│   readme.md
│   dataset.json
└───imagesTr
│   │   amos_xxxx.nii.gz
│   │   ...
└───imagesVa
└───imagesTs
└───labelsTr
└───labelsVa
└───labelsTs

```

### Citation

if you found this dataset useful for your research, please cite:

```
@article{ji2022amos,
  title={AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation},
  author={Ji, Yuanfeng and Bai, Haotian and Yang, Jie and Ge, Chongjian and Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhang, Lingyan and Ma, Wanling and Wan, Xiang and others},
  journal={arXiv preprint arXiv:2206.08023},
  year={2022}
  }
```

### Upcoming

We will publish more meta information and corresponding APIs in October, while more unlabeled data will be used to support more learning scenarios