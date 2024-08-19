# WSMCL
## Title: Weakly Supervised Multi-modal Contrastive Learning Framework for Predicting the HER2 Scores in Breast Cancer


## Authers:
Jun Shi, Dongdong Sun, Zhiguo Jiang, Jun Du, Wei Wang,  Yushan Zheng and Haibo Wu.


Haibo Wu and [Yushan Zheng](https://zhengyushan.github.io/) are the corresponding authors.


E-mail: wuhaibo@ustc.edu.cn and yszheng@buaa.edu.cn

## Framework:
![framework](images/framework.jpg)

## Installation:
### feature format:
```none
-feature_dir
  -slide-1_feature.pth
  -slide-1_coordinates.pth
  -slide-2_feature.pth
  -slide-2_coordinates.pth
  ......
  -slide-n_feature.pth
  -slide-n_coordinates.pth
 
 xxx_feature.pth -> shape: number_patches, feaure_dim
 xxx_coordinates.pth -> shape: number_patches, 2(row, col)
```

### Environment:
1. python_version: 3.8
2. install 3rd library:
```shell
pip install -r requirements.txt
```


## 4. Run:
```shell
python main.py
```