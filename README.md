## Paper title: Weakly Supervised Multi-modal Contrastive Learning Framework for Predicting the HER2 Scores in Breast Cancer


### Authers
<strong>Jun Shi, Dongdong Sun, Zhiguo Jiang, Jun Du, Wei Wang, Yushan Zheng and Haibo Wu.</strong>

[Haibo Wu](https://lcyx.ustc.edu.cn/2023/0615/c34245a605986/page.htm) and [Yushan Zheng](https://zhengyushan.github.io/) are the corresponding authors.

<strong>E-mail</strong>: <font color='blue'>wuhaibo@ustc.edu.cn</font> and <font color='blue'>yszheng@buaa.edu.cn</font>

### Overview
![framework](images/framework.jpg)

### Environment
![python](https://img.shields.io/badge/python-3.8-blue)
![torch](https://img.shields.io/badge/torch-1.8%2Bcu111-red)
![torchvision](https://img.shields.io/badge/torchvision-0.9.1+cu111-purple)
![numpy](https://img.shields.io/badge/numpy-1.22.2-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange)
![opencv-python](https://img.shields.io/badge/opencv--python-4.5.5.62-pink)
![einops](https://img.shields.io/badge/einops-0.6.6-brown)

#### install 3rd library
```shell
pip install -r requirements.txt
```

### Feature format
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

### Run
```shell
python main.py
```


### Citation
....