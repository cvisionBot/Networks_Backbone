# Networks Backbone
PyTorch 기반 다양한 Network를 논문을 참고하여 구현하는 프로젝트입니다.

## Implementations
- Netowrk Backbone
- Network Backbone Customizing
- Network Block Modulize

## 프로젝트 구조

```
backbones_factory
├─ .gitignore
├─ __README.md
├─ models # Networks 구현
│  ├─ MobileNetv1
│  ├─ ResNet
│  ├─ ResNeXt
│  └─ layers
├─ __init__.py
├─ initialize.py

```

## Requirements
`PyTorch` >= 1.10.1


## Reference
Networks
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - [Torchvision Github](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [ResNext: Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
  - [official Github](https://github.com/facebookresearch/ResNeXt)
- [MobileNetv1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
  - [Torchvision Github](https://github.com/osmr/imgclsmob/blob/956b4ebab0bbf98de4e1548287df5197a3c7154e/pytorch/pytorchcv/models/mobilenet.py)
- [DenseNet: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [Official Github](https://github.com/liuzhuang13/DenseNet)
- [MobileNetv2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
  - [Torchvision Github](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/mobilenet.py)
- [SENet: Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
  - [Torchvision Github](https://github.com/osmr/imgclsmob/blob/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models/senet.py)
