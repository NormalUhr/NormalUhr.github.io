---
layout:     post
title:      "Paper Notes - Intriguing properties of neural networks(ICLR 2014)"
subtitle:   "Adversarial Machine Learning"
date:       2021-01-02
author:     "Felix Zhang"
header-img: "img/in-post/2021-01-02-Intriguing/bg.JPG"
catalog: true
tags:
   - Deep Learning
   - Adversarial Machine Learning
   - Paper Notes
---

这篇文章被视作Adversarial Machine Learning的开山之作。这篇文章主要揭露了神经网络两个“不符合直觉”的性质：

1. > It suggests that it is the space, rather than the individual units, that contains the semantic information in the high layers of neural networks.

   神经网络中携带语义信息的并不是高层神经层的某（几）个神经元，而是整个网络或者整个一层的神经元。

2. > We can cause the network to misclassify an image by applying a certain hardly perceptible perturbation, which is found by maximizing the network’s prediction error

   神经网络的输入输出映射是相当不连续的，这种不连续性导致当输入被加入一些肉眼不可见的干扰攻击后，神经网络得到完全不一致的分类结果。更重要的是，这种攻击效果并不是完全随机的，同样的攻击应用到其他网络上，也有可能造成网络失效，即使这个网络使用的是不同的训练集、网络结构也完全不同。

## 引言

对于监督学习而言，网络的结果完全是由梯度的反向传播决定的，这造成了人们去直观理解神经网络时很困难。上述的第一个特性其实是围绕“单个神经元的语义信息”展开的。