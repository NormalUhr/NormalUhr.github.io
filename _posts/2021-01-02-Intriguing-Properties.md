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



对于监督学习而言，网络的结果完全是由梯度的反向传播决定的，这造成了人们去直观理解神经网络时很困难。以上两个特性表明，通过反向传播学习得到的神经网络具有盲点(blind spot)，并且这些盲点是和数据分布相关的，通常不容易发现也不直观。接下来分开讨论上述神经网络的第一个和第二个性质。

一、某一层而不是单个神经元承载着语义信息

这一个问题对应着第三节‘Units of $\phi(x)$’，展示了**语义信息其实并不是隐藏在某个单独的神经元中，而是该隐藏层或整个网络携带着语义信息。**

这句话的意思其实可以直观地解释为：假如有一层隐藏层网络有四个神经元，以前人们的想法是，可能这四个神经元分别对应着“圆形”、“矩形”、“三角形”和“五边形”，然后人们试图去确认到底这些神经元各自对应的语义信息是什么含义，但是经过作者测试后发现，实际上是代表着“形状”的语义信息整体隐藏在这一层网络中，而不是某个具体的神经元代表着一种语义信息。

在论文中作者将某一层网络视作由特征构成的空间（feature space），同时每个神经元各自都代表着这个特征空间的自然基（natural basis），同时这些自然基还可以线性组合为随机基（random basis）。首先我们对自然基（即单个神经元）进行分析，我们把所有让某个自然基（神经元，即$e_i$，$i$是神经元编号）的激活值达到最大的图片给挑出来，对比可以得到这些图片共同的特点——_Figure 1、3_所示——这些共同的图片都一定程度上表现出了相同的语义，这一点其实是说明了单个神经元是可以表现出一定语义信息的。那么同时，我们用同样的方法找出让所有随机基（这一层数个神经元的线性组合，即$v$）产生最大激活值的图片，同样可以发现这些图片都包含着一定的语义信息——_Figure 2、4_所示——这一现象就推翻了语义信息只包含在单个神经元之中这个结论。

借用上边的例子，如果语义信息只存在于单个神经元当中，那么我们把代表着“圆形”($e_1$)和“方形”($e_2$)的神经元做线性组合($v=e_1+e_2$)后，应该是没有意义的，但是实验却证明了所有让这个线性组合激活值最大的图片表现出了“椭圆形”这一语义，因而只有可能**更抽象的语义存在这整个一层网络中，单独的神经元代表的语义只是这个抽象语义的一种特化**。

二、微小的攻击可以使分类网络产生错误输出，攻击还对网络超参数、不同的训练集都具有一定的鲁棒性

对于大多数计算机视觉任务，算法都应具有“局部泛化特性”。即在输入上的一定范围内的微小偏差并不会改变分类输出结果，但是事实上绝大多数方法都不具有上述特性。事实上通过一些简单的优化算法即可找到成功攻击的样例。