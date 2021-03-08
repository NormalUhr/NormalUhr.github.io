---
layout:     post
title:      "Adversarial ML——Robustness and Unvertainty"
subtitle:   "Paper Summary"
date:       2021-03-03
author:     "Felix Zhang"
header-img: "img/in-post/2021-03-03-robustness-uncertainty/bg.JPG"
catalog: true
tags:
   - Adversarial Machine Learning
   - Paper Summary
   - Uncertainty
---

# Natural Robustness and Uncertainty in Machine Learning

在这篇文章中我们主要关心一个词——uncertainty。首先什么是深度神经网络的uncertainty？

## Background

在大多数训练过程中，我们训练集和数据集的数据分布都是一致的，因而在训练结束后，模型在测试集上正确率是很高的。那么遇到数据分布不一致的情况，模型的准确性就会大打折扣。这里所说的数据分布不一致可能出现在图像的方方面面，例如：

* 图像纹理不同，训练集的图片全部都是油画或素描，测试集如果用的是照片。
* 对象风格不同，训练集的街道全部是欧式建筑，但测试集的街道是中式建筑。
* 照片的时间不同：训练集的照片都是在早上/光线充足的照片，测试集的照片却包含了傍晚黄昏时光线柔和昏暗的照片。

总之，各种各样的原因导致了训练集和测试集数据分布之间的差异，我们称之为**distribution shift**，会使模型的识别结果大打折扣，这就是深度神经网络的uncertainty。从术语上，我们将和训练集分布一致的测试集成为in-distribution(IID) test set，与之相对的和训练集分布有较大差异的测试集称为out-of-distribution(OOD) test set，在OOD test set上模型维持高准确率的能力我们称为**OOD robustness**。同时如果OOD test set的数据分布差异是**自然产生**的而不是人为制造的，那么这种OOD robustness也叫**natural robustness**。

为了解决这一问题，各种文章提出了自己不同的看法，这篇文章将主要跟OOD robustness相关的七个假设汇总，并做了细致的测试，同时提出了三个全新的dataset和一种data augmentation的方式来让自己的测试更具说服力。这七个假设中有四个是关于如何提高OOD robustness的措施（methods），另外三个揭示了和OOD robustness相关的更抽象的性质（properties）。其中前四个为：

1. Larger Models：更大、更复杂的模型能够提高模型的OOD robustness。
2. Self-Attention：给模型增加self-attention layers能够增加模型的OOD robustness。
3. Diverse Data Augmentation：给训练数据施加更加diverse的data augmentation能够改善模型的OOD robustness。
4. Pretraining：在更大、更多样化的数据集上预训练，能够改善后期模型的OOD robustness。

后三个（properties）：

5. Texture Bias：深度神经网络在学习时总会记住训练数据的纹理特征，从而使模型的泛化能力下降，损害OOD robustness。

6. Only IID Accuracy Matters：测试集中的独立同分布的数据，唯一决定了该模型的natural robustness。

7. Synthetic does not lead to Natural：我们给模型施加的人为干预的data augmentation并不会对模型的natural robustness有很大帮助。

> 这篇文章讲的是，尽管我们做了很多努力提高模型的robustness，比如抵抗adversarial example。但这个时候作者提出了一个新的观点，就是这种robustness被称为synthetic robustness intervention，经过大量测试，作者发现无论是什么模型，在面对distribution shifts的时候，它的robustness都会下降，就是所谓的natural robustness。结果只有一种情况作者发现有所改善，那就是在训练集本身就包含各种各样的distribution的时候，而人为做data augmentation也不会有很大作用。

## New Benchmarks

为了测试以上七个假设，作者提出了三个新的数据集，之前的数据集为什么不能用，有什么问题呢？我们以ImageNet为例，ImageNet在创立数据集时就刻意将texture造成的差异排除在外，那么任何通过人为手段（data augmentation）施加上的texture就会违背natural distribution shift的设定，所测得的必然也不是natural robustness。同时，已有的数据集也没有diverse到能够测试以上所有假设。因此作者认为有必要提出新的数据集供测试之用。简而言之，这三个数据集的提出，都是为了测试不同场景下的natural distribution shift，它们分别体现在image texture，image capturing process以及camera operations。

### ImageNet-Redition

ImageNet-R和原始的ImageNet拥有相同的class label，只是在数据上ImageNet-R给数据加上了各种各样的texture。需要注意的是，这些texture不是人为加上去的，而是对现实世界中存在的texture拍成的照片，比如：剪纸、油画、雕塑、塑料制品、刺绣等。因此从数据特征角度上，这些没有人为改动过的天然texture能够测试出natural robustness。训练集包含30k张图片对应着200个类别。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-03-03-robustness-uncertainty/1.png?raw=true)

### StreetView StoreFronts(SVSF)

这个数据集是为了模拟在**照片采集过程中**可能出现的各种天然的distribution shift。大部分的CV算法其实都是对照片的**采集设备、时间以及照片的地理环境**非常敏感的，这个数据集就是为了测试算法对以上三个方面的natural robustness，统称为image capture process。这个dataset还没公开，分为一个训练集和五个测试集，其中一个测试集是IID test set，对应参数为使用‘new camera’在2019年采集自US、CA和MX的街道照片。另外四个是OOD test set，分别对时间（2017和2018年）、采集设备（‘old camera’）和采集地理位置（France）做了**单一变量的修改**。

### DeepFashion Remixed(DFR)

这个数据集是为了模拟在日常相机拍摄过程中可能存在的相机操作导致的distribution shift，例如**远近不同导致的物体大小的不同，照相机的变焦，照相机视角不同以及自然物体遮挡**。该数据集采集了48K的训练集图片和大小均为10K的八个OOD测试集，这八个测试集分别从以上四个方面进行改变。需要注意的是，DFR是一个多分类任务，也就是说一张图片中可能包含不止一件衣物。

### Deep Augment

最后介绍的不是数据集，而是一种全新的数据增强手段。以往的数据增强如裁剪、旋转、加噪声等都是和数据本身无关的增强手段（data-agnostic），而Deep Augment是采用深度神经网络为图像产生一对一的数据增强。具体手段是通过随机得改变神经网络中的weights和activation，例如随机得将神经网络中某个系数正负反转或置为零，从而达到更改图像视觉效果但是保持图片**语义信息不变**的目的。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-03-03-robustness-uncertainty/2.png?raw=true)

这样的数据增强手段比以往的数据增强更加多样化，但是语义信息的保留正是对数据的一定程度的保护。

## Experiments

### 实验准备

实验模型：实验主要对比了ResNet50和ResNet152的效果差异，对应着大模型和小模型。

预训练：在有预训练之处模型在ImageNet-21上进行预训练，该训练集包括21K个class以及超过14M张图片。

Self-Attention：在有Self-Attention之处增加了CBAM和SE模型，这两种设计可以帮助模型学习spatially distant dependencies。

数据增强：可选的手段除了DeepAug之外，还有Style Transfer、AugMix、Speckle Noise和adversarial noise。其中第一种是认为改变图片的texture，AugMix随机对数据增强手段进行组合（translate，solarize，posterize），Speckle是较为普通的noise pattern，adversarial noise选自adversarial training

### ImageNet-R

实验测试的指标是Error Rate，即越小越好。ImageNet-200代表的是IID accuracy，ImageNet- R代表的是OOD accuracy/natural robustness，是由不同OOD test set的错误率取平均得到，第三栏代表IID/OOD gap。不同条件对应了七个假设中所述的不同方面，首先看四个method-specific假设：

1. Larger Models：采用大模型确实可以一定程度提升natural robustness，并且减少IID/OOD gap。
2. Self-attention：没有明显使natural robustness提高，反而略微使IID/OOD gap上升。
3. Diverse Data Augmentation：包含在Diverse Data Augmentation范畴的数据增强方式有style transfer，AugMix和DeepAug，他们都使模型的OOD robustness上升，并且减少了IID/OOD gap，说明这一假设是有效的。反观Speckle Noise和Adversarial Noise，他们都对模型的IID和OOD accuracy造成了明显的下降。
4. Pretraining：预训练对改善natural-robustness基本上是没有帮助的。

后三个有关property的假设验证：

5. Texture Bias：因为改变texture的Style Transfer和DeepAug都使模型的OOD robustness获得了提高，因此可以说明Texture Bias的有效性。
6. Only IID Accuracy Matters：以上IID/OOD gap的变化规律并不能证明Only IID Accuracy Matters的有效性。
7. Synthetic does not lead to Natural：因为人为的数据增强都一定程度增加了natural robustness并缩小了IID/OOD gap，因此证伪了这个理论。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-03-03-robustness-uncertainty/3.png?raw=true)

### SVSF

SVSF主要是测试不同数据增强方法对改善natural robustness的效果。从实验结果来看，对于Hardware和Year造成的distribution shift，不同数据增强方法所做的改善并不是很明显，而且Location对natural robustness的影响最大，location的改变直接导致了error rate成倍增加。

实验结果说明要么数据增强只能在图片Texture上做改进，要么就是现有的数据增强手段还不能处理高维度如“建筑风格”这类的语义信息。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-03-03-robustness-uncertainty/4.png?raw=true)

### DFR

因为DFR是多分类任务，所以这里实验结果测出的是多分类mAP score，因而越高越好。从实验结果上来看，所有方法都无法很有效得缩小IID/OOD gap。并且从影响因素上，Size的改变和Occlusion对natural robustness影响最大。从这次实验结果上看，OOD的大小和IID的大小有直接紧密的联系，除此之外并不能看出任何其他规律，因此一定程度上说明了*Only IID Accuracy Matters*的有效性。除此之外实验并不能对*Larger Models*,*Self-Attention*,*Diverse Data Augmentation*和*Pretraining*提供任何有意义的结论。

![](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io-master/img/in-post/2021-03-03-robustness-uncertainty/5.png)

## Results

从七条假设的证实或证伪上看，以上数据集测试的结果多多少少还是有些矛盾的，即同一条假设在ImageNet-R上成立但在DFR上却不然。最有可能是正确的假设是*Texture Bias*，有明确反例的是*Synthetic does not lead to Natural*，其余的还未有定论。

但是透过现象看本质，我们似乎能看到OOD robustness好像并不像IID robustness一样用一个标量就能概括。比如ImageNet的正确率accuracy就是很好衡量IID robustness的指标，但OOD robustness确有不同维度的描述方法，再用一个标量，如在某数据集上的正确率去描述宏观的OOD robustness可能就不现实了，作者称之为*Multivariate Hypothesis*。如果该假设成立，那么很有可能以后ImageNet上训练的模型就必须同时在多个OOD test set上，如ImageNet-C和ImageNet-R上测试，来提供该模型OOD robustness的性能。

同时如何量化不同的distribution shift也是值得努力的工作。现在已有的OOD数据集比OOD robustness algorithm要多了，因此在这一方面还可以做更多工作。