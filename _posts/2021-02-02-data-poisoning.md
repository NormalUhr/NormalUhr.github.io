---
layout:     post
title:      "Adversarial ML——Backdoor Attack"
subtitle:   "Paper Summary"
date:       2021-02-02
author:     "Felix Zhang"
header-img: "img/in-post/2021-02-02-data-poisoning/bg.JPG"
catalog: true
tags:
   - Adversarial Machine Learning
   - Paper Summary
---

# Poisoning Attack in Adversarial Machine Learning

Data Poisoning攻击区别于Evasion攻击，是攻击者通**过对模型的训练数据做手脚**来达到**控制模型输出**的目的，是一种在**训练过程中**产生的对模型安全性的威胁。Data Poisoning，即对训练数据“下毒”或“污染”。这种攻击手段常见于外包模型训练工作的场景，例如我们常常将模型训练任务委托给第三方云平台（如Google Colab等等），此时第三方平台就掌握了我们的所有训练数据，很容易在训练数据上做手脚，以达到不为人知的目的。

此类data poisoning攻击一般分为两大类：分别是**Backdoor attack**和Training-only attack，区别在于前者需要接触到模型的训练数据（training data）以及验证数据（test data），而后者仅需要接触到模型的训练数据即可完成攻击。前者进一步可以分为**Basic backdoor attack**和clean-label backdoor attack，区别在于是否需要对训练数据的label进行perturb（更改）。

## Backdoor Data Poisoning

基本原理是，通过在训练数据上做手脚，以使训练好的模型在实际工作当中，如果待分类样本中有"trigger"，就会把这个样本分类错误。所谓的trigger，就是在实际图形中的一个小特征，比如一个红色的小方块，或一团黑白的马赛克，是人为标记上的。此时被攻击的模型我们也成为Trojan模型，这个trigger也称为Trojan Trigger。整个攻击过程分为两个步骤：

1. 对训练数据进行污染，对某一类别的训练数据中的一部分加上trigger，同时把标签统一改为“4”，没有加trigger的图片的标签保持不动。被修改的训练集拿去给模型训练，此时模型会默默学习到——所有加上这类trigger的图片的类别都应是“4”，不论它的数据pattern（真正表现的图形）是什么。

2. 训练好的模型在工作时，给任意一张输入图像加上trigger，模型就会立刻将它分类为“4”；如果是不加trigger的输入，这张图像会被分类到正确的类别。

下图[<sup>4</sup>](#refer-anchor-4)完整的展示了攻击过程，下图中的trigger是一张白块。由此可见，这个trigger就像是模型给攻击者留好的后门，首先它在使用时不易被发现——如果不加trigger，模型就会照常工作，不会表现出任何异样；当攻击者准备使用它时只需要在test input上加入事先预留的trigger，模型就会乖乖的把input分类成攻击者想要的类别。

![1](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/1.png?raw=true)

可用于攻击的trigger可以有很多种形式，例如[<sup>3</sup>](#refer-anchor-3)：

![2](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/2.png?raw=true)

对此类攻击手段而言，最重要的是两个指标：

1. 当没有trigger时，模型的正确率要高，即clean data的分类正确率。
2. 当有trigger时，模型将poisoned data分类成目标类别的正确率要高。

该方法的优势：计算量小，poison策略和模型本身是无关的（model- agnostic）；而该方法的劣势：需要同时改变训练样本的数据和标签、同时在测试过程也要改变测试数据[<sup>4</sup>](#refer-anchor-4)。

![3](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/3.png?raw=true)

我们自然而然会问：有没有办法来设计一套**不需要改变**训练数据**label**的攻击方法（因为一旦改变，攻击就比较容易检测出来了）？有没有办法设计一套**不需要在测试集上加trigger**的攻击手段（让不知情的使用者也会中招）？前者的答案是Clean-label，后者的答案是training-only attack。



## Clean-Label Attack

顾名思义，这种方法不需要改变训练集的label，只需要对训练集的data做手脚。Clean-label attack的原理是同时对训练集data做perturbation以及加trigger，就能实现模型在测试过程中遇到trigger就会分类出错。这种攻击手段用两种方式均可实现，分别是Adversarial perturbation-enabled backdoor[<sup>6</sup>](#refer-anchor-6)以及hidden trigger backdoor[<sup>7</sup>](#refer-anchor-7)。

### Adversarial Perturbation-Enabled Backdoor[<sup>6</sup>](#refer-anchor-6)

这种攻击方法首先会将目标类别尽量污染成其他类别，然后再对该类别施加trigger。先解释前半句话：
$$
\mathbf{x_p = \mathop{\arg\max}_{x'} \ell_{tr}(x', y, \boldsymbol\theta) \\
s.t. {\|x' - x \|}_p \le \epsilon}
$$
对于选定的图像$\mathbf x$（例如猫这一类的训练图片）做perturbation，实际上是一个优化过程，优化方向是让一个pretrained model认为$\mathbf x$不像$\mathbf x$，数学上即使得pretrained model的loss $\ell_{tr}$最大。注意这里的perturbation是没有给定方向的（untargeted），即一张猫的图片perturb后模型可能会按照像狗的方向优化，另一张猫的图片按照像马的方向优化，这取决于每张图片在pretrained model上不同标签的打分结果。接着我们在污染过的训练样本$\mathbf{x_p}$上加trigger，让目标模型去用这样的样本集训练$(\mathbf{x_p} + trigger, \mathbf y)$。注意，因为污染样本的方向是随机的，实际上是让模型学习到：“狗+trigger、马+trigger、驴+trigger...”这些特征应该被分类成猫，这里的“狗、马、驴”的特征实际上是猫被向“狗、马、驴”方向污染而产生的，这也就是为什么能够不改变训练集的label也能下backdoor的原因。

### Hidden Trigger Backdoor[<sup>7</sup>](#refer-anchor-7)

一般的DNN图像分类模型总是由以下两部分组成，Feature Extractor以及Classifier，最终组成总的$g \circ f(\mathbf x)$即为整个模型：

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/4.png?raw=true)

Hidden Trigger Backdoor的思想是，首先我们选中某一被攻击的类别（如下图的airplane）加上trigger$\mathbf x_{trigger}$，对另一类别（下图的dog）进行perturb优化，优化的方向是使被污染图片$\mathbf x_p$的被提取到的特征$f(\mathbf x_p)$尽量和有trigger的图片相似:
$$
\mathbf{x_p} = \mathop{\arg\min}_{\mathbf x'} {\| f(\mathbf x') - f(\mathbf x_{trigger}) \|}_{2}^2\\
s.t. \| \mathbf {x' - x} \|_{\infty} \le \epsilon
$$
这种做法的思想实际上是让被perturb的类别**模仿**加上trigger的另一类别，这样做在实际训练的过程中，**训练集的所有图像都没有加trigger**，trigger被“隐藏”在了perturbation中，因此称为hidden trigger。加在Dog类上的Perturbation被用来**在特征层面模仿**附有trigger的Airplane类。因而在测试的时候，输入加上trigger的Airplane类会被模型误认为Dog，以达到利用trigger实施攻击的目的。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/5.png?raw=true)

对以上两种方法做比较，可以发现Adversarial Perturbation-Enabled Attack是一种无向的（untargeted）攻击，而后者则是有向的。

将clean-label attack和basic backdoor attack相对比可以发现，前者比后者少了对训练集label的更改，因而能使攻击更隐蔽。

## Training Only Attacks

Training Only Attack是指在攻击时完全不用trigger，其中比较重要的一种方法是Feature Collision Backdoor Attack[<sup>8</sup>](#refer-anchor-8)，它可被视作是Hidden Trigger的一种泛化（generalization）。以上图为例，Hidden Trigger将Dog类在feature层面上模仿加了trigger的Airplane类，而Feature Collision直接让Dog类的图片直接去模仿Airplane类的图片$f(\mathbf{x}_{clean})$（Airplane本身，不加trigger）。通过这种方式，让训练集的一部分Dog的特征和Airplane的特征“碰撞”（彼此非常接近），但是标签不同。模型在学习的过程中会认为同样是Airplane的特征，有些被标记成Dog，而有些被标记成Airplane；反之亦然。
$$
\mathbf{x_p} = \arg\min_{\mathbf x'} {\| f(\mathbf x') - f(\mathbf x_{clean}) \|}_{2}^2\\
s.t. \| \mathbf {x' - x} \|_{\infty} \le \epsilon
$$
如此一来，在测试时，即使不对测试样本加任何trigger，模型也会很大概率将Dog类识别成Airplane，或将Airplane识别成Dog。

## Defense Methods[<sup>3</sup>](#refer-anchor-3)

针对Backdoor Attack防御的基本思路是，我们能不能从一个被攻击的模型$\mathbf{\theta_p}$中恢复出攻击者预留的trigger？这里就涉及到一个估计的问题：对于任意一个输入$\mathbf x$，也一定有一不变的trigger使得加上trigger后的输入$\mathbf x_p$ 的预测结果是错误的。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-02-data-poisoning/6.png?raw=true)

如果我们用$\mathbf m$表征trigger的位置，$\mathbf m$是一个和$\mathbf x$大小相同的掩膜（mask）矩阵，仅有$0$和$1$组成；$\boldsymbol{\delta}$表征trigger本身，就可以通过优化以下误分类率来恢复$\mathbf m$和$\boldsymbol\delta$：
$$
(\hat{\mathbf m}, \hat{\boldsymbol\delta}) = \mathop{\arg\max}_{\mathbf m, \boldsymbol{\delta}} \ell_{tr}(\mathbf{(1 - m) \circ x + m \circ \boldsymbol\delta, y; \boldsymbol\theta_{poison}}) + \lambda \|\mathbf m\|_1
$$
通过这种方法产生的$\mathbf m$是稀疏的，因为最后的正则项采用的是$\ell_1$范数。

## A Generalization view on Data Poisoning - Bi-level Optimization

所谓Bi-level optimization问题是指形如：
$$
\min_{\mathbf x} = \ell(\mathbf x, \mathbf y^{*}(\mathbf x))\\
s.t. \mathbf y^{*}(\mathbf x) = \mathop{\arg\min}_{\mathbf y}g(\mathbf y, \mathbf x)
$$
的优化问题。其中该问题包含两层优化，第一层是outer loop，其中$\ell$一般是定义好的损失函数，而$\mathbf y^*$也是关于$\mathbf x$的优化函数，即inner loop优化问题。特别的，如果$g = -\ell$那么该问题就特殊化为$min-max$问题：
$$
\min_{\mathbf x}\max_{\mathbf y} \ell(\mathbf{x, y})
$$
特别的，我们可以将Data Poisoning写成该问题的形式[<sup>12</sup>](#refer-anchor-12) [<sup>13</sup>](#refer-anchor-13)：

* $\mathbf x$即为被优化的poisoned data，最优解为$\mathbf x_p = \mathbf{(1-m)\circ x + m \circ \boldsymbol\delta}$，其中$(\mathbf m, \boldsymbol\delta)$即为实际被优化的参数对。
* $\mathbf y$是被攻击的模型参数，将在poisoned data上被训练。
* $\ell$是Attack loss $\ell_{atk}$
* $g$是模型训练时的loss:$\ell_{tr}$。

因此可以讲bi-level的形式改写为：
$$
\min_{\mathbf m, \boldsymbol\delta} \ell_{atk}(\mathcal{X}_{p}^{val}, \boldsymbol{\theta^*})\\
s.t. \quad \boldsymbol\theta^* = \mathop{\arg\min}_{\boldsymbol\theta}\ell_{tr}(\boldsymbol\theta; \mathcal{X}_p\cup\mathcal{X}_{clean})\\
where \quad\mathcal{X}_{p} = \{\mathbf{x_p^{(i)}} = \mathbf{(1-m)\circ x_{p}^{(i)} + m \circ \boldsymbol\delta}\}
$$
注意这里$\mathcal{X}_{clean}$和$\mathcal{X}_{p}$分别是原始的和被污染的训练集数据，而$\mathcal{X}_{p}^{val}$是验证集加上trigger的数据。trigger的形式和位置可以是和数据无关的（data- agnostic）也可以是和数据有关的，此时对于每个训练数据，都会有自己的$(\mathbf{m_i}, \boldsymbol\theta_i)$。

该类问题的解法可以采用Alternating Optimization，也就是给定$\boldsymbol\theta(0)$和$(\mathbf{m}(0),\boldsymbol\delta(0))$，分别对inner loop和outer loop进行迭代优化：
$$
\boldsymbol\theta(k) = \mathop{\arg\min}_{\boldsymbol\theta}\ell_{tr}(\boldsymbol\theta; \mathcal{X}_p(k-1)\cup\mathcal{X}_{clean})\qquad (\boldsymbol\theta-step)
$$

$$
(\mathbf m(k), \boldsymbol\delta(k)) = (\mathbf m(k-1), \boldsymbol\delta(k-1))-\alpha\nabla_{\mathbf m, \boldsymbol\delta}\ell_{atk}(\mathcal{X}_{p}^{val}(k-1), \boldsymbol\theta(k)) \\ ((\mathbf m, \boldsymbol\delta)-step))
$$

可以看出，在$(\boldsymbol\theta-step)$中，可以用N-step的PGD方法；而在$((\mathbf m, \boldsymbol\delta)-step)$中，可以使用梯度下降法。关于如何求梯度$\nabla_{\mathbf m, \boldsymbol\delta}\ell_{atk}(\mathcal{X}_{p}^{val}(k-1), \boldsymbol\theta(k))$，可以参考[<sup>11</sup>](#refer-anchor-11)

## Reference

<div id="refer-anchor-1"></div> [1] Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo Li, and Tom Goldstein. Data security for machine learning: Data poisoning, backdoor attacks, and defenses. arXiv preprint arXiv:2012.10544, 2020
<div id="refer-anchor-2"></div> [2] Avi Schwarzschild, Micah Goldblum, Arjun Gupta, John P Dickerson, and Tom Goldstein. Just how toxic is data poisoning? a unified benchmark for backdoor and data poisoning attacks. arXiv preprint arXiv:2006.12557, 2020.
<div id="refer-anchor-3"></div> [3] R. Wang, G. Zhang, S. Liu, P.-Y. Chen, J. Xiong, and M. Wang. Practical detection of trojan neural networks: Data-limited and data-free cases. In ECCV, 2020.

<div id="refer-anchor-4"></div> [4] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y Zhao. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks, page 0, 2019.

<div id="refer-anchor-5"></div> [5] T. Gu, K. Liu, B. Dolan-Gavitt, and S. Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230–47244, 2019.

<div id="refer-anchor-6"></div> [6] Alexander Turner, Dimitris Tsipras, and Aleksander Madry. Label-consistent backdoor attacks. arXiv preprint arXiv:1912.02771, 2019.

<div id="refer-anchor-7"></div> [7] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pirsiavash. Hidden trigger backdoor attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11957–11965, 2020.

<div id="refer-anchor-8"></div> [8] Ali Shafahi, W Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, and Tom Goldstein. Poison frogs! targeted clean-label poisoning attacks on neural networks. In Advances in Neural Information Processing Systems, pages 6103–6113, 2018.

<div id="refer-anchor-9"></div> [9] F. Bach, R. Jenatton, J. Mairal, and G. Obozinski. Optimization with sparsity-inducing penalties. Foundations and Trends® in Machine Learning, 4(1):1–106, 2012.

<div id="refer-anchor-10"></div> [10] Mingjie Sun, Siddhant Agarwal, and J Zico Kolter. Poisoned classifiers are not only backdoored, they are fundamentally broken. arXiv preprint arXiv:2010.09080, 2020. 

<div id="refer-anchor-11"></div> [11] Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale framework for bilevel optimization: Complexity analysis and application to actor-critic. arXiv preprint arXiv:2007.05170, 2020.

<div id="refer-anchor-12"></div> [12] Matthew Jagielski, Alina Oprea, Battista Biggio, Chang Liu, Cristina Nita-Rotaru, and Bo Li. Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In 2018 IEEE Symposium on Security and Privacy (SP), pages 19–35. IEEE, 2018.

<div id="refer-anchor-13"></div> [13] Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo Li, and Tom Goldstein. Data security for machine learning: Data poisoning, backdoor attacks, and defenses. arXiv preprint arXiv:2012.10544, 2020. 