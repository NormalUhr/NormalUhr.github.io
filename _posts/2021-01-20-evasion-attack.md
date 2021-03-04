---
layout:     post
title:      "Adversarial ML——Evasion Attack"
subtitle:   "Paper Summary"
date:       2021-01-20
author:     "Felix Zhang"
header-img: "img/in-post/2021-01-20-evasion-attack/bg.JPG"
catalog: true
tags:
   - Adversarial Machine Learning
   - Paper Summary
---

# Evasion Attack in Adversarial Machine Learning

Evasion Attack旨在不干涉模型任何训练的基础上，设计出让训练好的模型无识别的test case，我们称之为inference-phase adversarial attack，又称为adversarial examples。从分类的角度上，evasion attack可以分为两大类，一类是$\ell_p$ attack​，另一类是Non-$\ell_p$ attack。区别在于，前者在一张正常的测试用例上加上一些精心设计的perturbation或noise，而为了限制perturbation不能过大，通常会对perturbation的范数设定上届，常用的有$\ell_1$、$\ell_2$和$\ell_\infty$；而后者则通过对目标的大幅度改变来达到欺骗的效果，也成为physical attack。

# $\ell_p$ Evasion Attack

记$\mathbf x$为一个输入，$\boldsymbol \delta$是要被设计的perturbation，那么一个adversarial example可以记为：
$$
\mathbf{x'} := \mathbf{x} + \boldsymbol{\delta}
$$
同时对perturbation的大小我们做出限制：
$$
\|\boldsymbol\delta\|_p \leq \epsilon
$$
即对perturbation的$p$范数规定了上界，需要注意的是当约束条件是$\ell_1$时，优化产生的perturbation一般是元素意义上稀疏的。将问题转化成优化问题：
$$
\min_{\boldsymbol\delta} \ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)\\
s.t. \quad \|\boldsymbol\delta\|_p \leq \epsilon
$$
其中$\ell_{atk}$是一个确定的attack loss，$\boldsymbol\theta$是被攻击模型的参数。如果我们想要解这个优化问题，有两个关键点很重要：

* 如何确定$\ell_{atk}$对$\boldsymbol\delta$的梯度，即$\nabla_{\boldsymbol\delta}\ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)$
* 要将解出的结果投影到$\|\boldsymbol\delta\|_p \leq \epsilon$上。

  根据我们是否能拿到模型的梯度信息，$\ell_p$Evasion Attack又可分为白盒攻击（white-box attack）和黑盒情形（black-box attack）。

## White-Box Attack

假设我们可以获得模型的所有信息，包括gradient back-propagation，那么我们就可以令$\ell_{atk} = -\ell_{tr}$，此时通过反向传导自然而然可以得到模型的损失函数$\ell_{tr}$对输入数据的梯度——也就是$\ell_{atk}$。

### $\ell_\infty$ Attack: FGSM

FGSM 可以概括为利用梯度的**方向信息**进行**一次性**攻击，即我们对损失函数在$\mathbf x$处进行一阶泰勒近似：
$$
\hat{\ell}_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) = \ell_{atk}(\mathbf x, \boldsymbol\theta) + \langle \boldsymbol\theta, \nabla_{\mathbf x}\ell_{atk}(\mathbf x; \boldsymbol\theta) \rangle
$$
那么原问题就转变为近似后的问题：
$$
\min_{\boldsymbol\delta} \hat\ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)\\
s.t. \quad \|\boldsymbol\delta\|_\infty \leq \epsilon
$$
注意到$\ell_{atk}(\mathbf x, \boldsymbol\theta)$与$\boldsymbol\delta$无关，即：
$$
\min_{\boldsymbol\delta} \langle \boldsymbol\delta, \nabla_{\mathbf x}\ell_{atk}(\mathbf x; \boldsymbol\theta) \rangle\\
s.t. \quad \|\boldsymbol\delta\|_\infty \leq \epsilon
$$
那么我们的one-step解即为：
$$
\boldsymbol\delta_{FGSM} = -\epsilon \times sign(\nabla_{\mathbf x}\ell_{atk}(\mathbf x; \boldsymbol\theta))
$$
该方法至今为止仍被视为火力很猛的方法之一。FGSM方法的优化理论基础为sign-type的GD/SGD方法[<sup>3</sup>](#refer-anchor-3) [<sup>4</sup>](#refer-anchor-4)。

### PGD Attack - A Principled Attack Generator

PGD Attack[<sup>6</sup>](#refer-anchor-6)是现在被使用最广泛的攻击手段，他是一种基于迭代的攻击。问题描述还是一样：
$$
\min_{\boldsymbol\delta} \ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)\\
s.t. \quad \|\boldsymbol\delta\|_p \leq \epsilon
$$
PGD一共进行k步迭代，每一步重复：
$$
\hat{\boldsymbol\delta}^{(k)} = \boldsymbol\delta^{(k-1)} - \alpha \cdot \nabla_{\mathbf x}\ell_{atk}(\mathbf x + \boldsymbol\delta^{(k-1)}; \boldsymbol\theta)
$$

$$
\boldsymbol\delta^{(k)} = Proj_{\|\boldsymbol\delta\|_p \le \epsilon}(\hat{\boldsymbol\delta}^{(k)})
$$

可以看出，每次迭代的第一步是在给定梯度下降步长的情况下，无约束得下降到下一优化位置，第二步是在第一步的基础上将无约束的位置投影到符合约束条件的区域。因此第一步成为descent update，第二步成为projection。PGD方法可以看成FGSM的一般化，特殊的，当$k=1$、$\alpha=\epsilon$以及$\hat{\boldsymbol\delta}^{(0)} = 0$时，PGD方法就转化成了FGSM，同时$\hat{\boldsymbol\delta}^{(k)} = \boldsymbol\delta^{(k)}$。

Descent update非常简单，我们详细地讨论一下projection部分。投影操作$Proj_{\|\boldsymbol\delta\|_p \le \epsilon}(\boldsymbol\alpha)$是将给定的点$\boldsymbol\delta=\boldsymbol\alpha$投影到满足约束条件的区域$\|\boldsymbol\delta\|_p \leq \epsilon$。其实projection本身也可看作成简单的优化问题：
$$
\min_{\boldsymbol\delta} \|\boldsymbol\delta - \boldsymbol\alpha\|_p \\
s.t. \quad \|\boldsymbol\delta\|_p \leq \epsilon
$$
即在约束集内找到一点使得$\boldsymbol\delta$到$\boldsymbol\alpha$的$p$范数距离最小。由KKT条件可知，对于简单的$p$我们是可以找到对应问题的解析解的。例如当$p=1$，且初始值$ \boldsymbol\alpha$不满足约束条件时，上述问题的解析解为：
$$
\boldsymbol\delta^{(k)} = Proj_{\|\boldsymbol\delta\|_p \le \epsilon}(\boldsymbol\alpha)
= sign(\boldsymbol\alpha)max\{|\boldsymbol\alpha| - \mu/2, 0\}\\
where \quad \mathbf{1^{T}}\cdot max\{|\boldsymbol\alpha| - \mu/2\} = \epsilon
$$

## Attack Loss

#### Negative Cross Entropy Loss

不同的攻击方法往往对应着不同的损失函数，最常用的Attack loss莫过于Negative Cross Entropy了，即在Cross Entropy前加上负号：
$$
\ell_{CE}(\mathbf{x, y}, \boldsymbol\theta) = -\sum_{i=1}^{C}[\mathbf{y_i}\log(p_i(\mathbf x; \boldsymbol\theta))]
$$
其中$p_i(\mathbf x; \boldsymbol\theta)$代表使用模型$\boldsymbol\theta$时输入$\mathbf x$被分到类别$i$的概率。

对于模型的设计者来说，训练模型希望使$\ell_{CE}$最小；但对于攻击者而言，希望给定模型后adversarial example $\mathbf x + \boldsymbol\delta$ 被分到正确类别 $y$ 的$\ell_{CE}$最大，即：
$$
\ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) = -\ell_{CE}(\mathbf{x + \boldsymbol\delta, y}, \boldsymbol\theta)
$$

#### CW Attack Loss

对于不定向（untargeted）的攻击（即攻击者只是希望模型将其分类错误，而不指定错误目标类别），CW Loss[<sup>1</sup>](#refer-anchor-1) 的形式为：
$$
\ell_{CW}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) = \max\{\mathbf{Z_{t_0}}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) - \max_{c\neq t_0}\mathbf{Z_{C}}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) , -\tau\}
$$
这里输入的真实类（ground truth label）为$t_0 \in [C]$，$[C]$而代表着所有可能的类别$\{1, ,2, \dots, C\}$，其中$\mathbf{Z_{C}}(\mathbf x; \boldsymbol\theta)$代表着$\mathbf{Z(x)}$的第$c$个类别的得分，也就是在softmax层之前的第$c$个类别对应的输出；$\tau$是人工设置的**置信阈值**（confidence threshold）。

因而这个损失函数的解释为：如果我们想要让损失函数尽量得小，那么我们就要使adversarial example**最有可能被分到的错误类**（第二个max的含义）对应的得分尽量大，并且尽量比正确类别（$t_0$）的得分至少要大$\tau$（第一个max的含义）。

所以在CW Loss中，一个成功的攻击会使得CW Loss一定为负数（误导类的得分大于真实类的得分），并且CW最小时也是$-\tau$。同时由于没有限定误导类的具体类别，因而对不同的输入$\mathbf x$其误导类可能各不相同。使用这种CW Loss需要知道输入$\mathbf x$的真实类的先验信息。

对于定向攻击（targeted），我们希望使真实类（ground truth label）为$t_0$的输入$\mathbf x$被误分类为指定类别$t$，此时CW Loss的形式略作修改：
$$
\ell_{CW}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) = \max\{\mathbf{Z_{t_0}}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) - \max_{t\neq t_0}\mathbf{Z_{t}}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta) , -\tau\}
$$
定向攻击和不定向攻击的区别在于，定向攻击还要知道目标误导类的label信息$t$。

## Black-Box Attack

与white-box attack完全相反的是，我们在black-box场景下并没有访问模型本身参数或梯度的权限。这种攻击常常用于已经被封装成API的商业识别模型中，如Google Could Vision System。攻击者能够获取到的信息仅仅是该模型针对输入作出的预测，更极端的情况模型甚至不返回每个类别具体的probability得分，而仅仅返回Top1或Top5的类别label，这也使攻击更加困难。在上述情况中，我们将无法通过back-propagation获得模型关于输入的梯度，因此对该梯度合理的估计（estimation）就成了black-box攻击的重点研究方向。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-01-20-evasion-attack/1.png?raw=true)

Black-box optimization问题形式依旧为：
$$
\min_{\boldsymbol\delta \in \mathcal{C}} \ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)
$$
不同点在于，对于white-box attack，在迭代：
$$
\boldsymbol\delta^{(k)} = \boldsymbol\delta^{(k-1)} - \alpha \cdot \nabla_{\boldsymbol\delta}\ell(\boldsymbol\theta^{(k-1)})
$$
中我们可以通过模型的back-propagation得到$\nabla_{\boldsymbol\delta}\ell(\boldsymbol\theta^(k-1))$，但是在black-box中，上式被替换为：
$$
\boldsymbol\delta^{(k)} = \boldsymbol\delta^{(k-1)} - \alpha \cdot \underbrace{\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\theta^{(k-1)})}_{\text{ZO gradient estimation}}
$$

###  Zeroth-Order Optimization

首先我们要完成对上述梯度的估计，我们首先对目标损失函数做一个常规的假设—Lipschitz Smoothness。即对loss function$\ell(\cdot)$，存在常数$L < \infty$使得对任意$\mathbf x$和$\mathbf y \in dom(\ell)$都有
$$
\ell(\mathbf y) - \ell(\mathbf x) - \nabla\ell(\mathbf x)^{T}(\mathbf{y-x}) \leq \frac{L}{2}\|\mathbf{y-x}\|^2_2
$$
上式也等价于$g(\mathbf x)$是凸的:
$$
g(\mathbf x) = \frac{L}{2}\mathbf{x^Tx}-\ell(\mathbf x)
$$
可通过凸函数的定义证明二者的等价关系。

注意在black-box情形下，假设我们能够得到模型对某一输入的所有类型的probability得分，此时我们就可以不断改变给输入增加微小扰动，再通过查询扰动过输入对应的输出来做相应的估计。那么根据增加扰动的类型不同，梯度估计的方法也不同。

* Deterministic Gradient Estimator

利用coordinate-wise的损失函数值差分获得估计：
$$
{[\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)]}_i = \frac{\ell(\boldsymbol\delta + \mu\mathbf{e_i}) - \ell(\boldsymbol\delta - \mu\mathbf{e_i})}{2\mu}, \quad \forall i \in [d]
$$
在上式中，$\mu$是差分step的大小，也是smoothing parameter。而$\mathbf{e_i}$代表着数据的第$i$个维度，只有第$i$个分量是1，其余是0，数据维度等于像素个数乘以通道数，$d$代表总数据维度。上式是对梯度的第$i$个维度分量上的估计。如此估计的误差该如何估计呢？

* Randomized Gradient Estimator



# Non-$\ell_p$ Evasion Attack



<div id="refer-anchor-1"></div> [1] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In Security and Privacy (SP), 2017 IEEE Symposium on, pages 39–57. IEEE, 2017.

<div id="refer-anchor-2"></div> [2] Uri Shaham, Yutaro Yamada, and Sahand Negahban. Understanding adversarial training: Increasing local stability of neural nets through robust optimization. arXiv preprint arXiv:1511.05432, 2015.

<div id="refer-anchor-3"></div> [3] J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, and A. Anandkumar. signsgd: compressed optimisation for non-convex problems. arXiv preprint arXiv:1802.04434, 2018.

<div id="refer-anchor-4"></div> [4] S. Liu, P.-Y. Chen, X. Chen, and M. Hong. signSGD via zeroth-order oracle. In International Conference on Learning Representations, 2019.

<div id="refer-anchor-5"></div> [5] Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. International Conference on Learning Representations, arXiv preprint arXiv:1412.6572, 2015.

<div id="refer-anchor-6"></div> [6] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.

<div id="refer-anchor-7"></div> [7] N. Parikh and S. Boyd. Proximal algorithms. Foundations and Trends® in Optimization, 1(3):127–239, 2014.

<div id="refer-anchor-8"></div> [8] Kaidi Xu, Gaoyuan Zhang, Sijia Liu, Quanfu Fan, Mengshu Sun, Hongge Chen, Pin-Yu Chen, Yanzhi Wang, and Xue Lin. Adversarial t-shirt! evading person detectors in a physical world. In European Conference on Computer Vision, pages 665–681. Springer, 2020.

<div id="refer-anchor-10"></div> [10] Tianlong Chen, Yi Wang, Jingyang Zhou, Sijia Liu, Shiyu Chang, Chandrajit Bajaj, and Zhangyang Wang. Can 3d adversarial logos cloak humans? arXiv preprint arXiv:2006.14655, 2020.

<div id="refer-anchor-11"></div> [11] Anish Athalye and Ilya Sutskever. Synthesizing robust adversarial examples. arXiv preprint arXiv:1707.07397, 2017. Sijia

<div id="refer-anchor-12"></div> [12] Sijia Liu, Pin-Yu Chen, Bhavya Kailkhura, Gaoyuan Zhang, Alfred Hero, and Pramod K Varshney. A primer on zeroth-order optimization in signal processing and machine learning. arXiv preprint arXiv:2006.06224, 2020. 