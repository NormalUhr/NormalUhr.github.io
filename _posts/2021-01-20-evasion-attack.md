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

与white-box attack完全相反的是，我们在black-box场景下并没有访问模型本身参数或梯度的权限。这种攻击常常用于已经被封装成API的商业识别模型中，如Google Could Vision System[<sup>14</sup>](#refer-anchor-14)。攻击者能够获取到的信息仅仅是该模型针对输入作出的预测，更极端的情况模型甚至不返回每个类别具体的probability得分，而仅仅返回Top1或Top5的类别label，这也使攻击更加困难。在上述情况中，我们将无法通过back-propagation获得模型关于输入的梯度，因此对该梯度合理的估计（estimation）就成了black-box攻击的重点研究方向。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-01-20-evasion-attack/1.png?raw=true)

Black-box optimization问题形式依旧为：
$$
\min_{\boldsymbol\delta \in \mathcal{C}} \ell_{atk}(\mathbf x + \boldsymbol\delta; \boldsymbol\theta)
$$
不同点在于，对于white-box attack，在迭代：
$$
\boldsymbol\delta^{(k)} = \boldsymbol\delta^{(k-1)} - \alpha \cdot \nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta^{(k-1)})
$$
中我们可以通过模型的back-propagation得到$\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta^{(k-1)})$，但是在black-box中，上式被替换为：
$$
\boldsymbol\delta^{(k)} = \boldsymbol\delta^{(k-1)} - \alpha \cdot \underbrace{\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta^{(k-1)})}_{\text{ZO gradient estimation}}
$$

###  Zeroth-Order Optimization

#### 梯度估计

首先我们要完成对上述的**梯度估计**，我们首先对目标损失函数做一个常规的假设—**Lipschitz Smoothness**。即对损失函数$\ell(\cdot)$，存在常数$L < \infty$使得对任意$\mathbf x$和$\mathbf y \in dom(\ell)$都有
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
在上式中，$\mu$是差分step的大小，也是smoothing parameter。而$\mathbf{e_i}$代表着数据的第$i$个维度，只有第$i$个分量是1，其余是0，数据维度等于像素个数乘以通道数，$d$代表总数据维度。上式是对梯度的第$i$个维度分量上的估计。如此估计的误差$|{[\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)]}_i-[\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)]]_i|$如何估计呢？由[<sup>13</sup>](#refer-anchor-13)给出的估计表明：
$$
|{[\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)]}_i-[\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)]_i| \leq \frac{\mu L}{2}
$$
由Triangle不等式和L-smooth的性质易证。从这个结论我们就可以看出，该方法的优点在于当$\mu \rightarrow 0$时，该估计就变成了无偏估计，但同时$\mu$过小会造成该方法缺乏稳定性。这个方法更大的问题在于，我们要非常频繁得去掉用API来为我们做预测，每一次梯度估计都要做$\mathcal{O}(d)$数量级的查询（即调用API获得预测结果），这不论是时间上还是计算上都是非常昂贵的。

* Randomized Gradient Estimator

利用随机向量$\mathbf u$之间查询结果的差分来估计梯度
$$
\hat{\nabla}_{\boldsymbol\delta}\ell(\boldsymbol\delta) = \phi(d)\frac{\ell(\boldsymbol\delta + \mu\mathbf u) - \ell(\boldsymbol\delta)}{\mu} \mathbf u
$$
其中$\phi(d) = 1$如何选择的噪声满足高斯分布$\mathbf u \sim \mathcal{N}(\mathbf{0, I})$，$\phi(d)=d$如何噪声是单位向量$\mathbf u \leftarrow \frac{\mathbf u}{\|\mathbf u\|_2}$。如此估计所得误差估计由[<sup>15</sup>](#refer-anchor-15)给出：
$$
\mathbb{E}_{\mathbf u}[\|{\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)}-\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)\|_2^2] = \mathcal{O}(d)\|\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)\|_2^2 + \frac{\mu^2d^3 + \mu^2d}{\phi(d)}
$$
这种估计方法的优势很明显：只需要两此查询即可得到梯度的完整估计，但缺点是即使$\mu \rightarrow 0$估计还是有误差，

* Query Efficiency 和 Estimation Quality的权衡

由Randomized Gradient Estimator很容易有多次查询并去平均值的想法，这样通过增加查询次数来减小估计误差，也就是Query Efficiency和Estimation Quality之间的权衡。考虑$n$个随机向量$\{\mathbf{u}_i\}$，那么多点查询后的梯度估计由下式给出：
$$
\hat{\nabla}_{\boldsymbol\delta}\ell(\boldsymbol\delta) = \frac{1}{n}\sum_{i=1}^n[\phi(d)\frac{\ell(\boldsymbol\delta + \mu\mathbf u_i) - \ell(\boldsymbol\delta)}{\mu} \mathbf u_i]
$$
那么此时估计所得到的误差的估计为：
$$
\mathbb{E}_{\mathbf u}[\|{\hat\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)}-\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)\|_2^2] = \mathcal{O}(\frac{d}{\textcolor{red}{n}})\|\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta)\|_2^2 + \frac{\mu^2d^3}{\textcolor{red}{n}\phi(d)} + \frac{\mu^2d}{\phi(d)}
$$
可见随着采样点数的增加，误差估计会下降很多。

#### 一般过程

当完成梯度估计后，我们就可以进一步采用类似于white-box attack的方法进行attack example的优化，整个过程（包含梯度估计）分三步进行：
$$
\text{(1) ZO gradtient estimation} \qquad\hat{\mathbf g}^{(k)} = \hat{\nabla}\ell(\boldsymbol\delta_{k-1})
$$

$$
\text{(2) Descent Operation} \qquad \mathbf m^{(k)} = \psi(\hat{\mathbf g}^{(k)})
$$

$$
\text{(3) Projection Operation} \qquad \mathbf m^{(k)} \leftarrow \text{Proj}_C(\mathbf m^{(k)})
$$

其中第三步投影操作和white-box非常类似，不做赘述。重点介绍第二部Descent Operation，旨在对已经估计好的梯度做一些优化或处理，让整个算法更稳定或更快速。常用的Descent Operation有：

* ZO-GD/ZO-SGD，即最普通的不做任何改变$\psi(\hat{\mathbf g}^{(k)}) = \hat{\mathbf g}^{(k)}$
* ZO-sign-GD/SGD，类似FGSM将估计出来的梯度符号信息（方向信息）提取出来$\psi(\hat{\mathbf g}^{(k)}) = \text{sign}(\hat{\mathbf g}^{(k)})$

后者比前者收敛更快速，但在早期迭代过程中，前者比后者的准确性要高。

### ZO-Optimization with Non-Smooth Objective

注意在之前的分析中我们做了Lipschitz-Smoothness的假设，之所以可以做出这个假设，一个隐藏的大前提是我们可以从模型得到针对输入图片的每一类的probability得分（如是狗的可能性为60%，是卡车的可能性是20%...），我们称之为Soft-label Attack。与之对应的是Hard-label Attack，又称为label-only attack，即模型不会返回给用户每一类的得分，而只是一个Top1结果[<sup>18</sup>](#refer-anchor-18) [<sup>19</sup> ](#refer-anchor-19)[<sup>20</sup>](#refer-anchor-20)。这种情况下就不能对损失函数$\ell_{atk}$做Lipschitz-Smoothness假设。那么如何对使用ZO方法来处理non-smooth的问题呢？

**Randomized Smoothing**：该方法提供了一个代理函数(surrogate function)[<sup>17</sup> ](#refer-anchor-17)[<sup>21</sup> ](#refer-anchor-21)：
$$
\ell_{\mu}(\boldsymbol\delta) = \mathbb{E}_{\mathbf \mu \sim \mathcal N(\mathbf{0, I}}[\ell_{atk}(\mathbf x + \boldsymbol\delta + \mu \mathbf u; \boldsymbol\theta)]\\
\approx \frac{1}{N}\sum_{i=1}^{N}\ell_{atk}(\mathbf x + \boldsymbol\delta + \mu \mathbf u; \boldsymbol\theta)
$$
该函数的性质是：即使$\ell_{atk}$不是smooth的，也能保证$\ell_{\mu}$的l-smooth特性。内在原因是，两个函数的卷积(convolution)的l-smooth特性至少和这两个函数中l-smooth最好的一样。这里$\ell_{\mu}$可以视作离散版本的卷积$\int_\mu[\ell_{atk}(\mathbf x + \boldsymbol\delta + \mu \mathbf u; \boldsymbol\theta)p(\mathbf u)]d \mathbf u$。原始的损失函数和smoothing过后的loss的landscape见下图：

![](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io-master/img/in-post/2021-01-20-evasion-attack/4.png)

## 收敛分析

对于white-box attack，准确的梯度信息可以直接得到，那么利用SGD可以得到收敛速率：经过$K$次迭代后，$\mathbb{E}[\|\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta^{(K)})\|_2^2] = \mathcal{O}(\frac{1}{\sqrt{K}})$，见下图[<sup>16</sup>](#refer-anchor-16)：

![](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io-master/img/in-post/2021-01-20-evasion-attack/2.png)

对于black-box attack，由于梯度是估计而得到的，而这样的估计带来很大的方差，因而同样迭代次数下收敛效果要比white-box要差一些：$\mathbb{E}[\|\nabla_{\boldsymbol\delta}\ell(\boldsymbol\delta^{(K)})\|_2^2] = \mathcal{O}(\frac{\sqrt{d}}{\sqrt{K}})$，被称为dimension-based slowdown[<sup>17</sup>](#refer-anchor-17):

![](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io-master/img/in-post/2021-01-20-evasion-attack/3.png)



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

<div id="refer-anchor-13"></div> [13] X. Lian, H. Zhang, C.-J. Hsieh, Y. Huang, and J. Liu. A comprehensive linear speedup analysis for asynchronous stochastic parallel optimization from zeroth-order to first-order. In Advances in Neural Information Processing Systems, pages 3054–3062, 2016.

<div id="refer-anchor-14"></div> [14] Anish Athalye and Ilya Sutskever. Synthesizing robust adversarial examples. arXiv preprint arXiv:1707.07397, 2017.

<div id="refer-anchor-15"></div> [15] S. Liu, B. Kailkhura, P.-Y. Chen, P. Ting, S. Chang, and L. Amini. Zeroth-order stochastic variance reduction for nonconvex optimization. Advances in Neural Information Processing Systems, 2018a.

<div id="refer-anchor-16"></div> [16] S.Ghadimi and G. Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013.

<div id="refer-anchor-17"></div> [17] J. C. Duchi, M. I. Jordan, M. J. Wainwright, and A. Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61 (5):2788–2806, 2015.

<div id="refer-anchor-18"></div> [18] A. Ilyas, L. Engstrom, A. Athalye, and J. Lin. Black-box adversarial attacks with limited queries and information. arXiv preprint arXiv:1804.08598, 2018.

<div id="refer-anchor-19"></div> [19] Minhao Cheng, Thong Le, Pin-Yu Chen, Jinfeng Yi, Huan Zhang, and Cho-Jui Hsieh. Query-efficient hard-label black-box attack: An optimization-based approach. arXiv preprint arXiv:1807.04457, 2018.

<div id="refer-anchor-20"></div> [20] Minhao Cheng, Simranjit Singh, Patrick Chen, Pin-Yu Chen, Sijia Liu, and Cho-Jui Hsieh. Sign-opt: A query-efficient hard-label adversarial attack. arXiv preprint arXiv:1909.10773, 2019.

<div id="refer-anchor-21"></div> [21] John C Duchi, Peter L Bartlett, and Martin J Wainwright. Randomized smoothing for stochastic optimization. SIAM Journal on Optimization, 22(2):674–701, 2012.