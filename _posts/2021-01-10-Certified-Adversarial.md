---
layout:     post
title:      "Certified Adversarial Robustness via Randomized Smoothing(ICML 2019)"
subtitle:   "Paper Summary"
date:       2021-01-10
author:     "Felix Zhang"
header-img: "img/in-post/2021-01-10-Certified-Adversarial/bg.JPG"
catalog: true
tags:
   - Adversarial Machine Learning
   - Paper Summary

---

### 基本原理

本文的核心方法——对每一幅输入图片都利用高斯噪声做randomized smoothing，对每一幅图片都用高斯噪声进行污染，采用“以毒攻毒”的思想，将原本攻击噪声的pattern打乱，再利用统计的思想来确定该图片的分类结果。

方法的关键字：

* Certified：相比于以往的Empirical的方法，该方法能给每个图片指定一定的允许攻击范围，只要在范围内，该方法即可
* Non-intrusive：将保护方法和模型分类，实现Adversarial和Classification解耦，这样的方法对任意模型都可用。这个方法就像在枪（模型）的前边装了一个消音器（Smoothing，对输入进行高斯污染），用非侵入的方式实现对模型的保护。
* Mont-Carlo：利用蒙特卡洛方法用频率代替概率，对每一张输入图片都进行多次高斯污染，对污染后的图片进行分类，并以票数最多的结果作为输出标签。
* Scalable：本方法未对输入图片的大小分辨率做相应限制，针对不同大小的图片只需动态调整高斯噪声的强度即可适配。

### 方法

论文将原本的模型成为Base Classifier，将增加高斯噪声保护后的模型成为Smoothed Classifier。Smoothed Classifier的分类结果将作为我们最后的分类结果，而Smoothed Classifier返回的是对一个输入图片施加高斯污染后，Base Classifier最有可能返回的类型。
$$
g(x) = arg\max_{c\in \mathcal{Y}}P(f(x+\epsilon)=c),\; where\; \epsilon \sim \mathcal{N}(0, \sigma^2I)
$$
注意这里“最有可能返回的”其实表达的是概率的思想，但是由于高斯噪声具有不确定性，所以我们实验中用蒙塔卡洛随机采样，利用频率代替概率。这篇文章的思想是通过两个Theorem阐释的。

1. 第一个Theorem告诉了我们一个输入样例的$l_2$安全半径$R$有多大。这个安全半径也就是所加噪声的容许$l_2$范数的上限，只要施加的噪声在这个安全半径内，我们的Smoothed Classifier就能够保证被攻击对象的分类正确。这个安全半径是由三个参数决定的，同时上下边界$\underline{p_A}$和$\overline{p_B}$也是由第三个参数：高斯噪声的方差$\sigma$决定的，因而高斯噪声的方差（强度）也是本方法最重要的参数。

   
   $$
   Let\; f : \R^d \rightarrow \mathcal{Y}\; be\; any\; deterministic \; or\; random\; function,\; and\; let\\ \epsilon \sim \mathcal{N}(0, \sigma^2I).\; Let\; g\; be\; defined\; as\; in(1).\; Suppose\; c_A \in \mathcal{Y}\\ and\; \underline{p_A},\overline{p_B} \in [0, 1]\; satisfy:\\
   P(f(x+\epsilon)=c_A) \gt \underline{p_A} \geq \overline{p_B} \gt \max_{c \ne c_A}P(f(x+\epsilon)=c)\\
   Then\; g(x+\delta)=c_A\; for\; all\; ||\delta||_2 \lt R,\; where\\
   R = \frac{\sigma}{2}(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B}))
   $$
   
2. 第二个Theorem比较有意思，它说明了安全半径的意义是什么。需要注意的是，作者并不是声明了一个“传统的安全半径”——如果你施加的噪声大于安全半径，那么一定会分类错误。在这里作者声明：如果施加的攻击噪声大于安全半径，那么所有满足上述条件的Base Classifier一定有一个会分类错误——并不一定是我们手上的这个。所以这个说法就有点鸡肋了，他的这个safe radius一定程度上还是比较保守的，也就是说他是针对所有满足第一个Theorem的Base Classifier下的一个定论，而不是针对我们手上这个。

### 实验

那么问题在于怎么计算安全半径呢，上边提到了利用蒙特卡洛随机采样的方法来计算。具体的算法分为两部分，PREDICT和CERTIFY。

* PREDICT部分实际上就是Smoothed Classifier工作的部分。给定任何一个输入图片，我们对这个输入图片施加n次高斯噪声并作分类，分类结果投票最高的一项和次高的一项，我们做二项零假设检验，检验通过后，得票最高的那一项将是最后输出的结果，否则显示分类失败。

  > 零假设检验在这里说的是，我作出一个错误的假设，通过证明这个假设是错误的来证明目标结论是正确的。这里的假设是，得票最高的一项和次高的一项概率是相等的（都是0.5），如果我们能证明这个假设非常荒谬（在给定的容错率$\alpha$下），那么的票最高的一项的概率就应远大于次高的一项。具体操作就是仿真实验，零假设已经集成在python库中了。

  然而仅仅有PREDICT我们没办法证明Smoothed Classifier确实是对**所有**给定$l_2$范围$r$内的攻击都有效，因而接下来的CERTIFY部分就是为了证明这件事，即对抗算法确实是Certifiable的。

* CERTIFY相比PREDICT而言，除了返回预测的结果以外，还返回一个针对该输入的安全半径$R$，只有返回的安全半径$R$比预设的攻击半径$r$大的时候，我们才认为这次防御是有效的。这里计算时我们直接采用$\underline{p_A}=1-\overline{p_B}$来简化$R$的表达和计算：$R=\sigma \Phi^{-1}(\underline{p_A})$。

### 对训练过程的要求

理论上来说Adversarial部分和model应该是完全解耦的，就是说对抗部分应该不影响原模型的结构和训练过程，但是本方法经过测试，发现只有在训练时也加入一定高斯噪声才能将对抗算法达到最好的效果，因此该方法对原模型的训练过程还是有一定要求的，并不是完全的Non-Intrusive。

