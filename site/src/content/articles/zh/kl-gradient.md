---
title: "无偏的 KL 估计，为什么给不出正确的梯度？从常见的 KL Estimator 到异步 RL 的 trainer-generator 约束"
titleAlt: "Why Does an Unbiased KL Estimate Not Give the Right Gradient? From Common KL Estimators to the Trainer-Generator Constraint in Asynchronous RL"
date: 2026-11-11
category: post-training
series: grpo-line
part: 7
tags: ["GRPO"]
---

# 无偏的 KL 估计，为什么给不出正确的梯度？从常见的 KL Estimator 到异步 RL 的 trainer-generator 约束

<link rel="stylesheet" href="/css/kl-gradient.css">

我们之前写过一篇[读《Approximating KL Divergence》的笔记](https://normaluhr.github.io/zh/p/kl/)，讨论了 $k_1$、$k_2$、$k_3$ 这三个估计器在估计 KL 散度的数值时，各自有多大的偏差和方差。这篇文章是那篇笔记的续篇。在训练里，KL 的数值大多只用来记录，或者用来调节 KL 的权重 $\beta$。KL 真正影响模型更新的方式，是我们把它写进 loss 或者 reward 以后，它在 backward 中产生的梯度。一个估计器能把 KL 的数值估计得很准，并不代表它的梯度指向我们想要的方向。这篇文章就从这个问题讲起。

文章想回答两个问题。第一个问题是，常见的几种 KL 写法在 backward 之后分别产生了什么梯度，这个梯度对应的又是哪一个方向的 KL。第二个问题来自异步 RL。在异步 RL 里，生成数据的 generator 比正在训练的 trainer 落后若干步，我们在[介绍 Score Function Centering 加 KL 的文章](https://normaluhr.github.io/zh/p/score-function-centering-kl/)里的做法，是加一个 KL，把 trainer 拉向 generator。这个 KL 有好几种写法，我们想弄清楚每一种写法在做什么，又有什么代价。第一个问题的答案，会成为我们回答第二个问题的工具。最后，KL 和它的系数都是在有限精度下算出来的，我们也会讨论计算时需要注意的地方。

对第二个问题，这篇文章不会给出唯一的答案。不同的写法分别防范不同方向的失败，而哪一种失败更值得担心，在不同的设置下可能并不一样。我们会说明每一种写法在做什么、防范哪一类失败、要付出什么代价，具体怎么选，留给读者结合自己的设置来判断。

## 阅读前的准备

阅读这篇文章需要了解 PPO 或者 GRPO 的 loss 长什么样。下一节的前情回顾整理了 KL 的三个估计器，第 1 章开始就会用到它们。异步 RL 和 Score Function Centering 的背景，要到第 4 章才会用到，我们会在那里再回顾。

如果你遇到过下面这些情况，这篇文章可能会有帮助：

* 在 GRPO 的 loss 里用了 $k_3$，但不确定它实际优化的是哪个方向的 KL。
* 想弄清楚 KL 放在 reward 里和放在 loss 里有什么区别。
* 在做异步 RL，用了 trainer 和 generator 之间的 KL，想知道不同的写法会把模型拉向哪里。
* 在记录或者使用 KL 时，遇到过数值上说不通的结果，比如本该非负的 KL 算出了负数。

第 1 章先说明，为什么直接对 KL 的估计值调用 backward，得到的一般不是 KL 的梯度，并用一个词表只有两个 token 的例子，算出漏掉的那一项会带来多大的差别。为了不必每次都这样逐项计算，第 2 章把每一种 KL 写法的梯度都写成"一个系数乘以 score"，从系数就能读出 $k_1$、$k_2$、$k_3$ 当 loss 时对应的是哪个方向的 KL。同一个系数，放在 reward 里和放在 loss 里的效果也不一样，第 3 章讨论这个区别。前三章都假设样本来自当前模型，第 4 章去掉这个假设，先回顾异步 RL 和 Score Function Centering 的背景，再分别讨论 GRPO 一批数据更新多次和异步 RL 这两种情况。在异步 RL 里，拉近 trainer 和 generator 的写法不止一种，第 5 章把它们都写成对 advantage 的修改，第 6 章再比较它们怎样约束概率正在下降和正在上升的 token、最终会把分布拉到哪里，以及各自要付出什么代价。最后，第 7 章讨论在有限精度下怎样计算 KL 和它的系数，第 8 章是总结。

文中的图都可以交互。把鼠标移到图上可以读出数值，点击图例可以隐藏某一条曲线，有滑块的图可以拖动滑块改变参数。

## 前情回顾

为了让这篇文章可以单独阅读，这里先回顾后面要用到的三个估计器。

假设我们想估计 $\mathrm{KL}(q\|p)=\mathbb E_{x\sim q}[\log q(x)-\log p(x)]$，但手里只有从 $q$ 采出来的样本。这里的 $q$ 是采样用的分布，$p$ 是另一个分布。记 $r=p(x)/q(x)$，John Schulman 在他的博客[《Approximating KL Divergence》](http://joschu.net/blog/kl-approx.html)里给出了三个估计器：

$$
k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=(r-1)-\log r.
$$

$k_1$ 是无偏的，但方差较大。$k_2$ 的方差小，代价是有偏。$k_3$ 在 $k_1$ 上加了一个期望为 0 的控制变量 $r-1$，所以它既保持了无偏，方差又小。这三个估计器对任何两个分布都可以使用，后面用到它们时，我们会说明哪个分布处在 $q$ 的位置上，哪个分布处在 $p$ 的位置上。

## 1. 直接对 KL 的估计值求导，为什么得不到 KL 的梯度？

> **TL;DR：** KL 是一个期望，$\theta$ 既出现在被平均的估计值里，也出现在采样的分布里。直接对 KL 的估计值做 backward，只会对估计值本身求导，采样分布随 $\theta$ 变化带来的那一部分梯度会被漏掉，而且增加样本数也补不回来。所以一个估计器对 KL 的数值无偏，并不保证它当 loss 时给出正确的梯度。

这一章先说明，直接对 KL 的估计值调用 backward，得到的一般不是 KL 的梯度，原因是有一项梯度被漏掉了。然后我们用一个词表只有两个 token 的例子，算出漏掉这一项会带来多大的差别。在此之前，我们先约定第 1 章到第 3 章使用的记号。

为了简化，我们固定一个前缀（prefix），只看下一个 token 的分布。记模型在这个前缀上输出的 logit 为 $z$，它是一个长度等于词表大小的向量。正在训练的模型记为 $\pi_\theta$，它给 token $y$ 的概率是 $\pi_\theta(y)=\mathrm{softmax}(z)_y$。KL 正则里的参考分布是 reference model，记为 $\pi_{\text{ref}}$。$\pi_{\text{ref}}$ 一般是 RL 开始之前的模型，比如 SFT 之后的模型，它在整个训练过程中固定不变，所以除了训练刚开始的时候，它和 $\pi_\theta$ 并不相同。对一个 token $y$，记它的 log ratio 为

$$
\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y).
$$

把 score 记为 $s_\theta(y)=\nabla_\theta\log\pi_\theta(y)$，也就是 token $y$ 的 log 概率对模型参数的梯度。后面的例子只对 logit $z$ 求导，这时 score 有一个简单的形式。记 $e_y$ 为 token $y$ 的 one-hot 向量，也就是只在 $y$ 的位置上等于 1、在其余位置上都等于 0 的向量，再把 $\pi_\theta$ 在整个词表上的概率也排成一个向量，仍然记为 $\pi_\theta$。由于 $\log\pi_\theta(y)=z_y-\log\sum_v\exp(z_v)$，它对 $z_v$ 的偏导数在 $v=y$ 时是 $1-\pi_\theta(y)$，在 $v\neq y$ 时是 $-\pi_\theta(v)$。把这些偏导数排成一个向量，就是

$$
\frac{\partial\log\pi_\theta(y)}{\partial z}=e_y-\pi_\theta.
$$

KL 散度有两个方向：

$$
\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=\mathbb E_{y\sim\pi_\theta}[\rho(y)],\qquad
\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{ref}}}[-\rho(y)].
$$

在 RLHF 和 GRPO 里，我们通常想约束的是第一个。本文沿用[之前介绍 Score Function Centering 加 KL 的文章](https://normaluhr.github.io/zh/p/score-function-centering-kl/)的叫法，把参照分布在前、$\pi_\theta$ 在后的 KL 叫作 reverse KL，例如 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$，把 $\pi_\theta$ 在前的 KL 叫作 forward KL，例如 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$。有一些 RLHF 的论文和代码库的叫法正好相反，文末的附录专门列出了两种叫法的对应关系。

在前情回顾的三个公式里，$q$ 是采样用的分布，$p$ 是另一个分布。这里我们要估计的是 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$，样本来自 $\pi_\theta$，所以 $\pi_\theta$ 处在 $q$ 的位置上，$\pi_{\text{ref}}$ 处在 $p$ 的位置上。于是 $r=\pi_{\text{ref}}(y)/\pi_\theta(y)$，$\log r=-\rho(y)$。代入以后，三个估计器写成 $\rho$ 的函数分别是

$$
k_1=\rho,\qquad k_2=\tfrac12\rho^2,\qquad k_3=e^{-\rho}-1+\rho.
$$

第 1 章到第 3 章都使用这种写法，只用 $\rho$，不再单独使用 $r$。

### 1.1 backward 漏掉了哪一项？

为了同时讨论三个估计器，我们用 $f_\theta(y)$ 表示其中任意一个在样本 $y$ 上的值，比如用 $k_1$ 时，$f_\theta(y)=\rho(y)$。下标 $\theta$ 表示它依赖模型参数。对 $k_1$ 和 $k_3$ 这样的无偏估计器，KL 就等于 $\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]$，所以我们想要的 KL 的梯度，就是这个期望的梯度。把期望写成求和，

$$
\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]=\sum_y\pi_\theta(y)\,f_\theta(y).
$$

式子里的 $\theta$ 出现在两个地方。一个是估计器 $f_\theta(y)$ 本身，另一个是作为权重的 $\pi_\theta(y)$，也就是 token $y$ 被采到的概率。对乘积求导会得到两项。对 $f_\theta(y)$ 求导的那一项，下文称为估计器项。对 $\pi_\theta(y)$ 求导的那一项，用 $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$ 改写以后，下文称为采样项。这样就得到

$$
\nabla_\theta\,\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]=\underbrace{\mathbb E_{y\sim\pi_\theta}\big[\nabla_\theta f_\theta(y)\big]}_{\text{估计器项}}+\underbrace{\mathbb E_{y\sim\pi_\theta}\big[f_\theta(y)\,s_\theta(y)\big]}_{\text{采样项}}.
$$

常见的实现是这样计算 KL 这一项的。先从 $\pi_\theta$ 采出 $N$ 个 token $y_1,\dots,y_N$，再算出每个 token 的 $f_\theta(y_i)$，取平均得到 $\frac1N\sum_i f_\theta(y_i)$，乘上 $\beta$ 加进 loss，最后调用 backward。backward 得到的梯度是 $\frac1N\sum_i\nabla_\theta f_\theta(y_i)$，它的期望正是上式的估计器项。

采样项来自对权重 $\pi_\theta(y)$ 求导。在代码里，这个权重并没有作为一个数出现在 loss 中，它的作用是由采样来完成的，概率越大的 token 被采到的次数越多，在平均值里占的分量也就越大。而采样这一步输出的是 token 的编号。$\theta$ 变化一点点时，采到的编号要么不变，要么突然跳到另一个 token，所以单次采样的结果对 $\theta$ 的导数要么是 0，要么不存在，自动求导也不会经过这一步。$\theta$ 对各个 token 被采到的频率的影响，只体现在许多次采样的平均里，任何一次 backward 都看不到它。因此，backward 只会对 $f_\theta(y_i)$ 这个式子求导，并把采到的 $y_i$ 当成常数，采样项也就这样被漏掉了。需要说明的是，$f_\theta(y)$ 里也可能含有 $\pi_\theta(y)$，比如 $k_1$ 就是 $\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$，backward 会对这一处求导，它属于估计器项。被漏掉的只是作为权重的那个 $\pi_\theta(y)$。

采样项的形式和 policy gradient 完全一样。如果把 $f_\theta(y)$ 看成 token $y$ 拿到的 reward $R(y)$，采样项就是 $\mathbb E_{y\sim\pi_\theta}[R(y)\nabla_\theta\log\pi_\theta(y)]$，也就是 REINFORCE 的梯度。policy gradient 要解决的，正是"采样分布依赖 $\theta$ 时，怎样求期望的梯度"这个问题。对这个推导不熟悉的读者，可以参考[介绍 Score Function Centering 的文章](https://normaluhr.github.io/zh/p/score-function-centering/)的 1.1 节。所以，直接对 KL 的估计值求导，相当于只算了估计器本身的梯度，而把采样分布的变化带来的梯度整个丢掉了。

由于被漏掉的是一整项期望，增加样本数量只能让估计器项估计得更准，不能把采样项找回来，所以这里的偏差和样本数量无关。Tang 和 Munos（2025）以及 Zhang 等人（2025）都详细讨论过这个问题，并且指出不少开源实现都受它影响。

### 1.2 漏掉这一项，差别有多大？

下面在一个词表只有两个 token 的例子里，把三个梯度精确地算出来，看看漏掉采样项会带来多大的差别。

设词表里只有 A 和 B 两个 token，当前模型 $\pi_\theta(A)=0.7$，参考分布 $\pi_{\text{ref}}(A)=0.5$。为了简洁，记 $p_A=\pi_\theta(A)$，$p_B=\pi_\theta(B)$。我们比较三个量，它们都是对 A 的 logit $z_A$ 求导的结果，并且都对采样取了精确的期望。

第一个量是 forward KL 本身对 $z_A$ 的导数，也就是我们想要的结果。只有两个 token 时，$\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=p_A\rho(A)+p_B\rho(B)$。由上面的 $e_y-\pi_\theta$，$\log p_A$ 对 $z_A$ 的导数是 $1-p_A=p_B$，$\log p_B$ 对 $z_A$ 的导数是 $-p_A$。所以 $\rho(A)$ 和 $\rho(B)$ 对 $z_A$ 的导数分别是 $p_B$ 和 $-p_A$，而 $p_A$ 和 $p_B$ 本身的导数等于它们自己乘以对应的 log 概率的导数，分别是 $p_Ap_B$ 和 $-p_Ap_B$。对两个乘积分别求导以后，对 $\rho$ 求导的两项是 $p_A\cdot p_B+p_B\cdot(-p_A)=0$，正好抵消，这两项就是 1.1 节的估计器项。剩下的是对 $p_A$ 和 $p_B$ 求导的两项，也就是采样项，

$$
\frac{\partial}{\partial z_A}\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=p_Ap_B\big(\rho(A)-\rho(B)\big)=0.7\times0.3\times\ln\frac{0.7/0.5}{0.3/0.5}=0.21\times\ln\frac73\approx0.178.
$$

第二个量是把 $k_1$ 当成 loss 时，backward 得到的梯度的期望。$k_1=\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$，它对 $z_A$ 的导数就是 $\partial\log\pi_\theta(y)/\partial z_A$。由 $e_y-\pi_\theta$ 可知，采到 A 时这个导数是 $1-0.7=0.3$，采到 B 时是 $-0.7$。按 $\pi_\theta$ 取期望，结果是 $0.7\times0.3+0.3\times(-0.7)=0$。

第三个量是把 $k_3$ 当成 loss 时，backward 得到的梯度的期望。$k_3=e^{-\rho}-1+\rho$，它对 $z_A$ 的导数是 $(1-e^{-\rho})\,\partial\log\pi_\theta(y)/\partial z_A$，其中 $e^{-\rho}=\pi_{\text{ref}}(y)/\pi_\theta(y)$。采到 A 时，导数是 $(1-0.5/0.7)\times0.3\approx0.086$。采到 B 时，导数是 $(1-0.5/0.3)\times(-0.7)\approx0.467$。按 $\pi_\theta$ 取期望，结果是 $0.7\times0.086+0.3\times0.467=0.2$。

$k_1$ 和 $k_3$ 作为 KL 数值的估计都是无偏的，但把它们直接当成 loss，得到的期望梯度分别是 0 和 0.2，都不是 forward KL 的 0.178。这三个数都是精确的期望，所以不管 batch 有多大，这个差别都会存在。$k_1$ 的情况最极端，它的梯度平均下来是 0，也就是说这一项 KL 根本不会把模型往参考分布那边拉。

这个差别并不是 0.7 这个取值碰巧造成的。图 1 让 $\pi_\theta(A)$ 取遍 0 到 1 之间的值，画出了同样的几个量，图中的竖线标出的就是上面的例子。拖动滑块可以改变 $\pi_{\text{ref}}(A)$。图里还画了 $k_2$ 当 loss 的结果和 reverse KL 的梯度，2.2 节会解释它们为什么分别和 forward KL、$k_3$ 的曲线重合。

<figure data-chart="kl-two-token">
<noscript><p>图 1 需要 JavaScript 才能显示。</p></noscript>
</figure>
<script src="/js/kl-gradient-charts.js" defer></script>

$k_3$ 给出的 0.2 正好等于 $\pi_\theta(A)-\pi_{\text{ref}}(A)$。下一章会给出一个统一的方法，用它可以直接看出每一种写法对应的是哪个梯度，2.2 节会用它说明 0.2 是从哪里来的。

## 2. $k_1$、$k_2$、$k_3$ 当 loss 时，各自给出的是什么梯度？

> **TL;DR：** 把每一种 KL 写法在单个样本上的梯度写成"一个系数乘以 score"以后，比较系数就能判断它优化的是哪个 KL。样本来自当前模型时，$k_1$ 当 loss 的期望梯度为 0，$k_2$ 给出 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ 的梯度，也就是本文所说的 forward KL，$k_3$ 给出 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$ 的梯度，也就是 reverse KL。因此，InstructGPT 式的 PPO 把 $k_1$ 放进 reward，优化的是 forward KL，而 GRPO 把 $k_3$ 放进 loss，在样本来自当前模型的近似下，优化的是 reverse KL。

第 1 章的三个数是一个一个算出来的。这一章给出一个统一的方法，把每一种 KL 写法在单个样本上的梯度，都写成一个系数乘以 score。2.1 节介绍这个方法，2.2 节用它读出 $k_1$、$k_2$、$k_3$ 当 loss 时分别对应什么梯度，2.3 节再用这些结论比较 PPO 和 GRPO 里的 KL。

### 2.1 怎样从一个估计器读出它的梯度？

$k_1$、$k_2$、$k_3$ 都只依赖 $\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$，所以可以统一写成 $f(\rho)$。$\pi_{\text{ref}}$ 不随 $\theta$ 变化，因此 $\nabla_\theta\rho(y)=s_\theta(y)$。对于任何形如 $f(\rho)$ 的写法，单个样本上 backward 得到的梯度都是

$$
\nabla_\theta f(\rho(y))=f'(\rho(y))\,s_\theta(y).
$$

这个梯度就是一个标量系数 $f'(\rho)$ 乘以 score。对 loss 做梯度下降时，系数为正会降低这个样本的 log 概率，系数为负会提高它，系数的绝对值决定了变化的幅度。比较不同的写法，就是比较它们的系数。

样本来自当前模型时，系数为 $c(y)$ 的梯度，期望是 $\mathbb E_{y\sim\pi_\theta}[c(y)\,s_\theta(y)]$。score 的期望为 0，所以给 $c$ 加上任何一个和 $y$ 无关的常数，都不会改变这个期望。这也是 policy gradient 里可以减去 baseline 的原因，我们在[介绍 Score Function Centering 的文章](https://normaluhr.github.io/zh/p/score-function-centering/)的 2.1 节详细讨论过这一点，后面会反复用到。

再看 forward KL 的梯度应该是什么。在 1.1 节的公式里取 $f_\theta=\rho$，估计器项是 $\mathbb E[\nabla_\theta\rho]=\mathbb E[s_\theta]=0$，于是 forward KL 的梯度只剩下采样项 $\mathbb E[\rho\,s_\theta]$。也就是说，样本来自当前模型时，系数等于 $\rho$，或者 $\rho$ 加上任何常数，给出的就是 forward KL 的准确梯度。下一节会用这个标准来检查三个估计器。

这里说的"准确"指的是期望。系数加减一个常数不改变期望，但会改变梯度的方差。和 policy gradient 里减去 baseline 一样，我们可以从系数里减去一个和当前采到的 token 无关的量，来降低梯度的方差，5.1 节的 $k_4$ 也会用到这个想法。需要区分的是，[之前的笔记](https://normaluhr.github.io/zh/p/kl/)讨论的方差，指的是 KL 的估计值在样本之间的波动，它和这里说的梯度的方差并不是同一个量。

### 2.2 $k_1$、$k_2$、$k_3$ 的系数分别是什么？

先看 $k_1$。$k_1=\rho$，系数是 $f'(\rho)=1$，期望梯度是 $\mathbb E_{y\sim\pi_\theta}[s_\theta(y)]=0$。$k_1$ 作为 KL 数值的估计是无偏的，但作为 loss，它的梯度期望恒为 0。在单个 batch 上，它的梯度并不是 0，可是这些梯度平均下来不指向任何方向，只会给训练增加噪声。这就是 1.2 节里 $k_1$ 给出 0 的原因。

再看 $k_2$。$k_2=\tfrac12\rho^2$，系数是 $f'(\rho)=\rho$。按照 2.1 节的标准，它给出的就是 forward KL 的准确梯度。在 1.2 节的例子里，采到 A 时 $\rho=\ln(0.7/0.5)\approx0.336$，采到 B 时 $\rho=\ln(0.3/0.5)\approx-0.511$，乘上各自对 $z_A$ 的导数，再按 $\pi_\theta$ 取期望，得到的正好是 0.178。图 1 里 $k_2$ 的虚线和 forward KL 的曲线完全重合，也是这个原因。

这一点和 $k_2$ 作为数值估计时的表现正好相反。$k_2$ 作为 KL 数值的估计是有偏的，Schulman 的博客说明，在两个分布很接近时，它的期望和 KL 只在二阶以内相同。但是作为 loss，在样本来自当前模型的时候，$k_2$ 的梯度却是准确的。原因是 forward KL 的完整梯度只剩下采样项 $\mathbb E[\rho\,s_\theta]$，估计器项的期望是 0，而 $k_2$ 做 backward 以后，在单个样本上给出的恰好就是 $\rho\,s_\theta$。也就是说，$k_2$ 自己的 backward 恰好算出了 $k_1$ 漏掉的那一项。

最后看 $k_3$。$k_3=e^{-\rho}-1+\rho$，系数是

$$
f'(\rho)=1-e^{-\rho}=1-\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}.
$$

它的期望梯度是

$$
\mathbb E_{y\sim\pi_\theta}\Big[\Big(1-\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}\Big)s_\theta(y)\Big]=\mathbb E_{y\sim\pi_\theta}[s_\theta(y)]-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)]=-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)].
$$

而 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{ref}}}[\log\pi_{\text{ref}}(y)-\log\pi_\theta(y)]$ 的梯度也是 $-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)]$。因此 $k_3$ 当 loss 时，期望梯度是 reverse KL 的梯度。回到 1.2 节的例子，reverse KL 对 logit 的梯度是 $\pi_\theta-\pi_{\text{ref}}$，它在 A 上的分量是 $\pi_\theta(A)-\pi_{\text{ref}}(A)=0.2$，这就是 $k_3$ 给出的那个数。在图 1 里，$k_3$ 的虚线和 reverse KL 的曲线完全重合。

在 $\rho$ 接近 0 的时候，$1-e^{-\rho}\approx\rho-\rho^2/2$，和 $k_2$ 的系数一阶相同，所以 Liu 等人（2025）把 $k_3$ 当 loss 看作 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ 的一阶近似。这和"它是 reverse KL 的梯度"并不矛盾。模型离参考分布不远时，两个方向的 KL 梯度本来就很接近，离得越远，它们的差别越大。图 1 也能看出这一点，两条曲线在 $\pi_\theta(A)=\pi_{\text{ref}}(A)$ 的位置附近几乎重合，离这个位置越远，分开得越多。

$k_3$ 的系数在 $\rho$ 的两侧并不对称，图 2 画出了三个估计器在单个样本上的系数。在 $\rho>0$ 的一侧，也就是模型给某个 token 的概率高于参考分布时，$k_3$ 的系数最大只能到 1。在 $\rho<0$ 的一侧，也就是模型给某个 token 的概率低于参考分布时，系数的绝对值按 $e^{-\rho}$ 指数增长。比如 $\rho=2$ 时，$k_2$ 的系数是 2，$k_3$ 的系数是 0.865。$\rho=-2$ 时，$k_2$ 的系数是 $-2$，$k_3$ 的系数是 $1-e^2\approx-6.39$。

<figure data-chart="kl-coef">
<noscript><p>图 2 需要 JavaScript 才能显示。</p></noscript>
</figure>

要注意，这是单个样本上的系数。对一个 $\rho$ 远小于 0 的 token $v$，它在 $\pi_\theta$ 下很少被采到。乘上它被采到的概率以后，它的 logit 上的期望梯度是 $\pi_\theta(v)-\pi_{\text{ref}}(v)$，梯度下降会把它往回拉，但拉的幅度不会超过 $\pi_{\text{ref}}(v)$。所以指数增长的系数在期望上并没有带来很大的拉力，它带来的主要是方差，偶尔采到一个这样的 token，就会产生一个很大的梯度。4.1 节会看到，[DeepSeek-V3.2 的技术报告](https://arxiv.org/abs/2512.02556)对 $k_3$ 的批评说的就是这一点。拉力在两侧各有多大，第 6 章还会在 generator 的设定下专门分析。

### 2.3 PPO 和 GRPO 的 KL 有什么不同？

除了当 loss，KL 还可以放进 reward。InstructGPT（Ouyang 等人，2022）式的 PPO 就是这样做的，它把每个 token 的 $\rho$ 乘上 $\beta$，从这个 token 的 reward 里减掉。这时 $\rho$ 被当作 reward 的一部分，不参与求导，它对梯度的影响通过 policy gradient 体现，也就是作为系数乘在 score 前面。KL 这一部分的系数是 $\beta\rho$，所以它在期望上和 $k_2$ 当 loss 相同，给出的也是 forward KL 的梯度。这个结论只在单个位置上成立，放到整条序列上两者有什么区别，第 3 章会讨论。

有了这些结论，我们可以重新看 PPO 和 GRPO 里的 KL。我们在[之前的笔记](https://normaluhr.github.io/zh/p/kl/)里写过这样一句话："$k_3$ 正是 GRPO 在 KL 散度估计上区别于 PPO 的一大特点（PPO 使用的是 $k_1$）"。现在可以把这句话说得更完整一些。InstructGPT 式的 PPO 把 $k_1$ 放进 reward，它的梯度对应 forward KL。GRPO（Shao 等人，2024）把 $k_3$ 放进 loss，在样本来自当前模型的近似下，它的梯度对应 reverse KL。除了估计器不同以外，两者优化的 KL 方向也不同。2.2 节说过，模型离 reference model 不远时，两个方向的梯度很接近，所以这个差别在实践中的影响可能并不大。[DeepSeek-V3.2 的技术报告](https://arxiv.org/abs/2512.02556)后来修正了 GRPO 的这个 $k_3$ 项，4.1 节会介绍修正的方式。

到这里，我们讨论的都是单个位置上的梯度。$k_1$ 放进 reward 和 $k_2$ 放进 loss 在单个位置上期望相同，但在实际训练里，这两种放法仍然有区别，下一章就讨论这个区别。

## 3. KL 放在 reward 里和放在 loss 里，有什么区别？

> **TL;DR：** KL 放进 reward 时，会和任务 reward 一起经过 advantage 的标准化和 PPO 的 clip，所以同一个 $\beta$ 在不同 prompt 上实际起的作用并不相同。它还会通过 advantage 传给前面的 token，这样得到的是整条序列 KL 的准确梯度，但方差更大。KL 放进 loss 时，$\beta$ 就是这一项实际的权重，每个 token 只约束自己所在位置上的分布，方差较小，但得到的不是整条序列 KL 的梯度。

把 KL 放进 reward，是指把 $\beta\rho_t$（通常是 detach 之后的值）从第 $t$ 个 token 的 reward 里减掉，然后和任务 reward 一起计算 advantage，PPO 的经典实现就是这样做的。把 KL 放进 loss，是指 advantage 只由任务 reward 计算，KL 作为一个单独的、可以求导的项加进 loss，比如 $L=L_{\text{PG}}+\beta\cdot\mathrm{mean}_t\,k(\rho_t)$，GRPO 的做法属于这一种。

2.3 节说明，在单个位置上，$k_1$ 放进 reward 和 $k_2$ 放进 loss 的期望梯度相同。但这两种放法在实际训练中仍然有两个区别。第一个区别是，放进 reward 以后，KL 会和任务 reward 一起经过 advantage 的各种处理。第二个区别是，放进 reward 以后，KL 会沿着序列传给前面的 token。下面两节分别讨论。这两个区别在后半篇也会用到，第 5 章会看到，在 Score Function Centering 下，拉近 trainer 和 generator 的 KL 都可以写成对 advantage 的修改，那时同样要注意这两个问题。

### 3.1 放进 reward 以后，KL 会被怎样重新缩放？

把 KL 放进 reward 以后，advantage 里同时包含了"任务做得好不好"和"离参考分布近不近"两种信息，一个回答可能只是因为离参考分布更近，就拿到了更高的 advantage。除此之外，advantage 之后的每一步处理，也都会作用在 KL 上。比如在 reward 里加了 KL，又像 GRPO 那样按组的标准差做标准化，那么每个 prompt 上的 KL 都会被除以这个 prompt 所在组的标准差。组内 reward 差别小的 prompt 标准差也小，KL 会被放大，而组内 reward 差别大的 prompt，KL 会被缩小，所以同一个 $\beta$ 在不同的 prompt 上实际起的作用并不相同。PPO 的 clip 也会作用在 KL 上，如果一个 token 的梯度被 clip 掉了，它的 KL 那一部分的梯度也会一起消失。

把 KL 放进 loss 时，这些处理都不会影响 KL，$\beta$ 就是 KL 这一项实际的权重。

### 3.2 放进 reward 以后，KL 为什么会传给前面的 token？

先回顾一下 advantage 是怎样从每个 token 的 reward 算出来的。这里把第 $t$ 个 token 上的 reward 记为 $R_t$，把 critic 对第 $t$ 步状态的价值估计记为 $V(s_t)$。GAE（Generalized Advantage Estimation）先算出每一步的 TD 残差 $\delta_t=R_t+\gamma V(s_{t+1})-V(s_t)$，再把这一步和之后各步的残差按 $\gamma\lambda$ 的幂次加权求和，

$$
A_t=\sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}.
$$

其中 $\gamma$ 是折扣因子，$\lambda$ 控制 advantage 往后看多远，$\lambda=0$ 时只看一步，$\lambda$ 越接近 1，看得越远。我们在[介绍 GRPO 的文章](https://normaluhr.github.io/zh/p/grpo/)的 8.2 节详细介绍过 GAE。取 $\gamma=\lambda=1$ 时，相邻两步的 $V$ 两两抵消，$A_t=\sum_{k\ge t}R_k-V(s_t)$，也就是从第 $t$ 个 token 开始到序列结束的所有 reward 之和，再减去一个 baseline。

把 KL 放进 reward 时，第 $k$ 个 token 的 reward 里多了一项 $-\beta\rho_k$。按上面的式子，第 $t$ 个 token 的 advantage 里就多了 $-\beta\sum_{k\ge t}\rho_k$，它包含了从第 $t$ 个 token 开始到序列结束的所有 token 的 KL。用 2.1 节的说法，KL 这一部分在第 $t$ 个 token 上的系数是

$$
\beta\sum_{k\ge t}\rho_k.
$$

这个系数对应的是整条序列的 forward KL 的梯度。原因和 1.1 节的采样项一样。第 $t$ 个 token 的选择会改变之后所有 token 所处的前缀，而这些前缀出现的概率也依赖 $\theta$，一个早期的 token 因此要对它之后产生的所有 KL 负责。

把 KL 放进 loss 时，每个 token 的系数只取决于它自己的 $\rho_t$，比如 $k_2$ 的系数是 $\rho_t$，$k_3$ 的系数是 $1-e^{-\rho_t}$。这相当于在每个固定的前缀上分别约束下一个 token 的分布，但不考虑当前 token 对后面前缀的影响。Tang 和 Munos（2025）把这种做法得到的梯度称为部分梯度。

所以，$k_1$ 放进 reward 和逐个 token 的 $k_2$ 放进 loss，只在单个位置上等价。放到整条序列上，前者的系数是从当前 token 开始到序列结束的所有 token 的 $\rho$ 之和，它给出的是整条序列 KL 的准确梯度，但序列越长，这个和的波动越大，梯度的方差也就越大。后者每个 token 只用自己的 $\rho_t$，方差小，但它不是整条序列 KL 的梯度。

如果想在 loss 里得到整条序列 KL 的梯度，可以把 $k_2$ 用在整条序列的 log ratio 之和上，也就是 $\tfrac12\big(\sum_t\rho_t\big)^2$。它的梯度是 $\big(\sum_k\rho_k\big)\sum_t s_\theta(y_t)$，第 $t$ 个 token 的 score 乘上了所有位置的 $\rho_k$。其中 $k<t$ 的那些 $\rho_k$ 在采第 $t$ 个 token 之前就已经确定了，和这个 token 的 score 相乘以后期望为 0，所以它在期望上和 $k_1$ 放进 reward 相同，只是方差还要更大一些。

前三章都假设样本来自当前模型 $\pi_\theta$。GRPO 一批数据更新多次时，以及在异步 RL 里，这个条件都不成立。下一章讨论样本不来自当前模型时会发生什么。

## 4. 样本不来自当前模型时，会发生什么？

> **TL;DR：** 这一章去掉"样本来自当前模型"这个假设。GRPO 一批数据更新多次时，样本来自 $\pi_{\text{old}}$，$k_3$ 当 loss 的梯度既不是 forward KL 的梯度，也不是 reverse KL 的梯度。DeepSeek-V3.2 把 $k_3$ 乘上参与求导的比率 $\pi_\theta/\pi_{\text{old}}$，得到的才是 forward KL 的准确梯度。在异步 RL 里，样本来自固定的 generator，采样分布不依赖 $\theta$，没有任何一项会被漏掉，所以直接对 $k_1$ 或 $k_3$ 做 backward，得到的就是 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的准确梯度。

前三章的 KL 都以 reference model $\pi_{\text{ref}}$ 为参照，样本也都来自当前模型 $\pi_\theta$。从这一章开始，样本不再来自 $\pi_\theta$。在继续之前，我们先把文中的四个分布放在一起对照一下。

* $\pi_\theta$ 是正在训练的模型，每一步都在更新。
* $\pi_{\text{ref}}$ 是 reference model，一般是 RL 开始之前的模型，在整个训练过程中固定不变。PPO 和 GRPO 的 KL 正则以它为参照，限制模型离初始模型有多远。
* $\pi_{\text{old}}$ 是 PPO 和 GRPO 里生成当前这批数据的模型，也就是这批数据的 generator。
* $\pi_{\text{generator}}$ 是异步 RL 里生成数据的模型。它和 $\pi_{\text{old}}$ 的角色相同，只是落后得更多。

这几个分布之间的关系，取决于训练的方式。在完全 on-policy 的同步训练里，每批数据只更新一次，更新时的 $\pi_\theta$ 就是刚刚生成这批数据的模型，所以 $\pi_{\text{old}}$ 和 $\pi_\theta$ 的参数相同，区别只在于 $\pi_{\text{old}}$ 不参与求导。$\pi_{\text{ref}}$ 则不一样，它从训练开始就固定不变，而 $\pi_\theta$ 一直在更新，所以即使在同步训练里，两者也只在训练刚开始时相同。GRPO 一批数据更新多次时，从第二次更新开始，$\pi_\theta$ 已经比 $\pi_{\text{old}}$ 多更新了几步，两者不再相同。在异步 RL 里，generator 一边生成数据，trainer 一边更新，$\pi_{\text{generator}}$ 会落后 $\pi_\theta$ 几步甚至几百步。

异步 RL 里的这种落后，正是 Score Function Centering 要处理的问题。从 4.2 节开始，后面几章都在这个设定下讨论，所以这里先回顾它的做法，以及我们在它之外加的 KL。我们之前的文章把 trainer 和 generator 分别记为 $p_\theta$ 和 $q_\phi$，这篇文章记为 $\pi_\theta$ 和 $\pi_{\text{generator}}$。

**Score Function Centering。** policy gradient 里的 score $\nabla_\theta\log\pi_\theta(y)$，只有在 $y$ 来自 $\pi_\theta$ 时期望才为 0，这是 advantage 里可以减去 baseline 的前提。当 $y$ 来自 $\pi_{\text{generator}}$ 时，这个期望不再是 0，policy gradient 里就会多出一项。这一项等于平均 advantage 乘以 score 在 $\pi_{\text{generator}}$ 下的期望，它在 logit 上的方向是 $\pi_{\text{generator}}-\pi_\theta$，大小和正负都由平均 advantage 决定，而平均 advantage 并不反映哪个 token 更好。Score Function Centering（下文简称 SFC）把 score 改成以 $\pi_{\text{generator}}$ 为中心，从而去掉了这一项。如果只看对当前位置的 logit 求导的部分，普通的 score 是 $e_y-\pi_\theta$，SFC 之后的 score 是 $e_y-\pi_{\text{generator}}$，这里的 $e_y-\pi_\theta$，就是第 1 章算出的 $\partial\log\pi_\theta(y)/\partial z$。

计算 SFC 的 score，需要 generator 在这个位置上的完整分布。实际中，generator 只送回概率最高的若干个 token 和它们的概率，这部分称为 head，其余的部分称为 tail。tail 有两种处理方法，一种是再从 tail 里采几个 token 来估计，另一种是只记录 tail 的总概率，我们之前的实验默认用的是后一种。详细的推导见我们[介绍 SFC 的文章](https://normaluhr.github.io/zh/p/score-function-centering/)。

**在 SFC 上加 KL。** SFC 并不限制 trainer 和 generator 之间的距离。距离越大，在 generator 的样本上算出的更新，和在当前模型上应有的更新就差得越多。我们在[介绍 SFC 加 KL 的文章](https://normaluhr.github.io/zh/p/score-function-centering-kl/)里推导过这个差距的一个上界，它和 $\sqrt{\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)}$ 成正比，所以又在 SFC 之外加了一个 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$。它对 logit 的梯度是 $\pi_\theta-\pi_{\text{generator}}$，可以用一个稀疏的估计来计算。对一个从 generator 采到的 token $y$，这个估计只在 $y$ 自己的 logit 上取值 $\pi_\theta(y)/\pi_{\text{generator}}(y)-1$，在其他 logit 上都取 0，它的期望正好是 $\pi_\theta-\pi_{\text{generator}}$。

另外，每个 token 的 KL 权重会随它的 age 衰减。一个 token 的 age，指的是生成它的 generator 版本比当前 trainer 落后的步数，我们在之前的实验里把它称为 staleness。具体来说，KL 权重 $\beta_t$ 等于 $\beta_0$ 除以这个 token 的 age 的平方根，age 为 0 的 token 权重为 0。

这一章讨论两种样本不来自当前模型的情况。第一种是 GRPO 一批数据更新多次，样本来自 $\pi_{\text{old}}$，KL 的参照仍然是 $\pi_{\text{ref}}$。第二种是异步 RL，样本来自 $\pi_{\text{generator}}$，我们在 SFC 之外加的 KL 以 $\pi_{\text{generator}}$ 本身为参照，它限制的是每个 generator 版本之内的更新幅度。在完全 on-policy 的同步训练里，generator 就是 $\pi_\theta$ 自己，所以如果不考虑数值误差，这个 KL 恒等于 0，数值误差带来的问题留到 7.2 节讨论。

分析这两种情况时，要用到前面的两个结论，一个是 1.1 节对估计器项和采样项的拆分，另一个是 2.1 节的系数视角。第二种情况是后面几章的基础。

### 4.1 GRPO 一批数据更新多次时，$k_3$ 应该怎么修正？

一批数据更新多次时，从第二次更新开始，样本来自 $\pi_{\text{old}}$，而不是当前的 $\pi_\theta$。前面所有的期望都要改成在 $\pi_{\text{old}}$ 下计算，原来的结论也就不再严格成立。以 $k_3$ 当 loss 为例，期望梯度变成了 $\mathbb E_{y\sim\pi_{\text{old}}}[(1-\pi_{\text{ref}}/\pi_\theta)\,s_\theta]$，它既不是 forward KL 的梯度，也不是 reverse KL 的梯度。

要理解这一项该怎么修正，需要先看一个关于 $k_3$ 的恒等式。对每一个样本，都有

$$
\nabla_\theta k_3+k_3\,s_\theta=(1-e^{-\rho})s_\theta+(e^{-\rho}-1+\rho)s_\theta=\rho\,s_\theta.
$$

左边第一项是把 $k_3$ 当 loss 时 backward 得到的梯度，对应 1.1 节的估计器项。第二项是把 $k_3$ 当作 reward 时 policy gradient 给出的部分，对应 1.1 节里被漏掉的采样项。两者相加，正好是 forward KL 梯度的单样本形式 $\rho\,s_\theta$。这个恒等式说明，直接对 $k_3$ 做 backward 只保留了估计器项，补回采样项以后就能得到 forward KL 的梯度。

[DeepSeek-V3.2 的技术报告](https://arxiv.org/abs/2512.02556)在式 (7) 中修正了这一项，它把 $k_3$ 乘上了当前模型和旧模型的比率，

$$
\widehat{\mathrm{KL}}=w\cdot k_3,\qquad w=\frac{\pi_\theta(y)}{\pi_{\text{old}}(y)},\qquad y\sim\pi_{\text{old}}.
$$

如果 $w$ 参与求导，那么

$$
\nabla_\theta(w\,k_3)=w\,s_\theta\,k_3+w\,\nabla_\theta k_3=w\,(k_3\,s_\theta+\nabla_\theta k_3)=w\,\rho\,s_\theta,
$$

最后一步用的就是上面的恒等式。在 $\pi_{\text{old}}$ 下取期望，得到 $\mathbb E_{y\sim\pi_\theta}[\rho\,s_\theta]$，就是 forward KL 的准确梯度。

当 $w$ 参与求导时，从系数的角度看，这个修正同时做了两件事。第一件事是修正采样分布，把 $\pi_{\text{old}}$ 下的期望转成 $\pi_\theta$ 下的期望，这是比率本来的用途。第二件事不太明显。$w$ 对 $\theta$ 的导数是 $w\,s_\theta$，它的作用相当于补回了采样项。因此，即使在第一次更新时 $\pi_{\text{old}}$ 和 $\pi_\theta$ 完全相同，$w$ 的数值等于 1，只要它参与求导，$k_3$ 的梯度就会从 reverse KL 变成 forward KL。反过来，如果把 $w$ detach 掉，得到的仍然是 reverse KL 的梯度。

DeepSeek-V3.2 的报告也解释了原来的 $k_3$ 有什么问题。如果采到的 token 在当前模型下的概率远低于在 reference model 下的概率，$k_3$ 的梯度会给它一个很大而且没有上界的权重，使梯度的噪声变大。这对应 2.2 节里 $k_3$ 的系数的绝对值在 $\rho<0$ 一侧指数增长的那部分。

### 4.2 样本来自固定的 generator 时，为什么直接对 $k_1$ 求导反而是对的？

在我们之前讨论的[异步 RL 设定](https://normaluhr.github.io/zh/p/score-function-centering-kl/)里，样本来自 $\pi_{\text{generator}}$，KL 的参照也是 $\pi_{\text{generator}}$，我们要约束的是

$$
\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{generator}}}\big[\log\pi_{\text{generator}}(y)-\log\pi_\theta(y)\big].
$$

和前面不同的是，采样分布 $\pi_{\text{generator}}$ 不依赖 $\theta$。1.1 节的采样项来自"采样分布依赖 $\theta$"，现在这个依赖不存在了，也就没有东西会被漏掉。

从这一节开始，KL 的参照是 $\pi_{\text{generator}}$，所以我们把 log ratio 相应地定义为 $\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$，并记 $r=e^\rho=\pi_\theta(y)/\pi_{\text{generator}}(y)$。套用前情回顾里的三个估计器时，样本来自 $\pi_{\text{generator}}$，所以 $\pi_{\text{generator}}$ 处在 $q$ 的位置上，$\pi_\theta$ 处在 $p$ 的位置上，这里的 $r$ 正好就是前情回顾里的 $r$。

把 $k_1$，即 $-\log r=\log\pi_{\text{generator}}(y)-\log\pi_\theta(y)$，直接拿来 backward，单个样本的梯度是 $-s_\theta(y)$，它的期望

$$
-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta(y)]=\nabla_\theta\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)
$$

就是准确的梯度。在 logit 空间里，这个梯度是 $\pi_\theta-\pi_{\text{generator}}$，梯度下降的更新方向是 $\pi_{\text{generator}}-\pi_\theta$。用梯度下降最小化它，就是在 generator 采出来的 token 上最大化 $\log\pi_\theta$，也就是在 generator 的样本上做 SFT。

$k_3$ 也可以用。$k_3=(r-1)-\log r$，单个样本的梯度是 $(r-1)\,s_\theta$。它的期望

$$
\mathbb E_{y\sim\pi_{\text{generator}}}[(r-1)\,s_\theta]=\mathbb E_{y\sim\pi_\theta}[s_\theta]-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta]=-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta]
$$

和 $k_1$ 相同。$k_3$ 比 $k_1$ 多出来的 $r\,s_\theta$，在 $\pi_{\text{generator}}$ 下的期望是 $\mathbb E_{y\sim\pi_\theta}[s_\theta]=0$，因此它是一个控制变量。在 $\pi_\theta$ 和 $\pi_{\text{generator}}$ 接近时，它可以降低梯度的方差。在估计 KL 的数值时，$k_3$ 用控制变量降低了数值的方差，在这里，同样的思路降低的是梯度的方差。

把这一节和 2.2 节放在一起看，两处的代码几乎一样，都是对 $\pi_\theta$ 和另一个分布的 log 概率之差做 backward，只差一个符号。样本由当前模型自己采出来时，梯度的期望是 0。样本来自一个固定的 generator 时，它给出的却是 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的准确梯度。两者的区别只在于采样分布是否依赖 $\theta$。另外，按照第 1 章的叫法，$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 属于 reverse KL，和 GRPO 里 $k_3$ 当 loss 时的梯度是同一个方向。

这个更新方向在介绍 SFC 时也出现过。我们在[介绍 SFC 的文章](https://normaluhr.github.io/zh/p/score-function-centering/)里讨论过，异步 RL 里如果不做 SFC，平均 advantage 会乘上同一个更新方向 $\pi_{\text{generator}}-\pi_\theta$，大小和正负都由平均 advantage 决定。SFC 把这一项从 policy gradient 里去掉，再加上 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$，等于把同一个方向加了回来，只是权重变成了我们自己选的 $\beta$。这也是我们之前选择 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 这个方向的原因。

## 5. 在 Score Function Centering 下，拉近 trainer 和 generator 的 KL 怎样作用在 advantage 上？

> **TL;DR：** 记 $r=\pi_\theta(y)/\pi_{\text{generator}}(y)$。在 Score Function Centering 下，加上一个权重为 $\beta$ 的 trainer-generator KL，等价于在 advantage 全部算完以后，把每个采到的 token 的 advantage 减去 $\beta$ 乘以一个只依赖 $r$ 的系数。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数是 $r-1$。$k_4$ 是 $\log r$ 在 $\pi_{\text{generator}}$ 下的方差的一半，它的系数是 $\log r$。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数大约是 $r\log r$。三个系数在 $r=1$ 附近一阶相同，所以 trainer 和 generator 离得近时，几种写法的差别不大。

4.2 节说明，样本来自固定的 generator 时，$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的梯度可以准确地算出来。不过，梯度算得准，只说明我们准确地优化了这个 KL，并不说明它就是最合适的约束。把 trainer 拉向 generator 的写法还有好几种，它们在 trainer 和 generator 接近时作用差不多，离得远时，对概率正在下降和正在上升的 token 施加的力却很不一样。为了比较这些写法，这一章在 Score Function Centering（SFC）的设定下，把它们都写成"给 advantage 减去 $\beta$ 乘以一个系数"的形式。5.1 节推导这个形式，5.2 节用一个词表只有三个 token 的例子，看 trainer 和 generator 接近时这几种写法的差别有多大。

### 5.1 怎样把一个 KL 项写成对 advantage 的修改？

在 SFC 下，score 是以 $\pi_{\text{generator}}$ 为中心的版本，对当前位置 logit 的部分是 $e_y-\pi_{\text{generator}}$。从这一章开始，"系数"指的是乘在这个 score 上的量。设样本来自 $\pi_{\text{generator}}$，每个样本的系数是 $f(y)$，把 $f$ 在整个词表上的取值排成一个向量 $f$，那么期望是

$$
\mathbb E_{y\sim\pi_{\text{generator}}}\big[f(y)\,(e_y-\pi_{\text{generator}})\big]=\mathrm{diag}(\pi_{\text{generator}})\,f-\pi_{\text{generator}}\,(\pi_{\text{generator}}^\top f)=F_{\text{generator}}\,f,\qquad F_{\text{generator}}=\mathrm{diag}(\pi_{\text{generator}})-\pi_{\text{generator}}\pi_{\text{generator}}^\top.
$$

记 $\mathbf 1$ 为所有分量都等于 1 的向量。$F_{\text{generator}}\mathbf 1=\pi_{\text{generator}}-\pi_{\text{generator}}(\pi_{\text{generator}}^\top\mathbf 1)=0$，因此给 $f$ 的每个分量加上同一个常数，不会改变结果，这就是 SFC 下的 baseline 不变性。

policy gradient 那一项也可以写成这种形式。在当前前缀上选择 token $v$ 之后，记期望的 advantage 为 $A_v$，把它们排成向量 $A$。policy gradient 那一项的系数是采到的样本的 advantage，它给出的期望更新方向是 $F_{\text{generator}}A$。如果某个 KL 作为 loss，它的期望梯度也能写成 $F_{\text{generator}}f$，那么加上权重为 $\beta$ 的这个 KL 以后，梯度下降的期望更新方向是 $F_{\text{generator}}(A-\beta f)$。也就是说，加上这个 KL，等价于把 advantage 从 $A$ 改成 $A-\beta f$。

用修改 advantage 的方式来实现时，要注意第 3 章说的两个问题。上面的等价关系在每个位置上分别成立，$\beta$ 就是这一项实际的权重。所以对 advantage 的修改应该放在 advantage 全部计算完成之后，也就是组内标准化之后，并且只作用在当前 token 上，不沿序列向前累加。

下面依次看三种写法对应的 $f$。第一种是 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$。4.2 节说明，它在 logit 空间的梯度是 $\pi_\theta-\pi_{\text{generator}}$，而 $\pi_{\text{generator}}\odot r=\pi_\theta$，其中 $\odot$ 表示逐个分量相乘，所以

$$
F_{\text{generator}}(r-1)=\pi_{\text{generator}}\odot(r-1)-\pi_{\text{generator}}\Big(\sum_v\pi_\theta(v)-\sum_v\pi_{\text{generator}}(v)\Big)=\pi_\theta-\pi_{\text{generator}}.
$$

因此在 SFC 下，加一个权重为 $\beta$ 的 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$，在期望上等价于把每个样本的 advantage 从 $A$ 改成 $A-\beta(r-1)$。第 4 章开头回顾的稀疏估计用的也是 $r-1$ 这个数，只是它乘在 $e_y$ 上，只更新采到的那个 token 的 logit。两种做法在单个样本上的梯度不同，期望都是 $\pi_\theta-\pi_{\text{generator}}$。

第二种写法的出发点是，我们希望系数就是 $\log r$ 本身。和 $r-1$ 相比，$\log r$ 只按对数增长，在 $r$ 很大时也不会变得很大。可以验证，下面这个目标在 SFC 下对应的系数恰好就是 $\log r$，它是 $\log r$ 在 $\pi_{\text{generator}}$ 下的方差的一半，

$$
k_4=\tfrac12\,\mathrm{Var}_{y\sim\pi_{\text{generator}}}\big[\log r(y)\big].
$$

它的梯度是

$$
\nabla_z k_4=\mathbb E_{y\sim\pi_{\text{generator}}}\big[(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)\,(e_y-\pi_\theta)\big]=\pi_{\text{generator}}\odot(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)=F_{\text{generator}}\log r.
$$

$k_2$ 用的是 $\tfrac12(\log r)^2$ 的平均值，这里用的是减去均值以后的平方的平均值。为了和前三个估计器并列，下文把它记为 $k_4$。需要说明的是，Schulman 的博客里只有 $k_1$ 到 $k_3$，$k_4$ 是本文为了叙述方便起的名字。$\pi_\theta$ 和 $\pi_{\text{generator}}$ 很接近时，它和 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 在二阶以内相同，所以也可以把它看成 KL 的一种写法。梯度里 $\log r$ 减去了它在 $\pi_{\text{generator}}$ 下的均值，这相当于 2.1 节说的减去一个 baseline。在 SFC 下，这个均值减不减都一样，因为 $F_{\text{generator}}\mathbf 1=0$。

所以在 SFC 下，$k_4$ 等价于把 advantage 改成 $A-\beta\log r$。它的实现很简单，不需要单独的 loss 项，只要在每个采到的 token 上，用 detach 之后的 $\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$ 去修改 advantage。KLPO（Zhang, 2026）讨论过同一个目标，它把这种做法称为 token regression，并在附录 D 中把对应的目标写成了 advantage 加权的 log ratio，再加上 $\beta/2$ 倍的 log ratio 方差。

需要注意的是，$k_4$ 必须和 SFC 一起使用。如果 score 还是普通的 $e_y-\pi_\theta$，同样的代码实现的是不减均值的版本，也就是 $\tfrac12\mathbb E_{\pi_{\text{generator}}}[(\log r)^2]$ 的梯度。这个目标在 SFC 下对应的系数是 $\log r-\mathbb E_{\pi_{\text{generator}}}[\log r]\cdot r$，比 $k_4$ 多出了 $-\mathbb E_{\pi_{\text{generator}}}[\log r]\cdot r$ 这一项。

第三种是另一个方向的 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$。以 reference model 为参照时，RLHF 通常约束的就是这个方向，所以也值得看一下它用在 generator 上会怎样。它就是以 $\pi_{\text{generator}}$ 为参照的 forward KL，由 2.1 节，它的梯度是 $\mathbb E_{y\sim\pi_\theta}[\log r\,s_\theta]$。样本来自 $\pi_{\text{generator}}$ 时，乘上比率 $r$，就能把这个期望改写成 $\pi_{\text{generator}}$ 下的期望，所以用普通的 score、系数取 detach 之后的 $r\log r$，就能得到准确的梯度，

$$
\mathbb E_{y\sim\pi_{\text{generator}}}\big[r\log r\,(e_y-\pi_\theta)\big]=\pi_\theta\odot\log r-\pi_\theta\,\mathbb E_{\pi_\theta}[\log r]=\pi_\theta\odot\big(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})\big).
$$

在 SFC 下，对应的系数是 $r\,(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}}))$。当 $\pi_\theta$ 和 $\pi_{\text{generator}}$ 接近时，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 是一个二阶小量，这个系数约等于 $r\log r$。

总结一下，在 SFC 下，$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数是 $r-1$，$k_4$ 的系数是 $\log r$，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数大约是 $r\log r$。三者都是 $r$ 的函数，在 $r=1$ 处都等于 0，并且在 $r=1$ 附近一阶相同。三种写法都可以通过修改 advantage 来实现，只用到采到的 token 的 $\pi_\theta(y)$ 和 $\pi_{\text{generator}}(y)$，其中 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的准确系数还需要估计 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 本身。

### 5.2 trainer 和 generator 接近时，这几种写法的差别大吗？

5.1 节的三个系数在 $r=1$ 附近一阶相同，这意味着 trainer 和 generator 接近时，它们给出的梯度应该差别不大。下面用一个词表只有三个 token 的例子来看这一点，同时看看它们在细节上有什么不同。

设 $\pi_{\text{generator}}=(0.6,0.3,0.1)$，$\pi_\theta=(0.5,0.3,0.2)$，那么 $r=(0.833,1,2)$，$\log r\approx(-0.182,0,0.693)$，$\mathbb E_{\pi_{\text{generator}}}[\log r]\approx-0.040$。两个方向的 KL 分别是 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)\approx0.040$ 和 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})\approx0.048$。三种写法在 logit 上的期望梯度如下，图 3 把它们画在了一起。

* $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的梯度是 $\pi_\theta-\pi_{\text{generator}}=(-0.1,\ 0,\ +0.1)$。
* $k_4$ 的梯度是 $\pi_{\text{generator}}\odot(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)\approx(-0.085,\ +0.012,\ +0.073)$。
* $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的梯度是 $\pi_\theta\odot(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}}))\approx(-0.115,\ -0.014,\ +0.129)$。

<figure data-chart="kl-three-token">
<noscript><p>图 3 需要 JavaScript 才能显示。</p></noscript>
</figure>

在梯度下降下，第一个 token 的 logit 会上升，第三个 token 的 logit 会下降，三种写法在这一点上是一致的，都在把 $\pi_\theta$ 拉回 $\pi_{\text{generator}}$。区别在第二个 token 上。它的 $r=1$，在 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 下完全不动，在 $k_4$ 下会被轻微压低，在 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 下反而会被轻微抬高。原因是每种写法拿 $\log r$ 比较的基准不同。$k_4$ 拿它和 $\pi_{\text{generator}}$ 下的平均值比较，这个平均值是 $-0.040$，所以 $r=1$ 的 token 相对来说偏高了。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 拿它和 $\pi_\theta$ 下的平均值比较，这个平均值就是 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})=0.048$，所以同一个 token 相对来说偏低了。

这个例子里几种写法的差别都不大，因为 trainer 和 generator 离得不远。trainer 和 generator 会离多远，取决于 $\beta$ 和 advantage 的相对大小。当 $\beta$ 明显大于 advantage 之间的差别时，KL 会把 $r$ 维持在 1 附近，几种写法的差别不大。当 $\beta$ 和 advantage 之间的差别在同一个量级，或者比它更小时，$r$ 会离 1 很远，系数在远离 $r=1$ 处的形状就会让几种写法的行为差别很大。在我们之前的实验设置里，一个典型的 token 的 $\beta_t$ 只有 0.04 左右，而 advantage 之间的差别可以到 1 这个量级，正好属于后一种情况，6.1 节会给出这两个数的来历。下一章就讨论这种情况。

## 6. 三种写法分别怎样约束概率下降和概率上升的 token？

> **TL;DR：** 三种写法的差别主要出现在 $r$ 远离 1 的时候。对概率正在被压低的 token，$k_4$ 的拉力随着概率下降没有上限，能把这些 token 保住。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的拉力有上限，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的拉力会趋于 0，所以 $\beta$ 较小时，在这两种写法下，advantage 偏低的 token 都可能被一直压到 0。对概率正在被抬高的 token，情况正好反过来，$k_4$ 的约束最弱，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的约束最强。哪一种写法更合适，取决于训练中哪一类 token 的风险更大，以及 $\beta$ 和 advantage 的相对大小。

第 5 章把三种写法都写成了给 advantage 减去 $\beta$ 乘以一个系数的形式。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数是 $r-1$，$k_4$ 的系数是 $\log r$，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数大约是 $r\log r$。当 trainer 和 generator 离得比较远时，这三个系数的形状决定了它们的行为。我们分两类 token 来看，一类是概率正在被压低的 token，也就是 $r<1$ 的 token，另一类是概率正在被抬高的 token，也就是 $r>1$ 的 token。这一章先在 6.1 节比较三种写法在单个 token 上的拉力，再在 6.2 节用一个简化的模型看它们最终把分布拉到哪里。6.3 节讨论每种写法的代价，6.4 节讨论选择时可以考虑的因素。

### 6.1 一个 token 的概率被持续压低时，三种写法能把它拉回来吗？

KL 的权重是 $\beta$，但 KL 对训练的实际影响，还取决于 $\beta$ 乘上的那个系数，而且要和 advantage 的大小放在一起比较。为此，我们看某个 token $v$ 的 logit 在 SFC 下受到的"力"，这里的力指的是这个 logit 上的期望更新量。记 $\Delta A_v=A_v-\mathbb E_{\pi_{\text{generator}}}[A]$，其中 $A_v$ 是 5.1 节定义的、选择 token $v$ 之后的期望 advantage。由 5.1 节的工具，policy gradient 这一项在 $z_v$ 上的期望更新是 $(F_{\text{generator}}A)_v=\pi_{\text{generator}}(v)\,\Delta A_v$。这个量只和 $\pi_{\text{generator}}$ 有关，和 $\pi_\theta(v)$ 当前是多少无关。也就是说，在 generator 固定的情况下，SFC 会一直用同样的力去推或者拉这个 token。这一点和样本来自当前模型时不同，那时 policy gradient 在 $z_v$ 上的期望更新是 $\pi_\theta(v)\,(A_v-\mathbb E_{\pi_\theta}[A])$，token 的概率越低，推力越小。

三种写法把这个 token 往回拉的力分别如下。

* $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的拉力是 $\beta\,(\pi_{\text{generator}}(v)-\pi_\theta(v))$。当 $\pi_\theta(v)\to0$ 时，这个力趋于 $\beta\,\pi_{\text{generator}}(v)$，不会再增大。这和 2.2 节里 $k_3$ 当 loss 时的情况相同，那里的拉力也不会超过 $\pi_{\text{ref}}(v)$。
* $k_4$ 的拉力是 $\beta\,\pi_{\text{generator}}(v)\,(\mathbb E_{\pi_{\text{generator}}}\log r-\log r_v)$。当 $\pi_\theta(v)\to0$ 时，$-\log r_v$ 越来越大，这个力随着 logit 下降的幅度线性增长，没有上界。
* $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的拉力是 $\beta\,\pi_\theta(v)\,(\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})-\log r_v)$，写成 $\beta\,\pi_{\text{generator}}(v)$ 乘以系数的形式，系数大约是 $-r_v\log r_v$。函数 $r\log r$ 在 $r=1/e$ 时取到最小值 $-1/e$，之后随着 $r$ 继续变小又回到 0。所以它往回拉的力最多约为 $\beta\,\pi_{\text{generator}}(v)/e$，而且 token 被压得越低，这个力越弱。

把推力和拉力放在一起比较。对 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 来说，一个正在被压低的 token，往回拉的力不会超过 $\beta\,\pi_{\text{generator}}(v)$，而往下推的力是 $\pi_{\text{generator}}(v)\,|\Delta A_v|$。只要 $\Delta A_v<-\beta$，推力就一直大于拉力，这个 KL 无法阻止这个 token 的概率继续下降，它最终会被压向 0。样本来自当前模型时，推力会随着 $\pi_\theta(v)$ 一起变小，有上界的拉力足以把 token 拉住。在 SFC 下推力不会变小，有上界的拉力就可能拉不住。

图 4 把三种写法的拉力画成了 $r$ 的函数，单位是 $\beta\,\pi_{\text{generator}}(v)$，图中省略了 $\mathbb E_{\pi_{\text{generator}}}\log r$ 和 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 这两个小量。token 被压得越低，就越往图的左边走。虚线是推力 $|\Delta A_v|/\beta$，拖动滑块可以改变它。一条曲线和虚线相交的地方，就是拉力和推力平衡、token 停下来的位置。如果一条曲线一直在虚线下方，这种写法就拉不住这个 token。

<figure data-chart="kl-pull">
<noscript><p>图 4 需要 JavaScript 才能显示。</p></noscript>
</figure>

$\Delta A_v<-\beta$ 的情况在实际训练中是可能出现的。在我们[之前的实验设置](https://normaluhr.github.io/zh/p/score-function-centering-experiments/)里，KL 的权重 $\beta_t$ 等于 $\beta_0=0.3$ 除以 age 的平方根。在 Countdown 的 4-prompt slice 上，也就是 trainer 每次更新用 4 个 prompt 的 rollout 时，token 的平均 age 大约是 50，一个 age 为 50 的 token 的 $\beta_t=0.3/\sqrt{50}\approx0.042$。那些实验计算 advantage 时，答对的回答记为 $+1$，答错的记为 $-1$，再减去组内的平均值，所以单条回答的 advantage 在 $\pm1$ 这个量级。单个 token 的 $\Delta A_v$ 通常比这个小，因为大多数 token 换成别的选择，并不会改变回答的结果。但在少数决定回答走向的 token 上，$|\Delta A_v|$ 可能接近这个量级，远远超过 0.042。

不过，在那些实验里，SFC 加 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 一次都没有崩溃过，这和上面的分析可以同时成立。上面的比较假设推力一直持续，而在实际训练中，每一片 rollout 只用来做一次更新，generator 也在不断刷新，同一个前缀很少被反复更新，推力并不会一直持续。另外，一个 token 被压到 0，也不一定会让训练崩溃。拉力的上限说明的是，一旦推力持续存在，这个 KL 能提供的保护是有限的。

教科书里常说 $\mathrm{KL}(q\|p)$ 是 zero-avoiding 的，只要 $q(v)>0$ 而 $p(v)\to0$，KL 的数值就趋于无穷，所以最小化它会避免把 $p$ 的任何部分压到 0。这个说法对 KL 的数值是成立的。但 $\mathrm{KL}(q\|p)$ 的数值只随 $-\log p(v)$ 增长，换算到 logit 上大约是线性增长，它对 logit 的梯度是有界的。在一个持续不变的推力面前，数值趋于无穷并不能保证这个 token 不被压到 0。

### 6.2 一直训练下去，三种写法会把分布拉到哪里？

6.1 节只看了单个 token 受到的力。为了看三种写法最终把整个分布拉到哪里，我们用一个简化的模型。固定一个前缀和一个 generator 版本，假设每个 token $v$ 的期望 advantage $A_v$ 固定不变，然后让期望更新一直进行下去，看它最后停在哪里。

在优化里，梯度等于 0 的点通常称为驻点。这里的更新方向不一定是某一个目标函数的梯度，所以我们说的驻点，指的是期望更新量等于 0 的点，也就是动力系统里的不动点。到达驻点以后，期望更新不再改变 $\pi_\theta$，如果更新收敛，就会停在这里。驻点不一定能到达，有些 token 的概率会在更新中一直下降，永远停不下来，这时我们关心的是它的极限。下面的分析和例子都围绕驻点展开。

在 SFC 下，更新方向是 $F_{\text{generator}}(A-\beta f(r))$，其中 $f$ 是对应写法的系数。当 $\pi_{\text{generator}}$ 的每个分量都大于 0 时，满足 $F_{\text{generator}}x=0$ 的向量 $x$ 只有常数向量，因此驻点满足的条件是，$A_v-\beta f(r_v)$ 对所有 token 都相等。

对于 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$，$f(r)=r-1$，驻点是

$$
r_v^*=1+\frac{\Delta A_v}{\beta},\qquad \pi_\theta^*(v)=\pi_{\text{generator}}(v)\Big(1+\frac{A_v-\mathbb E_{\pi_{\text{generator}}}[A]}{\beta}\Big).
$$

驻点 $r_v^*$ 是 advantage 的线性函数。它也是 $\max_p\ \mathbb E_p[A]-\tfrac\beta2\chi^2(p\|\pi_{\text{generator}})$ 的驻点，其中 $\chi^2(p\|\pi_{\text{generator}})=\sum_v(p(v)-\pi_{\text{generator}}(v))^2/\pi_{\text{generator}}(v)$。也就是说，SFC 加 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 在一个 generator 版本之内的效果，相当于一个 $\chi^2$ 正则。$\chi^2$ 正则在 LLM 对齐里也有人讨论过，Huang 等人（2024）认为它比 KL 正则更能抑制对不完美 reward 的过度优化。这个驻点只有在所有 $\Delta A_v\ge-\beta$ 时才是一个合法的分布。否则，$\Delta A_v<-\beta$ 的 token 会被一直压向 0，这和 6.1 节的结论一致。

对于 $k_4$，$f(r)=\log r$，驻点是

$$
\pi_\theta^*(v)\propto\pi_{\text{generator}}(v)\,e^{A_v/\beta}.
$$

它是 $\max_p\ \mathbb E_p[A]-\beta\,\mathrm{KL}(p\|\pi_{\text{generator}})$ 的经典最优解，和 RLHF、DPO 里 $\pi^*\propto\pi_{\text{ref}}\,e^{R/\beta}$ 的形式相同，其中 $R$ 是 reward。它永远是一个合法的分布，任何 token 都不会被压到恰好为 0。对于被抬高的 token，驻点上的 $r_v^*$ 不会超过 $e^{\Delta A_v/\beta}$，同时也受概率归一化的限制，不会超过 $1/\pi_{\text{generator}}(v)$。如果每个 generator 版本都训练到驻点，再用当前的 trainer 作为新的 generator，那么把一轮一轮的结果连起来，就是 $\pi_{k+1}\propto\pi_k\,e^{A_k/\beta}$，和 mirror descent 的形式相同。另外，$k_4$ 用的是 generator 的样本，驻点却对应 $\mathrm{KL}(p\|\pi_{\text{generator}})$ 方向的正则，而且整个过程不需要任何 importance weight。

对于 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$，驻点没有简单的解析形式。按 6.1 节的分析，它对被压低的 token 的拉力最多约为 $\beta\,\pi_{\text{generator}}(v)/e$，并且 token 越低拉力越弱。粗略地说，当 $\Delta A_v$ 低于 $-\beta/e$ 左右时，这个 token 会被一直压到 0。准确的阈值是 $-\beta e^{K-1}$，其中 $K$ 是驻点上 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的值，所以它比 $-\beta/e$ 还要再低一些。$\beta$ 越大，这个阈值越低，被压到 0 的 token 也就越少。

这里有一个容易让人困惑的地方。$k_4$ 的驻点是 $\mathrm{KL}(p\|\pi_{\text{generator}})$ 正则问题的最优解，直接加 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$，驻点反而不是。原因在 policy gradient 那一项。在 SFC 下，这一项的期望是 $F_{\text{generator}}A$，它由 $\pi_{\text{generator}}$ 决定，不随 $\pi_\theta$ 变化。而在 $\max_p\mathbb E_p[A]-\beta\,\mathrm{KL}(p\|\pi_{\text{generator}})$ 这个问题里，$\mathbb E_p[A]$ 对 logit 的梯度是 $F_\theta A$，其中 $F_\theta=\mathrm{diag}(\pi_\theta)-\pi_\theta\pi_\theta^\top$。$k_4$ 的驻点条件是 $A-\beta\log r$ 为常数，正则问题最优解的条件也是这一个，所以两者重合。直接加 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 时，驻点条件是 $F_{\text{generator}}A=\beta F_\theta\log r$，两边乘的矩阵不同，驻点也就不再是那个最优解。

如果不加任何 KL，也就是 $\beta=0$，那么驻点的条件变成所有 $A_v$ 都相等，一般不会成立。这时没有驻点，概率会一直流向 advantage 最高的 token。

下面用两个例子把三种写法放在一起比较。这些数字是从 $\pi_\theta=\pi_{\text{generator}}$ 出发，在 logit 上反复做期望更新得到的。对于被压向 0 的 token，给出的是更新的极限。

第一个例子里，有的 token 的 advantage 偏低。设 $\pi_{\text{generator}}=(0.6,0.3,0.1)$，三个 token 的 advantage 是 $A=(0.5,0,-0.5)$，于是 $\mathbb E_{\pi_{\text{generator}}}[A]=0.25$，$\Delta A=(0.25,-0.25,-0.75)$。

* $\beta=1$ 时，$\beta$ 和 advantage 的差别在同一个量级。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 停在 $(0.75,0.225,0.025)$，$k_4$ 停在 $(0.733,0.222,0.045)$，两者很接近。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的极限是 $(0.803,0.197,0)$，第三个 token 被压到了 0。如果把 $\beta$ 增大到 3，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 会停在 $(0.653,0.276,0.071)$，第三个 token 保留了下来。
* $\beta=0.2$ 时，$\beta$ 比 advantage 的差别小。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的线性驻点是 $(1.35,-0.075,-0.275)$，它不是一个合法的分布，实际的更新最后趋于 $(1,0,0)$。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 也趋于 $(1,0,0)$。$k_4$ 停在 $(0.960,0.039,0.001)$，第二和第三个 token 被压得很低，但都没有到 0。

第二个例子里，一个很少被采到的 token 拿到了较高的 advantage。设 $\pi_{\text{generator}}=(0.9,0.09,0.01)$，$A=(0,0,0.5)$，$\beta=0.05$，$\beta$ 远小于 advantage 的差别。

* $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 停在 $(0.81,0.081,0.109)$，第三个 token 涨到了原来的约 11 倍。
* $k_4$ 停在 $(0.004,0.0004,0.996)$，第三个 token 几乎占据了全部概率，它的 $r$ 约为 100，已经到了概率归一化允许的上限 $1/\pi_{\text{generator}}(v)$。
* $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 停在 $(0.856,0.086,0.058)$，第三个 token 涨到了原来的约 5.8 倍。

图 5 画出了这两个例子的驻点。上方的按钮用来切换例子，拖动滑块可以把 $\beta$ 从 0.01 调到 10，看三种写法的驻点怎样随 $\beta$ 变化。$\beta$ 很大时，三种写法的驻点都贴近 generator。$\beta$ 越小，它们分开得越明显。

<figure data-chart="kl-fixed">
<noscript><p>图 5 需要 JavaScript 才能显示。</p></noscript>
</figure>

两个例子放在一起，可以看到三种写法的分工。在第一个例子里，$\beta$ 较小时，只有 $k_4$ 保留了 advantage 偏低的 token，在另外两种写法下，它们都被压到了 0。在第二个例子里，advantage 较高的稀有 token 在 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 下涨得最少，在 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 下涨得多一些，在 $k_4$ 下几乎占据了全部概率。

哪一种行为是我们想要的，取决于 advantage 有多可靠。如果第二个例子里那个稀有 token 的高 advantage 反映的是一个确实更好的行为，让它快速增长就是好事。如果它来自一个过时的样本、一个 reward 的漏洞，或者单纯的噪声，限制它的增长会更安全。第一个例子也是这样。如果低 advantage 的 token 确实不好，把它压到 0 并没有问题。如果它只是这一次运气不好，把它压到 0 就会损失探索。

最后要说明这个简化模型的适用范围。实际训练中，每个 generator 版本只会训练有限的步数，generator 会不断刷新，不同位置之间还共享参数。驻点只能说明每种写法往哪个方向拉、最多拉到哪里，并不能预测训练的具体结果。

### 6.3 每种写法的代价是什么？

6.1 节和 6.2 节讨论的是每种写法在期望上的行为。实际使用时，还要考虑它们的方差，以及它们的行为本身带来的风险。

5.1 节说明，三种写法都可以通过修改 advantage 来实现，只用到采到的 token 的 $\pi_\theta(y)$ 和 $\pi_{\text{generator}}(y)$，所以它们在实现上的差别并不大。方差上的差别主要来自系数的大小。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数 $r-1$ 里有 $1/\pi_{\text{generator}}(y)$，一个在 generator 下概率很小、在 trainer 下概率涨上来的 token，一旦被采到，系数就会很大。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数 $r\log r$ 比 $r-1$ 增长得更快，这和 importance sampling 的方差问题是同一类。$k_4$ 的系数 $\log r$ 只按对数增长，方差可能最小。

另一种降低方差的做法，是在 head 上精确计算期望，只对 tail 做估计，原则上三种写法都可以这样做。以 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 为例，记 head 在 generator 下的总概率为 $\alpha$，head 上的 token 可以精确计算 $\pi_\theta-\pi_{\text{generator}}$，tail 上采到的 token 用 $(1-\alpha)(\pi_\theta/\pi_{\text{generator}}-1)$ 来估计。如果连 tail 上的采样也省掉，只在 head 上精确计算，并把 tail 合并成一个类别，只用到 tail 的总概率，那么采样带来的方差就没有了，但会漏掉 tail 内部的差异，所以是有偏的，KLPO 把它称为 TopK-KL。我们之前的实验默认用的就是这一类实现，generator 送回概率最高的 128 个 token，不对 tail 采样。tail 的总概率取 generator 给出的值，再按 trainer 自己的概率比例分配给 tail 里的各个 token。这和把 tail 合并成一个类别是一回事。tail 的概率按 trainer 自己的比例分配以后，tail 里每个 token 的 $r$ 都相同，都等于 trainer 和 generator 在 tail 上的总概率之比，tail 内部的 KL 也就只剩下一项。

除了方差，$k_4$ 的不对称性本身也有好坏两面。好的一面是，它能防止 token 的概率在一个 generator 版本之内被压到 0，从而避免熵下降和探索变少，同时允许确实更好的 token 快速增长。坏的一面是，它对抬高这一侧的约束比较弱，如果一个 token 的高 advantage 来自过时的样本或者噪声，它也会被放任增长。我们之前的实验里有一个间接的参考。其中一个方法是在 SFC 的梯度上再乘一个截断到 2 的比率 $\min(\pi_\theta/\pi_{\text{generator}},2)$，也就是 truncated importance sampling，这是 Marek 和 Ryabinin（2026）使用的组合。这个方法在 Countdown 上崩溃时，熵持续上升，意味着原本概率很低的 token 整体上被抬高了，回答也越来越长，最后 trainer 和 generator 之间的 KL 突然涨了两个数量级，看起来更像是抬高这一侧出了问题。不过那个 run 里没有任何 KL 项，所以它只能作为一个间接的参考。

$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 和 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的风险正好在另一侧。6.1 节和 6.2 节已经看到，它们对被压低的 token 的保护都有上限，$\beta$ 较小时，低 advantage 的 token 可能被一直压到 0。

### 6.4 实际中应该怎么选？

在比较几种写法之前，可以先判断这个选择在自己的训练里重不重要。5.2 节说过，当 $\beta$ 明显大于 advantage 之间的差别时，KL 会把 $r$ 维持在 1 附近，几种写法的差别不大，选哪一个影响有限。只有当 $\beta$ 和 advantage 之间的差别在同一个量级或者更小时，比如 6.1 节里 age 为 50、$\beta_t$ 只有 0.042 的 token，选择才比较重要。

如果选择确实重要，接下来要看训练中更担心哪一类 token 出问题。如果观察到的是熵持续下降、某些 token 被压到几乎不再出现，那么保护被压低的 token 的写法，比如 $k_4$，可能更合适。如果观察到的是熵上升、回答越来越长、$r$ 的分布出现很长的尾巴，那么限制被抬高的 token 的写法，比如 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 或者 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$，可能更合适。在不同的模型、数据、optimizer 和 age 的分布下，哪一类 token 的风险更大可能并不相同，所以这个判断最好以自己训练中的观察为依据。

两类风险都需要防范时，可以把几种写法组合起来使用。比如把 $k_4$ 和 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 一起加上。按 6.2 节的结果，前者保护被压低的 token，后者对被抬高的 token 约束最强，两者的作用正好互补，代价是需要分别决定两者的权重。

方差也会影响选择。$k_4$ 的系数只按对数增长，方差可能也最小。$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数带 $1/\pi_{\text{generator}}$，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数增长得更快，偶尔采到一个 $r$ 很大的 token，就会带来很大的梯度。几种写法对数值精度的要求也不一样，这是下一章的内容。

另一个可以调节的是 $\beta$ 和 age 的关系。我们[之前的做法](https://normaluhr.github.io/zh/p/score-function-centering-kl/)是让 $\beta_t$ 随 age 衰减，理由是越旧的 generator 越不适合作为当前 trainer 的参照。反过来也可以这样考虑，样本越旧，离当前模型越远，越需要约束，因此 $\beta$ 应该随 age 增大。如果用 6.1 节的数字来看，age 为 50 时 $\beta_t$ 只有 0.042，在简化模型里，$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 已经拦不住 $\Delta A_v<-0.042$ 的 token。哪一种更合适，取决于旧 generator 的样本在训练中更容易带来哪一种风险。对于 age 为 0 的 token，我们之前把 $\beta$ 设为 0，原因和数值精度有关，7.2 节会解释。

最后，trainer 和 generator 之间的 KL，只能限制每一个 generator 版本之内的更新幅度。generator 一直在跟着 trainer 走，这个参照本身也在移动，所以它不能防止模型在很多轮之后离初始模型越来越远。如果需要限制长期的漂移，可以另外加一个以 $\pi_{\text{ref}}$ 为参照的 KL，两者的作用是互补的。

## 7. 有限精度下，KL 和它的系数应该怎么算？

> **TL;DR：** 计算 KL 和它的系数时，应该在 FP32 下先算出 log 概率之差 $\rho=\log\pi_\theta-\log\pi_{\text{generator}}$，再用 `expm1(rho)` 得到 $r-1$，记录 $k_3$ 的数值时用 `expm1(rho) - rho` 或者泰勒展开，不要先算出接近 1 的 $e^\rho$ 再减去 1。否则记录下来的 KL 可能是负数。如果 $r$ 以 BF16 存储，$r-1$ 还会把 $\rho$ 小于约 0.0039 的轻微上涨算成 0，让这些 token 完全不受约束。检查 trainer 和 generator 是否完全一致时，应该要求 $\rho^2$ 的最大值等于 0，不要看 $k_1$ 的平均值。

前面的分析都假设计算是精确的。实际训练中，KL 和它的系数都是在有限精度下算出来的。这一章说明，同一个公式用不同的方式计算，结果可能差别很大，而且第 5 章里不同的系数对精度的要求也不同。这一章的数值都按 FP32 或 BF16 的舍入规则实际算过，图 6 和图 7 里的曲线也是按同样的规则逐点计算的。

问题的根源是，浮点数在 0 附近很密，在 1 附近很稀。FP32 里，1 和比它大的下一个数相差 $2^{-23}\approx1.2\times10^{-7}$，而在 0 附近，FP32 可以表示小到 $10^{-38}$ 量级的数。BF16 更粗，1 上方相邻两个数相差 $2^{-7}\approx0.0078$，1 下方相差 $2^{-8}\approx0.0039$。如果我们要算的结果是一个很小的量，计算过程中最好不要出现"1 加上这个小量"这样的中间值。一旦中间值被存成了 1 附近的浮点数，那个小量的大部分信息就丢了，之后再减去 1 也找不回来。下面两节分别看这个问题在记录 KL 的数值和计算训练用的系数时会怎样出现。

### 7.1 记录 KL 的数值时，为什么会算出负数？

以 $k_3=e^\rho-1-\rho$ 为例，其中 $\rho=\log r=\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$。在 FP32 下直接按这个式子计算，$\rho=10^{-6}$ 时得到的是 $-4.6\times10^{-8}$。真值大约是 $\rho^2/2=5\times10^{-13}$，一个数学上非负的量被算成了负数，误差比真值大了五个数量级。原因是 $e^\rho$ 被存成了 1 附近最近的那个 FP32 数，这一步的舍入误差在 $10^{-7}$ 量级，减掉 1 以后，剩下的几乎全是舍入误差。改用 `expm1(rho) - rho` 来计算，`expm1` 直接算出 $e^\rho-1$，不经过 $e^\rho$ 这个接近 1 的中间值，得到的是 $4.5\times10^{-13}$，和真值只差 9%。剩下的这点误差来自 `expm1` 的结果本身的舍入。这个结果的相对误差在 $10^{-7}$ 量级，换算成绝对误差大约是 $6\times10^{-8}\rho$，所以 $\rho$ 越小，相对误差越大。如果需要在 $\rho$ 更小时也保持精度，可以在 $|\rho|$ 很小时直接按泰勒展开 $\rho^2/2+\rho^3/6$ 来计算，这样就完全没有相减的步骤。

图 6 在对数坐标下画出了 $k_3$ 的真值，以及三种算法的误差。直接计算的误差一直停在 $10^{-8}$ 到 $10^{-7}$ 这个量级，和 $\rho$ 的大小无关。当 $\rho$ 小于大约 $3\times10^{-4}$ 时，这个误差已经超过了真值本身，算出来的结果没有意义。用 `expm1` 计算时，误差和 $\rho$ 成正比，要到 $\rho$ 小于 $10^{-7}$ 左右才会和真值相当。用泰勒展开计算时，误差在 $\rho$ 很小时一直远小于真值，只有 $\rho$ 变大以后，被省略的高阶项才开始起作用。

<figure data-chart="kl-fp32">
<noscript><p>图 6 需要 JavaScript 才能显示。</p></noscript>
</figure>

同样的原因还会让一致性检查给出错误的结论。有些检查用的量写成了 1 加上一个二阶小量的形式，例如 $1+\rho\,(e^\rho-1)$，它在样本来自 generator 时的期望是 1 加上两个方向的 KL 之和。在 FP32 下，只要 $|\rho|\le2.4\times10^{-4}$，这个量就会被舍入成恰好等于 1，看起来就像 trainer 和 generator 完全一致。

$k_1$ 的平均值也不适合用来检查两者是否一致。$\rho$ 有正有负，平均值可能因为正负抵消而接近 0，甚至为负。如果想确认 trainer 和 generator 完全一致，更可靠的做法是直接计算 $\rho^2$，要求它的最大值恰好等于 0。$\rho^2$ 不会是负数，最大值等于 0 就说明每个 token 都完全一致。

### 7.2 训练用的系数受精度影响有多大？

训练时用到的系数同样会受这个问题影响，但几种系数受影响的程度不同。$k_4$ 的系数是 $\log r=\rho$。只要 $\rho$ 是用 FP32 的 log 概率相减得到的，即使之后转换成 BF16，相对误差也不超过 0.2%。但如果 log 概率本身就以 BF16 存储，相减这一步就会丢掉大部分信息。比如 $-2.3012$ 和 $-2.3042$ 存成 BF16 以后都是 $-2.296875$，相减得到 0，而真值是 0.003。

$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 的系数 $r-1$ 还多了一层问题。如果先算出 $r=e^\rho$，再减去 1，就会遇到这一章开头说的情况。假设 $r$ 是以 BF16 存储的，那么 $\rho=0.001$ 和 $\rho=0.003$ 时，$r$ 都被存成了恰好等于 1，$r-1$ 等于 0。$\rho=0.005$ 时，$r-1$ 变成 0.0078，而真值是 0.0050。$\rho=-0.003$ 时，$r-1$ 变成 $-0.0039$，而真值是 $-0.0030$。由于 1 上方和下方的间距不同，两侧的误差也不对称。$\rho$ 小于约 0.0039 的轻微上涨都会得到系数 0，完全不受约束，而轻微下降的 token 只要 $\rho$ 低于约 $-0.002$，就会被拉回来，并且拉力被放大了。图 7 画出了 BF16 存储的 $r$ 减去 1 以后得到的阶梯，两条竖线就是这两个阈值。

<figure data-chart="kl-bf16">
<noscript><p>图 7 需要 JavaScript 才能显示。</p></noscript>
</figure>

正确的做法是在 FP32 下计算 $\rho=\log\pi_\theta-\log\pi_{\text{generator}}$，$\log r$ 直接用这个 $\rho$，$r-1$ 则用 `expm1(rho)` 得到，6.3 节里 tail 上的估计也是如此。图 7 里的虚线就是先在 FP32 下算 `expm1`、再存成 BF16 的结果，它和真值几乎重合。$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 的系数 $r\log r=e^\rho\rho$ 里没有减去 1 的步骤，即使 $r$ 以 BF16 存储，相对误差也不超过 0.2%，受这个问题的影响较小。

还有一种数值差异，和存储的精度无关。即使 trainer 和 generator 用的是同一个版本的参数，两者算出来的 log 概率也不会完全相同，因为推理引擎和训练框架使用的 kernel、精度、并行方式、batch 形状都可能不同（Thinking Machines Lab, 2025）。这时 $\rho$ 并不是 0，任何一种 trainer 和 generator 之间的 KL 都会把 trainer 往推理引擎的数值误差上拉。所以对 age 为 0 的 token，也就是 trainer 和 generator 参数版本相同的 token，我们之前把 $\beta$ 设为 0。如果想从根本上消除这部分差异，需要让两边的计算方式完全一致，比如使用 batch invariant 的 kernel。这属于推理引擎的问题，不在这篇文章的范围之内。

## 8. 写在最后

这篇文章讨论了 KL 的写法和它产生的梯度之间的关系。分析一行 KL 代码时，可以先问下面几个问题：

* 样本是谁采的，是当前模型、旧模型，还是一个固定的 generator？
* $\theta$ 出现在哪几个地方，代码对哪几处求了导？
* 最后乘在 score 上的系数是什么，它对应的是哪个方向的 KL？
* 这个系数是在什么精度下算出来的？

对于异步 RL 中拉近 trainer 和 generator 的 KL，几种写法各有分工。$k_4$ 保护被压低的 token，$\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 约束被抬高的 token，$\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 对两类 token 都有作用，但对被压低的 token 的保护有上限。哪一种更合适，取决于训练中哪一类 token 的风险更大，也取决于 $\beta$ 和 advantage 的相对大小。

这篇文章讨论的是这几种写法在期望上的行为、它们的代价，以及计算时的精度问题。它们在实际训练中的表现，比如能不能防止训练崩溃，可能还和 head 的大小、tail 的处理方式以及模型的大小等因素有关。这些问题我们会在单独的一篇文章里用实验来讨论。

本文分析的是 token 级别的系数。之后的文章还会把同样的思路用到 prompt 级别。比如 GRPO 按组的标准差对 advantage 做标准化，相当于给每个 prompt 乘上了一个权重，这个权重决定了训练实际在优化哪一个目标。

## 附录：本文对 KL 方向的叫法，和其他公开资料有什么不同？

本文沿用了[之前介绍 Score Function Centering 加 KL 的文章](https://normaluhr.github.io/zh/p/score-function-centering-kl/)的叫法。那篇文章把 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ 的估计器叫作 sparse reverse-KL estimator，其中 $q_\phi$ 是 generator，$p_\theta$ 是 trainer。按照同一个规则，本文把参照分布在前、$\pi_\theta$ 在后的 KL 叫作 reverse KL，例如 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ 和 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$。$\pi_\theta$ 在前的 KL 则叫作 forward KL，例如 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 和 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$。

有一些公开资料的叫法正好相反。RLHF 的论文和代码库通常把正在训练的 $\pi_\theta$ 看作变分推断里那个用来做近似的分布，所以把 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ 叫作 reverse KL，把 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$ 叫作 forward KL。on-policy distillation 的文章也是这样，它们把学生在前、老师在后的 KL 叫作 reverse KL。两种叫法说的是同样的两个量，只是名字对调了。

对照其他资料读本文时，下面几处最容易混淆：

* 当 loss 用、样本来自当前模型时，$k_2$ 的梯度等于 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ 的梯度。本文说它给出的是 forward KL，叫法相反的资料会说是 reverse KL。
* 同样的条件下，$k_3$ 的梯度等于 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$ 的梯度。本文说它给出的是 reverse KL，叫法相反的资料会说是 forward KL。
* InstructGPT 式的 PPO 把 $k_1$ 放进 reward，梯度对应 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$，在本文里是 forward KL。GRPO 把 $k_3$ 当 loss，在样本来自当前模型的近似下，梯度对应 $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$，在本文里是 reverse KL。
* Liu 等人（2025）说"$k_3$ 当 loss 是 reverse KL 的一阶近似"，他们说的 reverse KL 是 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$，在本文里是 forward KL。
* 资料里常说"reverse KL 是 mode-seeking 的"，意思是 $\pi_\theta$ 在前的 KL 允许 $\pi_\theta$ 只保留参照分布的一部分取值。在本文的叫法下，这句话说的是 forward KL。6.2 节里，在 $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ 下，advantage 足够低的 token 会被一直压到 0，就是这个性质的表现。
* 之前那篇文章里的 KL 项是 $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$，它在那篇文章和本文里都叫 reverse KL，没有变化。

所以读这类材料时，最可靠的办法是对照公式。先看 KL 的前后两个位置上各是哪个分布，再看样本来自哪个分布，以及代码对哪几处求了导。这三点确定以后，名字就不会再造成误解。

## 参考文献

* John Schulman. [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html). 2020.
* Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. [Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization](https://arxiv.org/abs/2510.01555). 2025.
* Yunhao Tang, Rémi Munos. [On a few pitfalls in KL divergence gradient estimation for RL](https://arxiv.org/abs/2506.09477). 2025.
* Yifan Zhang et al. [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://arxiv.org/abs/2505.17508). 2025.
* Yifan Zhang. [On KL-Regularized Policy Optimization](https://arxiv.org/abs/2610.08963). 2026.
* Martin Marek, Max Ryabinin. [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807). 2026.
* DeepSeek-AI. [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556). 2025.
* Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024.
* Long Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). 2022.
* Audrey Huang et al. [Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization](https://arxiv.org/abs/2407.13399). 2024.
* Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference). 2025.
