---
title: "Score Function Centering：为什么 stale rollout 不一定需要 Importance Sampling"
date: 2026-10-11
category: post-training
series: grpo-line
part: 5
tags: ["GRPO"]
---

# Score Function Centering：为什么 stale rollout 不一定需要 Importance Sampling

这是 Score Function Centering 系列的第二篇。我相信读完本系列的[第一篇文章](https://normaluhr.github.io/zh/p/score-function-centering/)之后，你已经对 score function 是什么、为什么要做 Score Function Centering 以及它能达到什么效果有了初步的认识。在本文中，我希望给读者提供一个关于 Score Function Centering 的全新视角，看看它在 async RL 中解决了哪些问题，以及更重要的，它还没有解决哪些问题。同时，我希望在这篇文章里打破一个观念，那就是 stale rollout 一定要用 importance sampling 校正到 current policy 上，并且给出一些新的应对策略。

**人工与创新声明**：这篇文章并不是对已有工作和认知的总结，而是作者自己提出的对 Score Function Centering 的全新视角、理解和方法。和第一篇文章一样，本文由作者纯手工手搓，不是 AI 写的，当然如果你读的是英文版，很抱歉，它是 AI 翻译的。同时，作者强烈建议读者先阅读关于 Score Function Centering 的[第一篇文章](https://normaluhr.github.io/zh/p/score-function-centering/)，以确保理解本文使用的术语和持有的观点。


## 1. 从 Score Function Centering 没有解决的问题讲起

在这个章节，我们先谈谈 Score Function Centering，看看它修好了什么，又还没修好什么。

我们先来做一个简单的回顾。上一篇我们从 score function 本身讲起，最后解释了 async RL 里一个非常具体的问题：rollout 是 generator $q_\phi$ 产生的，但真正做 backward 的 trainer 已经变成了 $p_\theta$。

Standard score 虽然仍然是完全正确的 log-probability gradient：

$$s_\theta(a)=\nabla_\theta\log p_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]$$

但它在真正的数据生成分布 $q_\phi$ 下已经不再 zero-mean：

$$\mathbb E_{a\sim q_\phi}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.$$

Score Function Centering 做的事情是：既然 action 是从 $q_\phi$ 里 sample 出来的，那就不要再拿 $p_\theta$ 做 centering 了，而是改成：

$$\tilde s_{q,\theta}(a)=J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)].$$

于是：

$$\mathbb E_{a\sim q_\phi}[\tilde s_{q,\theta}(a)]=0.$$

到这里，上一篇最核心的问题解决了。我们重新拿回了 zero-mean property，也重新拿回了 baseline invariance。

但是这时候其实还有一个更大的问题没有解决：

**我们虽然把 score center 对了，但我们现在学到的，还是 generator $q_\phi$ 下的 learning signal。Trainer 明明已经是 $p_\theta$ 了，这真的没问题吗？**

这里，我们先区分一下 generator staleness 带来的两个问题。

Score Function Centering 修复的是 **score drift**。原来 standard score 在 $q_\phi$ 下有一个非零均值：

$$d_{q,\theta}:=\mathbb E_{q_\phi}[s_\theta]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta].$$

于是一个没有被 center 好的 reward update：

$$\mathbb E_{q_\phi}[A(a)s_\theta(a)]$$

可以拆成：

$$\mathbb E_{q_\phi}[A(a)s_\theta(a)]=\operatorname{Cov}_{q_\phi}(A,J_\theta)+\bar A_q\,d_{q,\theta},\qquad \bar A_q:=\mathbb E_{q_\phi}[A].$$

第二项的问题在于，$\bar A_q$ 只是一个平均 advantage，它并不描述 token 之间谁更好谁更差，却会直接控制沿着 trainer-generator mismatch direction 的更新。上一篇讲的 Score Function Centering，本质上就是把这一项拿掉：

$$g_q:=\mathbb E_{q_\phi}[A(a)\tilde s_{q,\theta}(a)]=\mathbb E_{q_\phi}[(A(a)-\bar A_q)J_\theta(a)]=\operatorname{Cov}_{q_\phi}(A,J_\theta).$$

这个式子非常重要。它告诉我们，SFC 之后留下来的更新只关心一件事情：**在真正生成数据的 sampler $q_\phi$ 下，哪些 action 的 advantage 相对更高，它们对应的 logit direction 是什么。**

但注意，它仍然是：

$$g_q=\mathbb E_{q_\phi}[(A-\bar A_q)s_\theta].$$

它不是 current trainer 下的：

$$g_p=\mathbb E_{p_\theta}[A s_\theta].$$

所以 SFC 并不能凭空把 $q_\phi$ 变成 $p_\theta$。它只是说：既然数据已经是 $q_\phi$ 生成的，那我们至少先把这个 distribution 下的 learning signal 定义干净。

## 2. Importance Sampling：从来如此，便对么？

### 2.1 Importance Sampling 的前世今生

接下来还没有解决的问题是：

**要不要再把这个 $q_\phi$ 下的 update，校正成 $p_\theta$ 下的 update？**

这个问题是什么意思？举一个极端一点的例子，假如 generator 用的是当前 trainer 10000 个 step 之前的版本（也就是说当前的 trainer 已经大学毕业，generator 还停留在小学生阶段），对于同样一个问题，比如 LLM 的耻辱问题：0.11 和 0.9 谁大，generator 给出来的回答是做了一堆分析，论证 11 比 9 大，最后得到答案 0.11 > 0.9。这个回答在当前 trainer 看来简直离谱得可笑，所以这个由 generator 给出的、置信度最高的答案，trainer 自己采样到它的概率几乎是 0。这个时候如果你去看梯度，你会发现这个答案错得离谱，reward 乃至 advantage 都非常低，正常来说由这个采样得到的梯度应该会带来一个比较大的矫正，但是它对 trainer 不仅可能没有意义，还有可能矫枉过正。如果你直接把梯度 apply 到 trainer 上，就类似于你逼着一个大学生对着一道 1+1=3 的错题学了一晚上，这个大学生应该这辈子都不会再犯这个错误了，但他可能也疯了。这就是为什么会有重要性采样（importance sampling）。它告诉你，当时采样到的这个样本，不管打分如何，对于当前版本的模型来说到底有多重要。

如果你学过一点 probability，看到上边的问题以后，第一反应大概率都是 importance sampling。假设我们真正想计算的是：

$$\mathbb E_{a\sim p_\theta}[f(a)].$$

把 expectation 展开：

$$\mathbb E_{a\sim p_\theta}[f(a)]=\sum_a p_\theta(a)f(a).$$

但现在我们手里的 sample 全部来自 generator：

$$a\sim q_\phi.$$

那怎么办？做一个非常简单的乘除：

$$\mathbb E_{a\sim p_\theta}[f(a)]=\sum_a q_\phi(a)\frac{p_\theta(a)}{q_\phi(a)}f(a)=\mathbb E_{a\sim q_\phi}\left[\frac{p_\theta(a)}{q_\phi(a)}f(a)\right].$$

于是定义：

$$r(a)=\frac{p_\theta(a)}{q_\phi(a)}.$$

这就是 importance ratio。所以 importance sampling 的本质其实很直白：**我虽然只能从 $q_\phi$ sample，但是给每个 sample 乘上 $p_\theta/q_\phi$，在 expectation 上就可以重新得到 $p_\theta$ 下的结果。**

对于上边的例子，如果大模型生成的回答 “0.11 > 0.9” 在 generator 看来概率是 0.99，但是在当前版本的 trainer 看来它的概率是 0.01，那么无论这个 rollout 带来的 advantage 有多正或者有多负，它都要乘上一个非常小的权重 0.01/0.99，从而变得不再重要。

把这个东西放回 policy gradient 里也完全一样。Current-policy update 是：

$$g_p=\mathbb E_{a\sim p_\theta}[A(a)s_\theta(a)].$$

如果数据来自 $q_\phi$，那么完整 importance weighting 给出：

$$g_p=\mathbb E_{a\sim q_\phi}\left[\frac{p_\theta(a)}{q_\phi(a)}A(a)s_\theta(a)\right].$$

从纯理论上讲，这当然是最标准的答案。数据来自旧 policy，但是我通过 $p/q$ 把它重新解释成 current policy 下的 expectation。其实 PPO 里大家非常熟悉的 ratio $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\mathrm{old}}(a_t|s_t)$，追根溯源也是同一个东西。$\pi_{\mathrm{old}}$ 产生 rollout，$\pi_\theta$ 是当前 policy，ratio 的作用就是做 distribution correction。

所以如果我们的目标明确是：**我一定要恢复 current-policy update**，importance sampling 完全合理。问题只是，到了 async LLM RL 里，这个技术就可能变得不那么合理了。

### 2.2 我们为什么开始不喜欢 Importance Ratio 了？

第一个问题非常现实：ratio 很容易变得很大。如果某个 token 在旧 generator 里 probability 是 $q_\phi(a)=10^{-6}$，但 trainer 已经把它学到了 $p_\theta(a)=10^{-3}$，那么：

$$r(a)=\frac{p_\theta(a)}{q_\phi(a)}=1000.$$

这意味着这个 token 在 generator 里几乎永远不会被 sample 到，但只要偶尔碰到一次，它的一条 gradient 就可能顶别人一千条。从理论推导出来的 expectation 本身没有问题，只是 Monte Carlo estimator 的 variance 会变得非常大。Long rollout 会让这个问题更明显。完整 trajectory 的 probability ratio 是：

$$\frac{p_\theta(y_{1:T})}{q_\phi(y_{1:T})}=\prod_{t=1}^{T}\frac{p_\theta(y_t|y_{<t})}{q_\phi(y_t|y_{<t})}.$$

哪怕每一个 token 上两个 policy 只差一点点，这些 ratio 连乘以后也可能变得非常极端。PPO 为什么要 clip ratio，本质上也是因为这一点：虽然 exact 的 importance weighting 理论上很正确，但直接拿来做 stochastic optimization，variance 可能大得完全不实用。所以[实际方法](https://arxiv.org/pdf/2609.20807)通常会做 truncation，例如：

$$w_\tau(a)=\min\left\{\frac{p_\theta(a)}{q_\phi(a)},\tau\right\}.$$

这当然会稳定很多，但一旦截断，你就不再是在做 exact importance sampling 了。比如一个 token 有 $q(v)=0.01$、$p_\theta(v)=0.10$，如果 $\tau=2$，那么完整 importance weighting 下它对应的 probability weight 是 $0.10$，截断以后只有 $0.01\times 2=0.02$。也就是说，这个 token 原本应该有的 update contribution 只保留了 20%。

这里尤其值得注意的一种情况是：trainer 已经学到了某个有效行为，所以这个 token 在 $p_\theta$ 里的 probability 上升了，但旧 generator $q_\phi$ 还没有跟上。那这个 token 恰好很容易进入高 ratio 区域。此时 clipping 有可能一边降低 variance，一边也削弱真正有价值的新行为。

注意，我这里说的是 **“有可能”**。Ratio 大并不自动等于这个 token 很重要，更不能直接推出 clipping 一定损害训练。它只是告诉我们：truncation 是按 distribution mismatch 来选择哪些 update 被削弱，而 task value 是否恰好和这种 mismatch 正相关，需要单独验证。这个 distinction 很重要。

但是除了 variance 之外，**Importance Sampling 还有一个和 Score Function Centering 本身更直接的冲突。**

### 2.3 Importance Sampling 和 Score Function Centering 不是好 buddy

Score Function Centering 构造的是一个专门在 $q_\phi$ 下 centered 的 score：

$$\tilde s_{q,\theta}(a)=J_\theta(a)-\mathbb E_{q_\phi}[J_\theta].$$

它满足：

$$\mathbb E_{a\sim q_\phi}[\tilde s_{q,\theta}(a)]=0.$$

如果这时候我们什么都不改，直接再乘一个 importance ratio：

$$\mathbb E_{a\sim q_\phi}\left[\frac{p_\theta(a)}{q_\phi(a)}\tilde s_{q,\theta}(a)\right],$$

根据 importance sampling identity，这个 expectation 会变成：

$$\mathbb E_{a\sim q_\phi}\left[\frac{p_\theta(a)}{q_\phi(a)}\tilde s_{q,\theta}(a)\right]=\mathbb E_{a\sim p_\theta}[\tilde s_{q,\theta}(a)]=\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{q_\phi}[J_\theta].$$

一般来说：

$$\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{q_\phi}[J_\theta]\neq 0.$$

也就是说，你刚刚花了半天在 $q_\phi$ 下把 score center 好，一个 $p/q$ 又把 expectation measure 从 $q_\phi$ 换成 $p_\theta$ 了。

**你费尽心思通过 Score Function Centering 找回来的 zero-mean property 又不再成立了。**

所以这里真正不能做的是：

**不能把一个已经按 $q_\phi$ center 好的 score，什么都不改就直接乘上 $p_\theta/q_\phi$。**

这不等于“importance sampling 和 score centering 永远不能组合”。当然可以重新推导一个 importance-weighted centering，让被加权后的 score 在 $q_\phi$ 下重新 zero-mean。[有的工作](https://arxiv.org/pdf/2609.20807)做的就是这一类事情，但是为了换来这样一个自洽的 estimator，这种做法更像是在理论上做了一次妥协，最后得到的结果反而两边都不够准确。

所以到这里其实出现了两条不同的路线。

第一条是 importance sampling 的路线：

$$a\sim q_\phi\quad+\quad r(a)=\frac{p_\theta(a)}{q_\phi(a)}\quad+\quad s_\theta(a)\quad\Longrightarrow\quad p_\theta\text{ 下的 current-policy update}.$$

直觉上的逻辑是：**我要把旧数据出现的概率校正成现在的。**

第二条是 Score Function Centering 的路线：

$$a\sim q_\phi\quad+\quad \tilde s_{q,\theta}(a)=J_\theta(a)-\mathbb E_{q_\phi}[J_\theta]\quad\Longrightarrow\quad q_\phi\text{ 下的 sampler-relative update}.$$

**我接受数据就是过去的 $q_\phi$ 产生的，我先保留它在自己 distribution 下表达的相对 reward signal。**

接下来，我们讲讲第二条路应该怎么走。


## 3. 如果不做 Importance Sampling，我该如何相信这个 update？

现在问题变得非常具体。我们决定保留 sampler-relative update：

$$g_q=\mathbb E_{q_\phi}[(A-\bar A_q)s_\theta],\qquad \bar A_q=\mathbb E_{q_\phi}[A].$$

但是我们真正关心的 trainer 已经是 $p_\theta$。Current-policy 的 local update 是：

$$g_p=\mathbb E_{p_\theta}[A s_\theta].$$

那 $g_q$ 和 $g_p$ 到底能差多远？我们先利用 standard score 的 zero-mean property：

$$\mathbb E_{p_\theta}[s_\theta]=0.$$

所以我们可以在 $g_p$ 里随便减去一个和 action 无关的 baseline。为了和 $g_q$ 对齐，我们故意减去同一个 $\bar A_q$：

$$g_p=\mathbb E_{p_\theta}[A s_\theta]=\mathbb E_{p_\theta}[(A-\bar A_q)s_\theta].$$

现在你会发现，两边已经非常接近了：

$$g_q=\mathbb E_{q_\phi}[(A-\bar A_q)s_\theta],\qquad g_p=\mathbb E_{p_\theta}[(A-\bar A_q)s_\theta].$$

它们的 integrand 一模一样。唯一的区别就是：一个在 $q_\phi$ 下取 expectation，一个在 $p_\theta$ 下取 expectation。把它们相减：

$$g_q-g_p=\sum_{v\in\mathcal V}\left(q_\phi(v)-p_\theta(v)\right)\left(A(v)-\bar A_q\right)s_\theta(v).$$

求和符号里边的每一项 $(q_\phi(v)-p_\theta(v))\times(A(v)-\bar A_q)s_\theta(v)$，就是每个 token 对 update mismatch 的贡献。第一项告诉你两个 policy 在这个 token 上概率差多少，第二项告诉你这个 token 对 learning direction 的影响有多大。

假设在我们关心的局部区域里，存在一个统一的 $B$，使得：

$$\left\|\left(A(v)-\bar A_q\right)s_\theta(v)\right\|\le B,\qquad \forall v\in\mathcal V.$$

那么用三角不等式：

$$\|g_q-g_p\|\le\sum_{v\in\mathcal V}|q_\phi(v)-p_\theta(v)|\left\|\left(A(v)-\bar A_q\right)s_\theta(v)\right\|\le B\sum_{v\in\mathcal V}|q_\phi(v)-p_\theta(v)|.$$

到这里，控制 gradient mismatch 已经变成了控制两个 distribution 的 $L_1$ distance。然后用 Pinsker inequality：

$$\sum_{v\in\mathcal V}|q_\phi(v)-p_\theta(v)|\le\sqrt{2D_{\mathrm{KL}}(q_\phi\|p_\theta)}.$$

于是我们得到：

$$\|g_q-g_p\|\le B\sqrt{2D_{\mathrm{KL}}(q_\phi\|p_\theta)}.$$

这个式子告诉我们一个非常重要的结论：**我们虽然没有通过 importance ratio 把 $q_\phi$ 下的 update 强行恢复成 $p_\theta$ 下的 update，但只要控制 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$，就可以控制 sampler-relative local update 和 current-policy local update 之间的误差。**

比如，如果我们能维持：

$$D_{\mathrm{KL}}(q_\phi\|p_\theta)\le\epsilon,$$

那么就能得到：

$$\|g_q-g_p\|\le B\sqrt{2\epsilon}.$$

所以这条路线和 importance sampling 的思路其实很不一样。Importance sampling 是说：我直接对 update 做 correction，希望恢复 $g_p$。而 Sampler-relative learning 加上 trainer-generator KL 是说：我保留 $g_q$，但是控制 $g_q$ 和 $g_p$ 的距离。

**一个是在 estimator 层面修复，一个是在 distribution 层面限制“不进行任何修复”而带来的误差。**

当然这个 bound 也不能过度解读。它是 fixed-prefix 的 local analysis，比较时固定了当前 prefix 和条件 advantage function，没有处理 policy 改变以后未来 prefix visitation distribution 也跟着变化的问题。所以这里的 $g_p$ 不是完整的 on-policy trajectory gradient。另外，$B$ 如果本身非常大，这个 bound 也可能很松。它给我们的是一个清楚的设计动机，而不是一个长期训练一定收敛的 theorem 或者 guarantee。

## 4. 等等，KL 不就是在把当前 Model 往过去拉吗？

没错。如果我们加的是：

$$D_{\mathrm{KL}}(q_\phi\|p_\theta),$$

而 $q_\phi$ 是旧 generator，$p_\theta$ 是当前 trainer，那么最小化这个 KL 确实会产生一个让 $p_\theta$ 靠近 $q_\phi$ 的力。

不过这里先停一下。**KL 是有方向的。** 我们这里写的是 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$，也就是以 generator $q_\phi$ 为 reference，去约束当前 trainer $p_\theta$。这个方向并不是随便选的，它和前面 Score Function Centering 去掉的那个 mismatch direction 恰好是同一个东西。

因为 $q_\phi$ 在 trainer 更新时是固定的：

$$D_{\mathrm{KL}}(q_\phi\|p_\theta)=\mathbb E_{a\sim q_\phi}\left[\log q_\phi(a)-\log p_\theta(a)\right].$$

对 $\theta$ 求梯度：

$$\nabla_\theta D_{\mathrm{KL}}(q_\phi\|p_\theta)=-\mathbb E_{a\sim q_\phi}\left[\nabla_\theta\log p_\theta(a)\right]=-\mathbb E_{q_\phi}[s_\theta].$$

而上一篇我们已经知道：

$$\mathbb E_{q_\phi}[s_\theta]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta].$$

所以：

$$\nabla_\theta D_{\mathrm{KL}}(q_\phi\|p_\theta)=\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{q_\phi}[J_\theta].$$

这就很有意思了。前面没有做 Score Function Centering 的时候，reward update 里多出来的那一项正是：

$$\bar A_q\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).$$

而最小化 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ 时，它提供的更新方向正好也是：

$$\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta].$$

区别只在于，这一次控制这个 direction 的不再是莫名其妙的平均 advantage $\bar A_q$，而是一个我们自己决定的 KL coefficient $\beta$。

所以这里用 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ 有一个非常自然的解释：**SFC 先把 reward update 里那个不受控制的 trainer-generator mismatch term 拿掉，然后 KL 再把同一个 direction 作为一个独立的 regularization term 加回来。**

这也是为什么 SFC + KL 这个分解非常巧妙。它并不只是简单地“加一个 KL 防止 trainer 和 generator 太不一样”，而是把原来混在 reward gradient 里的两件事情解耦了：

$$\mathbb E_{q_\phi}[A s_\theta]=\operatorname{Cov}_{q_\phi}(A,J_\theta)+\bar A_q\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).$$

SFC 保留第一项：

$$g_{\mathrm{reward}}=\operatorname{Cov}_{q_\phi}(A,J_\theta),$$

而 KL 单独负责第二个方向：

$$g_{\mathrm{KL}}=\beta\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).$$

于是最终：

$$g_{\mathrm{total}}=\operatorname{Cov}_{q_\phi}(A,J_\theta)+\beta\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).$$

这时候“学什么”由 reward 决定，而“允许 trainer 离 sampler 多远”由 $\beta$ 决定。

## 5. 最后一个问题：这个 Sampler 如果已经很旧了怎么办？

如果 KL 本来就在往 generator 的方向拉，那 generator 越旧，岂不是越把模型往过去拉？

没错。

所以真正做 async RL 的时候，每个 token 不应该共享同一个固定 KL strength。一个刚刚生成出来的 token，和一个几十个 optimizer step 以前生成出来的 token，它们对应的 sampler distribution 显然不是同一个质量的 reference。假设 token $t$ 是在 sampler step $\sigma_t$ 产生的，当前 trainer step 是 $\tau$，定义它的 age：

$$e_t=\tau-\sigma_t.$$

这里我提出一种在实际中可用的规则：

$$\beta_t=\begin{cases}0,&e_t=0,\\ \beta_0/\sqrt{e_t},&e_t\ge 1.\end{cases}$$

也就是说，token 越 stale，KL constraint 也就越弱。这个设计背后的 intuition 是：一个很新的 sampler 和当前 trainer 还很接近，它是一个比较可信的 local anchor；而一个已经很老的 sampler，本身就代表过去的 policy。我们当然还可以利用它产生 rollout 时留下来的 relative reward information，但没有必要再强迫当前 trainer 紧紧贴着它。

所以这里要区分两个东西：**“每个 token 用哪个 $q_t$ 做 reference”** 和 **“这个 reference 应该有多强的 KL weight”**。这是两个不同的问题。第一个问题由这个 token 真正的生成版本决定。第二个问题由它有多 stale 决定。

Fresh token 设 $\beta_t=0$ 也有一个很简单的理由：如果 generator 和 trainer 本来就是同一个 version，就没有必要把同版本下由于数值实现、推理 kernel 或 precision 带来的微小 distribution difference 再额外当成一个 imitation target。

但是这里也要明确：$1/\sqrt{e_t}$ 不是前面的 KL error bound 推导出来的答案。前面的理论告诉我们：$D_{\mathrm{KL}}(q_t\|p_\theta)$ 越小，sampler-relative local update 和 current-policy local update 的误差上界越小。它并没有告诉我们 $\beta_t$ 必须等于 $\beta_0/\sqrt{e_t}$。后者只是我推荐的一个 design choice，你也完全可以研究 constant coefficient、batch-level coefficient、其他 age schedule，甚至直接根据 observed KL 自适应调整。

最后还有一个很容易让人混淆的地方。前面我们说 reward-gradient 不再使用 importance weighting，但你可能在 sparse reverse-KL estimator 里仍然会看到 $p_\theta(v)/q_\phi(v)$。这并不矛盾。Reward-gradient importance ratio 的作用是：

$$\mathbb E_q\left[\frac pq f\right]=\mathbb E_p[f],$$

它在改变我们到底在哪个 distribution 下学习 reward signal。而 sparse KL 里的 ratio 是为了估计一个完全不同的目标：

$$\nabla_\theta D_{\mathrm{KL}}(q\|p_\theta).$$

比如对 logit $z_j$：

$$\frac{\partial D_{\mathrm{KL}}(q\|p_\theta)}{\partial z_j}=p_\theta(j)-q(j).$$

如果只 sample 一个 $v\sim q$，可以用：

$$\hat g_j(v)=\left(\frac{p_\theta(v)}{q(v)}-1\right)\mathbf 1[j=v],$$

因为：

$$\mathbb E_{v\sim q}[\hat g_j(v)]=q(j)\left(\frac{p_\theta(j)}{q(j)}-1\right)=p_\theta(j)-q(j).$$

所以这里的 $p/q$ 只是一个 sparse Monte Carlo gradient estimator 里的 coefficient。它没有把 reward update 从 $q$ measure 改成 $p$ measure。

## 6. 那这套方法到底 work 不 work？

前面全都是 intuition 和推导，所以这一节给一点真实的结果。

为了验证这套方法真的有效，我做了大量的实验，baseline 里也包括了 SFC 以及它已知的几个变种。受篇幅所限，这篇文章里只放其中很小的一部分。完整的实验 setup、结果、分析以及开源的代码，都会放在这个系列的第三篇里。

下面这张图里的七个 setting 都来自一个真正的异步 pipeline：generator 不停地产生 rollout，trainer 不停地更新，每个 token 都带着自己的 sampler 版本号，所以这里的 staleness 是实际测量出来的。任务是 [Countdown](https://arxiv.org/abs/2404.03683) 和 [GSM8K](https://arxiv.org/abs/2110.14168)，模型是 Qwen3-0.6B、Qwen3-4B、Qwen3-30B-A3B 和 GLM-4.5-Air（总参数 106B，激活参数 12B），completion budget 是 4096 个 token，每个 run 跑 3200 个 trainer step，实际测量到的平均 staleness 从 46 个 optimizer step 一直到 367 个。这两个任务的 reward 都只有 0 和 1 两种，比如 Countdown 要求模型用给定的几个数字写出一个结果等于目标值的算术表达式，算对了得 1 分，算错了得 0 分，所以图里的 reward 可以直接读成正确率。

图里的“4-prompt slice”说的是 trainer 消耗 rollout 的粒度。generator 一次产生 64 个 prompt 的 rollout，trainer 把这一批切成若干个 slice，每个 slice 走一个 optimizer step。4-prompt slice 就是每个 step 用 4 个 prompt，1-prompt slice 就是每个 step 只用 1 个 prompt。切得越细，同一批 rollout 要被消耗掉的 step 数就越多，sampler 相对 trainer 也就越旧，所以 1-prompt 的 setting 要比 4-prompt 的 setting stale 得多。图里每个 setting 下面都标了它实际测量到的平均 staleness。

为了和前面的讨论对齐，这里只放四个方法：只做 centering 的 SFC、SFC+TIS、SFC+MIS，以及本文的 SFC+KL($\beta_t$)，$\beta_0 = 0.3$。SFC+TIS 是在 center 过的 score 上再乘一个被截断的 importance ratio $\min(p_\theta(a)/q_\phi(a),\ 2)$，也就是 [truncated importance sampling](https://fengyao.notion.site/off-policy-rl) 的做法。SFC+MIS 则是只保留 ratio 落在 $[0.5,\ 5]$ 区间里的 token，其余的 token 会被直接 mask 掉，这个做法来自 [IcePop](https://arxiv.org/abs/2510.18855)。把这两种 ratio correction 和 score centering 组合起来，是前面提到的[那篇工作](https://arxiv.org/abs/2609.20807)的做法。

SFC+KL 是这七个 setting 里唯一一个每次都训练到最后的方法，而且在每一个 setting 上，最终的 held-out evaluation reward 也都是它最高。

<figure data-chart="sfc-methods">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-11-score-function-centering-kl/methods.png" alt="七个异步 setting 下四个方法最终的 held-out evaluation reward">
</noscript>
</figure>
<script src="/js/sfc-methods-chart.js" defer></script>

---

回到这一篇一开始的问题：**从 stale rollout 学习，是不是一定要先把它“修正成现在”以后才能用？**

答案是“不一定”。

Importance sampling 的回答是：数据虽然来自 $q_\phi$，但我要通过 $p_\theta/q_\phi$ 尽量恢复 current-policy expectation。而 Sampler-relative learning 的答案是：数据既然就是 $q_\phi$ 产生的，那我先保留它在自己 distribution 下表达的相对 reward signal；至于它和当前 trainer 已经差了多少，我用一个独立的 KL constraint 去控制。

简单来说，本文提出的方法淘汰掉了 importance sampling，同时可以看作是 Score Function Centering、sparse KL 以及 staleness scaling 的结合，三者各司其职：

* SFC：我接受数据来自 $q_\phi$，所以就在 $q_\phi$ 下 center。
* KL：但我不允许 $q_\phi$ 和 $p_\theta$ 离得太远。
* Staleness scaling：而且 $q_\phi$ 越老，我越觉得它的参考意义没有那么大。

我觉得这是理解 async RL 里 trainer-generator mismatch 的一条新的思路：**不要先假设 stale data 一定要被重新加权成 current-policy data。你也可以保留它本身携带的 learning signal，然后把“学什么”和“离当前 policy 多远”拆成两个独立的问题来处理。**
