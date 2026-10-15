---
title: "使用 Score Function Centering 和 Trainer-Generator KL 稳定你的异步 RL 训练"
date: 2026-10-14
category: post-training
series: grpo-line
part: 6
tags: ["GRPO"]
---

# 使用 Score Function Centering 和 Trainer-Generator KL 稳定你的异步 RL 训练

这是 Score Function Centering 系列的第三篇。在前两篇里，我们从 score function 出发，一步一步推出了 SFC+KL 这个方法，但实验只在第二篇的最后放了一张图。在这一篇里，我想把实验完整地讲一讲，看看前两篇的分析在实际训练里是什么样子，也回答一些前两篇没有回答的问题。

如果你还没有读过前两篇，我强烈建议先读一下[第一篇](https://normaluhr.github.io/zh/p/score-function-centering/)和[第二篇](https://normaluhr.github.io/zh/p/score-function-centering-kl/)，这一篇会直接沿用前两篇的符号和结论。为了方便，下面我先把前两篇的主要内容回顾一遍。

**开源**：这一篇用到的实验配置、训练曲线和代码都已经开源，链接：[github.com/NormalUhr/score-function-centering-trainer-generator-kl](https://github.com/NormalUhr/score-function-centering-trainer-generator-kl)

## 前情回顾

在 async RL 里，rollout 是 generator 产生的，而 trainer 在用上这些 rollout 之前，往往已经又更新了很多个 step。我们把产生 rollout 的 generator 记作 $q_\phi$，正在做 backward 的 trainer 记作 $p_\theta$。

[第一篇](https://normaluhr.github.io/zh/p/score-function-centering/)从 score function 讲起。Softmax 下的 score 等于被采样到的 action 的 logit Jacobian，减去所有 action 的 logit Jacobian 在 $p_\theta$ 下的平均：

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

所以只要 action 是从 $p_\theta$ 里 sample 出来的，score 的期望就是 0。正因为如此，我们才可以放心地给 reward 减去一个 baseline，也就是使用 advantage。但在 async RL 里，action 是从 $q_\phi$ 里 sample 出来的，这个 zero-mean property 就不再成立了：

$$
\mathbb E_{a\sim q_\phi}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

这个非零的期望在训练中的实际后果是，平均 advantage 会乘上这个方向，额外产生一个沿着 trainer 和 generator 之间差异方向的 gradient。平均 advantage 为正的时候，这个 gradient 会把 trainer 往旧 generator 拉，而拉多少完全由平均 advantage 的大小决定。Score Function Centering（下文简称 SFC）的做法是，改用真正生成 action 的 $q_\phi$ 来做 centering：

$$
\tilde s_{q,\theta}(a)=J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

这样 zero-mean property 就回来了。为了计算 $\mathbb E_{q_\phi}[J_\theta]$，generator 在生成每个 token 的时候，需要顺手把当时 distribution 的一部分信息送回给 trainer，比如概率最高的若干个 token，以及从剩下的 tail 里 sample 出来的几个 token。下文把这一份信息叫做 payload。

[第二篇](https://normaluhr.github.io/zh/p/score-function-centering-kl/)接着讨论了 SFC 没有解决的问题。SFC 把 score center 对了，但我们学到的仍然是 $q_\phi$ 下的 learning signal，而 trainer 已经变成了 $p_\theta$。最常见的做法是 importance sampling，给每个 token 乘上 $p_\theta/q_\phi$，把 update 校正回 $p_\theta$ 下。可是 ratio 很容易变得非常大，实际使用的时候只能截断，而截断又会带来 bias。更麻烦的是，在一个已经按 $q_\phi$ center 好的 score 上再乘 $p_\theta/q_\phi$，expectation 又被换回到了 $p_\theta$ 下，SFC 好不容易拿回来的 zero-mean property 也就跟着没了。

所以第二篇走了另一条路。我们接受数据来自 $q_\phi$ 这个事实，不再校正 update，转而控制 $q_\phi$ 和 $p_\theta$ 之间的距离。第二篇推导了一个局部的 bound：

$$
\|g_q-g_p\|\le B\sqrt{2D_{\mathrm{KL}}(q_\phi\|p_\theta)},
$$

其中 $g_q$ 是 SFC 在 $q_\phi$ 下给出的 update，$g_p$ 是 current policy 下的 update。也就是说，只要把 $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ 控制住，这两个 update 就不会差得太远。更巧的是，这个 KL 的梯度方向，恰好就是 SFC 从 reward gradient 里拿掉的那个方向。于是最终的 update 可以写成：

$$
g=\operatorname{Cov}_{q_\phi}(A,J_\theta)+\beta\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).
$$

第一项由 reward 决定学什么，第二项由 $\beta$ 决定允许 trainer 离 sampler 多远，这两件事被拆开了。另外，sampler 越旧，它就越不适合作为 reference，所以每个 token 的 KL 系数会随着这个 token 的 age $e_t$ 衰减：

$$
\beta_t=\begin{cases}0,&e_t=0,\\ \beta_0/\sqrt{e_t},&e_t\ge 1.\end{cases}
$$

这就是第二篇提出的方法，下文统一叫它 SFC+KL。它和 importance sampling 最大的区别在于，它在 reward gradient 上完全不使用 importance ratio。

第二篇最后的那张图里，七个异步 setting 上，只有 SFC+KL 每次都没有 collapse。不过那张图只给出了结果，这一篇想把每个方法在训练中发生了什么讲清楚。下面是这一篇会讨论的几个问题：

* 只做 SFC 够不够？
* 在 SFC 上加 importance sampling 会出什么问题？
* importance sampling 能不能干脆去掉？
* SFC 需要 generator 送回来多少信息？
* 换了 optimizer、数据集和模型以后，这些结论还成立吗？
* 如果 trainer 和 generator 的差异不是 staleness 造成的，结果会怎么样？

最后还有一个番外，把 SFC+KL 和 PPO、DAPO、GSPO、DPPO、TOPR 放在一起比较，也看看它们能不能组合起来用。如果你只想看结论，可以直接跳到最后的 takeaway。

## 1. 实验配置

所有实验都运行在一个真正的异步 pipeline 上。generator 不停地产生 rollout，trainer 不停地更新，每个 token 都记录着生成它的 sampler 版本，所以每个 token 的 staleness 都是实际测量出来的。

大部分实验用的是 Qwen3-0.6B 和 [Countdown](https://arxiv.org/abs/2404.03683) 数据集。Countdown 会给模型几个数字和一个目标值，模型要用这几个数字写出一个结果等于目标值的算术表达式，算对得 1 分，算错得 0 分，所以下文的 reward 可以直接理解成正确率。每个 run 训练 3,200 个 step，context length 最长 4,096 个 token，optimizer 是 SGD，SFC+KL 的 $\beta_0$ 是 0.3。

staleness 的大小由 trainer 消耗 rollout 的粒度决定。generator 每次产生 64 个 prompt 的 rollout，trainer 把这一批 rollout 切成若干个 slice，每个 slice 做一次 optimizer step。每个 slice 有 4 个 prompt 的时候，我们叫它 4-prompt slice，这时 token 的平均 staleness 大约是 50 个 step。每个 slice 只有 2 个 prompt 的时候，同样一批 rollout 要多用一倍的 step 才能消耗完，平均 staleness 也就变成了 100 个 step 左右。

下文主要比较四个基于 SFC 的方法。SFC 就是只做 centering。SFC+TIS 在 center 过的 score 上再乘一个截断过的 importance ratio $\min(p_\theta/q_\phi,\ 2)$，也就是 [truncated importance sampling](https://fengyao.notion.site/off-policy-rl) 的做法。SFC+MIS 只保留 ratio 落在 $[0.5,\ 5]$ 之间的 token，其余的 token 会被直接 mask 掉，这个做法来自 [IcePop](https://arxiv.org/abs/2510.18855)。把这两种 ratio correction 和 SFC 结合起来，是[这篇工作](https://arxiv.org/abs/2609.20807)的做法。最后一个就是这个 blog 系列提出的 SFC+KL。

比较方法的时候，看的是每个 run 在整个训练过程中最好的那一次 held-out evaluation reward，下文简称最好成绩。实际训练的时候，我们最后留下来用的，一般也就是 evaluation 最好的那个 checkpoint。这样一来，一个 run 即使后来 collapse 了，它在 collapse 之前的最好成绩也会被算进去，所以下文除了比较最好成绩，也会单独看每个 run 有没有 collapse。

## 2. 只做 SFC 够不够？

这个问题是为了探究是否仅仅 score function centering 本身就能起到稳定 async RL 训练的效果。答案是否定的。在 Countdown 的两个 setting 上，只做 SFC 的 run 基本没有学起来。

下面两张图分别是四个方法在 Countdown 上的 training reward 和 held-out evaluation reward，左边是 4-prompt slice，右边是 2-prompt slice。这一节和后面两节都会用到这两张图。

<figure data-chart="countdown-train">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/countdown-train.png" alt="Countdown 上四个方法的 training reward">
</noscript>
</figure>
<script src="/js/sfc-exp-charts.js" defer></script>

<figure data-chart="countdown-eval">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/countdown-eval.png" alt="Countdown 上四个方法的 held-out evaluation reward">
</noscript>
</figure>

先看 SFC。在两个 setting 上，它的 training reward 在最开始的一两百个 step 里一直在 0.1 附近，然后掉到了 0，之后就再也没有起来。这两个 run 分别在第 400 步和第 640 步被停掉了，它们的最好成绩就是训练开始之前的那一次 evaluation，分别是 0.080 和 0.094。

这和第二篇开头的分析是一致的。SFC 修好的是 score drift，但它并不限制 trainer 和 sampler 之间的距离。在 sampler 平均比 trainer 旧 50 到 100 个 step 的情况下，只做 SFC 的训练很快就 collapse 了。第二篇加上 KL，就是为了补上这一块。

## 3. 在 SFC 上加 importance sampling 会怎么样？

在 4-prompt slice 上，SFC+TIS 和 SFC+MIS 都能学一段时间，但都撑不到最后。到了 staleness 更大的 2-prompt slice 上，它们从一开始就没有学起来。另外，我在它们身上还看到了一个 SFC+KL 没有的现象，它们生成的回答特别长。SFC+TIS 尤其明显，它会让回答在训练中越变越长，被截断的比例也因此高得多。

先看 SFC+MIS。在 4-prompt slice 上，它一开始学得不错，第 800 步的 evaluation 是 0.646，同一时间的 SFC+KL 是 0.725。但是从那以后它就开始慢慢往下走，最后在第 3,168 步 collapse 了。在 staleness 更大的 2-prompt slice 上，它从一开始就没有学起来，第 400 步就被停掉了。

SFC+TIS 的情况不太一样。在 4-prompt slice 上，它的 training reward 在前 1,300 步里一度涨到 0.7 左右，和 SFC+KL 相差不大，只是波动大得多。但它的 evaluation 一直不高，第 800 步只有 0.389。到了第 1,350 步左右，它的 training reward 突然掉了下来，很快就掉到了 0，之后一直到第 3,200 步都没有恢复，evaluation 也一直是 0。在 2-prompt slice 上，它从头到尾都没有学起来，最好成绩就是训练开始时的 0.111。

再看回答的长度。统计训练最后的 200 步，在 4-prompt slice 上，因为超过 4,096 个 token 的 context length 而被截断的回答，SFC+TIS 占了 99%，SFC+MIS 占了 49%，而 SFC+KL 只有 19%。2-prompt slice 上没有学起来的三个 run，这个比例也都在 89% 到 97% 之间。

第二篇讲过 ratio correction 可能带来的两个问题，一个是截断带来的 bias，另一个是 ratio 和 SFC 不兼容，在 center 好的 score 上再乘 ratio，zero-mean property 又没了。MIS 的情况和 TIS 类似，只是它会把 ratio 超出区间的 token 整个 mask 掉。对于 SFC+TIS，我回头仔细看了训练日志，也抽查了模型生成的内容，3.2 节会具体讲它是怎么一步步崩掉的。

看到这么多回答被截断，我想到的第一个可能是 4,096 个 token 的 context length 太短了。如果 SFC+TIS 本来就需要更长的回答才能把题做对，长度一受限，reward 掉下来也就说得通了。下面这一小节专门来看这件事。

### 3.1 是不是 context length 的上限太小导致 SFC+TIS collapse 了？

不是。把 context length 的上限提高到 8,192 和 16,384 以后，被截断的回答少了很多，但 SFC+TIS 还是 collapse 了。

这一组实验固定在 Countdown 的 4-prompt slice 上，方法是 SFC+TIS，只把 context length 的上限分别提高到原来的两倍和四倍。

<figure data-chart="budget">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/budget.png" alt="三种 context length 上限下 SFC+TIS 的 training reward 和 held-out evaluation reward">
</noscript>
</figure>

下面这张表是这几个 run 在训练结束前最后 200 步里被截断的比例，以及它们的最好成绩，4,096 下的 SFC+KL 作为参照。

| 方法 | context length 上限 | 最后 200 步被截断的比例 | 最好成绩 | training reward 掉下来的时间 |
| --- | --- | --- | --- | --- |
| SFC+KL | 4,096 | 19% | 0.725 | 没有掉下来 |
| SFC+TIS | 4,096 | 99% | 0.389 | 第 1,400 步左右 |
| SFC+TIS | 8,192 | 31% | 0.639 | 第 900 步左右 |
| SFC+TIS | 16,384 | 15% | 0.650 | 第 1,100 步左右 |

上限提高以后，被截断的比例确实降了下来，16,384 的那一次甚至比 4,096 下的 SFC+KL 还低。多出来的长度也确实让 SFC+TIS 学得更好了一些，最好成绩从 0.389 提高到了 0.65 左右，不过仍然比 4,096 下 SFC+KL 的 0.725 低。

但是它们最后还是 collapse 了，而且 collapse 的时间和 4,096 的那一次差不多。

所以长度的上限并不是 SFC+TIS collapse 的原因。上限加大以后，它在 collapse 之前能多学一点，但 collapse 本身并没有被避免。回答为什么会越来越长，下面一小节来看。

### 3.2 为什么 SFC+TIS 的回答会越来越长？

简单来说，这些很长的回答是一段越来越没有方向的搜索，到最后变成了没有意义的循环，所以多给多少长度都不够用。

先看看 Countdown 上一个很长的回答里都写了什么。我取了 16,384 那个 run 在第 319 步、还没有开始 collapse 的模型，在三道 held-out 的题目上抽样，长度上限设成 2,048 个 token。比较简单的那道题是用 28、10、33、16 凑出 67，模型两次都做对了，分别用了 595 和 790 个 token，过程是一段一段地试：

```
What about:
33 - 16 = 17
28 - 10 = 18
18 + 17 = 35. No.
...
How about:
(16 + 33) + (28 - 10) = 49 + 18 = 67. Yes!
```

另外两道难一些的题，模型一直写到了 2,048 个 token 的上限，内容是反复出现的 Let me think、What about 和 Let me try，每一次都把同样的几个部分和重新算一遍。一个正常训练的 SFC+KL 在这三道题上也是这样，简单的那道先试再做对，难的两道也一直写到了上限。所以在 Countdown 上，一个很长的回答其实就是一段没有收敛的暴力搜索，它有多长，取决于这段搜索有多没有方向。

SFC+TIS 的问题在于，训练会让这段搜索变得越来越没有方向。从日志来看，它的 collapse 分成三个阶段，在 4,096 和 8,192 两个上限下都是一样的。

第一个阶段是 entropy 一路上涨。模型每个 token 的 entropy 连续几百个 step 都在涨，而 trainer 和 sampler 之间的 KL 一直很小。在 8,192 的上限下，entropy 从前 200 步的 0.52，涨到了第 576 到 639 步的 0.83，再到第 768 到 895 步的 1.4。回答的平均长度也跟着从 381 个 token 涨到了 4,896 个 token，而 training reward 这时还能维持在 0.5 到 0.75 之间。在 4,096 的上限下，entropy 在第 1,200 到 1,399 步之间从 0.40 涨到了 0.98。4,896 个 token 已经超过了 4,096 的上限，在 4,096 的上限下，这些回答就会被截断。再加上 collapse 以后模型会一直重复到上限，SFC+TIS 被截断的比例才会比 SFC+KL 高得多。

第二个阶段是 KL 失控。在一到两个 64 步的窗口里，trainer 和 sampler 之间的 KL 突然涨了两个数量级，8,192 的 run 涨到了 8.4，4,096 的 run 涨到了 5.0。与此同时，gradient norm 往下掉，reward 也掉到了 0。我的理解是，trainer 离 sampler 越远，sampler 生成的那些 token 在 trainer 下的概率就越低，它们的 ratio 不是接近 0，就是被截断，sampler 的数据几乎不再起作用，也就没有任何东西能把 trainer 拉回来。另外，被 $\min(\cdot,\ 2)$ 截断的 token 从头到尾只有 1.5% 到 3%，所以截断本身看起来并不是主要的原因。

第三个阶段是训练彻底停住。一个 batch 里所有的回答都是错的，group 内 center 之后的 advantage 全是 0，gradient norm 正好是 0，模型就再也动不了了。这也是为什么 4,096 的那一次 reward 掉到 0 以后，一直到第 3,200 步都没有恢复。这时候模型输出什么，取决于长度的上限。在 4,096 的上限下，模型几乎是确定性地重复同样的内容，直到撞上上限，这就是那 99% 被截断的回答。在 8,192 的上限下，模型会以 0.45 左右的 entropy 胡乱写上 2,300 个 token 左右，里面找不到一个能被解析的答案。所以多给的长度，最后只是被更多没有意义的内容填满了。

同一个 setting 上的 SFC+KL 完全没有走上这条路。在整整 3,200 个 step 里，它的 entropy 一直在 0.23 到 0.35 之间，和 sampler 之间的 KL 一直在 0.003 到 0.004 之间。我觉得原因在于，KL 约束的是每个位置上完整的 distribution，包括这个 distribution 有多分散，也包括结束这个 sequence 的那个 token 的概率。importance ratio 只作用在被 sample 到的那个 token 上，做不到这一点。

## 4. 那 importance sampling 能不能干脆去掉？

可以。SFC+KL 在 reward gradient 上完全不使用 importance ratio。在第 2 节的两张图里，它是唯一一个在两个 setting 上都稳定训练到最后的方法，最好成绩分别是 0.725 和 0.670。

放到这篇文章的所有 setting 里看，默认配置下的 SFC+KL 一次都没有 collapse 过，这也包括后面会讲到的换 optimizer、换数据集、换模型，以及量化 sampler 的那几组实验。在同样的这些 setting 里，SFC、SFC+TIS 和 SFC+MIS 都 collapse 过不止一次。

这也是第二篇最想说明的一件事。stale rollout 不一定要被重新加权成 current policy 下的数据，我们可以保留它在 $q_\phi$ 下的 learning signal，再用 KL 控制 trainer 离 sampler 有多远，训练一样可以稳定下来。

## 5. SFC 需要 generator 送回来多少信息？

在 Countdown 上，payload 的构成对最好成绩的影响不大，每个位置送回 16 个 token 就够了。区别在于稳定性，只送回 top-16 的 run 后来 collapse 了，同样是 16 个 token、但其中有一部分来自 tail sample 的配置没有。

先说一下这里在比较什么。前面所有实验用的默认 payload 是 top-128，generator 在每个位置只送回概率最高的 128 个 token 和它们的 logprob，剩下的 tail 不做 sample。trainer 端会把 tail 的总概率设成 sampler 给出的值，至于 tail 里每个 token 分到多少概率，则按照 trainer 自己的 distribution 来分配。另一种做法是第一篇 3.5 节讲过的 head 加 tail。head 照样送回 top-k，tail 部分则从 sampler 里额外 sample 几个 token 送回来，下文把它们叫做 tail sample。payload 的宽度，就是这两部分加起来一共有多少个 token。

这一组实验全部在 Countdown 的 4-prompt slice 上，方法是 SFC+KL，只改变 payload 的构成。

<figure data-chart="payload">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/payload.png" alt="不同 payload 下 SFC+KL 的最好成绩">
</noscript>
</figure>

九个配置的最好成绩都在 0.68 到 0.73 之间，所以单看最好成绩，我分不出这些配置的好坏。

区别出在稳定性上。只送回 top-16 的 run 在第 2,000 步的成绩是 0.713，和其他配置一样好，但它在第 2,992 步 collapse 了。同样是 16 个 token，top-8 加 8 个 tail sample 和 top-4 加 12 个 tail sample 都稳定地训练到了最后。

我的理解是这样的。只送回 top-k 的时候，tail 是用 trainer 自己的 distribution 补上的，补得准不准，取决于 tail 占了多少概率，也取决于 trainer 和 sampler 在 tail 上差了多少。top-k 越窄，留给 tail 的概率就越多，这个近似的误差也就越大。tail sample 直接来自 sampler 自己的 distribution，所以对 tail 的估计没有这个偏差，代价是多了一些 variance。

这个结果也说明，collapse 有时候会来得很晚。如果只训练 2,000 步，我们大概会很开心地得出“只送回 top-16 就够了”这个结论。

payload 的宽度还会影响速度。我对每种 payload 单独测试了 100 个 step，不做 evaluation，seed 也相同。

| payload | rollout 文件大小 | generator 每 decode 128 个 token 的时间 | trainer 每个 step 的计算时间 |
| --- | --- | --- | --- |
| top-128 | 2,074 MB | 4.28 秒 | 1.22 秒 |
| top-64 加 64 个 tail sample | 2,074 MB | 4.82 秒 | 1.20 秒 |
| top-32 加 96 个 tail sample | 2,074 MB | 5.03 秒 | 1.21 秒 |
| top-32 | 538 MB | 4.19 秒 | 1.24 秒 |
| top-16 加 16 个 tail sample | 538 MB | 4.32 秒 | 1.21 秒 |
| top-8 加 24 个 tail sample | 538 MB | 4.38 秒 | 1.22 秒 |
| top-16 | 282 MB | 4.14 秒 | 1.23 秒 |
| top-8 加 8 个 tail sample | 282 MB | 4.23 秒 | 1.22 秒 |
| top-4 加 12 个 tail sample | 282 MB | 4.26 秒 | 1.23 秒 |

trainer 每个 step 的计算时间几乎不受 payload 影响，一直在 1.2 秒左右。变化最大的是 rollout 文件。宽度从 128 降到 16，rollout 文件从 2,074 MB 降到了 282 MB，读取和上传的时间也从 1.4 秒到 1.5 秒降到了 0.2 秒左右。tail sample 的代价落在 generator 上，同样的宽度下，sample 的 token 越多，decode 就越慢。宽度是 128 的时候，96 个 tail sample 让 decode 的时间多了 18%，宽度是 16 的时候只多了 2% 到 3%。

如果想省带宽，我会选 top-8 加 8 个 tail sample 这样的配置。它的 rollout 文件大约是默认配置的七分之一，generator 的 decode 时间和默认配置差不多，最好成绩是 0.705，和其他配置相当。

## 6. 换 optimizer、换数据集、换模型以后呢？

在这五个 setting 上，SFC+KL 都训练到了最后，没有 collapse。不过看最好成绩的话，它并不是每次都最高，在 Qwen3-4B 和 GLM-4.5-Air 上，SFC+MIS 的最好成绩比它高。

这里一共有五个 setting，也就是第二篇那张图里除了 Countdown 以外的五个。第一个把 optimizer 换成了 AdamW，其他和 Countdown 的 4-prompt slice 一样。第二个把数据集换成了 [GSM8K](https://arxiv.org/abs/2110.14168)，模型仍然是 Qwen3-0.6B，用的也是 4-prompt slice。剩下三个换成了更大的模型，分别是 Qwen3-4B、Qwen3-30B-A3B 和 GLM-4.5-Air，其中 GLM-4.5-Air 的总参数是 106B，激活参数是 12B。这三个模型用的是 1-prompt slice，平均 staleness 大约在 90 到 370 个 step 之间。

<figure data-chart="settings">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings.png" alt="另外五个 setting 上四个方法的最好成绩">
</noscript>
</figure>

下面两张图是这五个 setting 上的 training reward 和 held-out evaluation reward 曲线。

<figure data-chart="settings-train">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings-train.png" alt="另外五个 setting 上四个方法的 training reward">
</noscript>
</figure>

<figure data-chart="settings-eval">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings-eval.png" alt="另外五个 setting 上四个方法的 held-out evaluation reward">
</noscript>
</figure>

先看 optimizer。换成 AdamW 以后，SFC+KL、SFC+TIS 和只做 SFC 的 run 在第 800 步都到了 0.7 以上，分别是 0.738、0.725 和 0.715，SFC+MIS 的最好成绩低一些，是 0.652。区别出在后面。SFC+TIS 和 SFC+MIS 后来都 collapse 了。只做 SFC 的 run 没有 collapse，但它的 evaluation 一路降到了 0.527。只有 SFC+KL 一直保持在 0.7 以上。这也是只做 SFC 唯一一次没有 collapse，我猜这是因为 AdamW 会对每个参数的更新幅度做归一化，这本身就起到了一些约束的作用，但这个猜测我没有验证过。

再看数据集。在 GSM8K 上，SFC+TIS 训练到了最后，最好成绩是 0.785，SFC+KL 是 0.803，两者差不多。这是 SFC+TIS 在这五个 setting 里唯一一次表现正常。它的回答依然很长，被截断的比例是 97%，SFC+KL 是 43%，只是在 GSM8K 上，这并没有妨碍它把题做对。只做 SFC 和 SFC+MIS 在 GSM8K 上都 collapse 了。

最后是三个更大的模型。SFC+KL 的最好成绩分别是 0.744、0.877 和 0.750。SFC+TIS 在 Qwen3-4B 和 GLM-4.5-Air 上 collapse 了，在 Qwen3-30B-A3B 上虽然没有 collapse，但也从来没有超过训练开始时的 0.680。只做 SFC 的 run 只在 Qwen3-4B 上运行过，也 collapse 了。

SFC+MIS 在这三个模型上的表现很有意思。它都没有 collapse，而且在 Qwen3-4B 上最好到了 0.871，在 GLM-4.5-Air 上最好到了 0.777，都比 SFC+KL 高。但它的成绩起伏很大，过了最高点以后就掉了下来，最后一次 evaluation 分别只有 0.668 和 0.574。SFC+KL 在 GLM-4.5-Air 上的成绩在第 400 步先掉了一下，之后就慢慢往上走，最好成绩出现在训练的最后。

这里有一点要说明一下。SFC+KL 在 Qwen3-4B 和 Qwen3-30B-A3B 上只在训练开始和结束的时候各做了一次 evaluation，SFC+MIS 则是每 400 步做一次。所以在这两个模型上，SFC+KL 的最好成绩其实就是它最后一次的成绩，它在训练中间有没有到过更高的地方，我们并不知道。

还有一件事我没有想明白。SFC+MIS 在 Qwen3-0.6B 的四个 setting 上全都 collapse 了，但到了这三个更大的模型上，staleness 并没有变小，它反而都撑住了。

## 7. 如果 trainer 和 generator 的差异不是 staleness 造成的呢？

那 KL 就没有明显的优势了。在量化 sampler 的这一组实验里，SFC+TIS 和 SFC+KL 的表现几乎一样。

前面的讨论都默认，trainer 和 generator 的差异来自 staleness。但在实际的系统里，还有一个常见的来源。generator 为了推理得更快，会使用量化过的模型，而 trainer 用的是没有量化的模型，所以即使两边用的是同一个版本的参数，它们给出的 distribution 也不完全一样。

这一组实验用的是 Qwen3-30B-A3B-Base 和 [INTELLECT-2](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-2-RL-Dataset) 里的数学题，generator 分别用了三种量化方式，每个 run 训练 800 个 step。和前面不同的是，这一组里 SFC+KL 的 $\beta_0$ 是 0.03。

| generator 的量化方式 | SFC+TIS | SFC+KL |
| --- | --- | --- |
| FP8 | 0.401 | 0.408 |
| FP8，以及 FP4 的 KV cache | 0.401 | 0.405 |
| INT8，以及 INT4 的 KV cache | 0.366 | 0.354 |

三组的差距都在 0.012 以内，我会把它们看成一样。在这一组实验上，TIS 和 KL 一样好用。作为参照，这几个 run 在训练开始时的成绩都在 0.33 到 0.35 之间。在 INT8 加 INT4 KV cache 的那一组里，两个方法训练 800 步之后都只提高了 0.01 左右，量化到这个程度，两个方法都几乎没有学起来。

我觉得原因在于，第二篇的整个论证都是围绕 staleness 展开的。sampler 越旧，它和 trainer 的距离就越大，KL 要控制的正是这个距离。量化带来的差异是另一种情况，两边用的是同一版参数，distribution 只在数值上有一些差别，KL 能约束的东西本来就不多。TIS 这边，ratio 大多应该在 1 附近，被截断的 token 不多，截断带来的 bias 也就不大。第二篇给 fresh token 设 $\beta_t=0$，也是出于同样的考虑，同一个版本下由数值精度带来的微小差异，没有必要再当成一个模仿的目标。

## 8. 番外：和其他方法放在一起，以及能不能组合

前面几节都只在基于 SFC 的四个方法之间比较。这一节把 PPO、DAPO、GSPO、DPPO 和 TOPR 也放进来。PPO 可以看[这一篇](https://normaluhr.github.io/zh/p/grpo/)，DAPO 和 GSPO 可以看[这一篇](https://normaluhr.github.io/zh/p/grpo-dapo-gspo/)。[DPPO](https://arxiv.org/abs/2602.04879) 用对 policy divergence 的约束代替了 PPO 的 ratio clipping。[TOPR](https://arxiv.org/abs/2503.14286) 对正 reward 和负 reward 的样本，采用了不同的 importance sampling 截断方式。

### 8.1 和其他方法放在一起

单独比较的话，SFC+KL 并不是在每个 setting 上都最好。下面这张表是六个 setting 上每个方法的最好成绩，第一列是这些 run 在训练开始时的平均成绩。

| setting | 训练开始时 | PPO | DAPO | GSPO | DPPO | TOPR | SFC+KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Countdown，4-prompt slice | 0.09 | 0.646 | 0.646 | 0.691 | 0.668 | 0.699 | **0.725** |
| Countdown，2-prompt slice | 0.10 | 0.531 | 0.342 | 0.656 | 0.078 | **0.676** | 0.670 |
| AdamW，4-prompt slice | 0.09 | 0.701 | 0.650 | 0.680 | 0.697 | 0.691 | **0.738** |
| GSM8K，4-prompt slice | 0.39 | 0.645 | 0.691 | 0.838 | 0.393 | **0.842** | 0.803 |
| Qwen3-4B，1-prompt slice | 0.41 | 0.416 | 0.543 | 0.557 | 0.398 | **0.762** | 0.744 |
| Qwen3-30B-A3B，1-prompt slice | 0.67 |  | 0.814 | 0.791 | 0.830 | 0.799 | **0.877** |

SFC+KL 在三个 setting 上最高，TOPR 在另外三个 setting 上最高。其中 Countdown 的 2-prompt slice 上两者只差 0.006，可以看成打平。GSM8K 上 TOPR 和 GSPO 都比 SFC+KL 高出 0.035 以上。Qwen3-4B 上 TOPR 比 SFC+KL 高 0.018，不过前面说过，SFC+KL 在这个模型上只做了最后一次 evaluation。稳定性方面，TOPR 在六个 setting 上都训练到了最后，这一点和 SFC+KL 一样。PPO、DAPO 和 DPPO 则都在不止一个 setting 上失败了，有的中途 collapse，有的从头到尾都没有超过训练开始时的成绩。

### 8.2 能不能组合

可以，而且加上以后很少变差。我把 SFC+KL 加到了 TOPR 和 GSPO 上，也就是在它们的基础上加上 SFC 的 centering 和 KL 这一项，在六个 setting 上一共得到 12 组对比。

<figure data-chart="compose">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/compose.png" alt="TOPR 和 GSPO 加上 SFC+KL 前后的最好成绩">
</noscript>
</figure>

12 组里有 10 组的最好成绩变高了，不过其中不少只高了 0.02 以内，差别不大。变化超过 0.02 的有五组。变好的四组里，最明显的是 Qwen3-4B。TOPR 从 0.762 提高到了 0.822。GSPO 自己在第 400 步到了 0.557，之后就掉到了接近 0，加上 SFC+KL 以后训练到了最后，最好成绩是 0.822。另外两组是 Countdown 2-prompt slice 上的 TOPR 和 Qwen3-30B-A3B 上的 GSPO。变差的一组是 GSM8K 上的 TOPR，低了 0.022。

所以 SFC+KL 也可以看成一个能加在其他方法上的部件，不一定要和它们二选一。加上以后很少变差，在 Qwen3-4B 这样的 setting 上还能帮上很大的忙。

## 9. 几点 takeaway

最后总结一下这些实验告诉我们的事情：

* 在 staleness 比较大的异步训练里，只做 SFC 不够。它修好了 score drift，但没有限制 trainer 离 sampler 有多远，用 SGD 训练的时候很快就会 collapse。
* 在 SFC 上加 TIS 或者 MIS 很不稳定。在 Qwen3-0.6B 上，它们大多数时候会 collapse 或者学不起来。SFC+MIS 在两个更大的模型上没有 collapse，最好成绩还超过了 SFC+KL，但过了最高点以后掉得很厉害。
* SFC+TIS 会让回答越来越长，被截断的比例也因此高得多。它的 entropy 在训练中一路上涨，搜索越来越没有方向，最后 trainer 和 sampler 之间的 KL 突然涨了两个数量级，训练彻底停住。把 context length 的上限提高到四倍也没能避免 collapse，多出来的长度只是被没有意义的内容填满了。
* SFC+KL 去掉了 importance sampling，改用 KL 控制 trainer 和 sampler 之间的距离，它在这篇文章的所有 setting 里都没有 collapse 过。在七个异步 setting 里，它有五个的最好成绩是 SFC 这几个方法里最高的。
* 在 Countdown 上，payload 的构成对最好成绩影响不大，压缩到每个位置 16 个 token 也够用，但要留一部分给 tail sample。只送回 top-16 的 run 在训练后期 collapse 了。
* 如果 trainer 和 generator 的差异主要来自量化，而不是 staleness，TIS 和 KL 一样好用。
* SFC+KL 可以加在 TOPR 和 GSPO 这样的方法上，12 组对比里有 10 组的最好成绩变高了，Qwen3-4B 上提高得最多。

## 10. 写在最后

受限于计算资源，这些实验没有对超参数做系统的 sweep，也没有对每个配置运行多个 seed，所以文中的结果仅供参考。代码和数据的链接放在了文章开头，欢迎大家在此基础上继续做下去。

最后简单回顾一下这个系列。第一篇讲的是，在 async RL 里应该用真正生成数据的 $q_\phi$ 来做 centering。第二篇讲的是，stale rollout 不一定要用 importance sampling 校正回 current policy，我们可以保留它在 $q_\phi$ 下的 learning signal，再用 KL 控制 trainer 离 sampler 有多远。这一篇用实验检验了这两件事。谢谢你读到这里。
