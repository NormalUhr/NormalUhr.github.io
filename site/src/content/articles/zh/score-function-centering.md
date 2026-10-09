---
title: "Score Function Centering 深入浅出：从 Score Function 理解异步 RL 中的 Trainer-Generator Mismatch"
date: 2026-10-08
category: post-training
series: grpo-line
part: 4
tags: ["GRPO"]
---

# Score Function Centering 深入浅出：从 Score Function 理解异步 RL 中的 Trainer-Generator Mismatch

我觉得网上关于 score function centering 的解释或多或少都有些绕，不够直接。我希望写这篇文章给大家提供一种从 intuition 讲起，对 score function centering 最直接的解释。这篇文章是我纯手工手搓，不是 AI 写的，当然如果你读的是英文版，很抱歉，它是 AI 翻译的。

这篇文章需要你知道 PPO 或者 GRPO 的使用方法（loss 长什么样）和一些简单的概率论知识（就够了）。

如果你满足以下任意一条，这篇文章应该是适合你的：

* 只会用 GRPO，但是不知道其内在原理
* 对 trainer-generator mismatch 想要进一步加深理解
* 做 async RL 并且 long rollout 训练但是还不确定该如何提升算法稳定性
* 不知道 score function centering 是个啥（甚至不知道这里的 score 是个啥）但是想了解的

Score Function Centering 并不是一个专有名词，我知道各家 frontier lab 对于这个技术有自己不同的叫法，alternative token、covariance optimization 等等，你想叫什么都行，这篇文章也会解释这些所谓的名字是怎么来的。

太长不看版：

* 读完第一章你会知道 score function 是计算 policy gradient 的一个重要桥梁，以及我们能从 auto-differentiation 中得到的最直接的产物。
* 读完第二章，你会知道 score function 的重要性质：它本身是中心化过了的，所以它的期望是 0，我们需要这个期望一直是 0，才能在 policy gradient 中用所谓的 advantage 来代替 reward，以及 trainer-generator mismatch 如何破坏了这个性质，所以我们需要修复它。
* 读完第三章，你会知道 score function centering 本质是在干什么，它为什么修复了这个 0 期望属性；以及实际在工程上我们应该如何使用它，以及在这个领域灌水的一个方向。

## 1. 我们从 score function 讲起，score 是啥？为啥要用它？

### 1.1 GRPO/PPO 里我们到底在求什么梯度？

> 下面我们对 policy gradient 做一个快速推导，如果你对这一部分相当熟悉，请跳过这一小节。

假设在某个 state，也就是 LLM 某个 prefix $x$ 上，policy 是：

$$
p_\theta(a|x)
$$

为了简单，下面暂时把 $x$ 省略，写成：

$$
p_\theta(a).
$$

我们的目标非常自然：

“让高 reward 的 action 更容易被模型采到。”

因此可以写一个最简单的目标：

$$
J(\theta) = \mathbb E_{a\sim p_\theta}[R(a)].
$$

这句话的意思就是：**“从当前模型 $p_\theta$ 里 sample action，计算 reward，希望它的期望越来越高。”**

问题是：$\nabla_\theta \mathbb E_{a\sim p_\theta}[R(a)]$ 怎么求？这里有一个麻烦：$\theta$ 并不直接出现在 $R(a)$ 里面。它藏在 $a\sim p_\theta$ 这个 sampling distribution 里面。也就是说，我们不是在求 $\nabla_\theta R(a)$。事实上，RL 里的 reward 完全可以不可导。比如：数学题答对给 1 分否则给 0 分，或者某个人类专家直接给打分。这都没关系。真正依赖 $\theta$ 的，是 $p_\theta(a)$。

把 expectation 写开：

$$
J(\theta) = \sum_a p_\theta(a)R(a).
$$

求导：

$$
\nabla_\theta J(\theta) = \sum_a R(a)\nabla_\theta p_\theta(a).
$$

然后用一个非常简单但极其重要的恒等式：

$$
\nabla_\theta p_\theta(a) = p_\theta(a) \nabla_\theta\log p_\theta(a).
$$

因为：

$$
\nabla \log p = \frac{1}{p}\nabla p.
$$

于是：

$$
\nabla_\theta J(\theta) = \sum_a p_\theta(a) R(a) \nabla_\theta\log p_\theta(a).
$$

重新写成 expectation：

$$
\nabla_\theta J(\theta) = \mathbb E_{a\sim p_\theta} \left[ R(a) \nabla_\theta\log p_\theta(a) \right].
$$

这里终于出现了 policy gradient 中最关键的东西：

$$
s_\theta(a) = \nabla_\theta\log p_\theta(a).
$$

这个东西叫做 **score function**。注意这里的 score 不是 reward score。不是“这个回答得了多少分”。它是概率统计中的一个专门术语：

$$
\text{score} = \nabla_\theta\log p_\theta(a).
$$

也就是 **某个被 sample 出来的 action 的 log probability，对模型参数的梯度。**

### 1.2 为什么我们要研究 score？

因为 policy gradient 最核心的结构其实就是：

**reward $\times$ score**

（我知道你想说我们实际用的都是 $\text{Advantage}\times\text{score}$，但你先别急）。

如果换成我们训练 LLM 时更熟悉的写法：

$$
g=R(a)\nabla_\theta\log p_\theta(a).
$$

所以可以把两部分理解成：$R(a)$ 告诉我们 **“这个 action 应该被强化还是削弱？强化多少？”** 而 $\nabla_\theta\log p_\theta(a)$ 告诉我们：**“如果我要提高或者降低这个 action 的 probability，参数应该朝哪个方向走？”**

这也是为什么我认为理解 policy gradient 最重要的一句话是：

**Score function 是把（你怎么也算不出来的）“概率分布期望的梯度”转化成“可以通过采样计算的梯度估计”的桥梁。**

更准确一点：

**distribution-level differentiation $\rightarrow$ sample-level gradient estimation**

我们原来想算：

$$
\nabla_\theta \mathbb E_{a\sim p_\theta}[R(a)].
$$

它看起来需要对整个 action distribution 求导。

但 score-function trick 把它变成：先 sample 一个 action

$$
a\sim p_\theta,
$$

然后算：

$$
R(a)\nabla_\theta\log p_\theta(a).
$$

多 sample 几次平均：

$$
\nabla_\theta J\approx\frac1N\sum_{i=1}^NR(a_i)\nabla_\theta\log p_\theta(a_i).
$$

这就是 REINFORCE / policy gradient 最核心的计算机制。而 GRPO 虽然加入了 group-relative advantage、importance ratio、clipping 等机制，它里面真正产生 policy gradient 的基本对象，仍然绕不开：

$$
\nabla_\theta\log p_\theta(a).
$$

同时，你会发现这个东西是你在真正使用 GRPO 的时候，一次 backward 给你的最直接的东西。

## 2. score function 与 trainer-generator mismatch

### 2.1 score function 有什么美好的性质？这个性质为什么重要？

我们现在来讨论讨论 advantage。我们都知道，在 GRPO/PPO 里，我们不会直接使用 reward，我们使用的是 advantage（[为什么？](/zh/p/grpo)）。advantage 就是 reward 减去一个常数：

$$
A(a)=R(a)-b,
$$

这里的 $b$ 可以是 value function，可以是一个 group mean，也可以是任何只和当前 state 有关、但是和具体 action 无关的 baseline。比如在 GRPO 里面，我们最熟悉的事情就是：同一个 prompt 采样很多条 response，然后用某种 group-relative 的 reward 作为 advantage。先不管具体 normalization 怎么做，本质上都是在做一件事情：从 raw reward 里减掉一个“大家共同拥有的部分”，只留下这个 action 相对平均水平好还是差。但是这里有一个非常基础、也非常容易被忽略的问题：

**我们凭什么能随便减这个 baseline？**

原来的 policy gradient 是：

$$
g=\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)].
$$

如果我们把 reward 换成 advantage：

$$
A(a)=R(a)-b,
$$

那么新的 gradient 就是：

$$
g'=\mathbb E_{a\sim p_\theta}[(R(a)-b)s_\theta(a)].
$$

把它展开：

$$
\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)] -b\mathbb E_{a\sim p_\theta}[s_\theta(a)].
$$

第一项就是我们原来的 policy gradient。也就是说，如果我们希望减掉 baseline 以后，policy gradient 的期望完全不变，那么第二项必须消失：

$$
b\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0.
$$

而 $b$ 显然不一定等于 0，所以我们真正需要的是：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0.
$$

这就是 score function 在 policy gradient 里面一个非常重要的性质。你可以先完全不管这个性质为什么成立，只需要记住它的现实意义：

**只要 score 的 expectation 是 0，我们就可以给 reward 减一个和 action 无关的 baseline，而不改变 expected policy gradient。**

这就是为什么我们可以把 reward 写成 advantage，也是为什么 PPO / GRPO 里面各种 reward centering、value baseline、group-relative baseline 能够成立。有些[文章](https://arxiv.org/abs/2609.20807)会从 covariance 开始解释这件事情，比如写：

$$
\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)] = \operatorname{Cov}(R(a),s(a)) + \mathbb E[R(a)]\mathbb E[s(a)],
$$

然后因为 $\mathbb E[s]=0$，所以最后只剩 covariance。这个写法当然没有问题，但是我个人觉得它反而把一件很简单的事情写绕了。我们真正关心的其实就是一句话：我想把 reward 减掉一个 baseline，但是我又不想让这个 baseline 本身产生 gradient。那自然就要求 $\mathbb E[s]=0$。就这么简单。

而且这个性质其实非常符合 intuition。假设两个 action 的 reward 是：

$$
R(A)=101,\qquad R(B)=99.
$$

和：

$$
R'(A)=2,\qquad R'(B)=0.
$$

从 policy optimization 的角度，这两组 reward 表达的是完全一样的信息：$A$ 都比 $B$ 好 2 分。我们显然不希望仅仅因为给所有 reward 同时加了 99，模型的 expected update direction 就发生变化。所有 reward 同时加减一个常数，都不会改变 expected policy gradient。这也是为什么 baseline 本质上是一个非常自然的操作，而不只是一个数学 trick。

好了，现在问题来了：

**为什么 score 的 expectation 恰好会等于 0？**

为了回答这个问题，我们终于需要看看 score 的庐山真面目到底是什么。

### 2.2 Score Function 到底是什么？

接下来，我们对 score function 做一些上过微积分就会做的推导，我们前面定义：

$$
s_\theta(a)=\nabla_\theta\log p_\theta(a).
$$

假设当前模型在某一个 prefix 上输出 vocabulary 中所有 token 的 logits：

$$
z_\theta(1),z_\theta(2),\ldots,z_\theta(V).
$$

我们默认输出的分布由 Softmax 给出：

$$
p_\theta(a) = \frac{e^{z_\theta(a)}}{\sum_v e^{z_\theta(v)}}.
$$

所以：

$$
s_\theta(a) = \nabla_\theta z_\theta(a) - \nabla_\theta \log\sum_v e^{z_\theta(v)}.
$$

现在给上式的第一项起一个好听点的名字，叫 logit Jacobian 吧：

$$
J_\theta(v)=\nabla_\theta z_\theta(v).
$$

这个 $J_\theta(v)$ 可以理解成：如果我稍微改变当前 trainer 的模型参数，token $v$ 的 raw logit 会朝哪个方向变化。

举个直观的例子。假设模型参数从 $\theta$ 变成：

$$
\theta+\Delta\theta,
$$

那么一阶近似下：

$$
z_{\theta+\Delta\theta}(v)
\approx
z_\theta(v)
+
J_\theta(v)^T\Delta\theta.
$$

所以 $J_\theta(v)$ 本质上就是 token $v$ 在参数空间里的一个“logit sensitivity direction”。它告诉我们：参数往哪里走，会把这个 token 的 logit 往上推或者往下拉。

接着我们处理上式的第二项：

$$
\nabla_\theta
\log\sum_v e^{z_\theta(v)}
$$

做一个小小的变形，得到：

$$
\frac{
\sum_v \color{orange}{e^{z_\theta(v)}} \color{blue}{\nabla_\theta z_\theta(v)}
}{
\color{orange}{\sum_v e^{z_\theta(v)}}
}.
$$

而我们恰好有：

$$
p_\theta(v) = \color{orange}{\frac{e^{z_\theta(v)}}{\sum_u e^{z_\theta(u)}}}, \quad J_\theta(v)= \color{blue}{\nabla_\theta z_\theta(v)}.
$$

所以上式第二项可以写作：

$$
\sum_v p_\theta(v)J_\theta(v).
$$

也就是：

$$
\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

因此：

$$
s_\theta(a) = J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

这个式子特别重要，因为它告诉我们 **在 score function 里，Softmax 自己就在给 logit Jacobian 做 centering**。$J_\theta(a)$ 是当前 sample 到的 action 的 logit gradient，而 $\mathbb E_{v\sim p_\theta}[J_\theta(v)]$ 是按照当前 policy $p_\theta$ 加权以后，所有 action 的平均 logit gradient。

所以 score 其实就是：

$$
\text{被采样到的 action 的 logit Jacobian} - \text{当前 policy 下平均的 logit Jacobian}.
$$

有了上边这个式子，score 的 expectation 为 0 这件事情几乎都不用证明了。因为 action 本身就是从：

$$
a\sim p_\theta
$$

采样出来的，所以：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{a\sim p_\theta}\left[J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]\right].
$$

把 expectation 拆开：

$$
=\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

这里的 $a$ 和 $v$ 只是两个名字不同的 dummy variable，它们都来自完全相同的 distribution $p_\theta$，所以这两项就是同一个东西：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]=0.
$$

所以我们现在可以把整件事情串起来了：**Softmax score 本身是 centered 的**，因此：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0,
$$

而这个 zero-mean property 又保证了：**我们可以在 policy gradient 中使用 reward 减 baseline 来代替原始的 reward，而不改变 expected policy gradient。**

到这里其实一切都非常美好。但是注意，上面整个推导里有一个我们一直在偷偷使用、但是还没有认真强调的假设：

**计算 expectation 的 distribution，和生成 action 的 distribution，都是 $p_\theta$。**

也就是说，我们默认：

**产生 rollout 的 generator，就是现在正在 backward 的 trainer。**

一旦这个假设不成立，后面的故事才真正开始。

### 2.3 trainer-generator mismatch 后到底破坏了什么？

肯定是破坏了 $\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0$ 这个 score function 的 zero-mean property 啊，不然我们上边讨论那么多不就白费功夫了。

但是这里最重要的问题不是“它被破坏了”，而是：**到底是哪一步发生了变化？**

我们先把 trainer 和 generator 明确区分开。假设当前正在做 backward、准备更新参数的 trainer policy 是：

$$
p_\theta(a).
$$

而真正生成 rollout 的 generator 是一个稍微旧一点的 policy：

$$
q_\phi(a).
$$

这里我故意用了不同的参数 $\theta$ 和 $\phi$。你可以把 $\theta$ 理解成 trainer 当前的参数，而 $\phi$ 是 generator 拿去做 rollout 时的那一版参数。在同步训练里，我们近似有：

$$
q_\phi \approx p_\theta,
$$

甚至可以直接认为：

$$
q_\phi=p_\theta.
$$

但在 async RL 里，尤其是 rollout 很长、generator 很多、trainer 又一直在更新的时候，这两个分布很自然就会开始产生 mismatch：

$$
q_\phi\neq p_\theta.
$$

现在回忆一下我们刚才证明 zero-mean property 时用的公式：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)]=0.
$$

为什么等于 0？本质上就是因为左右两边的 expectation 都是在同一个 $p_\theta$ 上算的。第一个 expectation 里，action $a$ 是从 $p_\theta$ sample 出来的；第二个 expectation 里，Softmax 本身又是用 $p_\theta$ 对所有 logit Jacobian 做 centering。所以两项正好完全一样。

但是现在 generator 换成了 $q_\phi$。真正送到 trainer 里的 action 已经不再满足：

$$
a\sim p_\theta,
$$

而是：

$$
a\sim \color{red}{q_\phi}.
$$

注意，trainer 并没有因此改变自己计算 score 的方式。我们 backward 的仍然是当前 trainer 的：

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

这里的 $J_\theta$ 仍然来自当前 trainer，Softmax centering 用的也仍然是当前 trainer distribution $p_\theta$。真正变化的只有一件事情：**外边这个 action 到底是从哪个 distribution sample 出来的。**

所以现在我们真正应该看的，不再是：

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)],
$$

而是：

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)].
$$

把 score 展开：

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)]=\mathbb E_{a\sim \color{red}{q_\phi}}\left[J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]\right].
$$

把 expectation 拆开：

$$
=\mathbb E_{a\sim \color{red}{q_\phi}}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

现在问题就非常明显了。原来我们是：

$$
\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)],
$$

左右两个 expectation 的 distribution 一样，所以等于 0。现在变成了：

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

左边在 generator distribution $q_\phi$ 下取平均，右边却仍然在 trainer distribution $p_\theta$ 下取平均。只要：

$$
q_\phi\neq p_\theta,
$$

我们一般就不能再指望：

$$
\mathbb E_{a\sim q_\phi}[J_\theta(a)]=\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

所以最终：

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

这就是 trainer-generator mismatch 对 score function 最直接的影响。

你会发现，这件事情其实一点都不神秘。**Score 自己没有突然变坏，Softmax 的公式也没有变，trainer 计算出来的 $J_\theta$ 也没有变。真正发生变化的，只是 sample action 的 distribution 从 $p_\theta$ 换成了 $q_\phi$。**

而 zero-mean property 恰好又依赖于“sample action 的 distribution”和“score 里面做 centering 的 distribution”是同一个 distribution。现在这两个东西不一样了：

$$
\underbrace{a\sim \color{red}{q_\phi}}_{\text{generator sampling distribution}},\qquad \underbrace{\mathbb E_{v\sim p_\theta}[J_\theta(v)]}_{\text{trainer centering distribution}}.
$$

所以原来那个：

$$
\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]=0
$$

就变成了：

$$
\mathbb E_{\color{red}{q_\phi}}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

到这里，我们其实已经把 trainer-generator mismatch 最关键的一层问题讲清楚了：

**在同步 RL 里，generator 和 trainer 是同一个 distribution，所以标准 score 天然是 zero-mean 的；而在 async RL 里，action 来自旧 generator $q_\phi$，score 却仍然由当前 trainer $p_\theta$ 来 center，于是这个 zero-mean property 被破坏了。**

## 3. Score Function Centering 到底在做什么？

答案呼之欲出了对不对！Score Function Centering 就是想让原本被破坏的 score function 的 zero-mean property 重新被满足，所以我们就又可以愉快地使用 baseline 啦！

我们刚才已经看到，在 trainer-generator mismatch 之后，真正的问题变成了：

$$
\mathbb E_{a\sim q_\phi}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

这里为什么不等于 0？原因非常简单：**action 是 generator 的 $q_\phi$ sample 出来的，但是 score 里面减掉的“中心”却还是 trainer 的 $p_\theta$。**

想要更正这一点，既然外边的 action 已经来自 $a\sim q_\phi$，那最直接的办法当然就是：**不要再用 $p_\theta$ 做 centering，而是改成用真正生成这些 action 的 distribution $q_\phi$ 做 centering。**

也就是说，把原来的标准 score：

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]
$$

改成：

$$
\tilde s_{\theta,\phi}(a)=J_\theta(a)-\mathbb E_{v\sim \color{red}{q_\phi}}[J_\theta(v)].
$$

**这就是 Score Function Centering 最核心的一步。**

注意这里有两个 distribution / model parameter，不要混在一起。$J_\theta(v)$ 仍然是：

$$
J_\theta(v)=\nabla_\theta z_\theta(v),
$$

所以它还是**当前 trainer** 的 logit Jacobian。我们并不是回到旧 generator 上去算 gradient，也不是要更新 $\phi$。真正变化的只有 centering 时使用的 probability distribution：

$$
\mathbb E_{v\sim p_\theta}[J_\theta(v)]\quad\longrightarrow\quad\mathbb E_{v\sim \color{red}{q_\phi}}[J_\theta(v)].
$$

你可以把这句话记成：**generator 决定“怎么平均”，trainer 决定“往哪个参数方向更新”。**

这样一改，zero-mean property 马上就回来了。因为真正的 action 本来就是：

$$
a\sim q_\phi,
$$

所以：

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=\mathbb E_{a\sim q_\phi}[J_\theta(a)]-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

$a$ 和 $v$ 还是两个 dummy variable，而且这一次它们终于又来自同一个 distribution 了：

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{q_\phi}[J_\theta]=0.
$$

于是我们重新得到：

$$
\mathbb E_{a\sim q_\phi}[(R(a)-b)\tilde s_{\theta,\phi}(a)]=\mathbb E_{a\sim q_\phi}[R(a)\tilde s_{\theta,\phi}(a)].
$$

也就是说，在**真正产生 rollout 的 distribution $q_\phi$ 下**，baseline 又不会自己产生一个额外的 expected gradient 了。

这就是 Score Function Centering 最本质的 intuition：**让 score 的 centering distribution 和真正生成 action 的 distribution 对齐。**

### 3.1 但是现实里有一个很大的问题：我们怎么得到 $\mathbb E_{q_\phi}[J_\theta]$？

你如果真的在训练 LLM，使用上边的方法，马上就会碰到一个非常现实的问题：

$$
\mathbb E_{v\sim q_\phi}[J_\theta(v)]=\sum_{v\in\mathcal V}q_\phi(v)J_\theta(v).
$$

假设 vocabulary 有 128K 个 token，那么这个 expectation 看起来要求我们知道 generator 在这个 prefix 上对**所有 128K 个 token**的 probability：

$$
q_\phi(1),q_\phi(2),\ldots,q_\phi(V).
$$

问题是，这些信息是在 generator 生成 token 的那一刻才有的，而真正做 backward 的 trainer 可能在另外一台机器、另外一个 pipeline stage，甚至已经过了几十个 train step。

当然你可以说，那我就把 generator 的整个 vocabulary distribution 都存下来，再发给 trainer 不就完了？

理论上可以，但是工程上不可以。

因为对于 rollout 里的**每一个 token position**，你都要额外保存一个长度为 vocabulary size 的 distribution。假设 vocab 是 128K，一个 rollout 又有几千甚至几万个 token，再乘上 batch size、generator 数量和并行 rollout 数，这个 communication 和 memory overhead 很快就会变得完全不可接受。

所以现实问题变成了：

**我们想知道 $q_\phi$ 下的 expectation，但是又不能把完整的 $q_\phi$ 搬到 trainer。怎么办？**

答案还是 Monte Carlo：**既然 expectation 本来就是对 $q_\phi$ 求平均，那我们就在 generator 端从 $q_\phi$ 里多 sample 几个 token。**

### 3.2 Alternative Tokens：不要把整个 distribution 给 trainer，给我几个 sample 就够了

假设真正被 generator 选中、进入 rollout 的 token 是：

$$
a\sim q_\phi.
$$

除了这个 primary action 以外，我们在**完全相同的 prefix**上，再额外从 generator distribution 里 sample $K$ 个 token：

$$
v_1,v_2,\ldots,v_K\sim q_\phi.
$$

这些 token 不会真的被 append 到 trajectory 里，也不会改变后续 generation。它们只是在告诉 trainer：

**“我当时在这个 prefix 上的 distribution 大概长这样。”**

不同地方会给这些 token 起不同的名字，alternative tokens、alter tokens、negative tokens、auxiliary samples……它们本质上都是在做同一件事情：**用少量 sample 来描述 generator 当时的 local action distribution。**

如果这些 $v_k$ 真的是 IID 从 $q_\phi$ 里 sample 出来的，那么：

$$
\mathbb E_{v\sim q_\phi}[J_\theta(v)]\approx\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k).
$$

于是 SFC score 就可以近似成：

$$
\hat s_{\theta,\phi}(a)=J_\theta(a)-\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k).
$$

而且这个 Monte Carlo estimator 对我们真正想要的 behavior-centered score 是 unbiased 的：

$$
\mathbb E_{v_1,\ldots,v_K\sim q_\phi}\left[\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k)\right]=\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

所以 generator 不需要把整个 $q_\phi$ 传给 trainer。它只需要在每一个 token position 附带一小撮额外的信息，比如：

$$
[a,v_1,v_2,\ldots,v_K].
$$

更实际一点，还会把这些 token 在 generator 下的 log probability 一起保存下来：

$$
[\log q_\phi(a),\log q_\phi(v_1),\ldots,\log q_\phi(v_K)].
$$

然后这些信息跟着 rollout 一路从 generator 传到 trainer。所以你可以把一个普通 RL rollout token 想成原来只携带：

$$
\text{token id}+\text{reward/advantage}+\log q_\phi(a),
$$

而 SFC 之后，它还额外携带了一个非常小的 **local policy sketch**：

$$
\{v_k,\log q_\phi(v_k)\}_{k=1}^K.
$$

这个 local policy sketch 就是在告诉 trainer：**generator 当时除了真正选中的 action 以外，还认为哪些 token 是可能的。**

**聪明的你发现了，如果想去灌水几篇学术会议，这个 generator distribution 的采样方法，你去换上几种，试一试谁比谁好，又是几篇 paper (LoL)。**

### 3.3 等一下，我们真的要在 trainer 里显式计算 $J_\theta(v)$ 吗？

看到这里你可能又要有点头大了，因为：

$$
J_\theta(v)=\nabla_\theta z_\theta(v)
$$

可是 $\theta$ 是整个 LLM 的所有参数。70B 模型的话，$J_\theta(v)$ 本身就是一个几十 billion 维的 vector。

难道我们真的要给每一个 alternative token 都 materialize 一个巨大的 Jacobian，然后再算：

$$
J_\theta(a)-\frac{1}{K}\sum_kJ_\theta(v_k)?
$$

当然不是。$J_\theta$ 在这里主要是一个**帮助我们理解算法的数学对象**。真正写代码的时候，我们只需要构造一个 scalar surrogate，然后让 autograd 帮我们把它求导。先回到 SFC 想要的 gradient：

$$
J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

因为：

$$
J_\theta(v)=\nabla_\theta z_\theta(v),
$$

所以：

$$
J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)]=\nabla_\theta\left(z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)]\right).
$$

于是我们真正需要构造的 scalar 非常简单：

$$
c_{\theta,\phi}(a)=z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)].
$$

如果用刚才的 $K$ 个 alternative samples：

$$
\hat c_{\theta,\phi}(a)=z_\theta(a)-\frac{1}{K}\sum_{k=1}^Kz_\theta(v_k).
$$

然后像普通 policy gradient 一样乘上 advantage：

$$
L_{\mathrm{SFC}}=-A(a)\hat c_{\theta,\phi}(a).
$$

对它做一次普通 backward：

$$
\nabla_\theta L_{\mathrm{SFC}}=-A(a)\left(J_\theta(a)-\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k)\right).
$$

这就是我们想要的 SFC gradient，所以在实际代码里，也是这样，大概逻辑是：

1. generator 告诉 trainer：primary token 是谁，另外几个 alternative token 是谁；
2. trainer 正常 forward 当前模型；
3. 从当前 trainer 的 logits 里 gather 出这几个 token 对应的 logits；
4. 算一个 `primary_logit - mean(alternative_logits)`；
5. 乘 advantage；
6. backward。

### 3.4 为什么这里甚至连完整的 log-softmax 都不需要？

我们前面明明一直在讲 score：

$$
\nabla_\theta\log p_\theta(a),
$$

为什么到了 SFC 实现的时候突然变成直接操作 logit $z_\theta(a)-\mathbb E_{q_\phi}[z_\theta(v)]$？

原因是 Softmax 的 normalization term 正好会被 cancel。我们可以先从下面这个 quantity 出发：

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)].
$$

因为：

$$
\log p_\theta(v)=z_\theta(v)-\log Z_\theta,
$$

其中：

$$
Z_\theta=\sum_u e^{z_\theta(u)},
$$

所以：

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)]=z_\theta(a)-\log Z_\theta-\mathbb E_{v\sim q_\phi}[z_\theta(v)-\log Z_\theta].
$$

对于固定 prefix，$\log Z_\theta$ 和 sample 到哪个 $v$ 没有关系，所以：

$$
\mathbb E_{v\sim q_\phi}[\log Z_\theta]=\log Z_\theta.
$$

于是两个 normalization term 直接消掉：

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)]=z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)].
$$

这件事情在工程上非常 nice：**为了这条 SFC loss path，我们只需要 selected token 的 logits，不需要为了它额外 materialize 整个 vocabulary 的 log-softmax。**

这里要稍微严谨一点：这并不代表 LLM forward 从此完全不需要 full vocabulary logits。它只意味着 **SFC 这一条 loss / gradient path 本身可以只依赖少量被选中的 token logits。**

### 3.5 纯随机 sample 还不够好：为什么实际会用 Head + Tail？

如果只是最简单地从 $q_\phi$ 里随机 sample：

$$
v_1,\ldots,v_K\sim q_\phi,
$$

当然已经可以得到一个 unbiased estimator。但是 LLM 的 next-token distribution 往往非常不均匀：少量 top tokens 可以占掉绝大部分 probability mass，而后面是一个非常长的 tail。

假设某个 prefix 下：

$$
q_\phi(A)=0.40,\qquad q_\phi(B)=0.25,\qquad q_\phi(C)=0.15,
$$

光前三个 token 就已经占了 80% 的 probability mass。

这时候如果我们还把所有 token 都当成普通 Monte Carlo sample 来处理，就有点浪费：明明 $A/B/C$ 几乎每次都很重要，我们却还在随机决定“这次有没有抽到它”。

所以一个很自然的做法是：**高概率的 head 直接精确算，只有 long tail 用 Monte Carlo sample。**

假设 generator 选出 top-$H$ token 组成 head set：

$$
\mathcal H=\operatorname{TopH}(q_\phi),
$$

它们的总 probability mass 是：

$$
\alpha=\sum_{h\in\mathcal H}q_\phi(h).
$$

那么任意一个 expectation 都可以拆成：

$$
\mathbb E_{v\sim q_\phi}[f(v)]=\sum_{h\in\mathcal H}q_\phi(h)f(h)+(1-\alpha)\mathbb E_{v\sim q_\phi(\cdot\mid v\notin\mathcal H)}[f(v)].
$$

第一项直接精确计算，第二项从 conditional tail distribution 里 sample $M$ 个 token：

$$
t_1,\ldots,t_M\sim q_\phi(\cdot\mid v\notin\mathcal H).
$$

于是：

$$
\mathbb E_{v\sim q_\phi}[f(v)]\approx\sum_{h\in\mathcal H}q_\phi(h)f(h)+(1-\alpha)\frac{1}{M}\sum_{m=1}^Mf(t_m).
$$

在 SFC 里，把：

$$
f(v)=J_\theta(v)
$$

代进去就行。

你可以把这个方法理解成一句朴素的话：

**最重要的几个 token 我全都要；剩下概率很小但是数量巨大的 long tail，我再用几个 sample 来代表。**

这个 head/tail sampling 本身并不是什么新东西，本质上就是 stratified Monte Carlo。但对于 LLM 这种 vocabulary 很大、概率又高度集中的分布来说，它非常实用。

### 3.6 所以 SFC 在实际系统里到底需要做什么？

到这里我们可以把整个操作过程用非常具体的方式重新说一遍。

Generator 在某个 prefix $x_t$ 上有自己的 behavior policy：

$$
q_\phi(\cdot|x_t).
$$

它正常 sample 一个真正进入 trajectory 的 token：

$$
a_t\sim q_\phi.
$$

与此同时，它额外保留少量能够描述当前 distribution 的 token。最简单可以直接从 $q_\phi$ sample $K$ 个 alternative tokens；更实际一点，可以保存 top-$H$ head tokens，再从 conditional tail 里 sample 几个 token。

然后 generator 把这些 token id、对应的 $q_\phi$ probability / log probability，以及自己当时的 model version 一起随着 rollout 传给 trainer。

Trainer 收到这些数据以后，用**当前参数 $\theta$** 正常 forward，然后只 gather：

$$
z_\theta(a_t),\qquad z_\theta(v_{t,1}),\ldots,z_\theta(v_{t,K}).
$$

再用这些 selected logits 构造对：

$$
\mathbb E_{q_\phi}[z_\theta(v)]
$$

的 sparse estimator，最后形成：

$$
z_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[z_\theta(v)].
$$

乘上 advantage：

$$
L_t=-A_t\left(z_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[z_\theta(v)]\right),
$$

然后正常 backward。

Autograd 自然就会给出：

$$
-A_t\left(J_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[J_\theta(v)]\right).
$$

所以如果非要用一句话总结，Score Function Centering 就是：**Generator 不再只告诉 trainer“我最后选了哪个 token”，而是顺手带回来一点“我当时的 local distribution 长什么样”的信息；trainer 再利用这些 sparse samples，把原本以 $p_\theta$ 为中心的 score，改成以真正生成数据的 $q_\phi$ 为中心。** 它解决的问题是：

**我们想计算 $\mathbb E_{q_\phi}[J_\theta]$，但是我们不可能把完整的 $q_\phi$ 从 generator 搬到 trainer。**

所以我们只能想办法在 generator 端保存一个足够便宜、但又足够有信息量的 sparse approximation。

最后再强调一个很重要的边界：SFC 恢复的是：

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=0,
$$

也就是 behavior distribution 下的 zero-mean / baseline-invariance property。它并不意味着：

$$
\tilde s_{\theta,\phi}(a)=\nabla_\theta\log p_\theta(a).
$$

只要：

$$
q_\phi\neq p_\theta,
$$

这两个东西一般就不是同一个 quantity。所以 SFC 不是“我们把 off-policy bias 修好了”。更准确地说，它做的是：

**在 trainer-generator mismatch 已经存在的情况下，重新构造一个和真实 sampling distribution $q_\phi$ 对齐的 centered gradient signal，让我们至少重新拿回 zero-mean score 以及 baseline invariance 这两个非常重要的结构。**
