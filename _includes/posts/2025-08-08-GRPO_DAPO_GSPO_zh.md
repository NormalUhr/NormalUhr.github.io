# 一篇文章从GRPO走向DAPO和GSPO

在大语言模型的强化学习阶段，PPO 曾经是主流方案，但其对 value model 的依赖在长文本和复杂任务中暴露出局限。GRPO 通过摆脱 value model，显著提升了可扩展性，但在效率和稳定性上仍有改进空间，于是有了 DAPO 对采样、clip、梯度计算等细节的精细优化。然而，在专家动态激活的 MoE 架构中，GRPO 框架下的 token-level 优化依旧难以稳定收敛，GSPO 则进一步将优化粒度提升到 sequence-level，从根本上缓解了高方差与结构性噪声的问题。本文将沿着这一演进路径，从 GRPO 出发，逐步讲清 DAPO 与 GSPO 背后的设计动机与实现思路。

## 前情回顾：GRPO

GRPO的训练目标函数是：

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)} \Bigg[\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg(\min \Bigg(r_{i,t}(\theta) A_{i},\ 
\text{clip}\Big(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\Big) A_{i}\Bigg) -\ \beta\ \mathbb{D}_{KL}\left(\pi_{\theta}\ \|\ \pi_{\text{ref}}\right)\Bigg) \Bigg]
\end{aligned}
$$

其中
$$
r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t}|q,o_{i,&lt;t})}{\pi_{theta_{\text{old}}}(o_{i,t}|q,o_{i,&lt;t})}
$$

$$
A_{i}=\frac{r_{i}-\text{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\text{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}
$$

在理解了 GRPO 的目标函数之后，我们需要先厘清其中最核心的一环——重要性采样（Importance Sampling）——的作用与局限。这不仅是理解 GRPO 的基础，也是后续 DAPO、GSPO 改进的切入点。

### Importance Ratio （重要性采样）到底在起什么作用？

重要性采样的的本质是，我们想要在新的分布下计算期望，但数据是从旧分布采的，于是用新旧策略下同一动作的概率比作为修正权重，以下为恒等式：

$$
\mathbb{E}_{p_\text{new}}[f(x)] = \mathbb{E}_{p_\text{old}}[\frac{p_\text{new}(x)}{p_\text{old}(x)} f(x)]  
$$

这样就能在离线数据（旧策略）上评估新策略的期望，避免每次更新都重新采数据（降低成本）。但是如果如果新旧策略差距太大，权重方差会很高，这就导致训练不稳定。

重要性采样存在的意义在于：我们想要估计一个预期的分布，但是我们手上只有另一个 behavior 分布，我们就只能在 behavior policy 下进行采样，通过这个样本，赋予这个重要性权重，来估计 出target policy 下函数的值。在 PPO/GRPO 中，我们不直接用新策略去采数据，而是先用旧策略生成数据（因为采样代价高），这个过程被称为 Rollout。所以更新时，我们需要修正分布差异，这就是重要性采样。定义每次采样后每个 token 的 importance ratio 为：$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$，那么PPO/GRPO的目标函数可以写为：

$$
L(\theta) = \mathbb{E}_t [\text{min}(r_tA_t, \text{CLIP}(r_t, 1-\epsilon, 1+\epsilon)A_t)]
$$

这里$A_t$是计算出来的优势，clip 限制了更新幅度，防止策略偏离旧策略太远。

有了重要性采样的直观认识，我们进一步思考它在 PPO/GRPO 中的实际影响：优势函数 $A_t$ 与比值 $r_t$ 的符号如何共同决定了策略更新的方向与幅度。

### $A_t$ 和 $r_t$ 的符号如何影响训练？

情况分析：假如 $A_t > 0$ (动作比期望好)，那么我们本意是希望增加该动作的概率。假设我们clip时候的参数 $\epsilon$ 设置成 $0.2$，那么当 $r_t > 1.2$ 时，我们通过min和clip的操作会把这个值给确定在1.2，当 $r_t &lt; 0.8$ 时，我们不会做任何clipping，因为有min操作。所以当优势为正时，增幅被限制。

反之，当 $A_t &lt; 0$ 时，这个时候我们采样生成的sequence动作比期望是差的，那么我们应该见效该动作的概率。当 $r_t &lt; 0.8$ 时，min 操作会限制 $r_t$ 进一步减小，所以会被限制到最小 $0.8A_t$，但是当 $t > 1.2$ 时，由于 `min` 操作，我们不会对其进行限制 （可以变成无穷大，带上负号就是无穷负）；即负优势动作的减幅被限制。

$A_t$ 衡量“当前动作/轨迹下比平均状态好还是差”。如果 $A_t$ 是正值，那么我们就应当鼓励；如果 $A_t$ 是负值，那么就应该惩罚，在未来少出现。重要性采样的值 $r_t$ 表明新策略和旧策略在该动作上的概率比，如果大于1则表明新模型更倾向于选择这个动作；反之则表明新模型更少选这个动作。所以在 $A_t$ 和 $r_t$ 的正负四种组合中，我们只希望两种情况，即 $A_t$ 和 $r_t$ 同号：当 $A_t$ 大于 $0$ 时，我们希望新模型加强这个动作；当 $A_t$ 小于 $0$ 时，我们希望旧模型减少这个动作，即模型在修正错误。

然而，光有 $A_t$ 与 $r_t$ 的方向一致性还不够，PPO/GRPO 中的 clip 操作 同样是稳定训练的关键，它决定了哪些 token 的梯度能真正参与更新。

### clip 操作对梯度和 token 效率的影响

对于 $A_t > 0$ 时，当 $r_t > 1 + \epsilon$，即涨幅达到了限制，我们会采用clip操作，这里梯度就变成了0，这里变相地把当前token对训练的影响给抹去了；类似的，当 A_t &lt; 0 时，如果 r_t &lt; 1 - \epsilon，即跌幅（修正）超过了限制，clip操作一样会使得当前token所带来的梯度变成0。这里有一个可能的误区就是，误以为clip会使用类似于straight-through的方法在back propagation的时候把clip后的值的gradient原封不动地copy到clip前的值，但实际上不会这样的，会直接让被clip前的gradient被置为0。

到这里，我们对 GRPO 的机制、优势与局限有了相对完整的认识。接下来，我们看 DAPO 如何在保留 GRPO 基本框架的前提下，引入更细致的改进，缓解效率与稳定性问题。


## 从GRPO到DAPO

DAPO 的出发点很直接：在实际训练中，GRPO 会因为 clip 范围不合理、采样冗余、长序列梯度稀释等问题浪费大量训练信号。DAPO 针对这些问题逐一提出优化，形成了四个关键改进点。

$$
\begin{aligned}
\mathcal{J}_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)}\Bigg[\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i = 1}^{G} \sum_{t=1}^{|o_i|} \min \Bigg(r_{i,t}(\theta) A_{i},\ 
\text{clip}\Big(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\Big) A_{i}\Bigg) \Bigg]
\end{aligned}
$$

$$
\text{s.t.}, 0 &lt; |\{o_i | \text{is_equivalent}(a, o_i)\}| &lt; G
$$

### DAPO 为什么选择把 $1-\epsilon_{\text{low}}$ 和 $1+\epsilon_{\text{high}}$ 中的后者提高？这里的原因是什么？

作者观察到，如果选择一个较小的 $\epsilon$ 作为clip 的上界，则会导致一个现象：如果 old policy 本身对一个采样到的 token 的概率很小，而此时 advantage 又是正的（old model 采样非常好），那么对于当前 policy model 留给他上涨空间并不高（上涨是我们期望看到的现象）。举个例子，如果 old policy 的概率是0.9，那么如果 $\epsilon=0.2$，那么他的上界就是 $1.08$，这已经超出概率值的上界 $1.0$ 了，policy model 再怎么采样都不会被 clip，old policy 的概率是 $0.2$，那么他的 clip 上界就是 $0.2 * 1.2 = 0.24$，也就是说此时如果 policy model 的概率是 $0.4$，本身是一个很好的提升，也会因为一个太小的 $\epsilon$ 导致 clip 上界太小，以至于此 token 作废。这也是为什么 DAPO 选择了 Clip-Higher，这也是为什么Clip-Higher 能够提升 token efficiency。

这一点其实就是我们所谓的“马氏加成”，可以形象得比喻为 “地主财产翻番易，穷苦佃户脱贫难”。假如 old policy 好不容易采样到了一个至关重要的token，比如 "Wait"，但是概率很低，但是当前模型在这个 token 上的概率明显增加了不少，去会非常轻易地就遭到clip，导致模型翻身的机会非常小。

Clip-Higher 解决的是“好 token 上涨受限”的问题，但它并未触及另一个常见浪费：采样的多样性缺失。为此，DAPO 引入了 Dynamic Sampling。

### DAPO - Dynamic Sampling

DAPO选择的第二个技术创新点被称为dynamical sampling。这个项技术的背景是来源于：假如一个query我们sample了10次，这10次每次都答得很好/或者很差，都取得了max reward/zero reward，这个时候由于GRPO的计算方法，导致这10次采样的advantage都是0，所以这些sample所带来的gradient就也都是0；这样做的一个后果就是，实际的有梯度的sample要远低于名义sample数，导致最后梯度汇集的时候没有收集到足够的信息，从而形成高方差，训练的不稳定性，以及sample的浪费。需要注意的是，这种现象是在训练初期；以及后期随着训练的进行在不断加强的，因为刚开始时模型效果很差，而训练越到后边模型效果越好，给出满分回答的几率就越大。因此，DAPO在采集样本时，额外做了一件事：保证每次采样出来的回答，reward不全是0或者1，如果采样出来的回答全是0或者1就继续采样，直到不满足为止。这也是损失函数中 $\text{s.t.}, 0 &lt; |\{o_i | \text{is_equivalent}(a, o_i)\}| &lt; G$ 的由来，它保证了针对同一个输入采样到的一组回答中，既有错误的也有正确的。

除了采样多样性，GRPO 在长回答的训练中还有一个隐性缺陷：token 梯度的权重会随回答长度变长而被稀释。DAPO 的第三个改进正是 Token-Level Gradient Loss。

### DAPO - Token-Level Gradient Loss

DAPO 第三个方面的创新是为了解决GRPO在训练长回答时token gradient的权重会随着采样回答的长度变长而下降的问题。首先解释为什么采样长度变长权重会下降。假设采样了2次，有一次回答一共有200个token，而另一次回答有10个token。那么根据GRPO的计算公式，每次回答的梯度先在sample内求平均，再在batch内求平均。第一次回答每个token的权重是 (1/200) * (1/2)，而第二个回答每个token的权重是 (1/10) * (1/2)，所以第二次回答的token的影响要明显高于第一次回答。再来说采样长度变长权重下降的危害：对于一些比较难的问题，长回答本身就很正常，如果这些回答本身非常好，那么由于长度平均就会导致本来非常有用的梯度信号被稀释；假如回答是不好的，长度长仅仅也是因为重复单词，或者回答冗余词太多，长度归一就导致这次采样本该带来的纠正信号没法正确传递到policy model上。所以DAPO采用的方法是：把一次梯度计算时所有采样生成的token总数加起来求平均，回到上边这个例子，第一次采样和第二次采样每个token的权重都是 1/(200 + 10)，即对不同回答中的token一视同仁。这样就能改善GRPO在长样本训练中的效率低下的问题。这对应着损失函数中的改变，对于loss的aggregation方式由原来GRPO的 $\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ 改为了DAPO的 $\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i = 1}^{G} \sum_{t=1}^{|o_i|}$。从最后的结果也可以看出，token-level的loss训练过程更加稳定，而且控制entropy不会过高（过高引起策略偏向随机，过低则引起探索不足，clip-high可以解决过低问题）。DAPO将sample-level的loss改为了token-level的loss计算，可以有效的让长回答更多影响最后的梯度。每个token的损失将直接对整体梯度产生影响，而不再依赖其所属样本的长度。

最后一个改进点与回答长度也有关，但切入角度不同：这次不是梯度稀释，而是过长回答对整体奖励的负面影响。

### DAPO - Overlong Reward Shaping

DAPO的第四个改进是读i过长回答的奖励进行调整，采用了一种soft punishment的机制对过长回答进行惩罚。特别的，它对处于过长去见的回答施加惩罚，体现在每个token的reward上（即advantage上）。在生成长度突破了预设好的第一个阈值后，这种惩罚开始随着生成长度进行线性增加，当突破了预设的第二个阈值，之后的token所接受的惩罚会抵消原有因回答正确而获得的所有reward，模拟了因回答长度太长而被“视作无效”这一场景。

至此，DAPO 在 Clip-Higher、Dynamic Sampling、Token-Level Gradient Loss 与 Overlong Reward Shaping 四个方面完成了对 GRPO 的“精细打磨”，显著提升了训练效率与稳定性。但在某些特殊架构下（尤其是 MoE），GRPO 本身存在的结构性问题依然没有完全解决，这就引出了 GSPO。

## GSPO：解决 MoE 训练中GRPO不稳定的问题。

如果说 DAPO 是在 GRPO 框架内做“微调与优化”，那么 GSPO 则是直接调整了优化目标的颗粒度——从 token-level 跳到 sequence-level。这一变化背后的动机，主要来自于在 MoE 结构上训练时，GRPO 的重要性采样会带来巨大方差和不稳定性。这个算法的核心观点在于优化奖励过程中对逐个token的依赖，同时加重了对采样sequence整体结果的影响。下面介绍GSPO的主要思想和内容。


> TLDR: 传统的机器学习算法，如PPO，GRPO普遍对模型输出的某个token逐个优化，会出现这个词的优化权重高一些，那个词的优化权重低一些。这种做法本意是更精细的优化，但在长文本，大模型的场景下，反而容易引入噪声和奖励偏差，导致模型训练迷失方向，甚至突然崩溃。问题的本质是，我们通常用回复的完整内容来评价模型，却用逐词的方法来训练他，奖励和优化目标之间出现了错配。GSPO 的核心思路就是把奖励和优化目标重新对齐，从给每个词打分，改为直接优化整个句子。这种切换带来的好处具体为：一是更稳定，GSPO直接对整句进行训练，减少了词级波动带来的训练噪声。二是更高效，GSPO 根据证据的分筛选样本，仅保留高质量，更纯净的样本参与优化，让模型更快收敛，效果更好。特别是在MoE架构下，GSPO 的优势更加突出，由于 MoE每次推理只激活少数几个专家模块，虽然效率更高，但是路径更加动态不可控。为了在训练时复现这种推理路径，传统算法往往需要引入 Routing Replay 的机制：即把推理时的专家模块调用记录下来，再还原到训练中使用，以保证前后一致。这种做法虽然有效，但大大增加了工程成本，还制约了模型性能。GSPO的优化逻辑，不优化每个token，只看总句的整体表现，自然就绕过了Routing replay的需求，这让GSPO在MoE上的训练更轻量化也更稳定。对于越来越多采用MoE价格的大模型来说，这是极具价值的突破。QWen3系列模型就已经使用了GSPO训练。从PPO，到GRPO，再到GSPO的发展，我们可以看到，大模型的强化学习优化目标要贴近任务的本身的性质，训练逻辑则要尽量简洁，可拓展，可落地。推动模型进步的，往往不是复杂的技巧，而是对问题本质的洞察和思考。

PPO 在长文本和复杂任务中面临的主要问题是对 value model 的依赖：当 policy model 输出很长时，value model 的估计精度会显著下降，难以在从简单任务到复杂任务的过程中保持良好泛化，因此其在长文本场景下的可扩展性受到限制。GRPO 的提出有效克服了这一依赖，使得训练过程能够摆脱 value model 的瓶颈。然而，GRPO 在 MoE 架构训练或长时间训练中仍存在稳定性问题：往往在训练达到某个阶段后模型会突然崩溃，即便尝试恢复训练（resume training）或调整参数（tune parameter）也难以挽回。下面，我们首先来分析一下这种现象的原因可能是什么，又如何解决它？


### Importance ratio 到底在起什么作用？在 GRPO 里会带来什么问题？ 

重要性采样存在的意义在于：我们想要估计一个预期的分布，但是我们手上只有另一个behavior分布，我们就只能在behavior policy下进行采样，通过这个样本，赋予这个重要性权重，来估计出target policy下函数的值。但是这种采样的前提在于多次采样，如果只有一次采样，并不能起到分布矫正的作用。问题在于大模型训练过程中，重要性采样都是per-token进行的，单个token进行的重要性采样是无法起到分布矫正的作用的，相反，这种采样手段反而会带来很大方差的噪声，尤其是在MoE这种不稳定的结构下。所以GRPO本身这种逐token的计算可能不太合理。

Per-token 采样和奖励回复的不匹配：我们的奖励其实是对每个回答整体给出的评价，但是在 per-token 的操作中，我们又把这个奖励平摊到每个 token 上（reward shaping），然后试图在 token 层面逐个做调整，所以这里就发生了一个优化的目标和奖励目标的颗粒度的差异。所以既然我们有了 sequence-level 的 reward，我们能不能也把 GRPO 的优化过程改成 sequence-level 的。

### GRPO在MoE结构上为什么难以收敛？(GRPO的局限性)

专家激活的波动性：因为新旧策略可能激活不同的专家，带来结构性偏差，引起噪声。当 $\pi_{\theta_{\text{old}}}$ 更新时，很有可能 Router 也发生了变化，导致新旧策略激活了不同的专家。虽然模型参数只更新了一步，但实际参与计算的专家组合完全不同，导致非常大的输出概率的波动，导致 clipping 被异常地、频繁地触发。Clip 过后的 token 往往就没有梯度，而最终留下来的token往往是有噪音的。所以这两个概率根本不是在相同结构下产生的，理想中的重要性比率应该反应模型在同一结构下参数变化导致的输出概率变化，但这个比率现在由于专家变化，导致高方差的波动，不可预测，与优化方向无关的噪声。这种高方差会导致提督估计严重失真，训练不稳定甚至崩溃。

### GSPO 之前的做法：Routing Replay

把 $\pi_{\theta_{\text{old}}}$ 采样时候的路由激活记录下来，在训练的时候，强制 $\pi_{\theta}$ 和 $\pi_{\theta_{\text{old}}}$ 使用相同的激活方案。问题：对 AI infra 带来非常大的开发工作量和开销；同时对于 $\pi_{\theta}$ ，我有可能已经有了更好的 routing path，但是现在却一定要走 $\pi_{\theta_{\text{old}}}$ 的routing path，这是 training 不是很高效。

传统方法会尝试通过 Routing Replay 来缓解专家激活的不一致，但这会带来工程复杂性与效率损失。GSPO 则选择直接规避这一依赖，从根本上降低了训练过程中的结构性方差。


### GSPO 的损失函数设计

$$
\begin{aligned}
\mathcal{J}_{GSPO}(\theta) = \mathbb{E}_{q \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)} \Bigg[\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg(\min \Bigg(s_{i}(\theta) A_{i},\ 
\text{clip}\Big(s_{i}(\theta), 1-\varepsilon, 1+\varepsilon\Big) A_{i}\Bigg) \Bigg]
\end{aligned}
$$

$$
s_i({\theta}) = \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}(o_i|q)}}^{\frac{1}{|o_i|}} \right) = \text{exp}\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \text{log} \frac{\pi_\theta(o_{i,t}|q,o_{i,&lt;t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,&lt;t})} \right)
$$

> “既然奖励是 sequence-level，那 importance ratio 也应该是sequence-level。”

从上边公式上看，对于重要性采样的 importance ratio，从GRPO原来的 $r_{i,t}(\theta)$ 变成了现在的 $s_i(\theta)$，可以发现这个比值不再和当前 step 的 step index $t$ 挂钩。GSPO的算法希望抛弃掉 GRPO 的 token-level objective，而是把 importance rate 直接用在 sequence-level 上，这也就自然得引入了 GSPO 的优化算法目标，即把token-level的importance rate 换成了 sequence-level 的 importance rate。这里对 sequence-level 的重要性做了长度归一化，这里主要是为了减少方差和统一数值范围。如果不做长度归一化，不同的问题可能回答长度是不一样的，因此importance rate可能会对长度很敏感。这里，由于所有属于同意采样的token用到的importance ratio 都是一样的，所以一旦 clipping 发生，所 clip 掉的将是整个采样到的 sequence，而不是一次采样中的某些 token。长度归一化 $\frac{1}{|o_i|}$ 避免长句子几个 token 波动就导致 ratio 爆炸。

**关于新的重要性采样比值 $s_i({\theta})$ 的讨论：为什么 GSPO 使用指数化的概率比值而不是直接使用对数似然差值？**

指数化是必要的，原因如下：重要性采样的核心公式为：

$$
\mathbb{E}_{z\sim \pi_{\text{tar}}}[f(z)] 
= \mathbb{E}_{z\sim \pi_{\text{beh}}} \left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right].
$$

这里的权重必须是**概率比值**（$\ge 0$），而不是对数概率差值。
如果直接使用对数似然差 $\Delta \log p$，实际上等价于：

$$
\mathbb{E}\left[ \Delta \log p \cdot A \right],
$$

这已经不再是无偏的重要性采样修正。

GSPO 在对数空间进行 $\frac{1}{|y_i|}$ 归一化后再指数化：

$$
s_i(\theta) = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,&lt;t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,&lt;t})} \right).
$$

这样可以保证不同长度序列的重要性比值处于一致的数值范围，不会因为长序列中少数 token 概率变化而导致比值极端放大或缩小。若直接停留在对数空间，长度差异会导致尺度变化很大，clip 范围也需随之调整。同时，PPO 与 GRPO 都是在概率空间中定义重要性比值。若改用对数比值，则需重新推导目标函数，并会破坏与现有 KL 正则项的兼容性。

这种设计不仅在数学上与 PPO/GRPO 的重要性采样公式保持一致，还通过长度归一化与指数化控制了方差，使得 clipping 范围在不同长度的序列间保持一致。


### GSPO 与 GRPO 在梯度上的理论分析


从优化目标的定义出发，GSPO 与 GRPO 的主要区别在于重要性比值的定义及其在梯度计算中的作用。

如果忽略掉clip机制，那么二者梯度本质上的区别在于，是否要对一个回复里边的不同token，他们的梯度做加权平均。GRPO是会对一个回复里边的不同token根据他们各自的重要性权重做加权，但是GSPO对一整个句子做相同importance ratio的放缩。具体而言，GSPO的梯度为：
\begin{equation}
\nabla_\theta J_{\text{GSPO}}(\theta) 
= \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G s_i(\theta) \, A_i \cdot \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,&lt;t}) \right].
\end{equation}

可以看出，GSPO 对同一条回复中的所有 token 赋予相同的权重 $s_i(\theta) {A}_i / |o_i|$，从而保证了序列内部梯度权重的一致性。相比之下，GRPO 的梯度为：
$$
\nabla_\theta J_{\text{GRPO}}(\theta) 
= \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\hat{A}_i}{|y_i|} \sum_{t=1}^{|y_i|} w_{i,t}(\theta) \, \nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,&lt;t}) \right].
$$
可以看出，GRPO 在同一条回复的不同 token 上采用不同的权重 $r_{i,t}(\theta) A_i / |o_i|$，这些权重会随 token 位置和上下文变化而波动，且可能出现较大方差，尤其在长序列或 MoE 模型中更为严重。


另外一个区别在于，GRPO原本的重要性采样权重对clip范围的影响。对于大于零的advantage的样本，GRPO允许的范围是零到一点几，但是对于advantage小于0的样本，clip的数值范围是零点几到正无穷，这是个很大的波动范围。当序列变长的时候，这个时候所携带的噪声是会不断积累的。这也是MoE模型在用GRPO训练时候崩溃的原因之一。

reward监控指标对于模型学偏这件事情是有一定滞后性的，就是模型学偏了一段时间以后，指标上才会有反馈。从实验结果上来看，GSPO实际用于训练的token比GRPO少很多（由于clipping），但同时达到了更高的训练效率。

GSPO-Token：给一个sequence的不同片段分配不同的重要性权重，为了能够进行更细粒度得调整每个token的权重，这里就有了GSPO-Token的变体。其实是变化了一下GRPO的重要性权重的公式。对sequence-level的重要性权重取了一个截断梯度的数值， 

总的来说，GSPO 在梯度计算上实现了序列内部权重的一致性，减少了 token 间的波动方差，尤其适合在长序列和 MoE 结构下进行稳定训练。它的出现，标志着从 PPO → GRPO → GSPO 的演化路线，从依赖 value model 的 token-level 优化，走向了直接面向任务性质的 sequence-level 优化。
