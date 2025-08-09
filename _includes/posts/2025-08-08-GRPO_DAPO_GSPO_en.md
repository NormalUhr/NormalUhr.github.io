# From GRPO to DAPO and GSPO


In the reinforcement learning stage of large language models, PPO was once the mainstream approach. However, its reliance on the value model reveals limitations when handling long text outputs and complex tasks. GRPO removes the dependency on the value model, significantly improving scalability, but still leaves room for optimization in efficiency and stability. This motivated DAPO, which refines details such as sampling, clipping, and gradient calculation. Yet, in MoE architectures with dynamically activated experts, token-level optimization under the GRPO framework still struggles to converge stably. GSPO takes this a step further by shifting the optimization granularity to the sequence level, fundamentally reducing high variance and structural noise. This article follows this evolutionary path: starting from GRPO and gradually unpacking the design motivations and implementation details behind DAPO and GSPO.


In the following article, you’ll discover:

1. Why GRPO breaks free from PPO’s dependency on the value model, yet can still “collapse” in certain scenarios.
2. How Clip-Higher fixes the hidden problem of good tokens being capped too early.
3. How Dynamic Sampling prevents massive computation waste from ineffective samples.
4. How Token-Level Gradient Loss ensures long responses no longer dilute valuable gradient signals.
5. Why GRPO’s per-token importance sampling creates huge variance in MoE architectures.
6. How GSPO replaces token-level optimization with sequence-level optimization to fundamentally improve stability and efficiency.

## A Recap on GRPO

The training objective of GRPO is:

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)} \Bigg[\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg(\min \Bigg(r_{i,t}(\theta) A_{i},\ 
\text{clip}\Big(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\Big) A_{i}\Bigg) -\ \beta\ \mathbb{D}_{KL}\left(\pi_{\theta}\ \|\ \pi_{\text{ref}}\right)\Bigg) \Bigg]
\end{aligned}
$$

where

$$
r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t}|q,o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,< t})}
$$

$$
A_{i}=\frac{r_{i}-\text{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\text{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}
$$

After understanding the GRPO objective, we first need to clarify the role and limitations of **importance sampling**, this is not only essential for understanding GRPO, but also the entry point for the improvements introduced in DAPO and GSPO.

### What role does the Importance Ratio play?

The essence of importance sampling is that we want to compute expectations under a new distribution, but our data is drawn from an old distribution. We therefore use the probability ratio of the same action under the new and old policies as a correction weight:

$$
\mathbb{E}_{p_\text{new}}[f(x)] = \mathbb{E}_{p_\text{old}}\left[\frac{p_\text{new}(x)}{p_\text{old}(x)} f(x)\right]  
$$

This allows us to evaluate the expected value under the new policy using offline data from the old policy, avoiding the need to resample after each update (thus lowering cost). However, if the gap between the new and old policies is too large, the variance of the weights can become very high, leading to unstable training.

The purpose of importance sampling is to estimate expectations under a target distribution when we only have samples from a behavior distribution. In PPO/GRPO, we do not directly sample data from the new policy; instead, we first generate data using the old policy (since sampling is expensive), this process is called **rollout**. When updating, we must correct for the distribution mismatch, and this is where importance sampling comes in. Defining the importance ratio for each token after sampling as:

$$
r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},
$$

the PPO/GRPO objective can be written as:

$$
L(\theta) = \mathbb{E}_t \left[\min(r_tA_t, \text{CLIP}(r_t, 1-\epsilon, 1+\epsilon)A_t)\right]
$$

Here, $A_t$ is the computed advantage, and the clipping limits the update magnitude to prevent the policy from drifting too far from the old policy.

With this intuition of importance sampling in mind, we can further consider its actual effect in PPO/GRPO: the sign of the advantage function $A_t$ and the ratio $r_t$ together determine the direction and magnitude of the policy update.

### How do the signs of $A_t$ and $r_t$ affect training?

Let’s analyze the scenarios. Suppose $A_t > 0$ (the action is better than expected); we want to increase the probability of this action. If we set $\epsilon = 0.2$ in clipping, then when $r_t > 1.2$, the `min` and `clip` operations will cap it at 1.2. When $r_t <  0.8$, no clipping occurs due to the `min` operation, so positive advantages have their upward change limited.

Conversely, when $A_t <  0$ (the action is worse than expected), we should reduce the probability of this action. If $r_t <  0.8$, the `min` operation limits it further, capping at $0.8A_t$; but when $r_t > 1.2$, the `min` operation imposes no restriction (it can go to $+\infty$, and with a negative sign becomes $-\infty$ ). Thus, the downward adjustment for negative advantages is also bounded.

$A_t$ measures whether the current action/trajectory is better or worse than average. If $A_t$ is positive, we encourage it; if negative, we penalize it so it appears less in the future. The importance ratio $r_t$ reflects how much more (or less) likely the new policy is to choose this action compared to the old policy. If $r_t > 1$, the new model prefers this action; if $r_t <  1$, it prefers it less. Among the four possible sign combinations of $A_t$ and $r_t$, we only desire two: when they have the same sign, positive $A_t$ with $r_t > 1$ (reinforce), or negative $A_t$ with $r_t <  1$ (correct mistakes).

However, matching the signs of $A_t$ and $r_t$ is not enough. In PPO/GRPO, the **clipping operation** is equally critical for stable training, as it decides which tokens’ gradients truly contribute to the update.

### Impact of clipping on gradients and token efficiency

For $A_t > 0$, when $r_t > 1 + \epsilon$, i.e., the increase hits the cap, we apply clipping, and the gradient becomes zero. This effectively nullifies the token’s contribution to training. Similarly, for $A_t <  0$, if $r_t <  1 - \epsilon$, i.e., the decrease exceeds the cap, the clipping also sets the gradient to zero. A common misconception is that clipping uses a straight-through estimator to pass the gradient of the clipped value back to the unclipped value; in reality, this does not happen: the gradient before clipping is directly set to zero.

At this point, we have a relatively complete understanding of GRPO’s mechanism, strengths, and limitations. Next, we will see how DAPO, while preserving GRPO’s basic framework, introduces more fine-grained improvements to address efficiency and stability challenges.


## From GRPO to DAPO

DAPO starts from a straightforward motivation: in practical training, GRPO often wastes a large amount of learning signal due to issues such as an unreasonable clip range, redundant sampling, and gradient dilution in long sequences. DAPO addresses these problems with four targeted improvements.

$$
\mathcal{J}_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)}\Bigg[\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i = 1}^{G} \sum_{t=1}^{|o_i|} \min \Bigg(r_{i,t}(\theta) A_{i},\ 
\text{clip}\Big(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\Big) A_{i}\Bigg) \Bigg]
$$

$$
\text{s.t.}, 0 < |\{o_i | \text{is\_equivalent} (a, o_i)\}| < G
$$

### Why does DAPO raise the upper bound $1+\epsilon_{\text{high}}$ while keeping $1-\epsilon_{\text{low}}$ fixed?

The authors observed that choosing a small $\epsilon$ as the clip upper bound can lead to the following problem: if the old policy assigns a very low probability to a sampled token, yet its advantage is positive (meaning the old model sampled something very good), the current policy is given little room to increase its probability, even though increasing it is exactly what we want.  

For example, if the old policy's probability is 0.9 and $\epsilon=0.2$, the upper bound is $0.9 \times 1.2 = 1.08$, which already exceeds the maximum probability of 1.0, so it will never be clipped. But if the old policy's probability is 0.2, the upper bound becomes $0.24$. In this case, even if the current policy raises the probability to 0.4 (a good improvement), the overly small $\epsilon$ causes it to be clipped, effectively discarding that token. This is why DAPO adopts **Clip-Higher**, raising the upper bound improves token efficiency.

This is essentially what we call the "Matthew Effect": *the rich get richer, the poor struggle to improve*. If the old policy barely manages to sample a crucial token, say, `"Wait"`, with very low probability, but the current model significantly increases that probability, it can still be clipped away, depriving the model of its chance to "turn the tables."

Clip-Higher solves the “good tokens being capped too early” problem, but it doesn’t address another common source of waste: lack of sampling diversity. To tackle that, DAPO introduces **Dynamic Sampling**.

### DAPO - Dynamic Sampling

The second innovation in DAPO is **Dynamic Sampling**. The motivation is as follows: suppose for a given query we sample 10 responses, and all 10 are either very good or very bad, consistently receiving max reward or zero reward. Due to GRPO’s computation method, all 10 samples will have an advantage of zero, and thus contribute zero gradient.  

This means the number of effective gradient-contributing samples is far lower than the nominal sample count, which leads to high variance, unstable training, and wasted samples. This effect is particularly strong at the start of training (when the model is poor) and again later (when the model is so good that it frequently produces perfect responses).  

To counter this, DAPO enforces an additional sampling rule: for each query, the set of sampled responses must not all have rewards of 0 or 1. If all samples are 0 or all are 1, additional samples are drawn until this condition is violated. This is expressed in the constraint:

$$
\text{s.t.}, 0 < |\{o_i | \text{is\_equivalent} (a, o_i)\}| < G
$$

which ensures that for the same input, the sampled set contains both correct and incorrect answers.

Beyond sampling diversity, GRPO has another hidden flaw for long responses: **token gradients are diluted as the response length increases**. DAPO’s third improvement addresses this through **Token-Level Gradient Loss**.

### DAPO - Token-Level Gradient Loss

The third innovation in DAPO fixes the problem that, in GRPO, the gradient weight for each token decreases as the sampled response length increases.  

Why does this happen? Suppose we sample twice: one response has 200 tokens, the other has 10 tokens. In GRPO’s formula, we first average the gradients within each sample, then average across the batch. This gives each token in the first response a weight of $(1/200) \times (1/2)$, while each token in the second response gets $(1/10) \times (1/2)$. The shorter response's tokens therefore have a much larger impact.

The downside is clear: for harder questions, long responses are common. If such responses are high-quality, their valuable gradient signals get diluted. If they are poor, and long simply due to repetition or verbosity, the corrective signal is also diminished.  

DAPO’s solution: average over the total number of tokens generated across all samples when computing gradients. In our example, both the long and short responses give each token a weight of $1/(200+10)$. This treats all tokens equally, improving efficiency in training with long samples.

This corresponds to changing the loss aggregation from GRPO’s:

$$
\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
$$

to DAPO’s:

$$
\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i = 1}^{G} \sum_{t=1}^{|o_i|}
$$

Empirically, token-level loss leads to more stable training, prevents entropy from becoming too high (which causes the policy to act randomly), and avoids exploration collapse when entropy is too low (which Clip-Higher also helps address). By shifting from sample-level to token-level loss, DAPO ensures that long responses contribute proportionally to the final gradient: each token directly impacts the overall gradient, independent of its sample length.

The final improvement also concerns response length, but approaches it from a different angle: **the negative impact of overly long responses on overall rewards**.

### DAPO - Overlong Reward Shaping

DAPO’s fourth improvement adjusts rewards for overly long responses using a **soft punishment** mechanism. Specifically, it penalizes tokens once the generated sequence exceeds a predefined first length threshold, with the penalty increasing linearly as length grows. If the length surpasses a second threshold, the penalty is large enough to cancel out the original reward from a correct answer, effectively simulating the scenario where an overly long response is considered invalid.

---

With Clip-Higher, Dynamic Sampling, Token-Level Gradient Loss, and Overlong Reward Shaping, DAPO delivers a fine-grained refinement of GRPO, significantly improving training efficiency and stability. However, in certain architectures, particularly MoE, GRPO still suffers from structural issues that DAPO cannot fully resolve, which leads us to GSPO.


## GSPO: Addressing GRPO Instability in MoE Training

If DAPO can be seen as a “fine-tuning and refinement” within the GRPO framework, GSPO takes a more fundamental step: it changes the optimization granularity from token-level to sequence-level. The motivation behind this shift stems from the fact that, during training with MoE architectures, GRPO’s importance sampling introduces large variance and instability. The core idea of GSPO is to reduce reliance on per-token optimization during reward processing while placing more emphasis on the overall sequence outcome. Below, we introduce the main concepts behind GSPO.

> **TL;DR:** Traditional algorithms such as PPO and GRPO typically optimize each token in the model’s output individually, giving some tokens higher weights and others lower. While this aims for fine-grained optimization, in long-text, large-model scenarios it can instead introduce noise and reward bias, causing the model to lose direction, or even collapse suddenly. The root of the problem is that we evaluate the model based on the full response, yet train it token-by-token, leading to a mismatch between the reward granularity and the optimization objective. GSPO aligns the reward and the optimization target by switching from per-token scoring to sequence-level optimization. This shift offers two main benefits: 

> 1. **Stability** – GSPO optimizes entire sequences, reducing the training noise from token-level fluctuations.  
> 2. **Efficiency** – GSPO filters and retains only high-quality samples for optimization, accelerating convergence and improving results.  
> In MoE architectures, the benefits are even greater: since only a small subset of expert modules is activated per inference, the routing path is dynamic and hard to control. Traditional methods often rely on **Routing Replay**, recording expert activations during inference and enforcing the same routing during training, to ensure consistency. While effective, this greatly increases engineering cost and limits performance. GSPO’s sequence-level logic naturally avoids the need for Routing Replay, making MoE training lighter and more stable. For the growing number of large MoE models, this is a valuable breakthrough. The QWen3 series, for example, has adopted GSPO. From PPO → GRPO → GSPO, we see that RL optimization objectives for LLMs should align closely with the nature of the task, while keeping the training logic simple, scalable, and deployable. Progress is often driven not by complex tricks, but by insights into the core problem.

PPO struggles in long-text and complex tasks primarily due to its reliance on the value model: when the policy model outputs long sequences, value estimates become inaccurate, making it hard to generalize from simple to complex tasks. GRPO removes this dependency, breaking free from the value model bottleneck. However, GRPO still faces stability issues in MoE training or during long training runs: at a certain point, the model can suddenly collapse, and even resuming training or tuning parameters often fails to recover it. Next, let’s analyze the possible causes and solutions.

### What role does the importance ratio play, and why is it problematic in GRPO?

Importance sampling allows us to estimate expectations under a target distribution when we only have samples from a behavior distribution. We do this by weighting samples according to the probability ratio between the target policy and the behavior policy. However, this correction assumes multiple samples, if there is only one sample, it cannot effectively adjust for the distribution shift.

The problem in large-model training is that importance sampling is performed **per-token**, and a single token’s ratio cannot meaningfully perform distribution correction. Instead, it introduces high-variance noise, especially in the unstable MoE setting. This suggests that GRPO’s token-level computation may be inherently suboptimal.

Another mismatch: our reward is given for the **entire response** (sequence-level), but in token-level importance sampling we spread this reward evenly across tokens (reward shaping) and try to adjust them individually. This creates a granularity mismatch between the reward signal and the optimization target. Given that we already have sequence-level rewards, why not also make GRPO’s optimization sequence-level?

### Why does GRPO struggle to converge in MoE architectures?

**Expert activation volatility:** New and old policies may activate different experts, introducing structural bias and noise. When $\pi_{\theta_{\text{old}}}$ is updated, the router may also change, so the two policies could activate completely different sets of experts, even if only one training step has passed. This causes large fluctuations in output probabilities, triggering clipping abnormally often. Clipped tokens contribute no gradient, and those that remain often contain noise.  

In theory, the importance ratio should reflect probability changes caused by parameter updates under the **same** structure. But expert changes lead to unpredictable, high-variance fluctuations unrelated to the optimization direction. This variance distorts policy gradient estimates, making training unstable and even causing collapse.

### Routing Replay before GSPO

Routing Replay records the expert activations during sampling from $\pi_{\theta_{\text{old}}}$ and forces $\pi_{\theta}$ to use the same routing path during training. The downside: high engineering and infrastructure cost, and inefficiency, $\pi_{\theta}$ might have found a better routing path but is forced to follow the old one.

While traditional methods use Routing Replay to mitigate expert activation mismatches, GSPO bypasses this dependency entirely, reducing structural variance at its root.

### GSPO loss design

$$
\begin{aligned}
\mathcal{J}_{GSPO}(\theta) = \mathbb{E}_{q \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q)} \Bigg[\frac{1}{G} \sum_{i = 1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg(\min \Bigg(s_{i}(\theta) A_{i},\ 
\text{clip}\Big(s_{i}(\theta), 1-\varepsilon, 1+\varepsilon\Big) A_{i}\Bigg) \Bigg]
\end{aligned}
$$

$$
s_i({\theta}) = {\left({\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}}\right)}^{\frac{1}{|o_i|}} = \text{exp}\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \text{log} \frac{\pi_\theta(o_{i,t}|q,o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,< t})} \right)
$$

> If the reward is sequence-level, the importance ratio should also be sequence-level.

From the above, GSPO replaces GRPO’s per-token ratio $r_{i,t}(\theta)$ with a sequence-level ratio $s_i(\theta)$, which is no longer tied to the step index $t$. The idea is to drop the token-level objective in favor of sequence-level scaling. This naturally leads to GSPO’s new optimization target: replacing token-level importance ratios with sequence-level ones.  

Sequence-level ratios are **length-normalized** to reduce variance and keep values on a consistent scale. Without normalization, answers of different lengths would make the ratio highly length-sensitive. Since all tokens from the same sequence share the same importance ratio, clipping (if triggered) will clip the **entire sequence**, not just certain tokens. The normalization factor $\frac{1}{|o_i|}$ also prevents a few volatile tokens in a long sequence from causing the ratio to explode.

**Why exponentiate instead of using log-likelihood differences directly?**

Exponentiation is necessary because the core formula for importance sampling is:

$$
\mathbb{E}_{z\sim \pi_{\text{tar}}}[f(z)] 
= \mathbb{E}_{z\sim \pi_{\text{beh}}} \left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right]
$$

Here, the weight must be a **probability ratio** ( $\ge 0$ ), not a log-probability difference. If we used $\Delta \log p$ directly, it would be equivalent to:

$$
\mathbb{E}\left[ \Delta \log p \cdot A \right],
$$

which is no longer an unbiased importance sampling correction.

GSPO normalizes in log space by $\frac{1}{\|o_i\|}$ and then exponentiates:

$$
s_i(\theta) = \exp\left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \right)
$$

This ensures consistent scaling of importance ratios across sequence lengths, avoiding extreme values from a few token probability shifts in long sequences. Staying in log space without exponentiation would make ratios length-sensitive, require clip range adjustment, and break compatibility with the KL regularization used in PPO/GRPO.

### Theoretical gradient analysis: GSPO vs. GRPO

From the objective definitions, the key difference lies in how importance ratios are defined and used in gradient computation.

Without clipping, the distinction is whether to weight tokens differently within the same response. GRPO assigns each token its own weight based on $r_{i,t}(\theta)$, while GSPO applies the same $s_i(\theta)$ to all tokens in a sequence.

GSPO’s gradient:

$$
\nabla_\theta J_{\text{GSPO}}(\theta) 
= \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G s_i(\theta) \, A_i \cdot \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,< t}) \right]
$$

Here, all tokens in a response share the same weight $s_i(\theta) A_i / \|o_i\|$, ensuring intra-sequence gradient consistency.

GRPO’s gradient:

$$
\nabla_\theta J_{\text{GRPO}}(\theta) 
= \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\hat{A}_i}{|o_i|} \sum_{t=1}^{|o_i|} r_{i,t}(\theta) \, \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,< t}) \right]
$$

Here, weights $r_{i,t}(\theta) A_i / \|o_i\|$ vary by token position and context, leading to higher variance, especially in long sequences or MoE models.

Another difference is how clipping interacts with these ratios. For positive-advantage samples, GRPO’s ratio range is roughly [0, ~1.x]; for negative-advantage samples, it can be [~0.x, ∞), a much wider range. Over long sequences, noise from this asymmetry can accumulate, contributing to MoE instability under GRPO.

Reward metrics also lag in detecting model drift, by the time the issue appears, the model may have diverged for a while. Experiments show GSPO trains with fewer effective tokens (due to more aggressive clipping) yet achieves higher training efficiency.

---

In summary, GSPO achieves consistent intra-sequence gradient weights, reduces variance between tokens, and is especially suited for stable training in long-sequence and MoE scenarios. Its introduction marks a shift from PPO → GRPO → GSPO, moving away from token-level optimization reliant on the value model toward sequence-level optimization aligned with the nature of the task.

