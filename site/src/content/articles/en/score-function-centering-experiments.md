---
title: "Stabilize Your Async RL Training with Score Function Centering and Trainer-Generator KL"
date: 2026-10-14
category: post-training
series: grpo-line
part: 6
tags: ["GRPO"]
---

# Stabilize Your Async RL Training with Score Function Centering and Trainer-Generator KL

*This English version was translated from Chinese by AI.*

This is the third article in the Score Function Centering series. In the first two articles we started from the score function and derived the SFC+KL method step by step, but the experiments only got one figure at the end of the second article. In this article I want to go through the experiments in full, to see what the analysis of the first two articles looks like in real training, and also to answer some questions that the first two articles left open.

If you have not read the first two articles yet, I strongly suggest reading [the first one](https://normaluhr.github.io/en/p/score-function-centering/) and [the second one](https://normaluhr.github.io/en/p/score-function-centering-kl/) first, because this article reuses their notation and conclusions directly. For convenience, I will first recap the main points of the first two articles below.

**Open source**: the experiment configurations, training curves and code used in this article are all open-sourced at [github.com/NormalUhr/score-function-centering-trainer-generator-kl](https://github.com/NormalUhr/score-function-centering-trainer-generator-kl).

## Recap

In async RL, rollouts are produced by the generator, and before the trainer uses these rollouts it has usually been updated for many more steps. We write the generator that produces a rollout as $q_\phi$, and the trainer that is doing the backward pass as $p_\theta$.

[The first article](https://normaluhr.github.io/en/p/score-function-centering/) started from the score function. Under a softmax, the score equals the logit Jacobian of the sampled action minus the average of the logit Jacobians of all actions under $p_\theta$:

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

So as long as the action is sampled from $p_\theta$, the expectation of the score is 0. This is exactly why we can safely subtract a baseline from the reward, which is to say, use an advantage. But in async RL the action is sampled from $q_\phi$, and this zero-mean property no longer holds:

$$
\mathbb E_{a\sim q_\phi}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

In training, the practical consequence of this nonzero expectation is that the average advantage multiplies this direction and produces an extra gradient along the difference between the trainer and the generator. When the average advantage is positive, this gradient pulls the trainer toward the old generator, and how hard it pulls is decided entirely by the size of the average advantage. What Score Function Centering (SFC below) does is to center with $q_\phi$, the distribution that actually generated the action:

$$
\tilde s_{q,\theta}(a)=J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

This brings the zero-mean property back. To compute $\mathbb E_{q_\phi}[J_\theta]$, the generator has to send part of its distribution back to the trainer along with every token it generates, for example the few tokens with the highest probabilities, plus a few tokens sampled from the remaining tail. Below, I call this information the payload.

[The second article](https://normaluhr.github.io/en/p/score-function-centering-kl/) went on to the problem that SFC does not solve. SFC centers the score correctly, but what we learn is still the learning signal under $q_\phi$, while the trainer has already become $p_\theta$. The most common approach is importance sampling: multiply every token by $p_\theta/q_\phi$ to correct the update back to $p_\theta$. But the ratio easily becomes very large, so in practice it has to be truncated, and truncation brings bias. Worse, multiplying a score that is already centered under $q_\phi$ by $p_\theta/q_\phi$ switches the expectation back to $p_\theta$, and the zero-mean property that SFC worked so hard to recover is gone again.

So the second article took another road. We accept the fact that the data comes from $q_\phi$, stop correcting the update, and control the distance between $q_\phi$ and $p_\theta$ instead. The second article derived a local bound:

$$
\|g_q-g_p\|\le B\sqrt{2D_{\mathrm{KL}}(q_\phi\|p_\theta)},
$$

where $g_q$ is the update that SFC gives under $q_\phi$ and $g_p$ is the update under the current policy. In other words, as long as $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ is kept under control, the two updates cannot be too far apart. Even better, the gradient direction of this KL is exactly the direction that SFC removed from the reward gradient. So the final update can be written as:

$$
g=\operatorname{Cov}_{q_\phi}(A,J_\theta)+\beta\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).
$$

The first term lets the reward decide what to learn, the second term lets $\beta$ decide how far the trainer may be from the sampler, and the two questions are separated. In addition, the older the sampler, the less suitable it is as a reference, so the KL coefficient of each token decays with the token's age $e_t$:

$$
\beta_t=\begin{cases}0,&e_t=0,\\ \beta_0/\sqrt{e_t},&e_t\ge 1.\end{cases}
$$

This is the method proposed in the second article, and below I call it SFC+KL throughout. Its biggest difference from importance sampling is that it uses no importance ratio at all in the reward gradient.

In the figure at the end of the second article, across seven asynchronous settings, SFC+KL was the only method that never collapsed. But that figure only gave the results, and this article wants to explain what happened to each method during training. These are the questions this article discusses:

* Is SFC alone enough?
* What goes wrong when importance sampling is added on top of SFC?
* Can importance sampling be dropped altogether?
* How much information does SFC need the generator to send back?
* Do these conclusions still hold with a different optimizer, dataset and model?
* What happens if the difference between the trainer and the generator is not caused by staleness?

At the end there is a bonus section that compares SFC+KL with PPO, DAPO, GSPO, DPPO and TOPR, and also checks whether they can be combined. If you only want the conclusions, you can jump straight to the takeaways at the end.

## 1. Experimental setup

All experiments run on a real asynchronous pipeline. The generator keeps producing rollouts, the trainer keeps updating, and every token records the version of the sampler that generated it, so the staleness of every token is measured.

Most experiments use Qwen3-0.6B and the [Countdown](https://arxiv.org/abs/2404.03683) dataset. Countdown gives the model a few numbers and a target, and the model has to write an arithmetic expression over these numbers whose result equals the target, 1 point when it is right and 0 when it is wrong, so the reward below can be read directly as accuracy. Every run trains for 3,200 steps, the context length is at most 4,096 tokens, the optimizer is SGD, and $\beta_0$ of SFC+KL is 0.3.

The size of the staleness is set by the granularity at which the trainer consumes rollouts. The generator produces the rollouts of 64 prompts at a time, the trainer cuts this batch of rollouts into a number of slices, and each slice takes one optimizer step. When each slice has 4 prompts, we call it a 4-prompt slice, and the mean staleness of the tokens is then about 50 steps. When each slice has only 2 prompts, the same batch of rollouts takes twice as many steps to consume, and the mean staleness becomes about 100 steps.

Below I mainly compare four methods built on SFC. SFC does centering only. SFC+TIS multiplies the centered score by a truncated importance ratio $\min(p_\theta/q_\phi,\ 2)$, which is what [truncated importance sampling](https://fengyao.notion.site/off-policy-rl) does. SFC+MIS keeps only the tokens whose ratio falls inside $[0.5,\ 5]$ and masks out all the other tokens, an approach that comes from [IcePop](https://arxiv.org/abs/2510.18855). Combining these two ratio corrections with SFC is what [this work](https://arxiv.org/abs/2609.20807) does. The last one is SFC+KL, which this blog series proposes.

When comparing methods, I look at the best held-out evaluation reward that each run reaches over the whole training, called the best score below. In real training, the checkpoint we keep and use in the end is usually the one with the best evaluation anyway. This way, even if a run collapses later, its best score before the collapse still counts, so besides comparing best scores, I also look separately at whether each run collapsed.

## 2. Is SFC alone enough?

This question asks whether score function centering by itself can stabilize async RL training. The answer is no. On the two Countdown settings, the runs that do only SFC basically did not learn anything.

The two figures below show the training reward and the held-out evaluation reward of the four methods on Countdown, with the 4-prompt slice on the left and the 2-prompt slice on the right. This section and the next two sections all use these two figures.

<figure data-chart="countdown-train">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/countdown-train.png" alt="Training reward of the four methods on Countdown">
</noscript>
</figure>
<script src="/js/sfc-exp-charts.js" defer></script>

<figure data-chart="countdown-eval">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/countdown-eval.png" alt="Held-out evaluation reward of the four methods on Countdown">
</noscript>
</figure>

Look at SFC first. In both settings, its training reward stayed around 0.1 for the first one or two hundred steps, then dropped to 0 and never came back. These two runs were stopped at step 400 and step 640, and their best scores are the evaluations from before training started, 0.080 and 0.094.

This agrees with the analysis at the beginning of the second article. What SFC repairs is the score drift, but it does not limit the distance between the trainer and the sampler. With the sampler 50 to 100 steps older than the trainer on average, training with SFC alone collapsed very quickly. The KL added in the second article is exactly what fills this gap.

## 3. What happens when importance sampling is added on top of SFC?

On the 4-prompt slice, SFC+TIS and SFC+MIS can both learn for a while, but neither makes it to the end. On the 2-prompt slice, where the staleness is larger, they did not learn from the start. I also saw something in them that SFC+KL does not have: the responses they generate are especially long. SFC+TIS is the clearest case. It makes the responses longer and longer during training, and the share of truncated responses is much higher as a result.

Look at SFC+MIS first. On the 4-prompt slice it learned well at first: its evaluation at step 800 was 0.646, while SFC+KL at the same point was 0.725. But from then on it slowly went down, and it finally collapsed at step 3,168. On the 2-prompt slice, where the staleness is larger, it did not learn from the start and was stopped at step 400.

SFC+TIS is a bit different. On the 4-prompt slice, its training reward rose to around 0.7 at one point within the first 1,300 steps, not far from SFC+KL, only with much larger fluctuations. But its evaluation stayed low, only 0.389 at step 800. Around step 1,350 its training reward suddenly fell and soon reached 0, and it did not recover all the way to step 3,200, with the evaluation also staying at 0. On the 2-prompt slice it never learned anything from beginning to end, and its best score is the 0.111 from the start of training.

Now the response lengths. Over the last 200 steps of training on the 4-prompt slice, the share of responses truncated for exceeding the 4,096-token context length was 99% for SFC+TIS and 49% for SFC+MIS, but only 19% for SFC+KL. For the three runs on the 2-prompt slice that did not learn, this share was also between 89% and 97%.

The second article discussed two problems that ratio correction can bring. One is the bias from truncation. The other is that the ratio is incompatible with SFC: multiplying a centered score by the ratio takes the zero-mean property away again. MIS is similar to TIS, except that it masks out entire tokens whose ratio falls outside the interval. For SFC+TIS, I went back and looked carefully at the training logs and also spot-checked what the model generated, and section 3.2 explains step by step how it broke down.

Seeing so many truncated responses, the first possibility that came to my mind was that a context length of 4,096 tokens is too short. If SFC+TIS needs longer responses to solve the problems in the first place, it makes sense that the reward drops once the length is capped. The next subsection looks at exactly this.

### 3.1 Did SFC+TIS collapse because the context length cap was too small?

No. After raising the context length cap to 8,192 and 16,384, far fewer responses were truncated, but SFC+TIS still collapsed.

This group of experiments stays on the Countdown 4-prompt slice with SFC+TIS, and only raises the context length cap to two times and four times the original.

<figure data-chart="budget">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/budget.png" alt="Training reward and held-out evaluation reward of SFC+TIS under three context length caps">
</noscript>
</figure>

The table below gives, for these runs, the share of truncated responses over the last 200 steps before training ended and their best scores, with SFC+KL at 4,096 as a reference.

| Method | Context length cap | Share truncated in the last 200 steps | Best score | When the training reward fell |
| --- | --- | --- | --- | --- |
| SFC+KL | 4,096 | 19% | 0.725 | It did not fall |
| SFC+TIS | 4,096 | 99% | 0.389 | Around step 1,400 |
| SFC+TIS | 8,192 | 31% | 0.639 | Around step 900 |
| SFC+TIS | 16,384 | 15% | 0.650 | Around step 1,100 |

With a higher cap, the truncated share did come down, and at 16,384 it is even lower than SFC+KL at 4,096. The extra length also did let SFC+TIS learn a bit better: its best score rose from 0.389 to around 0.65, though it is still below the 0.725 of SFC+KL at 4,096.

But they still collapsed in the end, and at about the same time as the 4,096 run.

So the length cap is not the cause of the SFC+TIS collapse. With a higher cap it can learn a little more before collapsing, but the collapse itself is not avoided. Why the responses get longer and longer is what the next subsection looks at.

### 3.2 Why do the responses of SFC+TIS get longer and longer?

In short, these very long responses are a search that becomes more and more aimless and finally turns into meaningless loops, so no amount of extra length is ever enough.

First, let us see what a very long response on Countdown actually contains. I took the model of the 16,384 run at step 319, before the collapse started, and sampled it on three held-out problems with the length cap set to 2,048 tokens. The easier problem was to make 67 from 28, 10, 33 and 16. The model got it right both times, using 595 and 790 tokens, by trying one piece at a time:

```
What about:
33 - 16 = 17
28 - 10 = 18
18 + 17 = 35. No.
...
How about:
(16 + 33) + (28 - 10) = 49 + 18 = 67. Yes!
```

On the other two, harder problems, the model kept writing until it reached the 2,048-token cap, with Let me think, What about and Let me try appearing over and over, and the same few partial sums recomputed each time. A normally trained SFC+KL behaves the same way on these three problems: it tries first and then solves the easy one, and it also keeps writing until the cap on the two hard ones. So on Countdown, a very long response is really a brute-force search that has not converged, and how long it gets depends on how aimless that search is.

The problem with SFC+TIS is that training makes this search more and more aimless. From the logs, its collapse goes through three stages, and they are the same under both the 4,096 and the 8,192 caps.

The first stage is a steady rise in entropy. The per-token entropy of the model kept rising for several hundred steps in a row, while the KL between the trainer and the sampler stayed very small. Under the 8,192 cap, the entropy rose from 0.52 in the first 200 steps to 0.83 at steps 576 to 639, and then to 1.4 at steps 768 to 895. The mean response length rose along with it, from 381 tokens to 4,896 tokens, while the training reward could still stay between 0.5 and 0.75 at this point. Under the 4,096 cap, the entropy rose from 0.40 to 0.98 between steps 1,200 and 1,399. A length of 4,896 tokens is already beyond the 4,096 cap, so under the 4,096 cap these responses get truncated. Add to that the fact that after the collapse the model keeps repeating itself up to the cap, and that is why SFC+TIS has a much higher truncated share than SFC+KL.

The second stage is the KL getting out of control. Within one or two windows of 64 steps, the KL between the trainer and the sampler suddenly rose by two orders of magnitude, to 8.4 for the 8,192 run and 5.0 for the 4,096 run. At the same time the gradient norm went down and the reward fell to 0. My understanding is this: the farther the trainer is from the sampler, the lower the probability that the trainer assigns to the tokens the sampler generated, so their ratios are either close to 0 or truncated, the sampler's data almost stops having any effect, and nothing is left that can pull the trainer back. In addition, the tokens truncated by $\min(\cdot,\ 2)$ were only 1.5% to 3% from beginning to end, so truncation itself does not look like the main cause.

The third stage is training coming to a complete stop. All the responses in a batch are wrong, the advantages after group centering are all 0, the gradient norm is exactly 0, and the model can no longer move at all. This is also why the 4,096 run never recovered all the way to step 3,200 after its reward fell to 0. What the model outputs at this point depends on the length cap. Under the 4,096 cap, the model repeats the same content almost deterministically until it hits the cap, and those are the 99% truncated responses. Under the 8,192 cap, the model rambles for about 2,300 tokens with an entropy of about 0.45, and no parseable answer can be found in it. So the extra length just ends up filled with more meaningless content.

SFC+KL on the same setting did not go down this road at all. Over the full 3,200 steps, its entropy stayed between 0.23 and 0.35, and its KL to the sampler stayed between 0.003 and 0.004. I think the reason is that the KL constrains the complete distribution at every position, including how spread out this distribution is, and also the probability of the token that ends the sequence. An importance ratio acts only on the token that was sampled, so it cannot do this.

## 4. Then can importance sampling be dropped altogether?

Yes. SFC+KL uses no importance ratio at all in the reward gradient. In the two figures of section 2, it is the only method that trains stably to the end on both settings, with best scores of 0.725 and 0.670.

Across all the settings in this article, SFC+KL in its default configuration never collapsed even once, and that includes the experiments later on with a different optimizer, a different dataset, a different model, and a quantized sampler. In the same settings, SFC, SFC+TIS and SFC+MIS each collapsed more than once.

This is also the one thing the second article most wanted to show. A stale rollout does not necessarily have to be reweighted into data under the current policy. We can keep its learning signal under $q_\phi$ and use KL to control how far the trainer is from the sampler, and training can be just as stable.

## 5. How much information does SFC need the generator to send back?

On Countdown, the composition of the payload has little effect on the best score, and sending back 16 tokens per position is enough. The difference lies in stability: the run that sends back only the top-16 collapsed later, while the configurations that also use 16 tokens but take part of them from tail samples did not.

Let me first say what is being compared here. The default payload used in all the experiments above is top-128: at every position the generator sends back only the 128 tokens with the highest probabilities and their logprobs, and the tail is not sampled. On the trainer side, the total probability of the tail is set to the value the sampler gives, and how much probability each token in the tail gets is allocated according to the trainer's own distribution. The other approach is the head plus tail described in section 3.5 of the first article. The head still sends back the top-k, and for the tail, a few extra tokens are sampled from the sampler and sent back, which I call tail samples below. The width of the payload is the total number of tokens in these two parts.

This group of experiments is all on the Countdown 4-prompt slice with SFC+KL, and only the composition of the payload changes.

<figure data-chart="payload">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/payload.png" alt="Best score of SFC+KL with different payloads">
</noscript>
</figure>

The best scores of the nine configurations are all between 0.68 and 0.73, so from the best score alone I cannot tell which of these configurations is better.

The difference is in stability. The run that sends back only the top-16 scored 0.713 at step 2,000, as good as the other configurations, but it collapsed at step 2,992. With the same 16 tokens, top-8 plus 8 tail samples and top-4 plus 12 tail samples both trained stably to the end.

My understanding is this. When only the top-k is sent back, the tail is filled in with the trainer's own distribution, and how accurate that is depends on how much probability the tail holds and on how much the trainer and the sampler differ on the tail. The narrower the top-k, the more probability is left to the tail, and the larger the error of this approximation. Tail samples come directly from the sampler's own distribution, so the estimate of the tail does not have this bias, at the cost of some extra variance.

This result also shows that a collapse can sometimes come very late. If we had trained for only 2,000 steps, we would probably have happily concluded that "sending back only the top-16 is enough".

The width of the payload also affects speed. I tested each payload separately for 100 steps, without evaluation and with the same seed.

| Payload | Rollout file size | Generator time per 128 decoded tokens | Trainer compute time per step |
| --- | --- | --- | --- |
| top-128 | 2,074 MB | 4.28 s | 1.22 s |
| top-64 plus 64 tail samples | 2,074 MB | 4.82 s | 1.20 s |
| top-32 plus 96 tail samples | 2,074 MB | 5.03 s | 1.21 s |
| top-32 | 538 MB | 4.19 s | 1.24 s |
| top-16 plus 16 tail samples | 538 MB | 4.32 s | 1.21 s |
| top-8 plus 24 tail samples | 538 MB | 4.38 s | 1.22 s |
| top-16 | 282 MB | 4.14 s | 1.23 s |
| top-8 plus 8 tail samples | 282 MB | 4.23 s | 1.22 s |
| top-4 plus 12 tail samples | 282 MB | 4.26 s | 1.23 s |

The trainer's compute time per step is almost unaffected by the payload and stays around 1.2 seconds. What changes most is the rollout file. As the width goes from 128 down to 16, the rollout file goes from 2,074 MB down to 282 MB, and the time to read and upload it goes from 1.4 to 1.5 seconds down to about 0.2 seconds. The cost of tail samples falls on the generator: at the same width, the more tokens are sampled, the slower the decoding. At a width of 128, 96 tail samples made the decode time 18% longer, while at a width of 16 they made it only 2% to 3% longer.

If I wanted to save bandwidth, I would pick a configuration like top-8 plus 8 tail samples. Its rollout file is about one seventh of the default configuration's, its generator decode time is about the same as the default configuration's, and its best score is 0.705, on par with the other configurations.

## 6. What about a different optimizer, dataset or model?

On these five settings, SFC+KL trained to the end without collapsing. But looking at the best score, it is not the highest every time: on Qwen3-4B and GLM-4.5-Air, SFC+MIS has a higher best score.

There are five settings here, which are the five settings in the figure of the second article other than Countdown. The first one changes the optimizer to AdamW and is otherwise the same as the Countdown 4-prompt slice. The second one changes the dataset to [GSM8K](https://arxiv.org/abs/2110.14168), still with Qwen3-0.6B and the 4-prompt slice. The remaining three change to larger models, Qwen3-4B, Qwen3-30B-A3B and GLM-4.5-Air, where GLM-4.5-Air has 106B total parameters and 12B active parameters. These three models use the 1-prompt slice, and their mean staleness is between about 90 and 370 steps.

<figure data-chart="settings">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings.png" alt="Best score of the four methods in the other five settings">
</noscript>
</figure>

The two figures below are the training reward and held-out evaluation reward curves on these five settings.

<figure data-chart="settings-train">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings-train.png" alt="Training reward of the four methods in the other five settings">
</noscript>
</figure>

<figure data-chart="settings-eval">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/settings-eval.png" alt="Held-out evaluation reward of the four methods in the other five settings">
</noscript>
</figure>

The optimizer first. With AdamW, SFC+KL, SFC+TIS and the SFC-only run all got above 0.7 at step 800, at 0.738, 0.725 and 0.715 respectively, while SFC+MIS had a lower best score of 0.652. The difference shows up later. SFC+TIS and SFC+MIS both collapsed later on. The SFC-only run did not collapse, but its evaluation kept falling, down to 0.527. Only SFC+KL stayed above 0.7 the whole time. This is also the only time that SFC alone did not collapse. My guess is that AdamW normalizes the update size of every parameter, which by itself acts as a kind of constraint, but I have not verified this guess.

Then the dataset. On GSM8K, SFC+TIS trained to the end with a best score of 0.785, and SFC+KL had 0.803, so the two are about the same. This is the only one of these five settings where SFC+TIS behaved normally. Its responses were still very long, with a truncated share of 97% against 43% for SFC+KL. It is just that on GSM8K this did not stop it from solving the problems. SFC alone and SFC+MIS both collapsed on GSM8K.

Finally, the three larger models. The best scores of SFC+KL are 0.744, 0.877 and 0.750. SFC+TIS collapsed on Qwen3-4B and GLM-4.5-Air. On Qwen3-30B-A3B it did not collapse, but it never went above the 0.680 from the start of training. The SFC-only run was run only on Qwen3-4B, and it also collapsed.

SFC+MIS on these three models is quite interesting. It collapsed on none of them, and its best was 0.871 on Qwen3-4B and 0.777 on GLM-4.5-Air, both higher than SFC+KL. But its scores swung a lot and dropped after the peak, and its last evaluations were only 0.668 and 0.574. The score of SFC+KL on GLM-4.5-Air first dropped a bit at step 400 and then climbed slowly, and its best score came at the end of training.

There is one thing to point out here. On Qwen3-4B and Qwen3-30B-A3B, SFC+KL was evaluated only once at the start and once at the end of training, while SFC+MIS was evaluated every 400 steps. So on these two models, the best score of SFC+KL is in fact its last score, and we do not know whether it reached higher somewhere in the middle of training.

There is also one thing I have not figured out. SFC+MIS collapsed on all four settings with Qwen3-0.6B, but on these three larger models, where the staleness did not get smaller, it held up on every one of them.

## 7. What if the difference between the trainer and the generator is not caused by staleness?

Then KL has no clear advantage. In this group of quantized-sampler experiments, SFC+TIS and SFC+KL performed almost the same.

The discussion so far has assumed that the difference between the trainer and the generator comes from staleness. But in real systems there is another common source. To run inference faster, the generator uses a quantized model while the trainer uses an unquantized one, so even when both sides use the same version of the parameters, the distributions they give are not exactly the same.

This group of experiments uses Qwen3-30B-A3B-Base and the math problems in [INTELLECT-2](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-2-RL-Dataset), with the generator using three kinds of quantization, and each run trains for 800 steps. Unlike before, $\beta_0$ of SFC+KL is 0.03 in this group.

| Generator quantization | SFC+TIS | SFC+KL |
| --- | --- | --- |
| FP8 | 0.401 | 0.408 |
| FP8, plus an FP4 KV cache | 0.401 | 0.405 |
| INT8, plus an INT4 KV cache | 0.366 | 0.354 |

The gaps in all three groups are within 0.012, and I would treat them as the same. On this group of experiments, TIS works as well as KL. For reference, these runs all scored between 0.33 and 0.35 at the start of training. In the INT8 plus INT4 KV cache group, both methods improved by only about 0.01 after 800 steps of training, so quantized to this degree, neither method really learned anything.

I think the reason is that the whole argument of the second article is built around staleness. The older the sampler, the larger its distance to the trainer, and that distance is exactly what the KL is meant to control. The difference that quantization brings is a different situation: both sides use the same version of the parameters, and the distributions differ only numerically, so there is not much for the KL to constrain in the first place. On the TIS side, the ratios should mostly be close to 1, few tokens get truncated, and the bias from truncation is small. The second article sets $\beta_t=0$ for fresh tokens for the same reason: a small difference caused by numerical precision under the same version does not need to be treated as a target to imitate.

## 8. Bonus: next to other methods, and whether they can be combined

The sections above only compare the four SFC-based methods with each other. This section brings in PPO, DAPO, GSPO, DPPO and TOPR as well. For PPO you can read [this article](https://normaluhr.github.io/en/p/grpo/), and for DAPO and GSPO [this one](https://normaluhr.github.io/en/p/grpo-dapo-gspo/). [DPPO](https://arxiv.org/abs/2602.04879) replaces PPO's ratio clipping with a constraint on the policy divergence. [TOPR](https://arxiv.org/abs/2503.14286) truncates the importance sampling differently for samples with positive and with negative rewards.

### 8.1 Next to other methods

Compared on its own, SFC+KL is not the best on every setting. The table below gives the best score of each method on six settings, and the first column is the average score of these runs at the start of training.

| Setting | At the start of training | PPO | DAPO | GSPO | DPPO | TOPR | SFC+KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Countdown, 4-prompt slice | 0.09 | 0.646 | 0.646 | 0.691 | 0.668 | 0.699 | **0.725** |
| Countdown, 2-prompt slice | 0.10 | 0.531 | 0.342 | 0.656 | 0.078 | **0.676** | 0.670 |
| AdamW, 4-prompt slice | 0.09 | 0.701 | 0.650 | 0.680 | 0.697 | 0.691 | **0.738** |
| GSM8K, 4-prompt slice | 0.39 | 0.645 | 0.691 | 0.838 | 0.393 | **0.842** | 0.803 |
| Qwen3-4B, 1-prompt slice | 0.41 | 0.416 | 0.543 | 0.557 | 0.398 | **0.762** | 0.744 |
| Qwen3-30B-A3B, 1-prompt slice | 0.67 | Still training | 0.814 | 0.791 | 0.830 | 0.799 | **0.877** |

SFC+KL is the highest on three settings, and TOPR on the other three. On the Countdown 2-prompt slice the two differ by only 0.006, which can be seen as a tie. On GSM8K, TOPR and GSPO are both more than 0.035 higher than SFC+KL. On Qwen3-4B, TOPR is 0.018 higher than SFC+KL, though as mentioned before, SFC+KL was evaluated only at the end on this model. In terms of stability, TOPR trained to the end on all six settings, just like SFC+KL. PPO, DAPO and DPPO each failed on more than one setting: some collapsed midway, and some never got above their score at the start of training.

### 8.2 Can they be combined?

Yes, and adding it rarely makes things worse. I added SFC+KL on top of TOPR and GSPO, that is, I added SFC's centering and the KL term on top of each of them, which gives 12 comparisons over the six settings.

<figure data-chart="compose">
<noscript>
<img src="https://normaluhr.github.io/img/in-post/2026-10-14-score-function-centering-experiments/compose.png" alt="Best score of TOPR and GSPO before and after adding SFC+KL">
</noscript>
</figure>

In 10 of the 12 comparisons the best score went up, though quite a few of them by less than 0.02, which is not much of a difference. Five comparisons changed by more than 0.02. Among the four that improved, the most obvious one is Qwen3-4B. TOPR went from 0.762 up to 0.822. GSPO on its own reached 0.557 at step 400 and then dropped to nearly 0, and with SFC+KL added it trained to the end with a best score of 0.822. The other two are TOPR on the Countdown 2-prompt slice and GSPO on Qwen3-30B-A3B. The one that got worse is TOPR on GSM8K, lower by 0.022.

So SFC+KL can also be seen as a component that can be added on top of other methods, and not necessarily as something you have to choose instead of them. Adding it rarely makes things worse, and on a setting like Qwen3-4B it can help a lot.

## 9. A few takeaways

Finally, let me summarize what these experiments tell us:

* In asynchronous training with large staleness, SFC alone is not enough. It repairs the score drift but does not limit how far the trainer is from the sampler, and with SGD it collapses very quickly.
* Adding TIS or MIS on top of SFC is very unstable. On Qwen3-0.6B, they collapse or fail to learn most of the time. SFC+MIS did not collapse on two of the larger models, and its best score even beat SFC+KL there, but it dropped hard after the peak.
* SFC+TIS makes the responses longer and longer, and the truncated share is much higher as a result. Its entropy keeps rising during training and the search becomes more and more aimless. In the end the KL between the trainer and the sampler suddenly rises by two orders of magnitude, and training comes to a complete stop. Raising the context length cap fourfold did not prevent the collapse either, and the extra length was just filled with meaningless content.
* SFC+KL removes importance sampling and uses KL to control the distance between the trainer and the sampler instead, and it never collapsed in any setting of this article. In five of the seven asynchronous settings, its best score was the highest among these SFC methods.
* On Countdown, the composition of the payload has little effect on the best score, and compressing it to 16 tokens per position is enough, but part of it should go to tail samples. The run that sent back only the top-16 collapsed late in training.
* If the difference between the trainer and the generator comes mainly from quantization rather than from staleness, TIS works as well as KL.
* SFC+KL can be added on top of methods like TOPR and GSPO. In 10 of the 12 comparisons the best score went up, and Qwen3-4B improved the most.

## 10. Closing words

Limited by computing resources, these experiments did not do a systematic hyperparameter sweep, and they did not run multiple seeds for each configuration, so the results in this article are for reference only. The link to the code and data is at the beginning of the article, and you are welcome to keep building on it.

Finally, a short look back at the series. The first article was about centering with $q_\phi$, the distribution that actually generated the data, in async RL. The second article was about how a stale rollout does not necessarily have to be corrected back to the current policy with importance sampling: we can keep its learning signal under $q_\phi$ and use KL to control how far the trainer is from the sampler. This article tested both ideas with experiments. Thank you for reading this far.
