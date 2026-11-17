---
title: "Why Does an Unbiased KL Estimate Not Give the Right Gradient? From Common KL Estimators to the Trainer-Generator Constraint in Asynchronous RL"
date: 2026-11-11
category: post-training
series: grpo-line
part: 7
tags: ["GRPO"]
---

# Why Does an Unbiased KL Estimate Not Give the Right Gradient? From Common KL Estimators to the Trainer-Generator Constraint in Asynchronous RL

<link rel="stylesheet" href="/css/kl-gradient.css">

We wrote [a reading note on "Approximating KL Divergence"](https://normaluhr.github.io/en/p/kl/) earlier. It discussed how large the bias and the variance of the three estimators $k_1$, $k_2$ and $k_3$ are when they estimate the value of the KL divergence. This article continues that note. In training, the value of the KL is mostly only logged, or used to adjust the KL weight $\beta$. What actually affects the model update is the gradient that the KL produces in the backward pass, after we write it into the loss or into the reward. An estimator that estimates the value of the KL accurately does not necessarily have a gradient that points in the direction we want. This article starts from this problem.

The article tries to answer two questions. The first question is what gradient each common form of the KL produces after the backward pass, and which direction of the KL that gradient corresponds to. The second question comes from asynchronous RL. In asynchronous RL, the generator that produces the data lags behind the trainer by some steps. In [the article that adds a KL to Score Function Centering](https://normaluhr.github.io/en/p/score-function-centering-kl/), our approach was to add a KL that pulls the trainer toward the generator. This KL can be written in several forms, and we want to understand what each form does and what it costs. The answer to the first question becomes the tool that we use to answer the second. Finally, the KL and its coefficients are computed in finite precision, so we also discuss what needs care in the computation.

This article does not give a single answer to the second question. Different forms protect against failures in different directions, and which failure is the bigger concern may differ from one setting to another. We explain what each form does, which kind of failure it protects against and what it costs. The choice is left to the reader, who can judge it in their own setting.

## Before you read

To read this article, you need to know what the loss of PPO or GRPO looks like. The recap in the next section collects the three KL estimators, which are used from Section 1 on. The background on asynchronous RL and Score Function Centering is not needed until Section 4, and we recap it there.

This article may help if you have been in one of the following situations.

* You used $k_3$ in the GRPO loss, but you are not sure which direction of the KL it actually optimizes.
* You want to understand the difference between putting the KL in the reward and putting it in the loss.
* You work on asynchronous RL, you use a KL between the trainer and the generator, and you want to know where the different forms pull the model.
* When you logged or used the KL, you got numbers that made no sense, for example a negative value for a KL that should be non-negative.

Section 1 first explains why calling backward directly on an estimate of the KL does not, in general, give the gradient of the KL. It then uses an example with a vocabulary of only two tokens to compute how large a difference the missing term makes. To avoid such term-by-term computation every time, Section 2 writes the gradient of every KL form as a coefficient times the score. From the coefficient, we can read which direction of the KL $k_1$, $k_2$ and $k_3$ correspond to when they are used as a loss. The same coefficient also has a different effect in the reward than in the loss, and Section 3 discusses this difference. Sections 1 to 3 all assume that the samples come from the current model. Section 4 removes this assumption. It first recaps the background of asynchronous RL and Score Function Centering, and then discusses two cases separately, GRPO with several updates on one batch and asynchronous RL. In asynchronous RL, there is more than one form that brings the trainer and the generator closer. Section 5 writes all of them as modifications of the advantage. Section 6 then compares how they constrain tokens whose probability is falling and tokens whose probability is rising, where they finally pull the distribution, and what each of them costs. Finally, Section 7 discusses how to compute the KL and its coefficients in finite precision, and Section 8 concludes.

All figures in this article are interactive. Move the mouse over a figure to read the values, and click a legend entry to hide a curve. In a figure with a slider, drag the slider to change a parameter.

## Recap

So that this article can be read on its own, we first recap the three estimators that are used later.

Suppose we want to estimate $\mathrm{KL}(q\|p)=\mathbb E_{x\sim q}[\log q(x)-\log p(x)]$, but we only have samples drawn from $q$. Here $q$ is the distribution we sample from, and $p$ is the other distribution. Write $r=p(x)/q(x)$. In his blog post ["Approximating KL Divergence"](http://joschu.net/blog/kl-approx.html), John Schulman gives three estimators:

$$
k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=(r-1)-\log r.
$$

$k_1$ is unbiased, but its variance is large. $k_2$ has a small variance, but at the price of a bias. $k_3$ adds to $k_1$ the control variate $r-1$, whose expectation is 0, so it stays unbiased and also has a small variance. These three estimators can be used for any two distributions. Whenever we use them later, we say which distribution is in the place of $q$ and which distribution is in the place of $p$.

## 1. Why does differentiating the KL estimate directly not give the gradient of the KL?

> **TL;DR:** The KL is an expectation, and $\theta$ appears both in the estimate that is averaged and in the distribution that is sampled. Calling backward directly on the KL estimate only differentiates the estimate itself. The part of the gradient that comes from the change of the sampling distribution with $\theta$ is lost, and more samples cannot recover it. So an estimator that is unbiased for the value of the KL is not guaranteed to give the correct gradient when it is used as a loss.

This section first shows which term of the gradient is lost, and then uses an example with a vocabulary of only two tokens to compute how large a difference the missing term makes. Before that, we fix the notation used in Sections 1 to 3.

To keep things simple, we fix a prefix and look only at the distribution of the next token. Write $z$ for the logits that the model outputs on this prefix. It is a vector whose length is the vocabulary size. The model being trained is $\pi_\theta$, and the probability that it gives to token $y$ is $\pi_\theta(y)=\mathrm{softmax}(z)_y$. The reference distribution in the KL regularization is the reference model, written $\pi_{\text{ref}}$. $\pi_{\text{ref}}$ is usually the model before RL starts, for example the model after SFT. It stays fixed throughout training, so it differs from $\pi_\theta$ except at the very beginning of training. For a token $y$, write its log ratio as

$$
\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y).
$$

Write the score as $s_\theta(y)=\nabla_\theta\log\pi_\theta(y)$, which is the gradient of the log probability of token $y$ with respect to the model parameters. The examples below differentiate only with respect to the logits $z$, and then the score has a simple form. Write $e_y$ for the one-hot vector of token $y$, that is, the vector that equals 1 at the position of $y$ and 0 at every other position. We also arrange the probabilities of $\pi_\theta$ over the whole vocabulary into a vector, and still write it as $\pi_\theta$. Since $\log\pi_\theta(y)=z_y-\log\sum_v\exp(z_v)$, its partial derivative with respect to $z_v$ is $1-\pi_\theta(y)$ when $v=y$, and $-\pi_\theta(v)$ when $v\neq y$. Arranged into a vector, these partial derivatives are

$$
\frac{\partial\log\pi_\theta(y)}{\partial z}=e_y-\pi_\theta.
$$

The KL divergence has two directions:

$$
\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=\mathbb E_{y\sim\pi_\theta}[\rho(y)],\qquad
\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{ref}}}[-\rho(y)].
$$

In RLHF and GRPO, the one that we usually want to constrain is the first. This article follows the naming of [our earlier article that adds a KL to Score Function Centering](https://normaluhr.github.io/en/p/score-function-centering-kl/). A KL with $\pi_\theta$ in the second place, such as $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$, is called the reverse KL. A KL with $\pi_\theta$ in the first place, such as $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, is called the forward KL. Some RLHF papers and codebases use the opposite names. The appendix at the end of the article lists how the two naming conventions correspond.

In the three formulas of the recap, $q$ is the distribution we sample from and $p$ is the other distribution. Here we want to estimate $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, and the samples come from $\pi_\theta$. So $\pi_\theta$ is in the place of $q$, and $\pi_{\text{ref}}$ is in the place of $p$. Then $r=\pi_{\text{ref}}(y)/\pi_\theta(y)$ and $\log r=-\rho(y)$. After this substitution, the three estimators, written as functions of $\rho$, are

$$
k_1=\rho,\qquad k_2=\tfrac12\rho^2,\qquad k_3=e^{-\rho}-1+\rho.
$$

Sections 1 to 3 all use this notation. They use only $\rho$ and do not use $r$ on its own.

### 1.1 Which term does backward miss?

To discuss the three estimators at once, we write $f_\theta(y)$ for the value of any one of them on the sample $y$. For example, with $k_1$, $f_\theta(y)=\rho(y)$. The subscript $\theta$ shows that it depends on the model parameters. For unbiased estimators such as $k_1$ and $k_3$, the KL equals $\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]$, so the gradient of the KL that we want is the gradient of this expectation. Written as a sum, the expectation is

$$
\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]=\sum_y\pi_\theta(y)\,f_\theta(y).
$$

In this expression, $\theta$ appears in two places. One is the estimator $f_\theta(y)$ itself. The other is the weight $\pi_\theta(y)$, which is the probability that token $y$ is sampled. Differentiating the product gives two terms. We call the term that differentiates $f_\theta(y)$ the estimator term. We call the term that differentiates $\pi_\theta(y)$ the sampling term, after we rewrite it with $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$. This gives

$$
\nabla_\theta\,\mathbb E_{y\sim\pi_\theta}[f_\theta(y)]=\underbrace{\mathbb E_{y\sim\pi_\theta}\big[\nabla_\theta f_\theta(y)\big]}_{\text{estimator term}}+\underbrace{\mathbb E_{y\sim\pi_\theta}\big[f_\theta(y)\,s_\theta(y)\big]}_{\text{sampling term}}.
$$

A common implementation computes the KL term as follows. It first samples $N$ tokens $y_1,\dots,y_N$ from $\pi_\theta$. It then computes $f_\theta(y_i)$ for each token, averages these values into $\frac1N\sum_i f_\theta(y_i)$, multiplies the average by $\beta$ and adds it to the loss. Finally it calls backward. The gradient from backward is $\frac1N\sum_i\nabla_\theta f_\theta(y_i)$, and its expectation is exactly the estimator term above.

The sampling term comes from differentiating the weight $\pi_\theta(y)$. In the code, this weight does not appear in the loss as a number. The sampling applies this weight instead. A token with a larger probability is sampled more often, so it takes a larger share of the average. The sampling step, however, outputs the index of a token. When $\theta$ changes slightly, the sampled index either stays the same or suddenly jumps to another token. So the derivative of a single sampling result with respect to $\theta$ is either 0 or does not exist, and automatic differentiation does not go through this step. The effect of $\theta$ on how often each token is sampled shows up only in the average over many samples, and no single backward pass contains this effect. Therefore backward differentiates only the expression $f_\theta(y_i)$ and treats the sampled $y_i$ as a constant, and this is how the sampling term is lost. Note that $f_\theta(y)$ may also contain $\pi_\theta(y)$. For example, $k_1$ is $\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$. Backward does differentiate this occurrence, and it belongs to the estimator term. Only the $\pi_\theta(y)$ that acts as the weight is lost.

The sampling term has the same form as a policy gradient. If we view $f_\theta(y)$ as the reward $R(y)$ that token $y$ receives, the sampling term is $\mathbb E_{y\sim\pi_\theta}[R(y)\nabla_\theta\log\pi_\theta(y)]$, which is the REINFORCE gradient. The problem that the policy gradient solves is how to take the gradient of an expectation when the sampling distribution depends on $\theta$. Readers who are not familiar with this derivation can see Section 1.1 of [the article that introduces Score Function Centering](https://normaluhr.github.io/en/p/score-function-centering/). So differentiating the KL estimate directly computes only the gradient of the estimator itself, and drops the whole gradient that comes from the change of the sampling distribution.

Because the lost part is a whole expectation term, more samples only make the estimate of the estimator term more accurate. They cannot bring back the sampling term, so this bias does not depend on the number of samples. Tang and Munos (2025) and Zhang et al. (2025) both discuss this problem in detail, and they point out that many open-source implementations are affected by it.

### 1.2 How large is the difference when this term is missing?

Below, we compute three gradients exactly in an example with a vocabulary of only two tokens, to see how large a difference the missing sampling term makes.

Suppose that the vocabulary has only two tokens, A and B. The current model has $\pi_\theta(A)=0.7$, and the reference distribution has $\pi_{\text{ref}}(A)=0.5$. To keep the formulas short, write $p_A=\pi_\theta(A)$ and $p_B=\pi_\theta(B)$. We compare three quantities. All of them are derivatives with respect to $z_A$, the logit of A, and all of them take the exact expectation over the sampling.

The first quantity is the derivative of the forward KL itself with respect to $z_A$, which is the result we want. With only two tokens, $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=p_A\rho(A)+p_B\rho(B)$. From $e_y-\pi_\theta$ above, the derivative of $\log p_A$ with respect to $z_A$ is $1-p_A=p_B$, and the derivative of $\log p_B$ with respect to $z_A$ is $-p_A$. So the derivatives of $\rho(A)$ and $\rho(B)$ with respect to $z_A$ are $p_B$ and $-p_A$, respectively. The derivative of $p_A$ or $p_B$ itself equals the probability times the derivative of its log probability, which gives $p_Ap_B$ and $-p_Ap_B$. When we differentiate the two products, the two terms that differentiate $\rho$ are $p_A\cdot p_B+p_B\cdot(-p_A)=0$, and they cancel exactly. These two terms are the estimator term of Section 1.1. What remains are the two terms that differentiate $p_A$ and $p_B$, and they form the sampling term:

$$
\frac{\partial}{\partial z_A}\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})=p_Ap_B\big(\rho(A)-\rho(B)\big)=0.7\times0.3\times\ln\frac{0.7/0.5}{0.3/0.5}=0.21\times\ln\frac73\approx0.178.
$$

The second quantity is the expectation of the gradient that backward gives when $k_1$ is used as the loss. Since $k_1=\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$, its derivative with respect to $z_A$ is just $\partial\log\pi_\theta(y)/\partial z_A$. From $e_y-\pi_\theta$, this derivative is $1-0.7=0.3$ when A is sampled, and $-0.7$ when B is sampled. The expectation under $\pi_\theta$ is $0.7\times0.3+0.3\times(-0.7)=0$.

The third quantity is the expectation of the gradient that backward gives when $k_3$ is used as the loss. Since $k_3=e^{-\rho}-1+\rho$, its derivative with respect to $z_A$ is $(1-e^{-\rho})\,\partial\log\pi_\theta(y)/\partial z_A$, where $e^{-\rho}=\pi_{\text{ref}}(y)/\pi_\theta(y)$. When A is sampled, the derivative is $(1-0.5/0.7)\times0.3\approx0.086$. When B is sampled, the derivative is $(1-0.5/0.3)\times(-0.7)\approx0.467$. The expectation under $\pi_\theta$ is $0.7\times0.086+0.3\times0.467=0.2$.

As estimates of the value of the KL, $k_1$ and $k_3$ are both unbiased. But when they are used directly as the loss, their expected gradients are 0 and 0.2, and neither of them equals the 0.178 of the forward KL. All three numbers are exact expectations, so the difference exists no matter how large the batch is. $k_1$ is the most extreme case. Its gradient averages to 0, which means that this KL term does not pull the model toward the reference distribution at all.

This difference is not caused by the particular value 0.7. Figure 1 lets $\pi_\theta(A)$ take every value between 0 and 1 and plots the same quantities. The vertical line in the figure marks the example above. Drag the slider to change $\pi_{\text{ref}}(A)$. The figure also shows the result of $k_2$ as the loss and the gradient of the reverse KL. Section 2.2 explains why the curve of $k_2$ coincides with the curve of the forward KL, and why the curve of the reverse KL coincides with the curve of $k_3$.

<figure data-chart="kl-two-token">
<noscript><p>Figure 1 needs JavaScript to display.</p></noscript>
</figure>
<script src="/js/kl-gradient-charts.js" defer></script>

The 0.2 from $k_3$ equals exactly $\pi_\theta(A)-\pi_{\text{ref}}(A)$. The next section gives a single method that shows directly which gradient each form corresponds to, and Section 2.2 uses it to explain where the 0.2 comes from.

## 2. What gradient do $k_1$, $k_2$ and $k_3$ each give when used as a loss?

> **TL;DR:** Once the gradient of each KL form on a single sample is written as a coefficient times the score, comparing the coefficients tells us which KL the form optimizes. When the samples come from the current model, $k_1$ as a loss has an expected gradient of 0. $k_2$ gives the gradient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, which this article calls the forward KL. $k_3$ gives the gradient of $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$, which is the reverse KL. So InstructGPT-style PPO, which puts $k_1$ in the reward, optimizes the forward KL. GRPO puts $k_3$ in the loss, and under the approximation that the samples come from the current model, it optimizes the reverse KL.

The three numbers in Section 1 were computed one at a time. This section gives a single method that writes the gradient of every KL form on a single sample as a coefficient times the score. Section 2.1 introduces the method. With it, Section 2.2 reads off the gradient that each of $k_1$, $k_2$ and $k_3$ gives as a loss, and Section 2.3 compares the KL in PPO with the KL in GRPO.

### 2.1 How can we read the gradient of an estimator?

$k_1$, $k_2$ and $k_3$ depend only on $\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$, so all of them can be written as $f(\rho)$. $\pi_{\text{ref}}$ does not change with $\theta$, so $\nabla_\theta\rho(y)=s_\theta(y)$. For any form $f(\rho)$, the gradient that backward gives on a single sample is

$$
\nabla_\theta f(\rho(y))=f'(\rho(y))\,s_\theta(y).
$$

This gradient is a scalar coefficient $f'(\rho)$ times the score. In gradient descent on the loss, a positive coefficient lowers the log probability of this sample, and a negative coefficient raises it. The absolute value of the coefficient sets the size of the change. Comparing different forms means comparing their coefficients.

When the samples come from the current model, a gradient with the coefficient $c(y)$ has the expectation $\mathbb E_{y\sim\pi_\theta}[c(y)\,s_\theta(y)]$. The score has expectation 0, so adding to $c$ any constant that does not depend on $y$ does not change this expectation. This is also why a baseline can be subtracted in the policy gradient. We discussed this in detail in Section 2.1 of [the article that introduces Score Function Centering](https://normaluhr.github.io/en/p/score-function-centering/), and we use it repeatedly later.

Next, consider what the gradient of the forward KL should be. Take $f_\theta=\rho$ in the formula of Section 1.1. The estimator term is $\mathbb E[\nabla_\theta\rho]=\mathbb E[s_\theta]=0$, so only the sampling term $\mathbb E[\rho\,s_\theta]$ remains in the gradient of the forward KL. In other words, when the samples come from the current model, a coefficient equal to $\rho$, or to $\rho$ plus any constant, gives the exact gradient of the forward KL. The next section uses this criterion to check the three estimators.

Here "exact" refers to the expectation. Adding a constant to the coefficient, or subtracting one, does not change the expectation, but it changes the variance of the gradient. As with subtracting a baseline in the policy gradient, we can subtract from the coefficient a quantity that does not depend on the token just sampled, to lower the variance of the gradient. The $k_4$ of Section 5.1 also uses this idea. Note that the variance discussed in [the earlier note](https://normaluhr.github.io/en/p/kl/) is a different quantity. It is how much the KL estimate varies from sample to sample, which is not the variance of the gradient discussed here.

### 2.2 What are the coefficients of $k_1$, $k_2$ and $k_3$?

Start with $k_1$. Since $k_1=\rho$, the coefficient is $f'(\rho)=1$, and the expected gradient is $\mathbb E_{y\sim\pi_\theta}[s_\theta(y)]=0$. As an estimate of the value of the KL, $k_1$ is unbiased, but as a loss, the expectation of its gradient is always 0. On a single batch its gradient is not 0, but on average these gradients point in no direction and only add noise to training. This is why $k_1$ gives 0 in Section 1.2.

Next comes $k_2$. Since $k_2=\tfrac12\rho^2$, the coefficient is $f'(\rho)=\rho$. By the criterion of Section 2.1, it gives the exact gradient of the forward KL. In the example of Section 1.2, $\rho=\ln(0.7/0.5)\approx0.336$ when A is sampled, and $\rho=\ln(0.3/0.5)\approx-0.511$ when B is sampled. Multiplying each $\rho$ by the derivative of $\log\pi_\theta(y)$ with respect to $z_A$ and taking the expectation under $\pi_\theta$ gives exactly 0.178. This is also why the dashed line of $k_2$ in Figure 1 coincides exactly with the curve of the forward KL.

This is the opposite of how $k_2$ behaves as an estimate of the value. As an estimate of the value of the KL, $k_2$ is biased. Schulman's blog post shows that when the two distributions are close, its expectation agrees with the KL only up to second order. As a loss, however, $k_2$ gives an exact gradient when the samples come from the current model. The reason is that the estimator term has expectation 0, so only the sampling term $\mathbb E[\rho\,s_\theta]$ remains in the full gradient of the forward KL, and the backward pass of $k_2$ gives exactly $\rho\,s_\theta$ on a single sample. In other words, the backward pass of $k_2$ happens to compute the term that $k_1$ misses.

Finally, consider $k_3$. Since $k_3=e^{-\rho}-1+\rho$, the coefficient is

$$
f'(\rho)=1-e^{-\rho}=1-\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}.
$$

Its expected gradient is

$$
\mathbb E_{y\sim\pi_\theta}\Big[\Big(1-\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}\Big)s_\theta(y)\Big]=\mathbb E_{y\sim\pi_\theta}[s_\theta(y)]-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)]=-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)].
$$

The gradient of $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{ref}}}[\log\pi_{\text{ref}}(y)-\log\pi_\theta(y)]$ is also $-\mathbb E_{y\sim\pi_{\text{ref}}}[s_\theta(y)]$. So when $k_3$ is used as a loss, its expected gradient is the gradient of the reverse KL. In the example of Section 1.2, the gradient of the reverse KL with respect to the logits is $\pi_\theta-\pi_{\text{ref}}$, and its component on A is $\pi_\theta(A)-\pi_{\text{ref}}(A)=0.2$. This is the number that $k_3$ gives. In Figure 1, the dashed line of $k_3$ coincides exactly with the curve of the reverse KL.

When $\rho$ is close to 0, $1-e^{-\rho}\approx\rho-\rho^2/2$, which agrees with the coefficient of $k_2$ to first order. For this reason, Liu et al. (2025) describe the $k_3$ loss as a first-order approximation of $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$. This does not contradict the statement that $k_3$ gives the gradient of the reverse KL. When the model is not far from the reference distribution, the gradients of the two directions of the KL are close anyway, and the farther apart the two distributions are, the larger the difference between the two gradients. Figure 1 shows this too. The two curves almost coincide near $\pi_\theta(A)=\pi_{\text{ref}}(A)$, and they separate more as $\pi_\theta(A)$ moves away from this point.

The coefficient of $k_3$ is not symmetric on the two sides of $\rho=0$. Figure 2 plots the coefficients of the three estimators on a single sample. On the side $\rho>0$, where the model gives a token a higher probability than the reference distribution does, the coefficient of $k_3$ is at most 1. On the side $\rho<0$, where the model gives a token a lower probability than the reference distribution does, the absolute value of the coefficient grows exponentially, like $e^{-\rho}$. For example, at $\rho=2$ the coefficient of $k_2$ is 2 and the coefficient of $k_3$ is 0.865. At $\rho=-2$ the coefficient of $k_2$ is $-2$ and the coefficient of $k_3$ is $1-e^2\approx-6.39$.

<figure data-chart="kl-coef">
<noscript><p>Figure 2 needs JavaScript to display.</p></noscript>
</figure>

This is the coefficient on a single sample. A token $v$ with $\rho$ far below 0 is rarely sampled under $\pi_\theta$. After we multiply by the probability that it is sampled, the expected gradient on its logit is $\pi_\theta(v)-\pi_{\text{ref}}(v)$. Gradient descent pulls the token back, but the size of this pull is never more than $\pi_{\text{ref}}(v)$. So the exponentially growing coefficient does not bring a large pull in expectation. What it mainly brings is variance. When such a token happens to be sampled, it produces a very large gradient. Section 4.1 shows that this is the criticism of $k_3$ in [the DeepSeek-V3.2 technical report](https://arxiv.org/abs/2512.02556). Section 6 analyzes how large the pull is on each side, in the setting with a generator.

### 2.3 How does the KL in PPO differ from the KL in GRPO?

Besides serving as a loss, the KL can also be put in the reward. InstructGPT-style PPO (Ouyang et al., 2022) does this. It multiplies the $\rho$ of each token by $\beta$ and subtracts the product from the reward of this token. Here $\rho$ is treated as part of the reward and is not differentiated. It affects the gradient through the policy gradient, as a coefficient that multiplies the score. The coefficient of the KL part is $\beta\rho$, so in expectation it is the same as $k_2$ used as a loss, and it also gives the gradient of the forward KL. This result holds only at a single position. Section 3 discusses how the two differ over a whole sequence.

With these results, we can look at the KL in PPO and GRPO again. In [the earlier note](https://normaluhr.github.io/en/p/kl/) we wrote that "$k_3$ is exactly the piece where, in practice, GRPO differs from PPO in how KL is estimated (PPO uses $k_1$)". We can now state this more completely. InstructGPT-style PPO puts $k_1$ in the reward, and its gradient corresponds to the forward KL. GRPO (Shao et al., 2024) puts $k_3$ in the loss, and under the approximation that the samples come from the current model, its gradient corresponds to the reverse KL. Besides using different estimators, the two methods also optimize different directions of the KL. As Section 2.2 said, when the model is not far from the reference model, the gradients of the two directions are close, so this difference may not matter much in practice. [The DeepSeek-V3.2 technical report](https://arxiv.org/abs/2512.02556) later corrected this $k_3$ term of GRPO, and Section 4.1 describes the correction.

So far, we have discussed only the gradient at a single position. $k_1$ in the reward and $k_2$ in the loss have the same expectation at a single position, but in real training the two placements still differ. The next section discusses this difference.

## 3. What is the difference between putting the KL in the reward and putting it in the loss?

> **TL;DR:** When the KL is in the reward, it goes through the normalization of the advantage and the PPO clip together with the task reward, so the same $\beta$ has a different actual effect on different prompts. The KL is also passed to earlier tokens through the advantage. This gives the exact gradient of the sequence-level KL, but with a larger variance. When the KL is in the loss, $\beta$ is the actual weight of this term. Each token constrains only the distribution at its own position. The variance is smaller, but the result is not the gradient of the sequence-level KL.

Putting the KL in the reward means subtracting $\beta\rho_t$, usually its detached value, from the reward of the $t$-th token, and then computing the advantage together with the task reward. The classic implementation of PPO does this. Putting the KL in the loss means computing the advantage from the task reward only, and adding the KL to the loss as a separate term that is differentiated, for example $L=L_{\text{PG}}+\beta\cdot\mathrm{mean}_t\,k(\rho_t)$. GRPO uses this second approach.

Section 2.3 showed that at a single position, $k_1$ in the reward and $k_2$ in the loss have the same expected gradient. But the two placements still differ in two ways in real training. The first difference is that, in the reward, the KL goes through all the processing of the advantage together with the task reward. The second difference is that, in the reward, the KL is passed along the sequence to earlier tokens. The next two sections discuss them in turn. Both differences matter again in the second half of the article. Section 5 shows that under Score Function Centering, every KL that brings the trainer and the generator closer can be written as a modification of the advantage, and the same two issues need attention there.

### 3.1 How is the KL rescaled once it is in the reward?

Once the KL is in the reward, the advantage carries two kinds of information at the same time, namely how well the task was done and how close the response is to the reference distribution. A response may get a higher advantage only because it is closer to the reference distribution. Besides this, every processing step that comes after the advantage also acts on the KL. For example, suppose that the KL is added to the reward, and that the advantage is then normalized by the standard deviation of the group, which is the normalization that GRPO uses. Then the KL on each prompt is divided by the standard deviation within the group of that prompt. A prompt whose rewards differ little within the group has a small standard deviation, so its KL is enlarged. A prompt whose rewards differ a lot within the group has its KL reduced. So the same $\beta$ has a different actual effect on different prompts. The PPO clip also acts on the KL. If the gradient of a token is clipped away, the gradient of its KL part disappears with it.

When the KL is in the loss, none of this processing affects the KL, and $\beta$ is the actual weight of the KL term.

### 3.2 Why is the KL passed to earlier tokens once it is in the reward?

First, recall how the advantage is computed from the reward of each token. Write $R_t$ for the reward on the $t$-th token, and $V(s_t)$ for the value that the critic estimates for the state at step $t$. GAE (Generalized Advantage Estimation) first computes the TD residual of each step, $\delta_t=R_t+\gamma V(s_{t+1})-V(s_t)$. It then sums the residuals of this step and of all later steps, weighted by powers of $\gamma\lambda$:

$$
A_t=\sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}.
$$

Here $\gamma$ is the discount factor, and $\lambda$ controls how far ahead the advantage looks. With $\lambda=0$ it looks only one step ahead, and the closer $\lambda$ is to 1, the farther it looks. We introduced GAE in detail in Section 8.2 of [the article that introduces GRPO](https://normaluhr.github.io/en/p/grpo/). With $\gamma=\lambda=1$, the $V$ terms of neighboring steps cancel in pairs, and $A_t=\sum_{k\ge t}R_k-V(s_t)$. This is the sum of all rewards from the $t$-th token to the end of the sequence, minus a baseline.

When the KL is in the reward, the reward of the $k$-th token gets an extra term $-\beta\rho_k$. By the formula above, the advantage of the $t$-th token then gets an extra $-\beta\sum_{k\ge t}\rho_k$, which contains the KL of every token from the $t$-th token to the end of the sequence. In the terms of Section 2.1, the coefficient of the KL part on the $t$-th token is

$$
\beta\sum_{k\ge t}\rho_k.
$$

This coefficient corresponds to the gradient of the sequence-level forward KL. The reason is the same as for the sampling term in Section 1.1. The choice of the $t$-th token changes the prefixes of all later tokens, and the probabilities of these prefixes also depend on $\theta$. So an early token is responsible for all the KL that is produced after it.

When the KL is in the loss, the coefficient of each token depends only on its own $\rho_t$. For example, the coefficient of $k_2$ is $\rho_t$, and the coefficient of $k_3$ is $1-e^{-\rho_t}$. This constrains the distribution of the next token separately on each fixed prefix, but it ignores the effect of the current token on later prefixes. Tang and Munos (2025) call the gradient obtained in this way a partial gradient.

So $k_1$ in the reward and a per-token $k_2$ in the loss are equivalent only at a single position. Over a whole sequence, the coefficient of $k_1$ in the reward is the sum of $\rho$ over the current token and all later tokens. It gives the exact gradient of the sequence-level KL, but the longer the sequence, the more this sum fluctuates, and the larger the variance of the gradient. The per-token $k_2$ in the loss uses only the $\rho_t$ of each token. Its variance is small, but it is not the gradient of the sequence-level KL.

To get the gradient of the sequence-level KL in the loss, we can apply $k_2$ to the sum of the log ratios over the whole sequence, that is, $\tfrac12\big(\sum_t\rho_t\big)^2$. Its gradient is $\big(\sum_k\rho_k\big)\sum_t s_\theta(y_t)$, so the score of the $t$-th token is multiplied by the $\rho_k$ of all positions. The $\rho_k$ with $k<t$ are already fixed before the $t$-th token is sampled, so their products with the score of this token have expectation 0. In expectation, this loss is therefore the same as $k_1$ in the reward, but its variance is even larger.

Sections 1 to 3 all assume that the samples come from the current model $\pi_\theta$. This does not hold when GRPO updates several times on one batch, and it does not hold in asynchronous RL either. The next section discusses what happens when the samples do not come from the current model.

## 4. What happens when the samples do not come from the current model?

> **TL;DR:** This section removes the assumption that the samples come from the current model. When GRPO updates several times on one batch, the samples come from $\pi_{\text{old}}$, and the gradient of $k_3$ as a loss is neither the gradient of the forward KL nor the gradient of the reverse KL. DeepSeek-V3.2 multiplies $k_3$ by the ratio $\pi_\theta/\pi_{\text{old}}$ and lets this ratio be differentiated, and only then is the result the exact gradient of the forward KL. In asynchronous RL, the samples come from a fixed generator. The sampling distribution does not depend on $\theta$, so no term is lost, and calling backward directly on $k_1$ or $k_3$ gives the exact gradient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$.

In Sections 1 to 3, the KL always compares the model with the reference model $\pi_{\text{ref}}$, and the samples always come from the current model $\pi_\theta$. From this section on, the samples no longer come from $\pi_\theta$. Before we continue, we put the four distributions of this article side by side.

* $\pi_\theta$ is the model being trained. It is updated at every step.
* $\pi_{\text{ref}}$ is the reference model, usually the model before RL starts. It stays fixed throughout training. The KL regularization of PPO and GRPO compares the model with it, to limit how far the model moves away from the initial model.
* $\pi_{\text{old}}$ is the model that generated the current batch of data in PPO and GRPO, that is, the generator of this batch.
* $\pi_{\text{generator}}$ is the model that generates the data in asynchronous RL. It has the same role as $\pi_{\text{old}}$, but it lags further behind.

How these distributions relate to each other depends on how the training is run. In fully on-policy synchronous training, each batch is used for only one update, and the $\pi_\theta$ at the update is the model that has just generated this batch. So $\pi_{\text{old}}$ and $\pi_\theta$ have the same parameters, and the only difference is that $\pi_{\text{old}}$ is not differentiated. $\pi_{\text{ref}}$ is different. It stays fixed from the start of training while $\pi_\theta$ keeps being updated, so even in synchronous training the two are the same only at the very beginning. When GRPO updates several times on one batch, from the second update on, $\pi_\theta$ has already taken a few update steps since $\pi_{\text{old}}$ generated the batch, and the two are no longer the same. In asynchronous RL, the generator produces data while the trainer is being updated, and $\pi_{\text{generator}}$ lags behind $\pi_\theta$ by a few steps or even hundreds of steps.

Score Function Centering deals with this lag in asynchronous RL. From Section 4.2 on, the rest of the article works in this setting, so we first recap how Score Function Centering works, and the KL that we added on top of it. Our earlier articles wrote the trainer and the generator as $p_\theta$ and $q_\phi$. This article writes them as $\pi_\theta$ and $\pi_{\text{generator}}$.

**Score Function Centering.** The score $\nabla_\theta\log\pi_\theta(y)$ in the policy gradient has expectation 0 only when $y$ comes from $\pi_\theta$, and this is what allows a baseline to be subtracted from the advantage. When $y$ comes from $\pi_{\text{generator}}$, this expectation is no longer 0, and the policy gradient gets an extra term. This term equals the mean advantage times the expectation of the score under $\pi_{\text{generator}}$. Its direction on the logits is $\pi_{\text{generator}}-\pi_\theta$, and its size and sign are both set by the mean advantage, which says nothing about which token is better. Score Function Centering, SFC for short below, centers the score at $\pi_{\text{generator}}$ instead, and so removes this term. If we look only at the part that differentiates the logits at the current position, the ordinary score is $e_y-\pi_\theta$, and the score after SFC is $e_y-\pi_{\text{generator}}$. Here $e_y-\pi_\theta$ is the $\partial\log\pi_\theta(y)/\partial z$ computed in Section 1.

Computing the SFC score needs the full distribution of the generator at this position. In practice, the generator sends back only its most probable tokens and their probabilities. This part is called the head, and the rest is called the tail. There are two ways to handle the tail. One is to sample a few more tokens from the tail to estimate it. The other is to record only the total probability of the tail, and our earlier experiments used this second way by default. The detailed derivation is in [our article that introduces SFC](https://normaluhr.github.io/en/p/score-function-centering/).

**Adding a KL to SFC.** SFC does not limit the distance between the trainer and the generator. The larger the distance, the more the update computed on the samples of the generator differs from the update that the current model should get. In [the article that adds a KL to SFC](https://normaluhr.github.io/en/p/score-function-centering-kl/), we derived an upper bound on this difference that is proportional to $\sqrt{\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)}$, so we added a $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ on top of SFC. Its gradient with respect to the logits is $\pi_\theta-\pi_{\text{generator}}$, and it can be computed with a sparse estimate. For a token $y$ sampled from the generator, this estimate takes the value $\pi_\theta(y)/\pi_{\text{generator}}(y)-1$ on the logit of $y$ itself, and 0 on every other logit. Its expectation is exactly $\pi_\theta-\pi_{\text{generator}}$.

In addition, the KL weight of each token decays with the age of the token. The age of a token is the number of steps by which the generator version that produced it lags behind the current trainer. In our earlier experiments we called it staleness. Concretely, the KL weight $\beta_t$ equals $\beta_0$ divided by the square root of the age of the token, and a token with age 0 gets the weight 0.

This section discusses two cases in which the samples do not come from the current model. The first case is GRPO with several updates on one batch. The samples come from $\pi_{\text{old}}$, and the KL still compares the model with $\pi_{\text{ref}}$. The second case is asynchronous RL. The samples come from $\pi_{\text{generator}}$, and the KL that we added on top of SFC compares the model with $\pi_{\text{generator}}$ itself. This KL limits how much the model is updated within each generator version. In fully on-policy synchronous training, the generator is $\pi_\theta$ itself, so apart from numerical error this KL is always 0. Section 7.2 discusses the problems that numerical error causes.

The analysis of the two cases uses two earlier results. One is the split into the estimator term and the sampling term in Section 1.1. The other is reading a gradient through its coefficient, as in Section 2.1. The second case is the basis of the later sections.

### 4.1 How should $k_3$ be corrected when GRPO updates several times on one batch?

When one batch is used for several updates, from the second update on, the samples come from $\pi_{\text{old}}$ instead of the current $\pi_\theta$. Every expectation above must now be taken under $\pi_{\text{old}}$, and the earlier results no longer hold exactly. Take the $k_3$ loss as an example. Its expected gradient becomes $\mathbb E_{y\sim\pi_{\text{old}}}[(1-\pi_{\text{ref}}/\pi_\theta)\,s_\theta]$, which is neither the gradient of the forward KL nor the gradient of the reverse KL.

To see how this term should be corrected, we first need an identity about $k_3$. For every sample,

$$
\nabla_\theta k_3+k_3\,s_\theta=(1-e^{-\rho})s_\theta+(e^{-\rho}-1+\rho)s_\theta=\rho\,s_\theta.
$$

The first term on the left is the gradient that backward gives when $k_3$ is used as a loss, and it corresponds to the estimator term of Section 1.1. The second term is the part that the policy gradient gives when $k_3$ is used as a reward, and it corresponds to the sampling term that was lost in Section 1.1. Their sum is exactly $\rho\,s_\theta$, the single-sample form of the gradient of the forward KL. The identity shows that calling backward directly on $k_3$ keeps only the estimator term, and that adding the sampling term back gives the gradient of the forward KL.

[The DeepSeek-V3.2 technical report](https://arxiv.org/abs/2512.02556) corrects this term in its Equation (7). It multiplies $k_3$ by the ratio of the current model to the old model:

$$
\widehat{\mathrm{KL}}=w\cdot k_3,\qquad w=\frac{\pi_\theta(y)}{\pi_{\text{old}}(y)},\qquad y\sim\pi_{\text{old}}.
$$

If $w$ is differentiated, then

$$
\nabla_\theta(w\,k_3)=w\,s_\theta\,k_3+w\,\nabla_\theta k_3=w\,(k_3\,s_\theta+\nabla_\theta k_3)=w\,\rho\,s_\theta.
$$

The last step uses the identity above. Taking the expectation under $\pi_{\text{old}}$ gives $\mathbb E_{y\sim\pi_\theta}[\rho\,s_\theta]$, which is the exact gradient of the forward KL.

Seen through its coefficient, this correction does two things at once when $w$ is differentiated. The first is to correct the sampling distribution, by turning an expectation under $\pi_{\text{old}}$ into an expectation under $\pi_\theta$. This is what the ratio is normally for. The second is less obvious. The derivative of $w$ with respect to $\theta$ is $w\,s_\theta$, and its effect is to add the sampling term back. So even at the first update, where $\pi_{\text{old}}$ and $\pi_\theta$ are identical and the value of $w$ is 1, the gradient of $k_3$ changes from the reverse KL to the forward KL as long as $w$ is differentiated. Conversely, if $w$ is detached, the result is still the gradient of the reverse KL.

The DeepSeek-V3.2 report also explains what was wrong with the original $k_3$. If a sampled token has a much lower probability under the current model than under the reference model, the gradient of $k_3$ gives it a large weight with no upper bound, which makes the gradient noisier. This corresponds to the side $\rho<0$ in Section 2.2, where the absolute value of the coefficient of $k_3$ grows exponentially.

### 4.2 When the samples come from a fixed generator, why is differentiating $k_1$ directly correct after all?

In [the asynchronous RL setting that we discussed earlier](https://normaluhr.github.io/en/p/score-function-centering-kl/), the samples come from $\pi_{\text{generator}}$, and the KL also compares the model with $\pi_{\text{generator}}$. What we want to constrain is

$$
\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)=\mathbb E_{y\sim\pi_{\text{generator}}}\big[\log\pi_{\text{generator}}(y)-\log\pi_\theta(y)\big].
$$

Unlike before, the sampling distribution $\pi_{\text{generator}}$ does not depend on $\theta$. The sampling term of Section 1.1 comes from the dependence of the sampling distribution on $\theta$. This dependence is now gone, so nothing can be lost.

From this section on, the KL compares the model with $\pi_{\text{generator}}$, so we redefine the log ratio accordingly as $\rho(y)=\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$, and we write $r=e^\rho=\pi_\theta(y)/\pi_{\text{generator}}(y)$. When we apply the three estimators of the recap, the samples come from $\pi_{\text{generator}}$. So $\pi_{\text{generator}}$ is in the place of $q$ and $\pi_\theta$ is in the place of $p$, and this $r$ is exactly the $r$ of the recap.

Now call backward directly on $k_1$, that is, on $-\log r=\log\pi_{\text{generator}}(y)-\log\pi_\theta(y)$. The gradient on a single sample is $-s_\theta(y)$, and its expectation

$$
-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta(y)]=\nabla_\theta\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)
$$

is the exact gradient. In logit space this gradient is $\pi_\theta-\pi_{\text{generator}}$, and the update direction of gradient descent is $\pi_{\text{generator}}-\pi_\theta$. Minimizing this KL with gradient descent means maximizing $\log\pi_\theta$ on the tokens that the generator sampled, which is SFT on the samples of the generator.

$k_3$ also works. Since $k_3=(r-1)-\log r$, the gradient on a single sample is $(r-1)\,s_\theta$. Its expectation

$$
\mathbb E_{y\sim\pi_{\text{generator}}}[(r-1)\,s_\theta]=\mathbb E_{y\sim\pi_\theta}[s_\theta]-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta]=-\mathbb E_{y\sim\pi_{\text{generator}}}[s_\theta]
$$

is the same as the expectation for $k_1$. Compared with $k_1$, $k_3$ has the extra term $r\,s_\theta$. Under $\pi_{\text{generator}}$, this term has the expectation $\mathbb E_{y\sim\pi_\theta}[s_\theta]=0$, so it is a control variate. When $\pi_\theta$ and $\pi_{\text{generator}}$ are close, it can lower the variance of the gradient. When the value of the KL is estimated, $k_3$ uses a control variate to lower the variance of the value. Here the same idea lowers the variance of the gradient.

Compare this section with Section 2.2. The code in the two places is almost the same. Both call backward on the difference between the log probability under $\pi_\theta$ and the log probability under another distribution, and they differ only by a sign. When the samples are drawn by the current model itself, the expected gradient is 0. When the samples come from a fixed generator, the same code gives the exact gradient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$. The only difference between the two cases is whether the sampling distribution depends on $\theta$. Also, by the naming of Section 1, $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is a reverse KL, like the $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$ whose gradient $k_3$ gives as a loss in GRPO.

This update direction also appeared when we introduced SFC. As we discussed in [the article that introduces SFC](https://normaluhr.github.io/en/p/score-function-centering/), if SFC is not used in asynchronous RL, the mean advantage multiplies the same update direction $\pi_{\text{generator}}-\pi_\theta$, and the size and sign of this term are both set by the mean advantage. SFC removes this term from the policy gradient. Adding $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ then adds the same direction back, but with a weight $\beta$ that we choose ourselves. This is also why we chose the direction $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ before.

## 5. Under Score Function Centering, how does a KL that brings the trainer and the generator closer act on the advantage?

> **TL;DR:** Write $r=\pi_\theta(y)/\pi_{\text{generator}}(y)$. Under Score Function Centering, adding a trainer-generator KL with the weight $\beta$ is equivalent to the following change. After the advantage is fully computed, subtract from the advantage of each sampled token $\beta$ times a coefficient that depends only on $r$. The coefficient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is $r-1$. $k_4$ is half the variance of $\log r$ under $\pi_{\text{generator}}$, and its coefficient is $\log r$. The coefficient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is approximately $r\log r$. The three coefficients agree to first order near $r=1$, so when the trainer and the generator are close, the forms differ little.

Section 4.2 showed that when the samples come from a fixed generator, the gradient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ can be computed exactly. However, an exact gradient only means that we optimize this KL exactly. It does not mean that this KL is the most suitable constraint. There are several other forms that pull the trainer toward the generator. They act in about the same way when the trainer and the generator are close. When the two are far apart, the forces that they put on tokens whose probability is falling and on tokens whose probability is rising are very different. To compare these forms, this section works in the setting of Score Function Centering (SFC) and writes each form as subtracting $\beta$ times a coefficient from the advantage. Section 5.1 derives this expression, and Section 5.2 uses an example with a vocabulary of only three tokens to see how much the forms differ when the trainer and the generator are close.

### 5.1 How can a KL term be written as a modification of the advantage?

Under SFC, the score is the version that is centered at $\pi_{\text{generator}}$, and its part on the logits at the current position is $e_y-\pi_{\text{generator}}$. From this section on, "coefficient" means the quantity that multiplies this score. Suppose that the samples come from $\pi_{\text{generator}}$ and that the coefficient of each sample is $f(y)$. Arrange the values of $f$ over the whole vocabulary into a vector $f$. Then the expectation is

$$
\mathbb E_{y\sim\pi_{\text{generator}}}\big[f(y)\,(e_y-\pi_{\text{generator}})\big]=\mathrm{diag}(\pi_{\text{generator}})\,f-\pi_{\text{generator}}\,(\pi_{\text{generator}}^\top f)=F_{\text{generator}}\,f,\qquad F_{\text{generator}}=\mathrm{diag}(\pi_{\text{generator}})-\pi_{\text{generator}}\pi_{\text{generator}}^\top.
$$

Write $\mathbf 1$ for the vector whose components all equal 1. Since $F_{\text{generator}}\mathbf 1=\pi_{\text{generator}}-\pi_{\text{generator}}(\pi_{\text{generator}}^\top\mathbf 1)=0$, adding the same constant to every component of $f$ does not change the result. This is the baseline invariance under SFC.

The policy gradient term can also be written in this way. Write $A_v$ for the expected advantage after choosing token $v$ on the current prefix, and arrange these values into a vector $A$. The coefficient of the policy gradient term is the advantage of the sampled token, and the expected update direction that it gives is $F_{\text{generator}}A$. Suppose that the expected gradient of some KL used as a loss can also be written as $F_{\text{generator}}f$. Then, after this KL is added with the weight $\beta$, the expected update direction of gradient descent is $F_{\text{generator}}(A-\beta f)$. In other words, adding this KL is equivalent to changing the advantage from $A$ to $A-\beta f$.

When this is implemented as a modification of the advantage, the two issues of Section 3 need attention. The equivalence above holds at each position separately, with $\beta$ as the actual weight of this term. So the modification should be applied after the advantage is fully computed, that is, after the normalization within the group. It should act only on the current token, and it should not be summed onto earlier tokens along the sequence.

Below we look at the $f$ of the three forms in turn. The first form is $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$. Section 4.2 showed that its gradient in logit space is $\pi_\theta-\pi_{\text{generator}}$. Also, $\pi_{\text{generator}}\odot r=\pi_\theta$, where $\odot$ means component-wise multiplication. So

$$
F_{\text{generator}}(r-1)=\pi_{\text{generator}}\odot(r-1)-\pi_{\text{generator}}\Big(\sum_v\pi_\theta(v)-\sum_v\pi_{\text{generator}}(v)\Big)=\pi_\theta-\pi_{\text{generator}}.
$$

So under SFC, adding $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ with the weight $\beta$ is equivalent in expectation to changing the advantage of each sample from $A$ to $A-\beta(r-1)$. The sparse estimate that we recapped at the start of Section 4 also uses the number $r-1$, but it multiplies $e_y$, so it updates only the logit of the sampled token. The two approaches give different gradients on a single sample, but both have the expectation $\pi_\theta-\pi_{\text{generator}}$.

The second form starts from the wish that the coefficient should be $\log r$ itself. Compared with $r-1$, $\log r$ grows only logarithmically, so it does not become very large even when $r$ is large. One can check that the coefficient of the following objective under SFC is exactly $\log r$. The objective is half the variance of $\log r$ under $\pi_{\text{generator}}$:

$$
k_4=\tfrac12\,\mathrm{Var}_{y\sim\pi_{\text{generator}}}\big[\log r(y)\big].
$$

Its gradient is

$$
\nabla_z k_4=\mathbb E_{y\sim\pi_{\text{generator}}}\big[(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)\,(e_y-\pi_\theta)\big]=\pi_{\text{generator}}\odot(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)=F_{\text{generator}}\log r.
$$

$k_2$ uses the mean of $\tfrac12(\log r)^2$, while this objective uses the mean of the square after the mean of $\log r$ is subtracted. To place it next to the first three estimators, we write it as $k_4$ below. Schulman's blog post has only $k_1$ to $k_3$, and the name $k_4$ is one that we chose in this article for convenience. When $\pi_\theta$ and $\pi_{\text{generator}}$ are very close, $k_4$ agrees with $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ up to second order, so it can also be seen as a form of the KL. In the gradient, the mean of $\log r$ under $\pi_{\text{generator}}$ is subtracted from $\log r$, which is the same as subtracting a baseline in Section 2.1. Under SFC, it makes no difference whether this mean is subtracted, because $F_{\text{generator}}\mathbf 1=0$.

So under SFC, $k_4$ is equivalent to changing the advantage to $A-\beta\log r$. It is simple to implement and needs no separate loss term. On each sampled token, we only need to modify the advantage with the detached value of $\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$. KLPO (Zhang, 2026) discusses the same objective. It calls this approach token regression. In its Appendix D, it writes the corresponding objective as the advantage-weighted log ratio plus $\beta/2$ times the variance of the log ratio.

Note that $k_4$ must be used together with SFC. If the score is still the ordinary $e_y-\pi_\theta$, the same code implements the version without the mean subtracted, that is, the gradient of $\tfrac12\mathbb E_{\pi_{\text{generator}}}[(\log r)^2]$. The coefficient of this objective under SFC is $\log r-\mathbb E_{\pi_{\text{generator}}}[\log r]\cdot r$, which has an extra term $-\mathbb E_{\pi_{\text{generator}}}[\log r]\cdot r$ compared with $k_4$.

The third form is the KL in the other direction, $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$. When the KL compares the model with the reference model, this is the direction that RLHF usually constrains, so it is worth seeing what happens when it is used with the generator. It is the forward KL that compares the model with $\pi_{\text{generator}}$, and by Section 2.1 its gradient is $\mathbb E_{y\sim\pi_\theta}[\log r\,s_\theta]$. When the samples come from $\pi_{\text{generator}}$, multiplying by the ratio $r$ turns this expectation into an expectation under $\pi_{\text{generator}}$. So the ordinary score with the detached coefficient $r\log r$ gives the exact gradient:

$$
\mathbb E_{y\sim\pi_{\text{generator}}}\big[r\log r\,(e_y-\pi_\theta)\big]=\pi_\theta\odot\log r-\pi_\theta\,\mathbb E_{\pi_\theta}[\log r]=\pi_\theta\odot\big(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})\big).
$$

Under SFC, the corresponding coefficient is $r\,(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}}))$. When $\pi_\theta$ and $\pi_{\text{generator}}$ are close, $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is a small quantity of second order, and this coefficient is approximately $r\log r$.

To summarize, under SFC the coefficient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is $r-1$, the coefficient of $k_4$ is $\log r$, and the coefficient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is approximately $r\log r$. All three coefficients are functions of $r$ that equal 0 at $r=1$ and agree to first order near $r=1$. Each form can be implemented by modifying the advantage, using only the $\pi_\theta(y)$ and the $\pi_{\text{generator}}(y)$ of the sampled token. The exact coefficient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ also needs an estimate of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ itself.

### 5.2 When the trainer and the generator are close, do these forms differ much?

The three coefficients of Section 5.1 agree to first order near $r=1$. This means that when the trainer and the generator are close, the gradients that they give should differ little. Below, an example with a vocabulary of only three tokens shows this, and it also shows how the forms differ in the details.

Let $\pi_{\text{generator}}=(0.6,0.3,0.1)$ and $\pi_\theta=(0.5,0.3,0.2)$. Then $r=(0.833,1,2)$, $\log r\approx(-0.182,0,0.693)$, and $\mathbb E_{\pi_{\text{generator}}}[\log r]\approx-0.040$. The two directions of the KL give $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)\approx0.040$ and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})\approx0.048$. The expected gradients of the three forms on the logits are listed below, and Figure 3 plots them together.

* The gradient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is $\pi_\theta-\pi_{\text{generator}}=(-0.1,\ 0,\ +0.1)$.
* The gradient of $k_4$ is $\pi_{\text{generator}}\odot(\log r-\mathbb E_{\pi_{\text{generator}}}\log r)\approx(-0.085,\ +0.012,\ +0.073)$.
* The gradient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is $\pi_\theta\odot(\log r-\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}}))\approx(-0.115,\ -0.014,\ +0.129)$.

<figure data-chart="kl-three-token">
<noscript><p>Figure 3 needs JavaScript to display.</p></noscript>
</figure>

Under gradient descent, the logit of the first token rises and the logit of the third token falls. The three forms agree on this, and all of them pull $\pi_\theta$ back toward $\pi_{\text{generator}}$. The difference is on the second token, whose $r$ is 1. Under $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ it does not move at all. Under $k_4$ it is pushed down slightly. Under $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ it is instead raised slightly. The reason is that each form compares $\log r$ with a different baseline. $k_4$ compares $\log r$ with its mean under $\pi_{\text{generator}}$, which is $-0.040$, so a token with $r=1$ is relatively high. $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ compares $\log r$ with its mean under $\pi_\theta$, which is $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})=0.048$, so the same token is relatively low.

In this example the forms differ little, because the trainer and the generator are not far apart. How far apart they get depends on the size of $\beta$ relative to the advantage. When $\beta$ is clearly larger than the differences between advantages, the KL keeps $r$ near 1, and the forms differ little. When $\beta$ is of the same order as the differences between advantages, or smaller, $r$ moves far from 1. Then the shapes of the coefficients far from $r=1$ make the forms behave very differently. In our earlier experimental setup, a typical token has a $\beta_t$ of only about 0.04, while the differences between advantages can be of order 1. This is the second case, and Section 6.1 explains where these two numbers come from. The next section discusses this case.

## 6. How do the three forms constrain tokens whose probability falls and tokens whose probability rises?

> **TL;DR:** The three forms differ mainly when $r$ is far from 1. For a token whose probability is being pushed down, the pull of $k_4$ has no upper bound as the probability falls, so $k_4$ can keep these tokens from being pushed to 0. The pull of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ has an upper bound, and the pull of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ goes to 0. So when $\beta$ is small, both of these forms may let tokens with a low advantage be pushed all the way down to 0. For a token whose probability is being raised, the situation is the reverse. $k_4$ gives the weakest constraint, and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ gives the strongest. Which form is more suitable depends on which kind of token carries more risk in training, and on the size of $\beta$ relative to the advantage.

Section 5 wrote the three forms as subtracting $\beta$ times a coefficient from the advantage, with the coefficient $r-1$ for $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$, $\log r$ for $k_4$, and approximately $r\log r$ for $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$. When the trainer and the generator are far apart, the shapes of these three coefficients determine how the forms behave. We look at two kinds of tokens. The first kind consists of tokens whose probability is being pushed down, that is, tokens with $r<1$. The second kind consists of tokens whose probability is being raised, that is, tokens with $r>1$. Section 6.1 first compares the pull of the three forms on a single token, and Section 6.2 then uses a simplified model to see where the forms finally pull the distribution. Section 6.3 discusses the cost of each form, and Section 6.4 discusses what to consider when choosing a form.

### 6.1 When the probability of a token keeps being pushed down, can the three forms pull it back?

The weight of the KL is $\beta$, but the actual effect of the KL on training also depends on the coefficient that $\beta$ multiplies, and this effect has to be compared with the size of the advantage. For this, we look at the "force" on the logit of a token $v$ under SFC, where the force means the expected update on this logit. Write $\Delta A_v=A_v-\mathbb E_{\pi_{\text{generator}}}[A]$, where $A_v$ is the expected advantage after choosing token $v$, as defined in Section 5.1. By the result of Section 5.1, the expected update of the policy gradient term on $z_v$ is $(F_{\text{generator}}A)_v=\pi_{\text{generator}}(v)\,\Delta A_v$. This quantity depends only on $\pi_{\text{generator}}$, and not on the current value of $\pi_\theta(v)$. In other words, while the generator is fixed, SFC keeps pushing or pulling this token with the same force. This differs from the case in which the samples come from the current model. In that case, the expected update of the policy gradient on $z_v$ is $\pi_\theta(v)\,(A_v-\mathbb E_{\pi_\theta}[A])$, so the lower the probability of the token, the smaller the push.

The forces with which the three forms pull this token back are as follows.

* The pull of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is $\beta\,(\pi_{\text{generator}}(v)-\pi_\theta(v))$. As $\pi_\theta(v)\to0$, this force approaches $\beta\,\pi_{\text{generator}}(v)$ and does not grow any further. This is the same as for $k_3$ used as a loss in Section 2.2, where the pull also never exceeds $\pi_{\text{ref}}(v)$.
* The pull of $k_4$ is $\beta\,\pi_{\text{generator}}(v)\,(\mathbb E_{\pi_{\text{generator}}}\log r-\log r_v)$. As $\pi_\theta(v)\to0$, $-\log r_v$ keeps growing, so this force grows linearly with how far the logit has fallen, and it has no upper bound.
* The pull of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is $\beta\,\pi_\theta(v)\,(\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})-\log r_v)$. When it is written as $\beta\,\pi_{\text{generator}}(v)$ times a coefficient, the coefficient is approximately $-r_v\log r_v$. The function $r\log r$ reaches its minimum, $-1/e$, at $r=1/e$, and it returns to 0 as $r$ decreases further. So this form pulls the token back with a force of at most about $\beta\,\pi_{\text{generator}}(v)/e$, and the lower the token is pushed, the weaker this force.

Now compare the push with the pull. For $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$, the force that pulls back a token that is being pushed down is never more than $\beta\,\pi_{\text{generator}}(v)$, while the force that pushes it down is $\pi_{\text{generator}}(v)\,|\Delta A_v|$. As long as $\Delta A_v<-\beta$, the push stays larger than the pull. This KL cannot stop the probability of this token from falling further, and the token is finally pushed toward 0. When the samples come from the current model, the push becomes smaller together with $\pi_\theta(v)$, and a bounded pull is enough to hold the token. Under SFC the push does not become smaller, so a bounded pull may not be enough to hold the token.

Figure 4 plots the pull of the three forms as a function of $r$, in units of $\beta\,\pi_{\text{generator}}(v)$. It leaves out the two small quantities $\mathbb E_{\pi_{\text{generator}}}\log r$ and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$. The lower a token is pushed, the farther left it moves in the figure. The dashed line is the push $|\Delta A_v|/\beta$, which you can change with the slider. Where a curve crosses the dashed line, the pull balances the push and the token stops. If a curve stays below the dashed line, that form cannot hold the token.

<figure data-chart="kl-pull">
<noscript><p>Figure 4 needs JavaScript to display.</p></noscript>
</figure>

The case $\Delta A_v<-\beta$ can occur in real training. In [our earlier experimental setup](https://normaluhr.github.io/en/p/score-function-centering-experiments/), the KL weight $\beta_t$ equals $\beta_0=0.3$ divided by the square root of the age. On the 4-prompt slice of Countdown, that is, when each update of the trainer uses the rollouts of 4 prompts, the mean age of a token is about 50, and a token with age 50 has $\beta_t=0.3/\sqrt{50}\approx0.042$. When those experiments computed the advantage, a correct response scored $+1$ and a wrong response scored $-1$, and then the mean within the group was subtracted. So the advantage of a single response is of order $\pm1$. The $\Delta A_v$ of a single token is usually smaller than that, because for most tokens, switching to a different choice does not change the outcome of the response. But on the few tokens that determine the outcome of the response, $|\Delta A_v|$ can come close to this order, far above 0.042.

However, in those experiments, SFC plus $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ never collapsed, and this does not contradict the analysis above. The comparison above assumes that the push lasts. In real training, each slice of rollouts is used for only one update, the generator keeps being refreshed, and the same prefix is rarely updated again and again, so the push does not last. Also, a token that is pushed down to 0 does not necessarily make training collapse. The upper bound of the pull shows that once the push persists, the protection that this KL can provide is limited.

Textbooks often say that $\mathrm{KL}(q\|p)$ is zero-avoiding. Whenever $q(v)>0$ and $p(v)\to0$, the value of the KL goes to infinity, so minimizing it avoids pushing any part of $p$ to 0. This statement is true for the value of the KL. But the value of $\mathrm{KL}(q\|p)$ grows only like $-\log p(v)$, which is roughly linear in the logit, and its gradient with respect to the logit is bounded. Against a push that stays constant, a value that goes to infinity does not guarantee that the token is kept away from 0.

### 6.2 If training continues, where do the three forms pull the distribution?

Section 6.1 looked only at the force on a single token. To see where the three forms finally pull the whole distribution, we use a simplified model. We fix a prefix and a generator version, and we assume that the expected advantage $A_v$ of each token $v$ stays fixed. Then we let the expected update continue indefinitely and see where it stops.

In optimization, a point where the gradient equals 0 is usually called a stationary point. The update direction here is not necessarily the gradient of some objective function, so by a stationary point we mean a point where the expected update equals 0, which is a fixed point in the sense of dynamical systems. Once a stationary point is reached, the expected update no longer changes $\pi_\theta$, and if the updates converge, they stop there. A stationary point may not be reachable. The probability of some tokens may keep falling during the updates and never stop, and then what we care about is the limit. The analysis and the examples below are all about stationary points.

Under SFC, the update direction is $F_{\text{generator}}(A-\beta f(r))$, where $f$ is the coefficient of the form. When every component of $\pi_{\text{generator}}$ is greater than 0, the only vectors $x$ with $F_{\text{generator}}x=0$ are constant vectors. So the condition for a stationary point is that $A_v-\beta f(r_v)$ is the same for all tokens.

For $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$, $f(r)=r-1$, and the stationary point is

$$
r_v^*=1+\frac{\Delta A_v}{\beta},\qquad \pi_\theta^*(v)=\pi_{\text{generator}}(v)\Big(1+\frac{A_v-\mathbb E_{\pi_{\text{generator}}}[A]}{\beta}\Big).
$$

The stationary point $r_v^*$ is a linear function of the advantage. It is also the stationary point of $\max_p\ \mathbb E_p[A]-\tfrac\beta2\chi^2(p\|\pi_{\text{generator}})$, where $\chi^2(p\|\pi_{\text{generator}})=\sum_v(p(v)-\pi_{\text{generator}}(v))^2/\pi_{\text{generator}}(v)$. In other words, within one generator version, SFC plus $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ has the same effect as a $\chi^2$ regularization. $\chi^2$ regularization has also been discussed for LLM alignment. Huang et al. (2024) argue that it suppresses the overoptimization of an imperfect reward better than KL regularization does. This stationary point is a valid distribution only when $\Delta A_v\ge-\beta$ for every token. Otherwise, the tokens with $\Delta A_v<-\beta$ keep being pushed toward 0, which agrees with the result of Section 6.1.

For $k_4$, $f(r)=\log r$, and the stationary point is

$$
\pi_\theta^*(v)\propto\pi_{\text{generator}}(v)\,e^{A_v/\beta}.
$$

This is the classic optimal solution of $\max_p\ \mathbb E_p[A]-\beta\,\mathrm{KL}(p\|\pi_{\text{generator}})$, and it has the same form as $\pi^*\propto\pi_{\text{ref}}\,e^{R/\beta}$ in RLHF and DPO, where $R$ is the reward. It is always a valid distribution, and no token is pushed to exactly 0. For a token that is raised, $r_v^*$ at the stationary point is not more than $e^{\Delta A_v/\beta}$. It is also limited by the normalization of the probabilities, so it is not more than $1/\pi_{\text{generator}}(v)$. Suppose that every generator version is trained to its stationary point and that the current trainer then becomes the new generator. Chaining the rounds together gives $\pi_{k+1}\propto\pi_k\,e^{A_k/\beta}$, which has the same form as mirror descent. Also, $k_4$ uses the samples of the generator, yet its stationary point corresponds to a regularization in the direction $\mathrm{KL}(p\|\pi_{\text{generator}})$, and the whole process needs no importance weight.

For $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$, the stationary point has no simple closed form. By the analysis of Section 6.1, the pull of this form on a token that is being pushed down is at most about $\beta\,\pi_{\text{generator}}(v)/e$, and the lower the token, the weaker the pull. Roughly speaking, when $\Delta A_v$ is below about $-\beta/e$, the token is pushed all the way to 0. The exact threshold is $-\beta e^{K-1}$, where $K$ is the value of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ at the stationary point, so the threshold is a little lower than $-\beta/e$. The larger $\beta$ is, the lower this threshold, and the fewer tokens are pushed to 0.

One point here is easy to get confused about. The stationary point of $k_4$ is the optimal solution of the problem that is regularized by $\mathrm{KL}(p\|\pi_{\text{generator}})$, while the stationary point that we get by adding $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ directly is not. The reason is the policy gradient term. Under SFC, the expectation of this term is $F_{\text{generator}}A$, which is determined by $\pi_{\text{generator}}$ and does not change with $\pi_\theta$. In the problem $\max_p\mathbb E_p[A]-\beta\,\mathrm{KL}(p\|\pi_{\text{generator}})$, however, the gradient of $\mathbb E_p[A]$ with respect to the logits is $F_\theta A$, where $F_\theta=\mathrm{diag}(\pi_\theta)-\pi_\theta\pi_\theta^\top$. The stationary condition of $k_4$ is that $A-\beta\log r$ is constant, and the optimality condition of the regularized problem is the same condition, so the two points coincide. When $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is added directly, the stationary condition is $F_{\text{generator}}A=\beta F_\theta\log r$. The two sides are multiplied by different matrices, so the stationary point is no longer that optimal solution.

Without any KL, that is, with $\beta=0$, the condition for a stationary point becomes that all $A_v$ are equal, which generally does not hold. Then there is no stationary point, and the probability keeps moving to the token with the highest advantage.

Below, two examples compare the three forms side by side. The numbers come from starting at $\pi_\theta=\pi_{\text{generator}}$ and applying the expected update to the logits again and again. For tokens that are pushed toward 0, the numbers are the limits of the updates.

In the first example, some tokens have a low advantage. Let $\pi_{\text{generator}}=(0.6,0.3,0.1)$, and let the advantages of the three tokens be $A=(0.5,0,-0.5)$. Then $\mathbb E_{\pi_{\text{generator}}}[A]=0.25$ and $\Delta A=(0.25,-0.25,-0.75)$.

* With $\beta=1$, $\beta$ is of the same order as the differences between advantages. $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ stops at $(0.75,0.225,0.025)$, and $k_4$ stops at $(0.733,0.222,0.045)$, so the two are close. The limit of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is $(0.803,0.197,0)$, where the third token has been pushed to 0. If $\beta$ is increased to 3, $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ stops at $(0.653,0.276,0.071)$, and the third token keeps a nonzero probability.
* With $\beta=0.2$, $\beta$ is smaller than the differences between advantages. The linear stationary point of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ is $(1.35,-0.075,-0.275)$. This is not a valid distribution, and the actual updates finally approach $(1,0,0)$. $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ also approaches $(1,0,0)$. $k_4$ stops at $(0.960,0.039,0.001)$. The second and third tokens are pushed very low, but neither of them reaches 0.

In the second example, a token that is rarely sampled gets a higher advantage. Let $\pi_{\text{generator}}=(0.9,0.09,0.01)$, $A=(0,0,0.5)$ and $\beta=0.05$, so $\beta$ is much smaller than the differences between advantages.

* $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ stops at $(0.81,0.081,0.109)$. The probability of the third token grows to about 11 times its original value.
* $k_4$ stops at $(0.004,0.0004,0.996)$. The third token takes almost all of the probability. Its $r$ is about 100, which is already the upper bound $1/\pi_{\text{generator}}(v)$ that the normalization of the probabilities allows.
* $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ stops at $(0.856,0.086,0.058)$. The probability of the third token grows to about 5.8 times its original value.

Figure 5 plots the stationary points of the two examples. The buttons at the top switch between the examples. Drag the slider to change $\beta$ from 0.01 to 10, and see how the stationary points of the three forms change with $\beta$. When $\beta$ is large, the stationary points of all three forms stay close to the generator. The smaller $\beta$ is, the more clearly they separate.

<figure data-chart="kl-fixed">
<noscript><p>Figure 5 needs JavaScript to display.</p></noscript>
</figure>

Together, the two examples show that the three forms have different roles. In the first example, when $\beta$ is small, only $k_4$ keeps the tokens with a low advantage, and under the other two forms these tokens are pushed to 0. In the second example, the rare token with a higher advantage grows the least under $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$, and somewhat more under $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$. Under $k_4$ it takes almost all of the probability.

Which behavior we want depends on how reliable the advantage is. If the high advantage of the rare token in the second example reflects a behavior that is really better, letting the token grow fast is good. If the high advantage comes from a stale sample, a flaw in the reward, or plain noise, limiting its growth is safer. The same holds for the first example. If a token with a low advantage is really bad, pushing it to 0 is fine. If it was only unlucky this time, pushing it to 0 loses exploration.

This simplified model also has a limited scope. In real training, each generator version is trained for only a limited number of steps, the generator keeps being refreshed, and different positions share parameters. The stationary points show only in which direction each form pulls, and how far it can pull at most. They cannot predict the actual outcome of training.

### 6.3 What does each form cost?

Sections 6.1 and 6.2 discussed how each form behaves in expectation. In practice, we also need to consider the variance of each form, and the risks that its behavior brings.

Section 5.1 showed that all three forms can be implemented by modifying the advantage, using only the $\pi_\theta(y)$ and the $\pi_{\text{generator}}(y)$ of the sampled token, so they differ little in implementation. Their difference in variance comes mainly from the size of the coefficients. The coefficient $r-1$ of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ contains $1/\pi_{\text{generator}}(y)$. Consider a token that has a small probability under the generator, but whose probability under the trainer has risen. Once this token is sampled, its coefficient is large. The coefficient $r\log r$ of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ grows faster than $r-1$, which is the same kind of problem as the variance of importance sampling. The coefficient $\log r$ of $k_4$ grows only logarithmically, so its variance is probably the smallest.

Another way to lower the variance is to compute the expectation exactly on the head and to estimate only the tail. In principle, all three forms can do this. Take $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ as an example, and write $\alpha$ for the total probability of the head under the generator. For the tokens in the head, $\pi_\theta-\pi_{\text{generator}}$ can be computed exactly. For a token sampled from the tail, the estimate is $(1-\alpha)(\pi_\theta/\pi_{\text{generator}}-1)$. Suppose that we also skip the sampling on the tail, compute exactly only on the head, and merge the tail into a single category that uses only the total probability of the tail. Then the variance that comes from sampling is gone, but the differences inside the tail are lost, so the result is biased. KLPO calls this TopK-KL. Our earlier experiments used this kind of implementation by default. The generator sent back its 128 most probable tokens, and the tail was not sampled. The total probability of the tail was set to the value that the generator gives, and it was split among the tokens of the tail in proportion to the probabilities that the trainer gives them. This is the same as merging the tail into a single category. Once the tail probability is split in the proportions of the trainer, every token in the tail has the same $r$, which equals the total probability of the tail under the trainer divided by the total probability of the tail under the generator. The KL inside the tail then reduces to a single term.

Apart from variance, the asymmetry of $k_4$ itself has a good side and a bad side. The good side is that $k_4$ prevents the probability of a token from being pushed to 0 within one generator version. This avoids a drop in entropy and a loss of exploration, and it still lets tokens that are really better grow fast. The bad side is that the constraint of $k_4$ on the raising side is weak. If the high advantage of a token comes from a stale sample or from noise, $k_4$ also does little to stop the token from growing. Our earlier experiments give one piece of indirect evidence. One method there multiplies the SFC gradient by the ratio truncated at 2, $\min(\pi_\theta/\pi_{\text{generator}},2)$, which is truncated importance sampling. This is the combination that Marek and Ryabinin (2026) use. When this method collapsed on Countdown, the entropy kept rising, which means that tokens that originally had a low probability were raised as a whole. The responses also became longer and longer, and at the end the KL between the trainer and the generator suddenly rose by two orders of magnitude. This looks more like a problem on the raising side. But that run had no KL term at all, so it is only indirect evidence.

The risk of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ is on the other side. Sections 6.1 and 6.2 showed that the protection these two forms give to tokens that are being pushed down is limited. When $\beta$ is small, tokens with a low advantage may be pushed all the way to 0.

### 6.4 How should one choose in practice?

Before comparing the forms, it helps to judge whether the choice matters in your own training. As Section 5.2 said, when $\beta$ is clearly larger than the differences between advantages, the KL keeps $r$ near 1, the forms differ little, and the choice has a limited effect. The choice matters only when $\beta$ is of the same order as the differences between advantages or smaller, for example for the token in Section 6.1 with age 50 and a $\beta_t$ of only 0.042.

If the choice does matter, the next question is which kind of token you worry about more in your training. Suppose you observe that the entropy keeps falling and that some tokens are pushed down until they almost never appear. Then a form that protects the tokens that are being pushed down, such as $k_4$, may be more suitable. Suppose instead you observe that the entropy rises, that the responses get longer and longer, and that the distribution of $r$ becomes very heavy-tailed. Then a form that limits the tokens that are being raised, such as $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ or $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$, may be more suitable. Which kind of token carries more risk may differ across models, data, optimizers and distributions of the age, so this judgment is best based on what you observe in your own training.

When you need to protect against both kinds of risk, you can combine the forms. For example, add both $k_4$ and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$. By the results of Section 6.2, $k_4$ protects the tokens that are being pushed down, and $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ gives the strongest constraint on the tokens that are being raised, so the two complement each other. The price is that the weights of the two terms have to be chosen separately.

Variance also affects the choice. The coefficient of $k_4$ grows only logarithmically, so its variance is probably the smallest of the three. The coefficient of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ contains $1/\pi_{\text{generator}}$, and the coefficient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ grows even faster. With either of these two forms, when a token with a very large $r$ happens to be sampled, it brings a very large gradient. The forms also have different requirements for numerical precision, which is the topic of the next section.

Another thing that can be tuned is how $\beta$ depends on the age. [Our earlier approach](https://normaluhr.github.io/en/p/score-function-centering-kl/) let $\beta_t$ decay with the age, on the grounds that an older generator is less suitable as the distribution that the current trainer is pulled toward. One can also argue the other way. The older a sample is, the farther it is from the current model and the more it needs a constraint, so $\beta$ should grow with the age. With the numbers of Section 6.1, $\beta_t$ is only 0.042 at age 50, and in the simplified model, $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ can no longer stop tokens with $\Delta A_v<-0.042$ from being pushed down. Which choice is more suitable depends on which kind of risk the samples of old generators are more likely to bring in training. For tokens with age 0, we set $\beta$ to 0 in our earlier experiments, for a reason related to numerical precision that Section 7.2 explains.

Finally, a KL between the trainer and the generator can only limit how much the model is updated within each generator version. The generator keeps following the trainer, so the distribution that this KL compares the model with also keeps moving. This KL therefore cannot prevent the model from moving farther and farther from the initial model over many rounds. If the long-term drift needs to be limited, one can add a separate KL that compares the model with $\pi_{\text{ref}}$. The two KL terms complement each other.

## 7. In finite precision, how should the KL and its coefficients be computed?

> **TL;DR:** To compute the KL and its coefficients, first compute the difference of log probabilities $\rho=\log\pi_\theta-\log\pi_{\text{generator}}$ in FP32, and then get $r-1$ with `expm1(rho)`. To log the value of $k_3$, use `expm1(rho) - rho` or a Taylor expansion. Do not first compute $e^\rho$, which is close to 1, and then subtract 1, because the logged KL can then be negative. If $r$ is stored in BF16, $r-1$ is also 0 for every token whose probability rose only slightly, with $\rho$ below about 0.0039, so these tokens get no constraint at all. To check whether the trainer and the generator agree exactly, require that the maximum of $\rho^2$ equals 0, and do not look at the mean of $k_1$.

The analysis so far assumes that the computation is exact. In real training, the KL and its coefficients are computed in finite precision. This section shows that the same formula, computed in different ways, can give very different results, and that the different coefficients of Section 5 have different requirements for precision. All numbers in this section were computed with the rounding rules of FP32 or BF16, and the curves in Figures 6 and 7 were computed point by point with the same rules.

The cause of the problem is that floating-point numbers are dense near 0 and sparse near 1. In FP32, the gap between 1 and the next larger number is $2^{-23}\approx1.2\times10^{-7}$, while near 0, FP32 can represent numbers as small as order $10^{-38}$. BF16 is coarser. The gap between neighboring numbers just above 1 is $2^{-7}\approx0.0078$, and the gap just below 1 is $2^{-8}\approx0.0039$. If the result we want is a very small quantity, the computation should avoid an intermediate value of the form 1 plus this small quantity. Once such an intermediate value is stored as a floating-point number near 1, most of the information in the small quantity is lost, and subtracting 1 afterwards cannot bring it back. The next two sections show how this problem appears when we log the value of the KL and when we compute the coefficients used in training.

### 7.1 Why can the logged value of the KL be negative?

Take $k_3=e^\rho-1-\rho$ as an example, where $\rho=\log r=\log\pi_\theta(y)-\log\pi_{\text{generator}}(y)$. Computing it directly by this formula in FP32 gives $-4.6\times10^{-8}$ at $\rho=10^{-6}$. The true value is about $\rho^2/2=5\times10^{-13}$. A quantity that is mathematically non-negative comes out negative, and the error is five orders of magnitude larger than the true value. The reason is that $e^\rho$ is stored as the FP32 number nearest to it, which lies near 1, and the rounding error of this step is of order $10^{-7}$. After 1 is subtracted, almost all that remains is rounding error. If we compute `expm1(rho) - rho` instead, `expm1` computes $e^\rho-1$ directly, without the intermediate value $e^\rho$ that is close to 1. The result is $4.5\times10^{-13}$, which differs from the true value by only 9%. The remaining error comes from rounding the output of `expm1`. That output has a relative error of order $10^{-7}$, which is an absolute error of about $6\times10^{-8}\rho$. So the smaller $\rho$ is, the larger the relative error of $k_3$. To keep the precision at even smaller $\rho$, one can compute with the Taylor expansion $\rho^2/2+\rho^3/6$ directly when $|\rho|$ is very small. This removes the subtraction step completely.

Figure 6 plots, on a logarithmic scale, the true value of $k_3$ and the errors of the three ways of computing it. The error of the direct computation stays at order $10^{-8}$ to $10^{-7}$, whatever the size of $\rho$. When $\rho$ is below about $3\times10^{-4}$, this error is already larger than the true value, and the computed result is meaningless. With `expm1`, the error is proportional to $\rho$, and it becomes comparable to the true value only when $\rho$ is below about $10^{-7}$. With the Taylor expansion, the error stays far below the true value while $\rho$ is small. Only when $\rho$ becomes large do the dropped higher-order terms start to matter.

<figure data-chart="kl-fp32">
<noscript><p>Figure 6 needs JavaScript to display.</p></noscript>
</figure>

The same cause can also make a consistency check reach a wrong conclusion. Some checks use a quantity that is written as 1 plus a small second-order quantity, for example $1+\rho\,(e^\rho-1)$. When the samples come from the generator, the expectation of this quantity is 1 plus the sum of the KL in the two directions. In FP32, as long as $|\rho|\le2.4\times10^{-4}$, this quantity is rounded to exactly 1, which looks as if the trainer and the generator agreed exactly.

The mean of $k_1$ is also not suitable for checking whether the two agree. $\rho$ can be positive or negative, so positive and negative values can cancel, and the mean can be close to 0 or even negative. To confirm that the trainer and the generator agree exactly, a more reliable approach is to compute $\rho^2$ directly and to require that its maximum is exactly 0. $\rho^2$ is never negative, so a maximum of 0 means that the trainer and the generator agree exactly on every token.

### 7.2 How much does precision affect the coefficients used in training?

The coefficients used in training are affected by the same problem, but to different degrees. The coefficient of $k_4$ is $\log r=\rho$. As long as $\rho$ is computed by subtracting FP32 log probabilities, its relative error stays within 0.2%, even if it is converted to BF16 afterwards. But if the log probabilities themselves are stored in BF16, the subtraction loses most of the information. For example, $-2.3012$ and $-2.3042$ both become $-2.296875$ when they are stored in BF16. Their difference is then 0, while the true value is 0.003.

The coefficient $r-1$ of $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ has one more problem. If $r=e^\rho$ is computed first and 1 is subtracted afterwards, we run into the situation described at the start of Section 7. Suppose that $r$ is stored in BF16. Then at $\rho=0.001$ and at $\rho=0.003$, $r$ is stored as exactly 1, and $r-1$ is 0. At $\rho=0.005$, $r-1$ becomes 0.0078, while the true value is 0.0050. At $\rho=-0.003$, $r-1$ becomes $-0.0039$, while the true value is $-0.0030$. Because the gaps above and below 1 differ, the errors on the two sides are not symmetric. Every token whose probability rose only slightly, with $\rho$ below about 0.0039, gets the coefficient 0, so it is not constrained at all. A token with a slight decrease is pulled back as soon as its $\rho$ is below about $-0.002$, and the pull is enlarged. Figure 7 plots the steps that we get when we subtract 1 from an $r$ stored in BF16, and the two vertical lines mark these two thresholds.

<figure data-chart="kl-bf16">
<noscript><p>Figure 7 needs JavaScript to display.</p></noscript>
</figure>

The correct approach is to compute $\rho=\log\pi_\theta-\log\pi_{\text{generator}}$ in FP32, to use this $\rho$ directly as $\log r$, and to get $r-1$ with `expm1(rho)`. The same applies to the estimate on the tail in Section 6.3. The dashed line in Figure 7 shows the result of computing `expm1` in FP32 first and storing the result in BF16 afterwards, and it almost coincides with the true value. The coefficient $r\log r=e^\rho\rho$ of $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ has no step that subtracts 1. Even if $r$ is stored in BF16, the relative error of this coefficient stays within 0.2%, so it is less affected by this problem.

There is another kind of numerical difference that does not depend on the storage precision. Even when the trainer and the generator use the same version of the parameters, the log probabilities that they compute are not exactly the same, because the inference engine and the training framework may use different kernels, precisions, parallelism strategies and batch shapes (Thinking Machines Lab, 2025). Then $\rho$ is not 0, and every form of the KL between the trainer and the generator pulls the trainer toward the numerical error of the inference engine. This is why, in our earlier experiments, we set $\beta$ to 0 for tokens with age 0, that is, tokens for which the trainer and the generator have the same parameter version. Removing this difference completely requires the two sides to compute in exactly the same way, for example with batch-invariant kernels. This is a problem of the inference engine, and it is outside the scope of this article.

## 8. Closing remarks

This article discussed how the form of a KL relates to the gradient that it produces. When you analyze a line of KL code, it helps to first ask the following questions.

* Which model drew the samples? Was it the current model, an old model, or a fixed generator?
* Where does $\theta$ appear, and which of these places does the code differentiate?
* What is the coefficient that finally multiplies the score, and which direction of the KL does it correspond to?
* In what precision is this coefficient computed?

For the KL that brings the trainer and the generator closer in asynchronous RL, each form has a different role. $k_4$ protects the tokens that are being pushed down. $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ constrains the tokens that are being raised. $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ acts on both kinds of tokens, but its protection of the tokens that are being pushed down is limited. Which form is more suitable depends on which kind of token carries more risk in training, and also on the size of $\beta$ relative to the advantage.

The analysis above covered how these forms behave in expectation, what they cost, and the precision issues in computing them. How they perform in real training, for example whether they can prevent training from collapsing, may also depend on other factors, such as the size of the head, how the tail is handled, and the size of the model. We will discuss these questions with experiments in a separate article.

The coefficients in this article are at the token level. A later article will apply the same idea at the prompt level. For example, when GRPO normalizes the advantage by the standard deviation of the group, this amounts to multiplying each prompt by a weight, and this weight determines which objective the training actually optimizes.

## Appendix: How does the naming of KL directions in this article differ from other public sources?

This article follows the naming of [our earlier article that adds a KL to Score Function Centering](https://normaluhr.github.io/en/p/score-function-centering-kl/). That article calls the estimator of $D_{\mathrm{KL}}(q_\phi\|p_\theta)$ the sparse reverse-KL estimator, where $q_\phi$ is the generator and $p_\theta$ is the trainer. By the same rule, this article calls a KL with $\pi_\theta$ in the second place the reverse KL, for example $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$ and $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$. A KL with $\pi_\theta$ in the first place is called the forward KL, for example $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$ and $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$.

Some public sources use the opposite names. RLHF papers and codebases usually view the model being trained, $\pi_\theta$, as the approximating distribution in variational inference. So they call $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ the reverse KL and $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$ the forward KL. Articles on on-policy distillation do the same, and they call the KL with the student first and the teacher second the reverse KL. The two naming conventions refer to the same two quantities, with the names swapped.

When you read this article next to other sources, the following points are the easiest to confuse.

* When $k_2$ is used as a loss and the samples come from the current model, its gradient equals the gradient of $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$. This article says that it gives the forward KL, while the sources with the opposite naming would say the reverse KL.
* Under the same conditions, the gradient of $k_3$ equals the gradient of $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$. This article says that it gives the reverse KL, while the sources with the opposite naming would say the forward KL.
* InstructGPT-style PPO puts $k_1$ in the reward, and its gradient corresponds to $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, which is the forward KL in this article. GRPO uses $k_3$ as a loss, and under the approximation that the samples come from the current model, its gradient corresponds to $\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)$, which is the reverse KL in this article.
* Liu et al. (2025) say that $k_3$ used as a loss is a first-order approximation of the reverse KL. The reverse KL that they mean is $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, which is the forward KL in this article.
* Sources often say that the reverse KL is mode-seeking. This means that a KL with $\pi_\theta$ first allows $\pi_\theta$ to keep only part of the values that the other distribution takes. In the naming of this article, this statement is about the forward KL. In Section 6.2, under $\mathrm{KL}(\pi_\theta\|\pi_{\text{generator}})$, tokens with a low enough advantage are pushed all the way to 0, which is an example of this property.
* The KL term in the earlier article is $\mathrm{KL}(\pi_{\text{generator}}\|\pi_\theta)$. It is called the reverse KL both in that article and in this one, so its name does not change.

So when you read such material, the most reliable approach is to check the formulas. First see which distribution is in each of the two places of the KL. Then see which distribution the samples come from, and which places the code differentiates. Once these three points are clear, the names no longer cause misunderstanding.

## References

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
