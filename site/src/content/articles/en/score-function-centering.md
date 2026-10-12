---
title: "Score Function Centering Made Simple: Understanding Trainer-Generator Mismatch in Async RL from the Score Function"
date: 2026-10-08
category: post-training
series: grpo-line
part: 4
tags: ["GRPO"]
---

# Score Function Centering Made Simple: Understanding Trainer-Generator Mismatch in Async RL from the Score Function

I find that the explanations of score function centering available online are all more or less roundabout, and not direct enough. I want this article to give you the most direct explanation of score function centering, starting from intuition. This article was written entirely by hand, not by AI. Of course, if you are reading the English version, I am sorry, that one was translated by AI.

This article assumes you know how to use PPO or GRPO (what the loss looks like) and some simple probability theory (that really is enough). If you have not read it yet, please take ten minutes to read [this article](https://normaluhr.github.io/en/p/grpo/).

If any one of the following describes you, this article should suit you:

* You only know how to use GRPO, but you do not know the principle behind it
* You want to deepen your understanding of trainer-generator mismatch
* You work on async RL with long rollout training and are still not sure how to improve the stability of the algorithm
* You have no idea what score function centering is (you may not even know what the score here is) but you want to find out

Score Function Centering is not an established term. I know that every frontier lab has its own name for this technique, alternative token, covariance optimization and so on. Call it whatever you like. This article will also explain where these names come from.

TL;DR:

* After section 1 you will know that the score function is an important bridge for computing the policy gradient, and the most direct thing we can get out of auto-differentiation.
* After section 2 you will know the important property of the score function: it is already centered, so its expectation is 0. We need that expectation to stay 0 in order to replace reward with the so-called advantage in the policy gradient. You will also see how trainer-generator mismatch breaks that property. And you will see what breaking it actually costs: the mean advantage produces an extra gradient that pulls the current trainer toward the old generator, which is why we need to repair it.
* After section 3 you will know what score function centering is really doing, why it repairs that zero-expectation property, how we should use it in practice, and one direction for churning out papers in this area.

## 1. Let us start from the score function. What is a score? Why do we use it?

### 1.1 In GRPO/PPO, what gradient are we actually computing?

> Below is a quick derivation of the policy gradient. If you are already quite familiar with this part, skip this subsection.

Suppose that at some state, that is, at some LLM prefix $x$, the policy is:

$$
p_\theta(a|x)
$$

For simplicity, let us drop $x$ for now and write:

$$
p_\theta(a).
$$

Our goal is very natural:

"Make high reward actions easier for the model to sample."

So we can write the simplest possible objective:

$$
J(\theta) = \mathbb E_{a\sim p_\theta}[R(a)].
$$

What this says is: **"Sample an action from the current model $p_\theta$, compute the reward, and hope its expectation keeps going up."**

The question is: how do we compute $\nabla_\theta \mathbb E_{a\sim p_\theta}[R(a)]$? There is a difficulty here: $\theta$ does not appear directly inside $R(a)$. It is hidden inside the sampling distribution $a\sim p_\theta$. In other words, we are not computing $\nabla_\theta R(a)$. In fact, the reward in RL is perfectly allowed to be non-differentiable. For example: 1 point for a correct answer to a maths problem and 0 otherwise, or a human expert handing out a score directly. None of that matters. What really depends on $\theta$ is $p_\theta(a)$.

Writing the expectation out:

$$
J(\theta) = \sum_a p_\theta(a)R(a).
$$

Differentiating:

$$
\nabla_\theta J(\theta) = \sum_a R(a)\nabla_\theta p_\theta(a).
$$

Then we use a very simple but extremely important identity:

$$
\nabla_\theta p_\theta(a) = p_\theta(a) \nabla_\theta\log p_\theta(a).
$$

Because:

$$
\nabla \log p = \frac{1}{p}\nabla p.
$$

So:

$$
\nabla_\theta J(\theta) = \sum_a p_\theta(a) R(a) \nabla_\theta\log p_\theta(a).
$$

Writing it back as an expectation:

$$
\nabla_\theta J(\theta) = \mathbb E_{a\sim p_\theta} \left[ R(a) \nabla_\theta\log p_\theta(a) \right].
$$

Here, at last, is the most critical object in the policy gradient:

$$
s_\theta(a) = \nabla_\theta\log p_\theta(a).
$$

This thing is called the **score function**. Note that the score here is not a reward score. It is not "how many points did this answer get". It is a specific term from probability and statistics:

$$
\text{score} = \nabla_\theta\log p_\theta(a).
$$

That is, **the gradient, with respect to the model parameters, of the log probability of some sampled action.**

### 1.2 Why do we study the score?

Because the core structure of the policy gradient really is just:

**reward $\times$ score**

(I know you want to say that what we actually use is $\text{Advantage}\times\text{score}$, but hold that thought.)

Written the way we are more used to when training LLMs:

$$
g=R(a)\nabla_\theta\log p_\theta(a).
$$

So you can read the two parts like this: $R(a)$ tells us **"should this action be reinforced or weakened, and by how much?"** while $\nabla_\theta\log p_\theta(a)$ tells us **"if I want to raise or lower the probability of this action, which direction should the parameters move in?"**

This is also why I think the most important sentence for understanding the policy gradient is:

**The score function is the bridge that turns "the gradient of an expectation over a distribution", which you have no way of computing, into "a gradient estimate you can compute by sampling".**

More precisely:

**distribution-level differentiation $\rightarrow$ sample-level gradient estimation**

What we originally wanted to compute was:

$$
\nabla_\theta \mathbb E_{a\sim p_\theta}[R(a)].
$$

It looks like it requires differentiating through the whole action distribution.

But the score-function trick turns it into this: first sample an action

$$
a\sim p_\theta,
$$

then compute:

$$
R(a)\nabla_\theta\log p_\theta(a).
$$

Sample a few more times and average:

$$
\nabla_\theta J\approx\frac1N\sum_{i=1}^NR(a_i)\nabla_\theta\log p_\theta(a_i).
$$

This is the core computational mechanism of REINFORCE and of policy gradient. And although GRPO adds group-relative advantage, importance ratio, clipping and so on, the basic object inside it that actually produces the policy gradient still comes down to:

$$
\nabla_\theta\log p_\theta(a).
$$

At the same time, you will notice that this is the most direct thing a single backward hands you when you actually use GRPO.

## 2. The score function and trainer-generator mismatch

### 2.1 What is the beautiful property of the score function, and why does it matter?

Let us now talk about the advantage. We all know that in GRPO/PPO we do not use the reward directly, we use the advantage ([why?](https://normaluhr.github.io/en/p/grpo/)). The advantage is the reward minus a constant:

$$
A(a)=R(a)-b,
$$

Here $b$ can be a value function, it can be a group mean, and it can be any baseline that depends only on the current state and not on the specific action. In GRPO, for instance, the thing we are most familiar with is sampling many responses for the same prompt and then using some group-relative reward as the advantage. Never mind exactly how the normalization is done, in essence it is all doing one thing: subtracting from the raw reward a part that everyone shares, leaving only whether this action is better or worse than average. But there is a very basic question here that is also very easy to overlook:

**What gives us the right to subtract this baseline at all?**

The original policy gradient is:

$$
g=\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)].
$$

If we swap the reward for the advantage:

$$
A(a)=R(a)-b,
$$

then the new gradient is:

$$
g'=\mathbb E_{a\sim p_\theta}[(R(a)-b)s_\theta(a)].
$$

Expanding it:

$$
\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)] -b\mathbb E_{a\sim p_\theta}[s_\theta(a)].
$$

The first term is our original policy gradient. In other words, if we want the expectation of the policy gradient to be completely unchanged after subtracting the baseline, the second term has to vanish:

$$
b\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0.
$$

And $b$ is clearly not necessarily 0, so what we really need is:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0.
$$

This is a very important property of the score function inside the policy gradient. You can ignore for now why the property holds and simply remember what it means in practice:

**As long as the expectation of the score is 0, we can subtract from the reward any baseline that does not depend on the action, without changing the expected policy gradient.**

This is why we can write the reward as an advantage, and why reward centering, value baselines and group-relative baselines all work inside PPO and GRPO. Some [papers](https://arxiv.org/abs/2609.20807) explain this starting from covariance, writing something like:

$$
\mathbb E_{a\sim p_\theta}[R(a)s_\theta(a)] = \operatorname{Cov}(R(a),s(a)) + \mathbb E[R(a)]\mathbb E[s(a)],
$$

and then, because $\mathbb E[s]=0$, only the covariance is left. There is of course nothing wrong with writing it this way, but personally I feel it makes something very simple sound roundabout. What we actually care about is one sentence: I want to subtract a baseline from the reward, and I do not want that baseline to produce a gradient of its own. That naturally requires $\mathbb E[s]=0$. It really is that simple.

And this property fits intuition very well. Suppose the rewards of two actions are:

$$
R(A)=101,\qquad R(B)=99.
$$

and:

$$
R'(A)=2,\qquad R'(B)=0.
$$

From the point of view of policy optimization, these two sets of rewards carry exactly the same information: $A$ is 2 points better than $B$ in both. We clearly do not want the expected update direction of the model to change simply because we added 99 to every reward. Adding or subtracting the same constant from all rewards never changes the expected policy gradient. This is also why a baseline is fundamentally a very natural operation and not merely a mathematical trick.

Now the question is:

**Why does the expectation of the score happen to equal 0?**

To answer that, we finally need to look at what the score really is.

### 2.2 What exactly is the score function?

Next we do some calculus on the score function, the kind anyone who has taken a calculus course can do. We defined earlier:

$$
s_\theta(a)=\nabla_\theta\log p_\theta(a).
$$

Suppose the current model outputs, at some prefix, the logits of all tokens in the vocabulary:

$$
z_\theta(1),z_\theta(2),\ldots,z_\theta(V).
$$

We assume as usual that the output distribution is given by Softmax:

$$
p_\theta(a) = \frac{e^{z_\theta(a)}}{\sum_v e^{z_\theta(v)}}.
$$

So:

$$
s_\theta(a) = \nabla_\theta z_\theta(a) - \nabla_\theta \log\sum_v e^{z_\theta(v)}.
$$

Let us give the first term above a nicer name, the logit Jacobian:

$$
J_\theta(v)=\nabla_\theta z_\theta(v).
$$

You can read this $J_\theta(v)$ as: if I change the current trainer's model parameters slightly, in which direction does the raw logit of token $v$ move.

Here is a concrete example. Suppose the model parameters change from $\theta$ to:

$$
\theta+\Delta\theta,
$$

then to first order:

$$
z_{\theta+\Delta\theta}(v)
\approx
z_\theta(v)
+
J_\theta(v)^T\Delta\theta.
$$

So $J_\theta(v)$ is essentially a "logit sensitivity direction" for token $v$ in parameter space. It tells us which way the parameters have to move to push this token's logit up or pull it down.

Next we deal with the second term:

$$
\nabla_\theta
\log\sum_v e^{z_\theta(v)}
$$

This needs the chain rule twice. First differentiate the outer log, which puts the sum inside the log into the denominator:

$$
\nabla_\theta\log\sum_v e^{z_\theta(v)}=\frac{1}{\sum_u e^{z_\theta(u)}}\nabla_\theta\sum_v e^{z_\theta(v)}.
$$

Then move the gradient inside the sum and use the chain rule once more on every $e^{z_\theta(v)}$:

$$
\nabla_\theta\sum_v e^{z_\theta(v)}=\sum_v\nabla_\theta e^{z_\theta(v)}=\sum_v e^{z_\theta(v)}\nabla_\theta z_\theta(v).
$$

Putting the two steps together gives:

$$
\frac{
\sum_v \color{orange}{e^{z_\theta(v)}} \color{blue}{\nabla_\theta z_\theta(v)}
}{
\color{orange}{\sum_u e^{z_\theta(u)}}
}.
$$

And it happens that:

$$
p_\theta(v) = \color{orange}{\frac{e^{z_\theta(v)}}{\sum_u e^{z_\theta(u)}}}, \quad J_\theta(v)= \color{blue}{\nabla_\theta z_\theta(v)}.
$$

So the second term can be written as:

$$
\sum_v p_\theta(v)J_\theta(v).
$$

That is:

$$
\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

Therefore:

$$
s_\theta(a) = J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

This expression is especially important, because it tells us that **inside the score function, Softmax is already centering the logit Jacobian by itself**. $J_\theta(a)$ is the logit gradient of the action we just sampled, while $\mathbb E_{v\sim p_\theta}[J_\theta(v)]$ is the average logit gradient over all actions, weighted by the current policy $p_\theta$.

So the score really is **the logit Jacobian of the sampled action minus the average logit Jacobian under the current policy**.

With that expression in hand, the fact that the expectation of the score is 0 barely needs proving. Because the action itself is sampled from:

$$
a\sim p_\theta
$$

we have:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{a\sim p_\theta}\left[J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]\right].
$$

Splitting the expectation:

$$
=\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

Here $a$ and $v$ are just two dummy variables with different names. They both come from exactly the same distribution $p_\theta$, so the two terms are the same thing:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]=0.
$$

So now we can put the whole story together: **the Softmax score is centered by construction**, therefore:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0,
$$

and this zero-mean property in turn guarantees that **we can use reward minus baseline in place of the raw reward in the policy gradient, without changing the expected policy gradient.**

Up to here everything is lovely. But notice that the whole derivation above quietly uses one assumption we have not yet stated carefully:

**The distribution the expectation is computed over, and the distribution the action is generated from, are both $p_\theta$.**

In other words, we assume:

**the generator that produces the rollout is the very trainer that is now running backward.**

The story only really begins once that assumption fails.

### 2.3 What exactly does trainer-generator mismatch break?

It must break $\mathbb E_{a\sim p_\theta}[s_\theta(a)] = 0$, the zero-mean property of the score function, otherwise all that discussion above would have been for nothing.

But the most important question here is not "it is broken", it is: **which step exactly changed?**

Let us first separate the trainer and the generator clearly. Suppose the trainer policy that is currently running backward and about to update the parameters is:

$$
p_\theta(a).
$$

And the generator that actually produced the rollout is a slightly older policy:

$$
q_\phi(a).
$$

I deliberately used different parameters $\theta$ and $\phi$ here. You can read $\theta$ as the trainer's current parameters, and $\phi$ as the version the generator took away to do the rollout. In synchronous training we approximately have:

$$
q_\phi \approx p_\theta,
$$

and we can even simply treat it as:

$$
q_\phi=p_\theta.
$$

But in async RL, especially when rollouts are long, there are many generators, and the trainer keeps updating, the two distributions naturally start to drift apart:

$$
q_\phi\neq p_\theta.
$$

Now recall the formula we just used to prove the zero-mean property:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)]=\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)]=0.
$$

Why is it 0? Essentially because both expectations are taken over the same $p_\theta$. In the first expectation, the action $a$ is sampled from $p_\theta$. In the second, Softmax itself uses $p_\theta$ to center all the logit Jacobians. So the two terms match exactly.

But now the generator has become $q_\phi$. The action that actually reaches the trainer no longer satisfies:

$$
a\sim p_\theta,
$$

but instead:

$$
a\sim \color{red}{q_\phi}.
$$

Note that the trainer has not changed the way it computes the score. What we run backward is still the current trainer's:

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

The $J_\theta$ here still comes from the current trainer, and the Softmax centering still uses the current trainer distribution $p_\theta$. Only one thing has actually changed: **which distribution the action on the outside was sampled from.**

So what we should really be looking at now is no longer:

$$
\mathbb E_{a\sim p_\theta}[s_\theta(a)],
$$

but:

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)].
$$

Expanding the score:

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)]=\mathbb E_{a\sim \color{red}{q_\phi}}\left[J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]\right].
$$

Splitting the expectation:

$$
=\mathbb E_{a\sim \color{red}{q_\phi}}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

Now the problem is obvious. Originally we had:

$$
\mathbb E_{a\sim p_\theta}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)],
$$

where both expectations are over the same distribution, so it equals 0. Now it has become:

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[J_\theta(a)]-\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

The left side averages under the generator distribution $q_\phi$, while the right side still averages under the trainer distribution $p_\theta$. As long as:

$$
q_\phi\neq p_\theta,
$$

we generally can no longer expect:

$$
\mathbb E_{a\sim q_\phi}[J_\theta(a)]=\mathbb E_{v\sim p_\theta}[J_\theta(v)].
$$

So in the end:

$$
\mathbb E_{a\sim \color{red}{q_\phi}}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

This is the most direct effect of trainer-generator mismatch on the score function.

You will find there is nothing mysterious about this at all. **The score did not suddenly go bad, the Softmax formula did not change, and the $J_\theta$ the trainer computes did not change either. The only thing that actually changed is that the distribution the action is sampled from went from $p_\theta$ to $q_\phi$.**

And the zero-mean property happens to depend on "the distribution the action is sampled from" and "the distribution the centering inside the score uses" being the same distribution. Those two things are now different:

$$
\underbrace{a\sim \color{red}{q_\phi}}_{\text{generator sampling distribution}},\qquad \underbrace{\mathbb E_{v\sim p_\theta}[J_\theta(v)]}_{\text{trainer centering distribution}}.
$$

So the old:

$$
\mathbb E_{p_\theta}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]=0
$$

has become:

$$
\mathbb E_{\color{red}{q_\phi}}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

At this point we have in fact explained the most critical layer of trainer-generator mismatch:

**In synchronous RL, the generator and the trainer are the same distribution, so the standard score is zero-mean by nature. In async RL, the action comes from the old generator $q_\phi$ while the score is still centered by the current trainer $p_\theta$, and so the zero-mean property is broken.**

### 2.4 What does this non-zero mean actually cause in practice?

At this point you may feel that $\mathbb E_{q_\phi}[s_\theta]\neq 0$ still sounds a bit abstract. We said above that it breaks baseline invariance, but what kind of gradient does it actually produce in training?

This becomes very intuitive if we look at it directly in logit space. For a Softmax, the gradient of the log probability of some action $a$ with respect to the whole logit vector $z$ is:

$$
\nabla_z\log p_\theta(a)=e_a-p_\theta,
$$

where $e_a$ is the one-hot vector of action $a$. In synchronous training $a\sim p_\theta$, so taking the expectation of it gives:

$$
\mathbb E_{a\sim p_\theta}[\nabla_z\log p_\theta(a)]=p_\theta-p_\theta=0.
$$

But now the action is sampled by the old generator $q_\phi$:

$$
\mathbb E_{a\sim q_\phi}[\nabla_z\log p_\theta(a)]=q_\phi-p_\theta.
$$

This formula makes the intuition behind it easier to see than $\mathbb E_q[J]-\mathbb E_p[J]$ did: **if some token is more common in the old generator than in the current trainer, that is $q_\phi(v)>p_\theta(v)$, this direction pushes its logit up. The other way around, if $q_\phi(v)<p_\theta(v)$, this direction pulls it down.**

Now let us go back to the baseline we have been discussing all along. Suppose you add a constant $c$ that has nothing to do with the action to every reward or advantage. In a normal on-policy setting this constant should not produce any expected gradient. But after trainer-generator mismatch it produces an extra:

$$
\Delta g_z=c\,\mathbb E_{a\sim q_\phi}[\nabla_z\log p_\theta(a)]=c(q_\phi-p_\theta).
$$

If $c>0$, what this gradient does is very direct: **it pulls the current trainer $p_\theta$ toward the distribution of the old generator $q_\phi$.**

The other way around, if $c<0$, the trainer is pushed away from the generator. That is a bad thing to have, because the $c$ here is playing the role of the baseline, and in theory a baseline is free to be positive or negative. So we have effectively added a perturbation to the distance between the trainer and the generator, and the direction and the size of that perturbation are decided entirely by the value of the baseline.

There is another way to look at this direction. Suppose we really do take the samples of the old generator and maximize the log likelihood of the current trainer:

$$
\mathbb E_{a\sim q_\phi}[\log p_\theta(a)],
$$

its gradient in logit space is exactly:

$$
\nabla_z\mathbb E_{a\sim q_\phi}[\log p_\theta(a)]=q_\phi-p_\theta.
$$

In other words:

$$
q_\phi-p_\theta=-\nabla_zD_{\mathrm{KL}}(q_\phi\|p_\theta).
$$

So when the mean reward or the mean advantage is positive, that thing which was supposed to be no more than "the part that everyone shares" quietly turns into **a gradient whose direction is exactly the direction of doing behavior cloning on the old generator**. This is quite strange: **the average says nothing about which action is better or worse than which, so what gives it the right to decide whether we move toward the old generator, and by how much?**

The same thing shows up in the parameter-space notation we used earlier. Split the advantage into:

$$
A(a)=(A(a)-\bar A_q)+\bar A_q,\qquad \bar A_q=\mathbb E_{q_\phi}[A].
$$

Then:

$$
\mathbb E_{q_\phi}[A(a)s_\theta(a)]=\mathbb E_{q_\phi}[(A(a)-\bar A_q)s_\theta(a)]+\bar A_q\,\mathbb E_{q_\phi}[s_\theta(a)].
$$

And the second term is:

$$
\bar A_q\left(\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\right).
$$

This is one of the most practical consequences of breaking the zero-mean property: **the term that really describes how good one token is relative to another is the first one, but the mean advantage additionally multiplies a trainer-generator mismatch direction.**

Looking at Score Function Centering now, the motivation is more direct. We do not only want a mathematical expectation to look like 0 again, we also want to take this old-policy cloning direction, the one quietly controlled by "the average reward level", out of the reward gradient. How far apart the trainer and the generator should be allowed to drift is a separate question, and $\bar A_q$ should not get to decide it for us in passing here.

## 3. What is Score Function Centering actually doing?

The answer is almost shouting at you by now. Score Function Centering simply wants the broken zero-mean property of the score function to hold again, so that we can go back to happily using a baseline.

We just saw that after trainer-generator mismatch, the real problem becomes:

$$
\mathbb E_{a\sim q_\phi}[s_\theta(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{p_\theta}[J_\theta]\neq 0.
$$

Why is this not 0? The reason is very simple: **the action was sampled from the generator's $q_\phi$, but the "center" subtracted inside the score is still the trainer's $p_\theta$.**

To correct that, given that the action on the outside already comes from $a\sim q_\phi$, the most direct approach is of course: **stop centering with $p_\theta$, and center with $q_\phi$, the distribution that actually generated these actions.**

That is, take the original standard score:

$$
s_\theta(a)=J_\theta(a)-\mathbb E_{v\sim p_\theta}[J_\theta(v)]
$$

and change it to:

$$
\tilde s_{\theta,\phi}(a)=J_\theta(a)-\mathbb E_{v\sim \color{red}{q_\phi}}[J_\theta(v)].
$$

**This is the core step of Score Function Centering.**

Note that there are two distributions and model parameters here, and they should not be mixed up. $J_\theta(v)$ is still:

$$
J_\theta(v)=\nabla_\theta z_\theta(v),
$$

so it is still the logit Jacobian of the **current trainer**. We are not going back to the old generator to compute gradients, and we are not updating $\phi$. The only thing that changes is the probability distribution used for centering:

$$
\mathbb E_{v\sim p_\theta}[J_\theta(v)]\quad\longrightarrow\quad\mathbb E_{v\sim \color{red}{q_\phi}}[J_\theta(v)].
$$

You can remember it as this: **the generator decides "how to average", the trainer decides "which parameter direction to update".**

With that single change, the zero-mean property comes straight back. Because the real action was from:

$$
a\sim q_\phi,
$$

we get:

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=\mathbb E_{a\sim q_\phi}[J_\theta(a)]-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

$a$ and $v$ are still two dummy variables, and this time they finally come from the same distribution again:

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=\mathbb E_{q_\phi}[J_\theta]-\mathbb E_{q_\phi}[J_\theta]=0.
$$

So we recover:

$$
\mathbb E_{a\sim q_\phi}[(R(a)-b)\tilde s_{\theta,\phi}(a)]=\mathbb E_{a\sim q_\phi}[R(a)\tilde s_{\theta,\phi}(a)].
$$

In other words, under **the distribution $q_\phi$ that actually produced the rollout**, the baseline once again stops producing an extra expected gradient of its own.

This is the most essential intuition of Score Function Centering: **align the centering distribution of the score with the distribution that actually generates the actions.**

### 3.1 But there is a big problem in practice: how do we get $\mathbb E_{q_\phi}[J_\theta]$?

If you are really training an LLM and you use the method above, you immediately run into a very practical problem:

$$
\mathbb E_{v\sim q_\phi}[J_\theta(v)]=\sum_{v\in\mathcal V}q_\phi(v)J_\theta(v).
$$

Suppose the vocabulary has 128K tokens. Then this expectation appears to require knowing the generator's probability for **all 128K tokens** at this prefix:

$$
q_\phi(1),q_\phi(2),\ldots,q_\phi(V).
$$

The problem is that this information only exists at the moment the generator produces the token, while the trainer that actually runs backward may be on another machine, in another pipeline stage, or even dozens of train steps later.

Of course you could say: then I will just store the generator's entire vocabulary distribution and send it to the trainer, problem solved.

In theory you can. In engineering you cannot.

Because for **every token position** in the rollout you would have to store an extra distribution of length vocabulary size. Suppose the vocab is 128K and a rollout has thousands or even tens of thousands of tokens, then multiply by batch size, by the number of generators, and by the number of parallel rollouts. That communication and memory overhead very quickly becomes completely unacceptable.

So the practical problem becomes:

**We want the expectation under $q_\phi$, but we cannot move the complete $q_\phi$ to the trainer. What do we do?**

The answer is Monte Carlo again: **since the expectation is an average over $q_\phi$ in the first place, let us sample a few more tokens from $q_\phi$ on the generator side.**

### 3.2 Alternative tokens: do not give the trainer the whole distribution, a few samples are enough

Suppose the token actually chosen by the generator and entered into the rollout is:

$$
a\sim q_\phi.
$$

Besides this primary action, at **exactly the same prefix** we additionally sample $K$ tokens from the generator distribution:

$$
v_1,v_2,\ldots,v_K\sim q_\phi.
$$

These tokens are never actually appended to the trajectory, and they do not change the subsequent generation. They only tell the trainer:

**"This is roughly what my distribution looked like at this prefix."**

Different places give these tokens different names, alternative tokens, alter tokens, negative tokens, auxiliary samples, and so on. They are all doing the same thing in essence: **using a small number of samples to describe the generator's local action distribution at the time.**

If these $v_k$ really are sampled IID from $q_\phi$, then:

$$
\mathbb E_{v\sim q_\phi}[J_\theta(v)]\approx\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k).
$$

So the SFC score can be approximated as:

$$
\hat s_{\theta,\phi}(a)=J_\theta(a)-\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k).
$$

And this Monte Carlo estimator is unbiased for the behavior-centered score we actually want:

$$
\mathbb E_{v_1,\ldots,v_K\sim q_\phi}\left[\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k)\right]=\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

So the generator does not have to hand the whole $q_\phi$ to the trainer. It only needs to attach a small amount of extra information at each token position, such as:

$$
[a,v_1,v_2,\ldots,v_K].
$$

More realistically, the log probability of these tokens under the generator is stored alongside them:

$$
[\log q_\phi(a),\log q_\phi(v_1),\ldots,\log q_\phi(v_K)].
$$

This information then travels with the rollout all the way from the generator to the trainer. So you can picture an ordinary RL rollout token as originally carrying only:

$$
\text{token id}+\text{reward/advantage}+\log q_\phi(a),
$$

while after SFC it also carries a very small **local policy sketch**:

$$
\{v_k,\log q_\phi(v_k)\}_{k=1}^K.
$$

This local policy sketch tells the trainer: **besides the action it actually chose, which other tokens the generator considered possible at the time.**

**Clever as you are, you will have noticed that if you want to churn out a few conference papers, you can swap in a few different sampling methods for this generator distribution, see which beats which, and there are several papers right there (LoL).**

### 3.3 Hold on, do we really have to compute $J_\theta(v)$ explicitly in the trainer?

At this point your head may be starting to hurt, because:

$$
J_\theta(v)=\nabla_\theta z_\theta(v)
$$

and $\theta$ is every parameter of the whole LLM. For a 70B model, $J_\theta(v)$ is itself a vector with tens of billions of dimensions.

Do we really have to materialize a huge Jacobian for every alternative token and then compute:

$$
J_\theta(a)-\frac{1}{K}\sum_kJ_\theta(v_k)?
$$

Of course not. Here $J_\theta$ is mainly a **mathematical object that helps us understand the algorithm**. When we actually write the code, all we need is to construct a scalar surrogate and let autograd differentiate it for us. Go back to the gradient SFC wants:

$$
J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)].
$$

Because:

$$
J_\theta(v)=\nabla_\theta z_\theta(v),
$$

we have:

$$
J_\theta(a)-\mathbb E_{v\sim q_\phi}[J_\theta(v)]=\nabla_\theta\left(z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)]\right).
$$

So the scalar we actually need to construct is very simple:

$$
c_{\theta,\phi}(a)=z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)].
$$

With the $K$ alternative samples from before:

$$
\hat c_{\theta,\phi}(a)=z_\theta(a)-\frac{1}{K}\sum_{k=1}^Kz_\theta(v_k).
$$

Then multiply by the advantage, just like an ordinary policy gradient:

$$
L_{\mathrm{SFC}}=-A(a)\hat c_{\theta,\phi}(a).
$$

Run one ordinary backward on it:

$$
\nabla_\theta L_{\mathrm{SFC}}=-A(a)\left(J_\theta(a)-\frac{1}{K}\sum_{k=1}^KJ_\theta(v_k)\right).
$$

This is exactly the SFC gradient we wanted, and real code works the same way. The rough logic is:

1. the generator tells the trainer which token is the primary one and which are the alternative ones;
2. the trainer runs a normal forward of the current model;
3. gather the logits of those tokens from the current trainer's logits;
4. compute `primary_logit - mean(alternative_logits)`;
5. multiply by the advantage;
6. backward.

### 3.4 Why do we not even need a full log-softmax here?

All along we have been talking about the score:

$$
\nabla_\theta\log p_\theta(a),
$$

so why does the SFC implementation suddenly work directly on logits, $z_\theta(a)-\mathbb E_{q_\phi}[z_\theta(v)]$?

The reason is that the Softmax normalization term cancels exactly. We can start from this quantity:

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)].
$$

Because:

$$
\log p_\theta(v)=z_\theta(v)-\log Z_\theta,
$$

where:

$$
Z_\theta=\sum_u e^{z_\theta(u)},
$$

we get:

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)]=z_\theta(a)-\log Z_\theta-\mathbb E_{v\sim q_\phi}[z_\theta(v)-\log Z_\theta].
$$

For a fixed prefix, $\log Z_\theta$ has nothing to do with which $v$ was sampled, so:

$$
\mathbb E_{v\sim q_\phi}[\log Z_\theta]=\log Z_\theta.
$$

The two normalization terms therefore cancel outright:

$$
\log p_\theta(a)-\mathbb E_{v\sim q_\phi}[\log p_\theta(v)]=z_\theta(a)-\mathbb E_{v\sim q_\phi}[z_\theta(v)].
$$

This is very nice from an engineering point of view: **for this SFC loss path, we only need the logits of the selected tokens, and we do not need to materialize the log-softmax over the whole vocabulary for its sake.**

To be a little more rigorous: this does not mean the LLM forward never needs full vocabulary logits again. It only means that **the SFC loss and gradient path itself can depend on a small number of selected token logits.**

### 3.5 Purely random sampling is not good enough: why do people use head plus tail in practice?

If we simply sample at random from $q_\phi$:

$$
v_1,\ldots,v_K\sim q_\phi,
$$

we do of course already get an unbiased estimator. But the next-token distribution of an LLM is often extremely uneven: a small number of top tokens can take up the vast majority of the probability mass, and behind them is a very long tail.

Suppose that at some prefix:

$$
q_\phi(A)=0.40,\qquad q_\phi(B)=0.25,\qquad q_\phi(C)=0.15,
$$

the first three tokens alone already account for 80% of the probability mass.

If we still treat every token as an ordinary Monte Carlo sample here, it is a bit wasteful: $A/B/C$ clearly matter almost every time, and yet we are still leaving it to chance whether we happen to draw them.

So a very natural approach is: **compute the high-probability head exactly, and use Monte Carlo sampling only for the long tail.**

Suppose the generator picks the top-$H$ tokens to form the head set:

$$
\mathcal H=\operatorname{TopH}(q_\phi),
$$

whose total probability mass is:

$$
\alpha=\sum_{h\in\mathcal H}q_\phi(h).
$$

Then any expectation can be split as:

$$
\mathbb E_{v\sim q_\phi}[f(v)]=\sum_{h\in\mathcal H}q_\phi(h)f(h)+(1-\alpha)\mathbb E_{v\sim q_\phi(\cdot\mid v\notin\mathcal H)}[f(v)].
$$

The first term is computed exactly, and for the second we sample $M$ tokens from the conditional tail distribution:

$$
t_1,\ldots,t_M\sim q_\phi(\cdot\mid v\notin\mathcal H).
$$

So:

$$
\mathbb E_{v\sim q_\phi}[f(v)]\approx\sum_{h\in\mathcal H}q_\phi(h)f(h)+(1-\alpha)\frac{1}{M}\sum_{m=1}^Mf(t_m).
$$

In SFC, substitute:

$$
f(v)=J_\theta(v)
$$

and you are done.

You can understand this method with one plain sentence:

**I want all of the few most important tokens; for the long tail, which has tiny probabilities but an enormous number of entries, I use a few samples to stand in for it.**

This head/tail sampling is nothing new in itself, it is essentially stratified Monte Carlo. But for a distribution like an LLM's, with a large vocabulary and probability highly concentrated, it is very practical.

### 3.6 So what does SFC actually require in a real system?

At this point we can walk through the whole procedure again in very concrete terms.

At some prefix $x_t$, the generator has its own behavior policy:

$$
q_\phi(\cdot|x_t).
$$

It samples, as usual, one token that really enters the trajectory:

$$
a_t\sim q_\phi.
$$

At the same time it keeps a small number of extra tokens that describe the current distribution. The simplest option is to sample $K$ alternative tokens directly from $q_\phi$. More realistically, it can keep the top-$H$ head tokens and sample a few more from the conditional tail.

The generator then passes these token ids, their $q_\phi$ probabilities or log probabilities, and its own model version at the time, along with the rollout, to the trainer.

Once the trainer receives this data, it runs a normal forward with the **current parameters $\theta$** and then gathers only:

$$
z_\theta(a_t),\qquad z_\theta(v_{t,1}),\ldots,z_\theta(v_{t,K}).
$$

It uses these selected logits to build a sparse estimator of:

$$
\mathbb E_{q_\phi}[z_\theta(v)]
$$

and finally forms:

$$
z_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[z_\theta(v)].
$$

Multiply by the advantage:

$$
L_t=-A_t\left(z_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[z_\theta(v)]\right),
$$

and then run a normal backward.

Autograd will naturally give:

$$
-A_t\left(J_\theta(a_t)-\widehat{\mathbb E}_{q_\phi}[J_\theta(v)]\right).
$$

So if I had to sum up Score Function Centering in one sentence: **the generator no longer tells the trainer only "which token I finally chose", it also brings back a little information about "what my local distribution looked like at the time"; the trainer then uses these sparse samples to turn a score that was centered on $p_\theta$ into one centered on $q_\phi$, the distribution that actually generated the data.** The problem it solves is:

**We want to compute $\mathbb E_{q_\phi}[J_\theta]$, but we cannot move the complete $q_\phi$ from the generator to the trainer.**

So all we can do is find a way to keep, on the generator side, a sparse approximation that is cheap enough and still informative enough.

One last boundary is worth stressing. What SFC restores is:

$$
\mathbb E_{a\sim q_\phi}[\tilde s_{\theta,\phi}(a)]=0,
$$

that is, the zero-mean and baseline-invariance property under the behavior distribution. It does not mean that:

$$
\tilde s_{\theta,\phi}(a)=\nabla_\theta\log p_\theta(a).
$$

As long as:

$$
q_\phi\neq p_\theta,
$$

these two are generally not the same quantity. So SFC is not "we fixed the off-policy bias". More precisely, what it does is:

**given that trainer-generator mismatch already exists, rebuild a centered gradient signal that is aligned with the real sampling distribution $q_\phi$, so that we at least get back the two very important structures of a zero-mean score and baseline invariance.**
