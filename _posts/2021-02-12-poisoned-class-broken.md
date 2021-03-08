---
layout:     post
title:      "Poisoned Classifiers Are Not Only Backdoored, They Are Fundamentally Broken!"
subtitle:   "Paper Summary"
date:       2021-02-12
author:     "Felix Zhang"
header-img: "img/in-post/2021-02-12-poisoned-class-broken/bg.JPG"
catalog: true
tags:
   - Adversarial Machine Learning
   - Paper Summary
   - Backdoor Attack

---

# Poisoned Classifiers Are Not Only Backdoored, They Are Fundamentally Broken!

针对backdoor攻击所产生的一个疑问是，是不是只有参加训练的特定的那个backdoor trigger才能在test时触发test攻击，换言之，training过程中所学习到的backdoor trigger pattern和所加入的trigger pattern有没有差异呢？本篇文章的引入点就在这里。

> 举一个例子，有一个电影院每次都有很多人在排队，而电影院有一个后门，看后门的是个傻儿子（待训练的model），电影院老板为了教傻儿子放别人不用排队走后门（有backdoor的学习），就向傻儿子展示，每次有人走到后门给了他100美元（trigger），他就放别人过去这个后门（training过程触发trigger），不给100美元的只能去正常排队（正常识别）。老板带了傻儿子三天，发现傻儿子学的差不多了，老板就换傻儿子来看门，发现每次有人给100美元，傻儿子就放人走后门了，老板觉得傻儿子学会了，就放心的走了。但问题是，傻儿子真的学会了吗？
>
> 老板走后的第二天有一个顾客拿着100人民币（fake trigger）过来了，傻儿子竟然放顾客过去了；第三天这个顾客拿着100泰铢（fake trigger）过来了， 傻儿子依然放顾客过去了；第四天这个顾客拿着一张纸上写着100（fake trigger），傻儿子又放顾客过去了。这个顾客发现，傻儿子并没有完全学到“100美元”这个trigger的所有特征，只是学到了“100”或者“写着100的纸”这个特征。但是如果老板测试的方法只是让人拿着100美元看傻儿子是否会让他通过，那老板永远也不会发现傻儿子学到的trigger pattern（100美元）和他希望傻儿子学到的trigger pattern（）存在差异。这就会引发trigger泄露。

基于这一想法我们继续推理，如果模型学习到的trigger pattern不是training set中的trigger，那么在test set中触发backdoor的最优pattern应该另有其人。如果能找到这个alternative optimal trigger，在测试中它的效果应该会比training set中的trigger更好。在我们认同了这一基本想法后，就只需要做实验验证它。整个实验最困难的部分就是如何找到真正的trigger pattern，也是本文的关键。

## Perceptually Aligned Gradients

首先从Robustified Model的特点讲起。如果我们对一个没有任何防御功能的model优化其adversarial example，那么我们得到的很可能是肉眼上无规律的细微噪声。现在我们面对一个已经防御过后的model——称为robustified model——它的adversarial example呈现出哪些特点呢？第一个特点是perturbation的L范数很大，因为小的perturbation已经被防御手段阻挡在外了，另一个特点是它的perturbation在肉眼上看起来已经不再像之前一样没有规律了，它呈现出很强的模仿其他class视觉特点的pattern，这就是所谓的 *perceptually aligned gradients*。对于一个被backdoor攻击了的模型，这里所谓的其他class既包含原始的其他类别，也包含特殊的那一个backdoor触发的类别，后者在一定程度上是离原始图片最近的一类（因为把一只猫perturb成一只狗和perturb成一只带trigger的猫，显然后者更简单）。

> 一般不具有抗攻击性的模型，它们的adversarial example都是很隐蔽的，比如将狗图攻击成猫图，人眼一般是看不出来这个狗图中有猫的特征的，perturbation非常隐蔽或者被攻击类别不会和目标类别的视觉特征直接关联。但是已经robustified过的模型就不一样了，一般的攻击是攻不进去的，能攻进去的adversary视觉上一般都具有target class的明显的特征，不再具有隐蔽性了。

因此，如果我们把一个被backdoor攻击过的模型做防御（robustify），同时再对这个防御加强后的模型在大/无 L范数的约束下优化出adversarial example，那么这个adversarial example上很有可能就带有隐藏的trigger pattern。具体过程如下图：

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-12-poisoned-class-broken/1.png?raw=true)

剩下的几个问题：

* 如何robustify一个backdoored模型？
* 用什么方法去优化出robustified model的adversarial example？
* 如何从探索到的adversarial example中提取trigger pattern？
* 对众多backdoor attack这种假设都成立吗？如Clean Label 和 Hidden Trigger Backdoor Attack。

在本文中，作者采用robustify的方法是randomized smoothing + denoiser（Denoised Smoothing），将一个含有backdoor的普通模型增强成鲁棒模型。

第一个初步的实验，作者使用混接彩块（下图中的Trigger A）作为training trigger，使用上述方法求出robustified model的adversarial example，结果发现真的会有肉眼可见的彩色斑块出现，且随着perturbation大小限制放宽($\epsilon$从20到60)彩斑越来越明显。但是与之形成对比的是，没有backdoor的clean classifier的adversarial example却未发现类似的彩斑。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-12-poisoned-class-broken/2.png?raw=true)

进一步实验，作者将单色彩块作为training trigger，结果发现adversarial example中这种视觉特点更加明显。

![](https://github.com/StarkSchroedinger/StarkSchroedinger.github.io/blob/master/img/in-post/2021-02-12-poisoned-class-broken/3.png?raw=true)

这也印证了之前的猜想，说明adversarial example的visual trigger确实和training trigger有很强的关联。

## Breaking Poisoned Classifiers

现在从上述robustified model的adversarial example中的noise pattern构造alternative trigger。作者采用手动构造的方式——虽然直白简单，但是却很有效。作者从adversarial example中选出一个有代表性的pixel或者直接在adversarial example的图上截取一部分来构造新的trigger。这种做法取得了显著的成效，新构造的trigger成功得让原模型出发了backdoor，效果并不比原模型差。

为了避免此中效果是源自transferability of adversarial examples，作者对Clean model也做了相似的实验，但并没有发现类似的效果。

## Conclusion

本文揭示了“trigger 泄漏”的现象，即一个被backdoor attack攻击过的模型，可能会遭到诸多潜在的替代trigger攻击。这些替代trigger的攻击效果可能并不亚于、甚至优于training trigger。作者发现对robustified model优化得到的adversarial example中包含该模型可能存在的trigger pattern，作者进一步成功地从中提取出了alternative trigger。

另外作者指出的模型在遭受backdoor攻击时学习到a spectrum of potential backdoor的观点我是持保守观点的，多种trigger都对poisoned模型有用是由于都没有达到模型真正学习到的trigger pattern，应该有一种最优trigger能够使触发效果最好。

另一个角度，频域上的backdoor是否能避免这一攻击？

