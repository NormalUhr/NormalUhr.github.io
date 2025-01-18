# A Review on the Evolvement of Load Balancing Strategy in MoE LLMs: Pitfalls and Lessons

Hello everyone, welcome to my nerdy-yet-fun exploration of how **Mixture-of-Experts (MoE)** has evolved over the years—particularly focusing on the clever, sometimes messy, always interesting ways researchers have tackled load balancing. This post is very much a “lab notebook” of my ongoing exploration: a mix of academic analysis and personal reflections.

The story begins with GShard—back when people realized that models with billions (or trillions) of parameters can be smartly *“sparsified”* to train faster while preserving high accuracy. Since then, we’ve witnessed a cascade of innovations. Here, I want to piece together how we went from **GShard** to the latest innovations like **DeepSeek-V3**—what each one contributed, what pitfalls came up, and what big questions remain unanswered.

## Introduction

### Why Sparse Mixture-of-Experts (MoE)?

So, let’s start with a little bit of context. MoE architectures took the world by storm when folks realized you could dramatically increase model capacity (parameters) without linearly inflating the amount of computation (FLOPs). The big idea is that, for each token, we only “activate” a small subset of the total parameters—i.e., a few experts—rather than forcing every parameter to take part.

However, the dark side of this approach quickly reared its head: if you’re only sending tokens to a subset of experts, how do you keep the load “balanced” so that no single expert gets hammered with tokens while others stay idle? This is load balancing in a nutshell, and it’s quite the puzzle to solve at scale.

###  What This Post Is About

I’m going to walk through some of the landmark MoE systems, starting with GShard (the earliest large-scale MoE system that went mainstream) and meandering all the way to the brand-new DeepSeek-V3. While we’ll cover the usual suspects (like Switch Transformer and GLaM), I want to highlight the pitfalls that each system ran into—and how newer architectures overcame them.

If you’re reading this to glean practical takeaways, great: I’ll try to maintain enough academic rigor so it’s useful for advanced practitioners and researchers. But hopefully it’ll stay lighthearted enough that you don’t nod off after a few paragraphs—this is a blog, not a final exam!

### Key Themes I’ve Noticed

1. **Routing Approaches**: top-2 gating, single-expert gating, top-K gating, correlation-aware gating… yes, we love gating terminology!
2. **Auxiliary Loss**: helps push balanced usage of experts, but can also hamper performance if it’s too heavy-handed.
3. **Capacity Constraints**: “capacity factor” is a fancy name for “how many tokens can each expert handle before we drop the extras.”
4. **Implementation Details**: from “random dispatch” to hierarchical all-to-all. The HPC (high-performance computing) perspective is super relevant.
5. **Scalability**: we’re talking thousands of experts in some cases, so distributed computing overhead is non-trivial.


## Historical Progression: From GShard to Switch

### GShard: The Pioneer

**GShard** (introduced by Google) is widely cited as among the first large-scale, super-sparse MoE frameworks. It changed the conversation by showing that you could train ~600B parameter models if you carefully sharded the layers and balanced tokens among experts.

GShard’s gating approach typically selects the **top-2** experts for each token. Let’s denote:

$$\text{GATE}(x)=\text{Top2}(W_{gate}\cdot x)$$
where $x$ is the token embedding and $W_{gate}$ is the router's weight matrix. Only the top 2 experts get activated. However, to keep each expert from being overloaded, we need to introduce:

1. **Expert capacity**, $C \approx \frac{2N}{E}$ if $N$ tokens and $E$ experts. If an expert is overloaded beyond capacity, some tokens are dropped (or overflowed to the next layer).
2. **An auxiliary load-balancing loss**, often of the form
$$ \mathcal{L}_{\text{aux}} \ \sum_{e=1}^E f_e P_e$$
    where $f_e$ is the fraction of tokens routed to expert $e$, and $P_e$ is the average gating probability for expert $e$. This loss nudges the system toward distributing tokens more evenly across experts.
**3. Local groups** so that not every token competes with every other token globally.

**Pitfall**: You guessed it—dropping tokens is not super glamorous. If tokens exceed capacity, they might get incomplete processing. Also, the overhead of top-2 gating and random dispatch can get heavy at scale. Also, the over-dependence on an auxiliary loss sometimes forced a “fake” distribution of tokens, hurting specialized learning. But still, GShard proved that MoE could be done and that it’s worth the trouble. The concept of capacity constraints was spot on and we still see that in almost every subsequent MoE method.

### Switch Transformer: When “Less is More”

Switch Transformer essentially said, “Hey, let’s only route each token to one expert.” This made the gating simpler (pick whichever expert has the highest gating logit) and drastically reduced the compute overhead. The gating function goes as:
$$g_i(x) = \text{softmax}(W_{\text{router}} \cdot x)_i$$
and we pick
$$\text{expert_index}(x)={\text{argmax}}_i g_i(x)$$.

The primary innovation of Switch Transformer is its single-expert routing, as fewer experts activated in general gives you simpler code, and further typically faster training speeds. In order to better balance the load, they keep an auxiliary load-balancing loss akin to GShar's approach. They also define a **capacity factor** to let experts handle more tokens than naive fraction. For example,
$$C = \text{CF} \times \frac{\text{tokens per batch}}{\text{number of experts}}$$

The **gains vs. trade-offs** of Switch Transformer is rather obvious: you have better speed because you only do one feed-forward path per token, but you might risk bigger token overflow (you only have one expert to handle them!). Some tokens are "dropped" or forcibly passed to a residual pathway.

**Pitfalls and Lessons**: Single-expert routing is conceptually simpler and often faster. But if the CF (capacity factor) is set incorrectly, you might get too many tokens dropped or too many tokens assigned to one expert. Switch Transformer basically spelled out how a bit of well-chosen hyperparameter tuning can do wonders. Switch simplified MoE gating—showing that scaling up is possible even with top-1 routing. This spurred follow-up work on “which K is best?” and “how do we best handle overflow?”

## Refinements and Variations: GLaM, DeepSpeed-MoE, ST-MoE, Mixtral

### GLaM: Revisiting Top-2 with Efficiency in Mind

**GLaM** (Generalist Language Model) reintroduced **top-2 gating** but with a new spin on **energy efficiency**—reporting that it uses roughly 1/3 of GPT-3’s training energy with better zero-shot performance. They used:
$$y = \sum_{i=1}^2 g_i \cdot E_i(x),$$
where $g_i$ are gating weights and $E_i(x)$ are the two selected experts. Similarly, GLaM introduces a carefully tuned auxiliary loss to encourage an even distribution of tokens across experts. This auxiliary loss penalizes imbalanced routing by optimizing the utilization of experts:
$$\mathcal{L}_{\text{aux}}=\alpha \cdot \sum_{i=1}^E f_i \cdot p_i,$$
where $f_i$ is the fraction of tokens routed to expert $i$, $p_i$ is the average gating probability for expert $i$, and $\alpha$ is a weighting factor. To prevent overloading experts, GLaM also introduces capacity constraints, where the maximum token capacity per expert is defined as:
$$C = \frac{\text{tokens per batch}}{\text{number of experts}} \cdot \text{capacity factor}.$$
Tokens exceeding this capacity will be dropped and passed through residual connections to the next layer. A capacity factor of $1.25$ is typically used to balance token overflow and computational efficiency.

**Pitfalls and Lessons**: GLaM emphasized just how big the energy savings can be when you only activate a small fraction of the model parameters at a time. (They compared with GPT-3 and said, “Look, we’re using a fraction of the energy. Y’all should pay attention!”) Although GLaM discovered that you can indeed overshadow the cost of dense computations, you must watch out for potential imbalances in expert usage—particularly on real-world text distributions. The model’s carefully tuned gating and capacity constraints helped keep experts from overloading.


### DeepSpeed-MoE: Focusing on Inference

**DeepSpeed-MoE**, by Microsoft, is a prime example of how load balancing has matured to handle both the challenges of **token distribution** during training and efficient **expert utilization** during inference. Building on the pitfalls of earlier MoE systems, DeepSpeed-MoE introduces several innovations to address token load imbalance.

**Core Idea.** At its heart, DeepSpeed-MoE extends the MoE framework with a **flexible multi-expert and multi-data parallelism design** to optimize load balancing, particularly focusing on token-level distribution across experts. The goal is clear: ensure that no expert is overloaded while keeping training efficient and scalable across distributed GPUs.

Following Switch Transformer, DeepSpeed-MoE employs a top-1 gating mechanism. This simplifies routing and reduces computational overhead compared to top-2 or top-k gating. To prevent token imbalance, an **auxiliary load-balancing loss** is added. The loss nudges the distribution of tokens to be more uniform across experts:
$$\mathcal{L}_{aux} = \alpha \sum_{i=1}^E |f_i - \frac{1}{E}|,$$
where $𝑓_i$ is the fraction of tokens routed to expert $i$, $E$ is the total number of experts, and $\alpha$ is a tunable weight. This term discourages over-concentration of tokens on a few experts. DeepSpeed-MoE also adopts a **dynamic token redistribution strategy**: During training, DeepSpeed-MoE dynamically redistributes tokens to prevent any single expert from becoming a bottleneck. Tokens that exceed an expert's capacity are rerouted to other, less-busy experts rather than being dropped or passed to a residual pass way. To further mitigate the impact of uneven token distribution, DeepSpeed-MoE introduces the **Residual-MoE architecture**. Here, the output of the dense MLP is combined with the output from the selected expert, treating the expert output as a “residual correction”:
$$y=\text{MLP}(x) + g \cdot E(x),$$
where $g$ is the gating score and $E(x)$ is the expert output. This ensures that even underutilized experts contribute meaningfully to the model's overal output.

**Load Balancing Across GPUs.** Leveraging the observation that deeper layers benefit more from large numbers of experts, DeepSpeed-MoE utilizes more experts in later layers. While this ensures efficient parameter usage and improved model quality, this can lead to varying number of experts across layers. In such a case, a uniform degree of parallelism is inefficient because:
* Setting parallelism to the smallest number of experts leads to reduced batch sizes and increased memory requirements for GPUs handling larger layers.
* Setting parallelism to the largest number of experts causes load imbalance, where some GPUs process more experts than others.

The DeepSpeed-MoE system solves this problem by **dynamically adjusting the parallelism degree** across layers and distributing workloads optimally. For a given model, the system allows different parts of the model to use different degrees of expert and data parallelism. For example, layers with 32 experts might use 32-way expert parallelism and 4-way data parallelism, while layers with 128 experts might use 128-way expert parallelism and 1-way data parallelism. This ensures that **each GPU processes exactly one expert per layer** regardless of the total number of experts in the layer. By aligning the expert parallelism with the number of experts in each layer, the system avoids scenarios where some GPUs handle more experts than others. This avoids bottlenecks and ensures maximum utilization of resources.

**Pitfalls and Lessons.** While DeepSpeed-MoE achieves impressive results in balancing token loads, a few trade-offs remain:
* **Complexity of Configuration**: Balancing the capacity factor, auxiliary loss weight, and expert parallelism settings requires careful tuning.
* **Edge Cases in Real-World Data**: Text distributions in NLP tasks can be highly skewed, which can still strain the gating mechanism if not tuned carefully. 

Nevertheless, DeepSpeed-MoE demonstrated that token load balancing isn’t just a theoretical optimization—it’s a practical necessity for training large-scale MoE systems. By combining routing innovations with system-level optimizations, it set a new standard for efficiency and scalability in MoE training. Even if you have an amazing training pipeline, you still need to handle inference well—especially if you want real-time or interactive applications.

### ST-MoE: Capacity Factor Tuning & Router Z-Loss

**ST-MoE (Stable and Transferable Mixture-of-Experts)** marks a significant leap forward in sparse expert models, offering solutions to some of the long-standing challenges in training stability and transferability. While previous models like Switch Transformer and GLaM laid the groundwork, ST-MoE refined these ideas, addressing pitfalls with a blend of architectural innovations and hyperparameter optimizations.

One of ST-MoE's standout contributions is the **router z-loss**, designed to stabilize training without degrading quality. Sparse models often grapple with instability due to the exponential functions in routing, which amplify small numerical errors. The router z-loss mitigates this by adding a penalty for large logits in the routing network, effectively controlling their magnitude:
$$\mathcal{L}_z = \frac{1}{B} \sum_{i=1}^B(\text{log}\sum_{j=1}^N \text{exp}(x_{ij}))^2$$
Here, $B$ is the batch size, $N$ is the number of experts and $x_{ij}$ are the logits for routing. This loss not only reduces instability but also slightly improves model quality - a win-win for sparse model training.

**Tuning the Capacity Factor.** ST-MoE also emphasizes the critical role of the capacity factor (CF) in balancing efficiency and performance. To further improve load balancing, ST-MoE incorporates an auxiliary loss similar to DeepSpeed-MoE, that ensures tokens are evenly distributed across experts. 

**Pitfalls and Lessons**: ST-MoE achieves an improved Stability vs. Quality Trade-offs: Earlier approaches like GLaM and DeepSpeed-MoE made progress on load balancing but often required compromises in model quality or scalability. ST-MoE's router z-loss shows that it's possible to achieve stability without such trade-offs. However, ST-MoE is not without limitations. The complexity of tuning hyper-parameters like CF and z-loss weight demands careful experimentation. In summary, ST-MoE represents a new chapter in the evolution of MoE architectures, combining robust design principles with innovative solutions to long-standing challenges. 


### Mixtral 8x7B: Temporal Locality & Specialized Sparse Kernels

**Mixtral 8x7B** stands out as an innovative Sparse Mixture-of-Experts (SMoE) language model, built to address some of the long-standing challenges in load balancing for MoE architectures. Let’s dive into its unique approach to the per-expert token load-balancing problem and uncover the lessons it provides.

At its core, Mixtral employs a **Top-2 gating** mechanism for routing tokens: each layer includes 8 experts, with only 2 experts activated per token at a given time. This approach ensures that by limiting each token to only two experts, Mixtral effectively caps the active parameter count at 13B per token, offering a significant reduction compared to dense models like Llama 2 70B. In the meantime, experts selected can vary across tokens and layers, enhancing the model’s adaptability to different input patterns.

**Temporal Locality in Expert Assignment.** One of the most striking findings from the routing analysis is the observed temporal locality in expert assignments. Tokens often retain the same expert assignments across consecutive positions, particularly in deeper layers: In layers 15 and 31, consecutive tokens are assigned the same experts much more frequently than random distribution would predict. This phenomenon is termed **Higher Repetition Rates** and indicates structured behavior, likely tied to the input's syntactic or positional features. Temporal locality offers both opportunities and challenges: it ensures smoother transitions in token assignments, minimizing abrupt workload spikes for specific experts, but it also can lead to over-concentration of tokens on a subset of experts, especially in datasets with syntatic or positional regularities. Similar to DeepSpeed-MoE, Mixtral also adopts the **Dynamic Token Redistribution** strategy: When an expert exceeds its token capacity, excess tokens are efficiently handled by redistributing them to other less-loaded experts. 


**Mitigating GPU Overload with Sparse Kernels.** Mixtral employs specialized sparse kernels (e.g., Megablocks) to alleviate token overload. Megablocks handle variable token assignments efficiently, leveraging high arithmetic intensity to speed up computations. Tokens destined for specific experts are dynamically routed across GPUs. This partitioning strategy, while effective, requires careful load balancing to avoid GPU overloading.

**Pitfalls and Lessons.** Mixtral’s analysis of expert usage across diverse datasets underscores the importance of understanding domain-specific token distributions. If the dataset distribution changes (like if you go from news articles to code), the “locality” might vanish. So each approach has assumptions about your data.

## Next-Generation Approaches: OpenMoE, DeepSeekMoE, JetMoE, & More

### OpenMoE: Context-Independent Specialization & Drop-Towards-the-End

OpenMoE is another interesting spin on the standard top-k gating formula, with capacity constraints and an auxiliary balancing loss. But it’s famous for identifying certain quirky behaviors that arise in MoE systems over large training runs, namely **Context-Independent Specialization** and **Drop-Towards-the-End**. 
* **Context-Independent Specialization**: Tokens might bet routed more by token ID or surface-level patterns, rather than deeper semantic attributes, especailly early on in pretraining.
* **Drop-Towards-the-End**: In long sequences, capacity constraints often get triggered late in the sequence, so those later tokens are more likely to be dropped. This obviously hurts performance on tasks that rely on end-of-sequence context.

Like many other MoEs, OpenMoE adopts a top-k selection with $k=2$. Similar to GShard and Switch, a load-balance loss was adopted in the form of:
$$\mathcal{L}_b = E \cdot \sum_{i=1}^E m_i \cdot P_i,$$
where $m_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average gating probability for expert $i$. To stabalize the training, they also introduce the router loss by penalizing large logits:
$$ \mathcal{L}_z = \frac{1}{B} \sum_{j=1}^B (\text{log} \sum_{i=1}^E (f(x_j)_i)).$$

To maintain a balanced workload, OpenMoE enforces capacity constraints on each expert. This can ensure the throughput when training and deploying the MoE model with expert parallelism, i.e., distributing different experts to different GPUs. However, OpenMoE for the first time identifies the **Drop-Towards-the-End** issue, that the later tokens would be dropped if the previous tokens have filled the expert. In decoder-only MoE architecture, due to the auto-regressive nature, the later tokens in a sequence may be dropped more. This is particularly problematic for
sequential tasks like instruction-following, where later tokens may carry critical information.

**Pitfalls and Lessons:** OpenMoE taught us to watch out for distributional quirks, especially if you are focusing on tasks that rely on full sequence coverage or want strong domain adaptation. If the gating function picks up superficial patterns (like token IDs), it might not adapt well to new domains. Because capacity constraints are a per-batch mechanism, tokens at the tail end of a batch can get starved.

### DeepSeekMoE: Fine-Grained Experts & Shared Experts

Before we get to the latest version (DeepSeek-V3), let’s discuss **DeepSeekMoE**. It’s recognized for splitting each expert into finer sub-experts and isolating some **“shared experts”** that are **always activated** (i.e., bypass gating). This approach aims to reduce parameter redundancy while still giving enough diversity for specialized sub-experts.

**Fine-Grained Expert Segmentation.** DeepSeekMoE introduces the concept of fine-grained expert segmentation to enhance expert specialization. This is achieved by splitting each expert into smaller units while maintaining the total number of parameters and computational cost constant:
$$h_t^l = \sum_{i=1}^{mN} g_{i,t} \cdot \text{FFN}_i (u_t^l) + u_t^l,$$

where $mN$ denotes the total number of fine-grained experts and $g_{i,t}$ is the gating value for expert $i$. The routing mechanism selects the top-$mK$ experts for each token.

Suppose you have $mN$ total sub-experts,  with $N_r=mN$ "routed" experts plus $N_s$ "shared" experts. For the $t$-th token $u_t^l$ at layer $l$:
$$h_t^l = u_t^l + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)} (u_t^l) + \sum_{j=1}^{N_r} g_{j,t} \cdot \text{FFN}_j^{(r)} (u_t^l),$$
where $g_{j,t}$ is a gating value for sub-expert j, typically chosen among top-$K_r$.

DeepSeekMoE emplys two levels of load-balance losses to address potential routing collapes and computational bottlenecks: 
* **Expert-Level balance loss**: this loss encourages uniform token distribution across experts:
$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{mN-K_s} f_i \cdot P_i,$$
where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the average routing probability for expert $i$.
* **Device-Level balance loss**: it ensures balanced compuation across devices:
$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f'_i \cdot P'_i,$$
where $D$ is the number of devices, $f'_i$ and $P'_i$ represent the average token fractions and probabilities for device $i$, respectively.

### JetMoE: Dropless MoE & Pipeline Parallelism

Where most MoE approaches consider dropping tokens when capacity is exceeded, JetMoE tries a “dropless” approach. The design ensures that no tokens are ever flat-out discarded:
1. **Dropless MoE**: The gating mechanism is carefully managed to not exceed each expert's maximum capacity.
2. **Pipeline Parallelism**: instead of scattering experts across many devices, JetMoE keeps all experts of a layer on the same device, forming a pipeline for different layers

JetMoE adopts the top-2 routing and mechanism and has all the load balancing features previously defined, such as frequency-based auxiliary load-balancing loss and the router z-Loss. Unlike previous methods using a fixed capacity factor, JetMoE inherits from MegaBlocks, which replaces the traditional token-dropping approach with block-sparse matrix operations. MegaBlocks implements custom block-sparse GPU kernels to handle the dynamic and load-imbalanced nature of MoE computation efficiently. By constructing a block-sparse matrix topology dynamically based on expert assignments, the framework ensures all tokens are processed without being dropped, unlike traditional methods that use a fixed capacity factor.

**Pitfalls and Lessons**: Implementing dropless can get complicated. You might see overhead or suboptimal gating. If you do dropless well, you have consistent token coverage. This is attractive for tasks where dropping tokens is disastrous (like QA or code generation). But you must handle the complexities of capacity-limited gating in real time.


### Skywork-MoE: Gating Logit Normalization & Adaptive Auxiliary

**Skywork-MoE** is a high-performance Mixture-of-Experts (MoE) model with 146 billion parameters and 16 experts. The model leverages the architecture of Skywork-13B, a dense language model, using its pretrained dense checkpoints for initialization. Skywork-MoE incorporates advanced techniques like gating logit normalization and adaptive auxiliary loss coefficients to improve expert diversification and layer-specific load balancing. It introduced two neat ideas to address unbalanced experts:  
1. **Gating Logit Normalization**: They standardized gating logits before softmax, controling the "sharpness".
2. **Adaptive Auxiliary Loss Coefficients**: If a layer is dropping too many tokens, the balancing penalty is automatically increased.

The MoE layer replaces standard FFNs in transformers with multiple experts, selectively activating the top-k most relevant experts for each input token.


**Auxiliary Loss for Load Balancing.** To prevent routing collapse, where a few experts dominate, Skywork-MoE employs an auxiliary loss:

$$\mathcal{L}_{\text{aux}}=\sum_{j=1}^n (\frac{1}{n} - \frac{1}{T}\sum_{i=1}^T g_{ij})^2,$$

where $n$ is the number of experts, $T$ is the token batch size, and $g_{ij} is the probability of token $i$ being routed to expert $j$. The auxiliary loss ensures even token distribution across experts.

**Gating Logit Normalization.** To improve expert discrimination, Skywork-MoE introduces gating logit normalization:

$$z=Wx+b, z'=\lambda \frac{z-\mu}{\sigma},$$

$$g = \text{softmax}(z'),$$

where $\mu$ and $\sigma$ are the mean and standard deviation of $z$; $\lambda$ is a scaling factor controlling the sharpness of the output distribution. This normalization enhances the gating mechanism’s ability to differentiate experts, reducing entropy in the gate outputs.

**Adaptive Auxiliary Loss Coefficients.** Skywork-MoE employs a dynamic approach to adjust auxiliary loss coefficients $\alpha^{(l)}$ for each MoE layer $l$:

$$\alpha_{i+1}^{(l)} = \beta \alpha_i^{(l)} + (1- \beta)\xi d_i^{(l)},$$

where $d_i^{(l)}$ is the token drop rate for layer $l$ at iteration $i$, $\xi$ is a sensitivity parameter, and $\beta$ is a smoothing factor. This adaptation ensures balanced load distribution without over-regularizing already balanced layers.


**Pitfalls and Lessons.** On the one hand, Adapting $\alpha^{(l)}$ is helpful—some layers might be balanced already, while others need a stronger push to distribute tokens. So a one-size-fits-all auxiliary loss can be suboptimal. In the meantime, the hyper-parameter tuning in gating logit normalization could be tricky. if $\lambda$ is set too high, gating probabilities might become too “sharp,” forcing tokens into an extreme distribution. Too low and experts might not specialize enough.


### DeepSeek-V3: Bias-Based Auxiliary-Loss-Free Strategy

Finally, **DeepSeek-V3** is the latest iteration, and it’s considered cutting-edge because it tries to remove large auxiliary losses and replace them with a more direct, bias-based balancing approach. If you want to talk about advanced load balancing, DeepSeek-V3 is a prime example.

**Model Architecture.** DeepSeek-V3 employs the DeepSeekMoE architecture for Feed-Forward Networks (FFNs). Compared with
traditional MoE architectures like GShard, DeepSeekMoE introduces finer-grained experts and isolates some experts as shared ones. The FFN output for the $t$-th token, denoted as $h′_t$, is computed as follows:

$$h_t^l = u_t^l + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)} (u_t^l) + \sum_{j=1}^{N_r} g_{j,t} \cdot \text{FFN}_j^{(r)} (u_t^l),$$

where

$$ g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}},$$ 

$$g'_{i,t}=s_{i,t}, \, \text{if} \, s_{i,t}\in \text{TopK}(\{s_{j,t}|1 \leq j \leq N_r\}, K_r), \, \text{else} \, 0$$

$$s_{i.t}=\sigma(u_t^\top e_i).$$

Here, similar to DeepSeek-MoE, $N_s$ and $N_r$ are the number s of shared and routed experts; $K_r$ is the number of activated routed experts; $g_{i,t}$ is the gating value; $s_{i,t}$ represents the token-to-expert affinity; and $e_i$ is the centroid vector of the $i$-th routed expert; and $\sigma$ is the activation function. 


**Auxiliary-Loss-Free Load Balancing Strategy.**  Traditional MoE models often experience routing collapse due to unbalanced expert loads, reducing computational efficiency. Conventional solutions utilize auxiliary losses to encourage balance, which can impair performance if overly emphasized. To address this, DeepSeek-V3 introduces an auxiliary-loss-free strategy, adding a bias term $b_i$ for each expert to adjust affinity scores:

$$g'_{i,t}=s_{i,t}, \, \text{if} \, s_{i,t} + b_i\in \text{TopK}(\{s_{j,t} + b_i|1 \leq j \leq N_r\}, K_r), \, \text{else} \, 0.$$

The bias term $b_i$ is dynamically updated during training:
$$b_i \leftarrow b_i - \gamma, \quad \text{if expert } i \text{ is overloaded},$$

$$b_i \leftarrow b_i + \gamma, \quad \text{if expert } i \text{ is underloaded},$$

where $\gamma$ is the bias update speed. This strategy ensures balanced expert loads throughout training without the performance degradation associated with auxiliary losses.

**Complementary Sequence-Wise Auxiliary Loss.** To prevent extreme imbalance within individual sequences, a sequence-wise balance loss is also employed:
$$\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i,$$
where:
$$f_i = \frac{N_r}{K_r T}\sum_{t=1}^T \mathbb{I}(s_{i,t} \in \text{TopK}(\{s_{j,t} | i \leq j \leq N_r\}), K_r),$$
$$s'_{i,t}=\frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}, P_i = \frac{1}{T}\sum_{t=1}^T s'_{i,t}.$$
Here, $\alpha$ is a hyper-parameter with a small value, $\mathbb{I}$ is an  indicator function, and $T$ denotes the sequence length.

**Dynamic Routing and Node-Limited Strategy.** DeepSeek-V3 also employs a node-limited routing mechanism to reduce communication costs during training. Each token is sent to at most $M$ nodes, determined by the highest $K_r/M$ affinity scores for experts distributed on each node. This approach maintains nearly full computation-communication overlap while improving scalability.

**Pitfalls and Lessons.** If $\gamma$ (the bias update speed) is too large, the gating might thrash around. If it’s too small, you might not adapt quickly to changes in token distribution. Nevertheless, this approach can maintain balanced loads with minimal interference to the main training objective. It's arguably a cleaner approach than a heavy-handed global auxiliaryterm. DeepSeek-V3 exemplifies a new wave of MoE thinking-stepping away from large auxiliary regularizations to more subtle, dynamic, and locally corrective balancing.
















## Emerging Trends & Observations

In tracing the path from GShard to DeepSeek-V3, a few overall trends have become clear:

- **Gating is Getting Craftier**  
  We started with simple top-2 gating (GShard), moved to single-expert gating (Switch), and have since explored correlation-based, bias-based, and more elaborate routing. Researchers are continually seeking that sweet spot between complexity and efficiency.

- **Rethinking Auxiliary Loss**  
  Early on, methods like GShard and Switch heavily relied on auxiliary losses to prevent expert overload. Lately, some (like DeepSeek-V3) are minimizing or dropping it in favor of more direct, dynamic solutions to manage balancing.

- **Capacity Constraints & Dropping**  
  There’s a spectrum between “dropless” approaches like JetMoE and designs that rely heavily on capacity factors (Switch, GLaM). Neither extreme is a one-size-fits-all solution; each dataset or use case may tilt the balance differently.

- **Training vs. Inference**  
  Training-era load balancing doesn’t always solve inference-era bottlenecks. Systems like DeepSpeed-MoE highlight specialized strategies (token grouping, dynamic node parallelism) to keep inference from becoming a nightmare.

- **Multi-Dimensional Parallelism**  
  Pipeline parallel, tensor parallel, expert parallel: HPC is now the norm for MoE. We’re seeing more flexible ways to combine these parallelisms, adjusting them per layer to squeeze out every bit of performance.


## Pitfalls & Lessons Learned

Load balancing in MoE is a double-edged sword—go too far, and you hamper the model’s main objective; go too light, and half your experts might sit idle. Here are the key pitfalls and what we can learn:

- **Routing Collapse & Over-Specialization**  
  If a few experts take in most tokens, you’re wasting parameters. Good gating plus mild balancing losses (or bias corrections) can stave off collapse.

- **Capacity Factor Tuning**  
  Set it too high and you get minimal drops but waste compute. Set it too low and you drop tokens left and right. Tuning CF is an art form—especially with large or skewed datasets.

- **Over-Reliance on Auxiliary Loss**  
  Strong balancing losses can overshadow the language modeling objective. Balancing is critical, but using it too aggressively can stunt specialized learning.

- **Inference-Time Bottlenecks**  
  Balancing for training doesn’t automatically translate to balanced inference. If certain experts get hammered at inference, that kills latency. Strategies like hierarchical routing and dynamic token grouping (à la DeepSpeed-MoE) can help.

- **Domain Adaptation Challenges**  
  Gating often locks in certain patterns after pretraining. If the domain shifts (e.g., from news to code), that gating logic might not adapt well unless you carefully re-train or tune.


## Conclusion

The journey from **GShard** to **DeepSeek-V3** has shown that **load balancing in MoE** has grown from a side note into a central piece of the puzzle. GShard popularized the top-2 gating approach and capacity constraints; Switch Transformer simplified routing with top-1; GLaM zeroed in on energy efficiency; DeepSpeed-MoE demonstrated robust balancing for both training and inference; ST-MoE introduced z-loss for stability; Mixtral leveraged temporal locality; and so on—culminating in more dynamic, bias-based, or correlation-based approaches such as **DeepSeek-V3**.

**Main Takeaway**: Perfect load balancing is a moving target. Push it too hard, and you hurt model performance. Ignore it, and your super-giant model ends up idling half its experts. We’ll likely see further integration with HPC strategies, more adaptive gating mechanisms, and new solutions for the ever-pesky inference bottleneck.

As the field marches onward, we’ll likely see more synergy with HPC techniques, more adaptive gating networks, and new ways to handle inference-time constraints. It’s an exciting time for MoE researchers—and I’m definitely looking forward to the next wave of breakthroughs.

Thanks for reading, and feel free to drop me a line if you have any thoughts, questions, or improvements to share. Until the next MoE adventure—happy gating!

