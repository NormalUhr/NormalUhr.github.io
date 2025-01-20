# From Zero to Reasoning Hero: How DeepSeek-R1 Leverages Reinforcement Learning to Master Complex Reasoning

*It is well said that 2024 was the year of the agent, but 2025 is shaping up to be the year of reinforcement learning. And DeepSeek-R1 just proves that. It also underscores how an "open" AI company makes much more contributions than OpenAI to the open-source community* 

## 1. Introduction

Since the blockbuster release of DeepSeek-V3, DeepSeek has been the shining star of the LLM community. Enthusiasts and experts alike have eagerly awaited the open-source preview of "DeepSeek-R1-Lite." And here it comes, making a grand entrance in the first month of 2025—ready to redefine how we think about AI reasoning. DeepSeek-R1 breaks the norm. This new approach uses massive reinforcement learning (RL)—sometimes **without any** supervised warm-up—to unlock emergent reasoning capabilities, including extended chain-of-thought (CoT), reflection, verification, and even “aha moments.”

In this post, we explore two groundbreaking models in the DeepSeek lineage:
- **DeepSeek-R1-Zero**: A model that learns complex reasoning behaviors purely through reinforcement learning without any supervised fine-tuning, showing emergent abilities like extended chain-of-thought, reflection, and self-correction.
- **DeepSeek-R1**: Building on R1-Zero, this version incorporates a small amount of high-quality "cold-start" data alongside iterative reinforcement learning and supervised fine-tuning to produce more coherent, user-friendly outputs while maintaining state-of-the-art reasoning performance.

By comparing these models, their training strategies, and the underlying mathematics, we highlight how reinforcement learning is transforming LLM capabilities.

In this post, we will delve into:
- **How DeepSeek-R1-Zero** achieved near state-of-the-art reasoning performance *without any supervised data*.
- **Why DeepSeek-R1** combines a small “cold-start” dataset with iterative RL and supervised fine-tuning to achieve even better user-friendly outputs.
- **How distillation** from DeepSeek-R1’s advanced reasoning patterns can transform smaller dense models into powerful mini “reasoning engines.”
- **Lessons** learned from exploring different RL mechanisms and why certain approaches fell short in large-scale experiments.

Consider this blog a technical lens into the biggest leaps (and near misses) of the DeepSeek-R1 pipeline.

---

## 2. Motivations and Background

### 2.1. Why Pure RL for Reasoning?

Traditionally, major leaps in LLM reasoning have come from providing large amounts of carefully annotated data. DeepSeek-R1 questions that assumption. The key hypothesis is simple yet bold: *Can we just reward the model for correctness and let it discover the best way to think on its own?* By eliminating SFT from the start (in the DeepSeek-R1-Zero case), the research team lets the LLM find its own chain-of-thought patterns purely from reward signals.

The DeepSeek-R1-Zero approach uses the Group Relative Policy Optimization (GRPO) algorithm, which optimizes the policy without a critic model, saving computational resources. The core of GRPO's update rule is as follows:

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Bigg[
& q \sim P(Q),\ \{o_{i}\}_{i = 1}^{G} \sim \pi_{\theta_{\text{old}}}(O | q) \\
& \cdot \frac{1}{G} \sum_{i = 1}^{G}\Bigg(\min \Bigg(\frac{\pi_{\theta}\left(o_{i} | q\right)}{\pi_{\theta_{\text{old}}}\left(o_{i} | q\right)} A_{i},\ 
\text{clip}\Big(\frac{\pi_{\theta}\left(o_{i} | q\right)}{\pi_{\theta_{\text{old}}}\left(o_{i} | q\right)}, 1-\varepsilon, 1+\varepsilon\Big) A_{i}\Bigg) \\
& \qquad \qquad \qquad -\ \beta\ \mathbb{D}_{KL}\left(\pi_{\theta}\ \|\ \pi_{\text{ref}}\right)\Bigg) \Bigg]
\end{aligned}
$$

Here, the advantage \(A_i\) for each sample in a group is calculated as:

$$
A_{i}=\frac{r_{i}-\text{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\text{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}
$$

These equations encapsulate the mathematical backbone of how the model learns—optimizing its policy in groups and normalizing rewards to refine decision making without explicit step-by-step guidance.


### 2.2. Emergent Behaviors: The “Aha Moment” Phenomenon

One of the fascinating outcomes of large-scale RL training on LLMs is the spontaneous emergence of complex, self-reflective behaviors. DeepSeek-R1-Zero shows that, with enough updates, the model starts to:
- **Extend** its chain-of-thought length for difficult problems,
- **Re-evaluate** steps if an early approach seems likely to fail,
- **Show** an actual “aha moment,” where it steps back, spots mistakes, and corrects itself.

For experts used to conventional fine-tuning, it’s quite striking to see an LLM spontaneously “learn to think better” purely via RL signals. This finding alone points to major opportunities in RL-driven self-improvement.

---

## 3. DeepSeek-R1-Zero: Reinforcement Learning Without a Net

DeepSeek-R1-Zero starts from a base LLM and, crucially, does **no** supervised fine-tuning. The research team introduced:
1. **Accuracy Rewards:** Checking if the model’s final answer is correct (for math, code, logic).
2. **Format Rewards:** Incentivizing a structured chain-of-thought, e.g., `<think> ... </think>` tags.

By optimizing these rewards, the model’s pass@1 on the AIME 2024 math benchmark skyrocketed from 15.6% to 71.0%—competitive with established top-tier models. Even more surprisingly, with majority-vote sampling, it reached 86.7%—overtaking OpenAI’s o1-0912 on the same dataset.

**Why it matters:**  
- The model *learned* how to reason through a set of tasks with zero “handholding.”  
- The improvement trajectory suggests a self-discovery of problem-solving techniques (like reflection, verification, etc.) that many believed required curated data.

**But there’s a drawback:** The output was often in tangles—mixing languages, lacking user-friendly structure, and occasionally showing bizarre rhetorical flourishes. Enter “cold-start” data for the next iteration.

---

## 4. DeepSeek-R1: Merging Cold Start with Large-Scale Reinforcement Learning

The next question was whether injecting **a small supervised “cold-start” dataset** (thousands of curated chain-of-thought samples) might fix the readability and language-mixing issues—and perhaps improve final performance. The team designed a multi-stage pipeline:

1. **Cold Start:** Fine-tune a base model on a few thousand curated, human-friendly long CoTs.  
2. **Reasoning-Focused RL:** Scale up RL with math, coding, and logic tasks. This time, add *language-consistency rewards* to push the model into staying coherent in a single language.  
3. **Rejection Sampling + SFT:** Sample correct, well-structured chains-of-thought from the RL model, augment them with general capabilities data (writing, Q&A, self-cognition), and train a new base checkpoint.  
4. **RL Across Scenarios:** A second RL stage includes both reasoning tasks *and* general tasks for “helpfulness” and “harmlessness.”

**Key Achievements:**  
- The final model, *DeepSeek-R1*, now competes closely with OpenAI-o1-1217 on math and coding tasks.  
- It significantly improves upon its predecessor (DeepSeek-V3) in knowledge benchmarks such as MMLU and GPQA Diamond—especially in STEM-heavy topics.

**Note:** The synergy of minimal curated data + large-scale RL is a potent alternative to the heavy upfront SFT used by many leading LLM pipelines.

---

## 5. Distillation: Transferring Advanced Reasoning Patterns to Smaller Models

**Why Distillation?** Training a 70B model (like DeepSeek-R1) with large-scale RL is expensive—often out of reach for smaller research labs or organizations. However, the final DeepSeek-R1 can generate vast correct solutions for a wide range of tasks. So the authors exploit a simple but powerful approach: **train smaller models (1.5B, 7B, 8B, 14B, 32B) directly from DeepSeek-R1’s curated outputs.**

**Highlights:**
- Distilled Qwen-based 7B model beats some *much larger* open-source models on math and code tasks.  
- Distilled 14B sets new records on certain reasoning benchmarks—proving that, if you have a strong teacher, smaller dense students can replicate advanced reasoning with surprisingly high fidelity.

**Takeaway:** Reinforcement learning on smaller base models (like a 7B or 32B) from scratch simply cannot compete with distillation from a more capable teacher model. The smaller model, left to RL alone, plateaus much lower and at higher cost. Distillation emerges as the “secret weapon” to swiftly propagate advanced reasoning behaviors to new architectures or smaller footprints.

---

## 6. Pitfalls and Unsuccessful Attempts

Experiments with:
- **Process Reward Models (PRM)** found that it was difficult to robustly define or train step-wise correctness signals at massive scale.  
- **Monte Carlo Tree Search (MCTS)** for hierarchical solution exploration faced combinatorial explosion in the generation space and a fragile value model.  
- These methods are not necessarily doomed, but they proved too unwieldy in the large-scale RL context used for DeepSeek-R1.

For professionals considering *internal* RL pipelines: these experiences highlight the complexity of applying search or step-by-step reward systems to sequences as large as LLM outputs.

---

## 7. Broader Implications and Future Directions

### 7.1. General Capabilities vs. Specialized Reasoning

DeepSeek-R1 sometimes trails older siblings (like DeepSeek-V3) on complex dialogues, role-playing, or structured JSON outputs. How do we unify advanced chain-of-thought “brains” with full-fledged interactive features? The authors suggest the next wave of RL expansions could incorporate multi-turn tasks and advanced APIs directly into the chain-of-thought.

### 7.2. Language Mixing and Multi-lingual Support

DeepSeek-R1’s training optimizes specifically for English and Chinese, occasionally leading to “linguistic collisions.” Future expansions might incorporate fine-grained language-detection rewards or multi-lingual chain-of-thought alignment.

### 7.3. Software Engineering Use Cases

While the coding results are strong, the authors note that engineering tasks requiring large contexts or specialized reasoning are still a big RL frontier. Speeding up the RL evaluation loop on code correctness is non-trivial but highly impactful. Asynchronous or more incremental reward mechanisms could be the next big leap.

### 7.4. Prompt Engineering Sensitivities

Unlike older models, few-shot prompts tend to *hurt* DeepSeek-R1’s performance. Leaner, zero-shot instructions appear to work better. This is a curiosity for advanced users—worth exploring in your own environment if you adopt a chain-of-thought-based RL model.

---

## 8. Concluding Thoughts

The DeepSeek-R1 family, in particular **DeepSeek-R1-Zero**, fundamentally proves that massive RL can organically nurture strong reasoning patterns—even without any supervised “crutch.” Yet, the final version of DeepSeek-R1 shows the practical synergy of a small curated dataset plus multi-stage RL to ensure both *power* and *usability*.

For experts researching LLM training pipelines, distillation from a thoroughly RL-optimized teacher is one of the most cost-effective ways to spread advanced reasoning across model sizes. At the same time, the experiences with reward hacking, MCTS complexities, and partial success with process-reward approaches are cautionary tales.

**In short, DeepSeek-R1 is a hallmark that invites us to rethink the role of reinforcement learning in shaping truly “intelligent” LLMs—and underscores how an open AI company makes much more contributions than OpenAI to the open-source community.**