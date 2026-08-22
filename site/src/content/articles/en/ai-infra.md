---
title: "A Role Shift for AI Infra: From Foundational Support to a Core Engine of Innovation"
titleAlt: "AI Infra 的角色转变：从辅助转为主力输出"
subtitle: "AI Market Insights"
date: 2025-10-02
category: systems
---


# A Role Shift for AI Infra: From Foundational Support to a Core Engine of Innovation

Amidst the brilliance of today's large language models (LLMs), the vast and intricate systems that underpin them often go unnoticed. This system, AI Infrastructure (AI Infra), is what truly determines the pace, cost, and ultimate ceiling of technological advancement. It has evolved from a traditional supporting role into a core engine, standing on par with algorithms and data, and its strategic evolution is profoundly reshaping the competitive landscape of the AI industry.

## The Essence and Evolution of AI Infrastructure

The architecture of AI Infrastructure can be understood through the classic cloud computing stack. At the base is hardware, or **IaaS (Infrastructure as a Service)**, which includes not only core AI accelerators like GPUs but also high-speed networking and large-scale storage systems: the three pillars of compute, communication, and storage. Above this lies **PaaS (Platform as a Service)**, responsible for resource orchestration and management, a layer that also encompasses Model-as-a-Service (MaaS). The top layer, **SaaS (Software as a Service)**, is closest to the application, primarily involving the optimization of training and inference frameworks.

As a distinct field, AI Infra is relatively young, with the advent of AlphaGo marking a significant turning point in its development. The first generation of pioneers were often researchers with backgrounds in algorithms, who built tools out of necessity to advance their own work. They were succeeded by a "second generation" of practitioners who entered the field as industry began to adopt deep learning at scale, focusing on systematic engineering for thousand-GPU clusters and beyond.

In the LLM era, the strategic value of AI Infra has magnified exponentially. The situation is strikingly similar to the rise of search engines, where Google's dominance was secured by its world-class infrastructure for processing internet-scale data. Today's LLM competition is likewise a contest of managing unprecedented data and compute demands. The principle is clear: the most advanced models must be built upon the most advanced infrastructure.

## A High-Stakes Bet on "Efficiency": How DeepSeek Redefined "Fast"

While all players pursue "efficiency," the very definition of the term has become a point of strategic divergence. A critical industry case study reveals a profound paradigm shift, where an unconventional bet on a different metric led to a decisive advantage.

The prevailing dogma of the pre-2024 era was the supremacy of **MFU (Model FLOPs Utilization)**. This metric represents a "production efficiency" mindset, focused on maximizing the use of expensive GPU hardware during the pre-training phase. The logic was straightforward: higher MFU meant more effective compute was utilized within the same timeframe, leading to a stronger foundational model at a lower cost. In this race, the goal was to be the most efficient "foundry" for forging base models. However, **DeepSeek pursued an unconventional path.** They did not prioritize achieving the highest MFU. Instead, they optimized for a different goal: **inference cost and decoding speed - a metric of "operational efficiency."** The analogy is akin to auto manufacturing: while competitors focused on who could assemble a car fastest on the production line (high MFU), DeepSeek engineered a highly fuel-efficient, fast-accelerating engine. The engine's complexity might have increased the car's assembly time (lower MFU), but its performance on the road was superior.

The paradigm shifted with the rise of **Reinforcement Learning from Human Feedback (RLHF)** as a critical step for fine-tuning state-of-the-art models. This transformed the nature of training. It was no longer a one-shot "forging" process but a high-frequency iterative loop of generation, feedback, and learning. The model had to act as an apprentice, repeatedly performing tasks (generating responses), receiving scores (rewards), and optimizing itself.

Suddenly, a new bottleneck emerged. The speed of the entire training loop was no longer dictated by the initial pre-training efficiency (MFU) but by the "apprentice's" speed of practice, but its inference and decoding velocity.

At this juncture, DeepSeek's strategic foresight was realized. Its architecture, designed for fast inference, could complete the RLHF loop multiple times while others were still processing a single iteration. The model that was less efficient in the initial build-out phase became the most efficient learner in the new training regime. Their victory was a triumph of correctly anticipating the future of AI development. They shifted their focus from "how to efficiently build a model" to "how to efficiently use and iterate on a model."

## Organization, Ecosystem, and the Path Forward

This paradigm shift also highlights the ideal organizational structure for AI development: a tight collaboration between the “Triangle of Algorithm, Data, and Infra”. In this model, responsibilities are clearly delineated:

* The **Data team** is accountable for the model's performance ceiling, as data quality is paramount.
* The **Infra (system) team** is responsible for efficiency and cost, and should therefore lead model architecture design (and also participate in the algorithm design).
* The **Algorithm team**'s core focus is on pioneering and optimizing the training methodologies themselves.

**This contrasts with many large enterprises where infrastructure is often relegated to a passive, supporting role, creating a misalignment between organizational structure and modern AI development needs.**

For third-party AI Infra companies, the business ecosystem presents its own challenges. They risk being squeezed between model developers and hardware vendors. The path to a sustainable advantage lies in vertical integration: either by deeply integrating with hardware to gain cost and technical benefits, or by aligning closely with model development to offer unique value, much like a gaming console needs exclusive titles to thrive.

Looking ahead, the ultimate opportunity in AI Infra lies in achieving **true co-design of models and hardware, potentially disrupting the current compute paradigm.** This aligns with the core message of Rich Sutton's "The Bitter Lesson": in the long run, methods that best leverage computation will prevail. The gap between different countries, such as US and China, and between companies, will primarily lie in scale and the ability to execute such vertical integration. To address this, leading teams should explore innovative strategies to enable deep adaptation support for chip manufacturers, using model architecture innovation to enhance the competitiveness of local hardware.

For all practitioners in the field, the path forward is clear: one must operate at the intersection of models and hardware. In this era, AI Infrastructure is no longer a silent foundation but the active, driving force shaping the future of artificial intelligence.