# DualPipe Explained in Simple Terms: A Comprehensive Guide to DualPipe That Anyone Can Understand—Even Without a Distributed Training Background

Last week, DeepSeek announced an "OpenSourceWeek" on social media, releasing a series of open-source software libraries for five consecutive days. Over the first few days, they introduced **FlashMLA** (an efficient Hopper GPU MLA decoding kernel), **DeepEP** (an expert parallel communication library for MoE), and **DeepGEMM** (a GEMM library with FP8 support). On the fourth day, they open-sourced three major components in one go: **DualPipe**, **EPLB**, and **profile-data**. Among these, DualPipe—due to its core idea of "bidirectional pipeline parallelism"—sparked widespread discussion.

This blog post will focus on the core concept of DualPipe: **how to fully overlap forward and backward passes in large-model training**, thereby greatly reducing the "bubble time" in pipeline execution. To help you grasp these ideas, we’ll start with a straightforward analogy—**"process optimization in a machine workshop"**—introducing each concept first through the lens of mechanical processing, and then mapping it back to parallel training in deep learning. By the end of this article, you’ll have a clear mental picture of how these ideas work. We’ll also delve into DualPipe’s source-code-level details, exploring how it further reduces pipeline bubbles, overlaps forward and backward passes, minimizes communication pressure, and integrates into more complex hybrid parallelism scenarios.

---

## Introduction

With today’s buzz around large language models (LLMs) such as GPT-3, PaLM, and LLama, distributed training has become an essential technique for pushing beyond the limits of a single GPU and successfully training ultra-large models. Terms like **data parallel**, **model parallel**, and **pipeline parallel** come up often, yet it can be challenging—especially for beginners—to see how they differ and connect. And when you encounter advanced features like **DualPipe** in DeepSeek-V3, it might feel even more daunting.

On the other hand, in industrial settings, **optimizing a production process** often involves countless trials, while in AI, **training a large language model** similarly requires numerous parameter adjustments. These two activities may seem unrelated, but they share a remarkable resemblance. Let’s walk through a story from Lao Wang’s machine workshop to see how a production line with multiple machine tools can help us understand the four major types of parallelism in large-model training.

### The Single-GPU Era: The Small Handicraft Workshop

In Suzhou Industrial Park, Lao Wang owns a mid-sized mechanical company that focuses on optimizing manufacturing processes such as casting temperature, quenching time, cutting angles, etc. When a new order arrives, Lao Wang first designs a set of initial manufacturing parameters (the "process manual"), proceeds with machining, then inspects the final part and back-propagates the adjustments: if the part has void defects, he tweaks the casting temperature upward, and so on.

> Lao Wang’s process is remarkably similar to large-model training: the "process manual" is like the model’s parameters, the parts being machined are akin to training data, each individual process (casting, heat treatment, etc.) corresponds to different layers in a neural network, and the machine tools are analogous to GPUs. Finally, the quality check is akin to computing the loss function, and adjusting parameters based on the inspection is just like backpropagation.

When Lao Wang started his business, he took on relatively simple orders, like machining screws. For such parts, you only need a single multifunctional machine to handle two steps: cutting and polishing. If the part turned out unevenly polished, you’d adjust the polishing step; if the cutting was off, you’d adjust the cutting angle. It was all done on a single machine—like a **small handicraft workshop**. That was enough for basic parts, but there was no scalability.

> This corresponds to **single-GPU training**. All model layers (all "process steps") run on the same GPU (the same machine). Both the forward pass (machining) and the backward pass (adjusting parameters) happen on a single device. It’s simple and reliable, but once the tasks become more complex, a single device becomes the bottleneck.

### Model Parallelism: The Art of Splitting the Process Manual

One day, Lao Wang received a much bigger order for optimizing the process of manufacturing an engine crankshaft. He quickly realized that a single machine could not handle all the needed steps. So he split the processes (casting, heat treatment, precision machining) across three specialized machines, each equipped with its own operation instructions. He also had to keep track of how adjusting casting parameters might affect subsequent processes. It introduced a new problem—machine idle time. While the first machine was busy casting, the other machines might be waiting. Plus, moving items from one machine to the next took additional time. If not planned carefully, these machine transitions could cause extra idle periods.

> In large language models, this is **model parallelism**. When a model is too large for a single GPU’s memory, you split it across multiple GPUs (e.g., different layers or different modules for each GPU). In this analogy, the casting is like an input layer, heat treatment is the intermediate layer, and precision machining is the output layer. As you train, each GPU is responsible for a segment of the model and must communicate intermediate outputs to others. This inevitably leads to idle time across GPUs and frequent cross-device data transfer. The problems Lao Wang faces resemble the scheduling and communication challenges among GPUs.

### Data Parallelism: A Plan to Clone the Workshop

To further speed up the process-parameter optimization, Lao Wang—now with more funding—built three identical workshops next door. Each workshop has the same entire pipeline, just working on different batches of turbine discs (data shards). By the end of the day, the four workshop managers come together to compare notes and unify the process standards. An order of 10,000 raw parts that used to take a month now only needs about two weeks. Lao Wang wonders, "Why did quadrupling my workshop capacity only double the speed?" After talking to the managers, he found that each batch of raw materials might encounter unique problems, causing some workshops to finish later. When they’re done, they have to wait for the slowest one before summarizing the day’s results.

> This is **data parallelism**. Each workshop (GPU) holds a full copy of the process manual (model parameters) but processes a different portion of the data (different mini-batches). Once each workshop computes its local gradient (inspecting the part and figuring out parameter adjustments), they must gather and average these gradients (through **All-Reduce**) to form a unified set of parameters. The bottleneck is that the slowest workshop (straggler GPU) holds everyone up, and the communication overhead (All-Reduce bandwidth) increases dramatically with more parallel workshops.

### Tensor Parallelism: Collaborating on an Oversized Single Component

One day, Lao Wang leveled up and received a **massive** project to optimize processes for airplane components—for instance, an aircraft wing. Even just the wing itself is so big that multiple identical machines must work together on the same sub-step. In other words, though this sub-step belongs to one particular process stage, it still exceeds the capacity of a single machine. So Lao Wang had multiple identical machines collaborate on one single huge part.

> This is akin to **tensor parallelism** in large-model training. Even after splitting the model across different layers or modules, you might still find a single module too large for one GPU. In that case, you split the module’s **tensors** themselves—like partitioning a large matrix across multiple GPUs to perform parallel matrix multiplication. Once you finish each partial multiplication, you merge the results. This approach distributes the workload of a single, very large layer across multiple GPUs.

At this point, Lao Wang realized that **scheduling** had become the key to higher efficiency. Machines had to coordinate closely, as many of them handled only a piece of a part. That part might be moved from one machine to another multiple times; subsequent steps often can’t start until a previous step finishes, and in the feedback loop, earlier steps can’t finalize parameter adjustments until later steps finish their own. This all leads to idle periods everywhere: some machines wait for their "brother machines" to finish, downstream processes wait for upstream results, upstream processes wait for feedback. Lao Wang felt he needed to do something to further boost efficiency.

> This is how tensor parallelism really works. When even a single process (a single layer in the model) exceeds a single GPU’s capacity, you split the large tensor into smaller chunks that a GPU group can handle. For example, polishing an airplane wing might require four machines working on different wing areas, then stitching them back together. That level of collaboration introduces communication overhead (merging partial outputs), synchronization overhead (one slow machine can hold up the rest), and additional "feedback loops" for gradient synchronization. Collectively, these can introduce new idle times—**"collaboration bubbles."**

### Pipeline Parallelism: Making Different Machines Work Simultaneously

To deal with the collaboration challenges among different processes, Lao Wang devised an ingenious pipeline system. If the original processing route was **Casting → Forging → Heat Treatment → Polishing**, the moment the casting machine finished its first batch, that batch was immediately moved on to the forging machine; casting then started work on the second batch. By the time the first batch left forging for heat treatment, the second batch arrived at forging, and the third batch could head to casting—like **dominoes**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="img/in-post/2025-02-27-dualpipe/naive.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview of the data flow when not using pipeline parallelism: when data is passed to different parts of the model (on different GPUs), only one GPU works (either in a forward or backward way) at one time. This makes the GPU efficiency very low and thus not perferred.
</div>

Before implementing this pipeline system, the workflow in each workshop looked like the diagram above. During the first batch’s lifecycle (T1~T4), only one machine was working at a time; when it came time for inspection and parameter feedback (T5~T12), again just one machine was active while the others were idle (gray areas in the figure). Lao Wang quickly spotted the inefficiency: once the first batch moved on to the second stage, the first stage could be processing the second batch, and so on. So the pipeline became:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="img/in-post/2025-02-27-dualpipe/1F1B.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview on the 1F1B pipeline parallelism strategy.
</div>


> The diagram above illustrates the well-known **1F1B (one-forward-one-backward)** pipeline parallelism scheme. Its principle is: whenever a GPU (or a machine) is ready to perform the backward pass on a recent micro-batch, it will prioritize the backward pass. For instance, at time T5, Device 4 must choose between doing a forward pass for the second batch or a backward pass for the first batch, and it prioritizes the backward pass. Meanwhile, all data are backpropagated in micro-batch order; for example, the second micro-batch only starts its backward pass after the first micro-batch’s backward pass has begun. Also, each device keeps a limited number of forward activations (4 in the figure) to avoid storing too many intermediate results for backward computation. Returning to our analogy, these intermediate "activation data" are like the manufacturing logs that the workshop stores to assist with final quality assessment and parameter updates. As you can see, there are still idle periods in the 1F1B pipeline. In large-model training, such idle times are called **bubbles**—periods when GPUs are waiting rather than working. We aim to reduce these bubbles as much as possible.

After closer observation, Lao Wang discovered that a key source of pipeline "bubbles" was how much time the parameter feedback process took—nearly twice the actual processing time for each batch. That meant once the first stage had processed the fourth batch, it had to wait a long time to receive the feedback updates on the first batch. To address this, he came up with a novel idea: since feedback consumes so much time, **split it into two independent parts** so they can be decoupled. For example, each process might involve **fixture design** and **manufacturing design**. If the fixture design can be updated immediately based on the quality inspection report, independent of the manufacturing design updates, we can eliminate some idle time. Based on this idea, Lao Wang designed the improved pipeline:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="img/in-post/2025-02-27-dualpipe/ZB1P.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview on the ZB1P pipeline parallelism strategy.
</div>

Here, each batch’s feedback is split into two phases: fixture design adjustment (light blue) and manufacturing design adjustment (dark blue). Within the same process stage, there is a dependency among the same feedback phases (e.g., forging’s fixture design must finish before casting can update its fixture design), but different phases don’t depend on each other. This decoupling allows each stage’s fixture design update to be done earlier, while the manufacturing design updates can be delayed, thus reducing idle bubbles compared to the original 1F1B approach.

> The scheme above is the **ZB1P** (Zero Bubble 1F1B) baseline mentioned in the DeepSeek-V3 technical report. It divides the backward pass into two sub-steps in deep learning:
> 1. **Input gradient computation**: Passing gradients from the current layer back to the previous layer.
> 2. **Parameter gradient computation**: Computing the parameter gradient for the current layer so it can be updated.
>
> For a linear layer \( \mathbf{y} = \mathbf{W}\mathbf{x} \) with loss \( L \), once we receive \( \frac{\partial L}{\partial \mathbf{y}} \) from the subsequent layer, we need to compute:
> - \( \frac{\partial L}{\partial \mathbf{x}} \): to pass the gradient backward to the preceding layer;
> - \( \frac{\partial L}{\partial \mathbf{W}} \): for the layer’s own parameter update.
>
> Interestingly, these two computations do not depend on each other in a strict order. Even if a layer has only finished computing (1) but not (2), the gradient can still propagate to the previous layer. Leveraging this fact, ZB1P decouples (1) and (2) so that **input gradient** (1) can be performed as early as possible while **parameter gradient** (2) is postponed, thus greatly increasing pipeline scheduling flexibility and efficiency.

By now, you understand the two baseline pipeline schemes in the DeepSeek-V3 report that are compared against DualPipe. Below is a summary table comparing their efficiency:

| Method | Bubble                             | Activation       |
| ------ | ---------------------------------- | ---------------- |
| 1F1B   | \((PP - 1)(F + B)\)                | \(1 \times PP\)  |
| ZB1P   | \((PP - 1)(F + B - 2W)\)            | \(1 \times PP\)  |
<!-- | DualPipe | \(( \frac{PP}{2} - 1) (F\&B + B - 3W)\)| \(2 \times PP + 1\)| -->

**Key Parameters**:

- \(PP\): Pipeline depth, i.e., the number of process stages involved in parallel.
- \(F\): Time needed for a forward pass (e.g., each workshop’s initial machining).
- \(B\): Time needed for a backward pass (e.g., each workshop’s feedback adjustments).
- \(W\): The window size for activation data accumulation—i.e., the upper limit of stored intermediate activations for backprop.

From this table, you can see that **ZB1P** reduces bubbles substantially compared to **1F1B**, at the same level of activation memory usage. By decoupling the backward computations, ZB1P achieves more overlap between forward and backward, thus cutting down idle time. More advanced scheduling strategies (like **DualPipe**) push this idea further, aiming to maximize resource utilization and parallel performance in large-model training.

In short, Lao Wang’s evolving pipeline schemes—from the original 1F1B to ZB1P and beyond to DualPipe—mirror how large language model training has advanced. Each new innovation reduces bubbles (idle times) and pushes the system to higher performance and better resource utilization.

---

### When the Pipeline Still Has “Blind Spots”: Limitations of ZB1P

Although ZB1P’s decoupling approach significantly shortens idle periods, Lao Wang’s workshop still experiences some lingering gaps. Imagine a pipeline of **Casting → Forging → Heat Treatment → Polishing**: once the casting machine finishes its "fixture design update," it hands that info off to forging, but a deeper stage’s "manufacturing design update" might still have to wait on all preceding updates to be done. Because parts flow through multiple tightly coupled stages, even small delays in one stage can ripple through and create new bubbles.

In large-model training, ZB1P does boost overlap between forward and backward, but it can’t achieve **truly simultaneous** forward and backward passes. Why?

1. **Pipeline bubbles are still present**  
   Traditional pipeline implementations often strictly separate forward and backward: process all micro-batches in forward mode, then do backward. This leads to idle GPUs during each phase.

2. **Manual scheduling is complex**  
   Conventional pipeline schemes require writing extensive logic: When to send activation tensors? When to receive gradients? If not carefully arranged, you’ll face extra waits or communication bottlenecks.

3. **Forward and backward can interfere**  
   As more micro-batches accumulate in forward passes, the backward passes can eventually starve the forward pass of GPU resources (e.g., memory or bandwidth). Poor resource management can lead to additional stalls.

4. **Insufficient front-back overlap**  
   Ideally, we want other micro-batches’ forward passes to run concurrently with backward passes on different GPUs. Doing so, however, requires a sophisticated scheduling design.

Returning to Lao Wang’s workshop example: if both the casting and forging machines are fully utilized at a certain moment, but the heat treatment or polishing machine reduces output for some reason (akin to GPU load imbalance), you may quickly revert to a "front-waits-for-back, back-waits-for-front" scenario. ZB1P already improves scheduling flexibility, but there remain "blind spots" of idle time.

Seeking an even better approach, Lao Wang aimed to further **break down** the dependencies between forward and backward so they could fully interleave in adjacent pipeline nodes. In other words, if a machine is handling a backward pass, it could still process the next micro-batch’s forward pass. This approach greatly increases pipeline concurrency and minimizes downtime. Lao Wang’s new design for his workshop parallels a new scheduling strategy for model training, called **DualPipe**, the main focus of this article.

---

## DualPipe: A Two-Pronged Pipeline Strategy

### Bidirectional Scheduling: Pushing “Front and Back” onto the Production Line Together

In **traditional (single-direction)** pipelines such as 1F1B or ZB1P, a machine either performs forward processing or waits for a feedback signal to do backward adjustments. These two modes are usually **mutually exclusive**. In **DualPipe**, however, Lao Wang equips each machine with a **"timesharing mode"** and a **flexible front-back transport system** that allows the same machine to handle **both** forward and backward tasks simultaneously. The machine can receive new raw materials from the "front" while also getting a feedback report from the "back."

By introducing this dual transport system, a machine can keep receiving new parts to process (the forward pass) while at the same time handling the backward pass updates from downstream. In a large language model, this translates to letting forward and backward passes truly occur **in parallel**, significantly boosting GPU utilization.

Unlike a single-direction pipeline, **DualPipe** injects micro-batches from **both ends** of the pipeline. If you picture a linear conveyor belt, previously we only fed data from one side, passing all the way to the end. DualPipe, however, opens up the opposite end as well, so a machine can do forward processing for tasks coming from the left (upstream) while simultaneously receiving backward gradients from the right (downstream). This approach greatly increases overall pipeline efficiency and cuts down idle periods. GPUs can process forward tasks from the "front" and handle backward tasks from the "back" at the same time, thereby avoiding the strict one-way flow that caused waiting times in earlier methods.

Below is a table comparing **1F1B**, **ZB1P**, and **DualPipe** in terms of pipeline bubbles and activation usage, illustrating how each approach tackles latency and scheduling:

| Method   | Bubble                                                   | Activation           |
| -------- | -------------------------------------------------------- | -------------------- |
| 1F1B     | \((PP - 1) (F + B)\)                                     | \(1 \times PP\)      |
| ZB1P     | \((PP - 1) (F + B - 2W)\)                                | \(1 \times PP\)      |
| DualPipe | \(\left(\frac{PP}{2} - 1\right) (F\&B + B - 3W)\)        | \(2 \times PP + 1\)  |

Where:
- \(PP\) is pipeline depth (number of pipeline stages),
- \(F\) and \(B\) are forward and backward times,
- \(W\) is the activation window size.

By comparing the three:

- **1F1B**  
  The simplest one-way pipeline. Bubble time is \((PP - 1)(F + B)\). Each stage must wait in a mostly serial fashion. The activation storage is \(1 \times PP\).

- **ZB1P**  
  By decoupling the backward steps (splitting input gradient and parameter gradient), bubble time shrinks to \((PP - 1)(F + B - 2W)\). Activation usage remains \(1 \times PP\).

- **DualPipe**  
  Uses a **bidirectional** pipeline plus overlapping compute and communication to drastically reduce idle times. The bubble time drops further to \(\left(\frac{PP}{2} - 1\right) (F\&B + B - 3W)\). However, it demands **increased activation storage**, up to \(2 \times PP + 1\), because each GPU holds extra activation data to support simultaneous forward and backward.

In short, **DualPipe** trades off a bit more memory usage for a significantly higher overlap between forward and backward, thus achieving greater throughput in large-scale distributed training—just as Lao Wang’s workshop gains higher utilization by letting machines work in **both directions** simultaneously.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="img/in-post/2025-02-27-dualpipe/dualpipe.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview of the DualPipe pipeline parallalism.
</div>

### “Compute-Communication” Overlap Inside the GPU: Doing Calculations While Data Moves

Another key to DualPipe’s efficiency is **“chunk-based” or “segmented” transport**. In large-scale distributed scenarios, both **computation** and **communication** can be major time sinks. If they happen strictly one after another (compute → comm → compute → comm), we’ll inevitably face idle GPU cycles during communication, and idle network bandwidth during compute. 

DualPipe splits data transfers into smaller chunks—**“micro-batch streaming”**—and interleaves them with partial compute tasks, so GPUs can start computing as soon as a portion of the data arrives, rather than waiting for the entire dataset to transfer. Here’s why chunk-based shipping is vital:

#### Why “Chunked Transport” Improves Efficiency

Imagine Lao Wang’s **casting** machine has 1,000 parts to send to the **forging** machine. If we do a **one-shot** transfer, forging can’t start until all 1,000 are delivered. During that time, forging is idle. After delivery, forging might quickly finish some tasks, only to wait again. This can lead to “you wait for me, I wait for you.” 

But if we **break** the 1,000 parts into several smaller shipments (say 4 or 10 chunks), forging can start working on the first chunk immediately while the second chunk is in transit, removing idle time and ensuring a steady flow of tasks.

In GPUs, this is akin to splitting large tensor transfers or **all-to-all** communications into smaller pieces, enabling partial results to be computed while the remaining data is still streaming.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="img/in-post/2025-02-27-dualpipe/dualpipe_communication.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview of the communication design in DualPipe.
</div>

#### How “Compute-Communication” Overlap Works on GPUs

1. **Resource Partitioning**  
   Modern GPUs have multiple SMs (Streaming Multiprocessors). We can allocate some to handle communication (sending/receiving packets) while others handle compute kernels. If communication is chunked into smaller messages, the compute SMs can stay busy while the communication SMs handle the data transfer.

2. **Fine-Grained Splitting of Tensors**  
   Instead of waiting for the entire model layer’s data to be transferred, we break each layer’s output or gradient into smaller segments. The GPU can compute on the first segment while the next segment is still being transferred.

3. **Parallel Scheduling**  
   Frameworks like PyTorch support **multi-stream** asynchronous operations. You issue a communication call (e.g., `cudaMemcpyAsync` or collectives) in one stream while continuing computation in another. You only synchronize when you actually need the results.

4. **Pipeline Mechanism**  
   As DualPipe introduces bidirectional flow, each GPU both **receives** data from the previous stage and **sends** updates to the next stage. By carefully scheduling smaller chunks, communication from "front to back" and from "back to front" can overlap with local compute, forming a tight pipeline across micro-batches and GPUs.

All these enable what we call **time overlap** between compute and communication—**the fundamental reason** GPUs don’t sit idle waiting for large transfers to finish. 

In a “single-chunk” scenario, you might see:

```makefile=
Time: ======================> Comm: [send all data] -> (wait) -> Compute:[start only after entire data is available]
```

Resulting in wasted GPU cycles during comm and wasted network bandwidth during compute. By contrast, **chunked**:


```makefile=
Time:     ==================================>
Info:      [comm chunk1]  [comm chunk2]   ...
Compute:         [compute chunk1]   [compute chunk2]  ...
```

They overlap like interlocking gears, drastically reducing idle periods and ensuring a continuous workflow. This is how DualPipe sustains high throughput even as model size and parallelism scale up. 

Just like Lao Wang’s multi-stage workshop—where partial shipments of parts go back and forth and get processed without waiting for an entire batch to travel—DualPipe leverages chunk-based data flows to keep GPUs near full utilization, effectively **hiding** communication overhead behind concurrent computation.

---

## How the Source Code Implements “Two-Pronged” Parallelism: Key Logic of DualPipe

In the previous section, we used a manufacturing analogy to illustrate how **DualPipe** enables forward and backward passes to truly run “at the same time” in the same pipeline, thereby maximizing GPU utilization. Here, we’ll explore how DualPipe’s **core source code** implements these ideas. Specifically, we’ll look at how it handles batch splitting, manages communication, and cleverly interweaves forward-backward tasks to achieve deep overlap and minimal pipeline bubbles.

> Note that we won’t parse the code line by line. Rather, we’ll focus on the critical functions and workflows, tying them back to the “machine workshop” analogy to help you understand them more intuitively.

---

### 1. Basic Class Structure and Initialization

    class DualPipe(nn.Module):
        def __init__(
            self,
            modules: Tuple[nn.Module, nn.Module],
            ...
        ) -> None:
            super().__init__()
            
            ...
            self.module = nn.ModuleList(modules)
            self.overlaped_forward_backward = ...
            ...

- **`modules`**: This parameter is a tuple of two `nn.Module` instances, typically representing the “front half” and “back half” of a pipeline. In our mechanical analogy, these could correspond to a combined “forward processing” set of machines (e.g., casting + forging) and another combined set (e.g., quality inspection + parameter adjustment).
- **`overlaped_forward_backward`**: This is a specialized function that checks whether the two `Module` objects support forward-backward overlap. Only if both `Module`s implement an `overlaped_forward_backward` method (and are of the same type) will the subsequent workflow apply true forward-backward interleaving for the same batch.

Additionally, the code sets up parameters and flags related to distributed training, such as:

- **`self.group` and `self.rank`**: These tie into `torch.distributed`, used to manage the pipeline rank (i.e., the stage) of the current process.
- **`self.is_first_rank`, `self.is_last_rank`, `self.is_in_second_half`**: Flags that mark whether this node (machine) is at the “left end” or “right end” of the pipeline, i.e., whether it belongs to the first half or the second half.

These flags and mappings resemble the labels Lao Wang might attach to each machine in his workshop, e.g. “Casting Line,” “Polishing Line,” or “Second-to-Last Step.”

---

### 2. State Management and Reset

Below is the `_reset_states` method:

    def _reset_states(self) -> None:
        WeightGradStore.clear()

        self.input_chunks = ([], [])
        self.output_chunks = ([], [])
        self.input_grad_chunks = ([], [])
        self.output_grad_chunks = ([], [])
        self.labels = None
        self.loss_chunks = []
        self.criterion = None

        ...
        # Various counters for tracking which chunk is being processed
        self.current_f_chunk_id = [0, 0]
        self.current_b_chunk_id = [0, 0]
        ...
        self.comm_ops = []
        self.to_free = []

- **`_reset_states`** is similar to clearing the workshop and re-laying out tools and logbooks whenever Lao Wang gets a new order:

  - `WeightGradStore.clear()` removes any stored “parameter gradient” callbacks, preparing for Zero-Bubble or partial-overlap strategies.
  - `input_chunks`, `output_chunks`, `input_grad_chunks`, `output_grad_chunks`: These are like the “work in progress” parts and their “gradient info.” Initializing them to empty lists so we can fill them with each micro-batch and move them along.
  - Various counters track the number of forward chunks, backward chunks, send/receive events, etc.

---

### 3. Forward and Backward Computation: Achieving Overlap on the Same Device

In large-model training, “forward computation” and “backward computation” essentially involve **running a forward or backward pass on a tensor** and then passing gradients to the previous layer. In the mechanical analogy, forward is “machining the part,” while backward is “checking for defects and adjusting production parameters.” DualPipe wraps these processes in a few methods:

#### 3.1 `_forward_compute_chunk(self, phase)`

    def _forward_compute_chunk(self, phase: int) -> None:
        phase ^= self.is_in_second_half  # dynamically correct the phase
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        ...
        outputs = self.module[phase](*inputs)
        ...

- Here, the `phase` is adjusted to choose which sub-module (the “front” vs. the “back” part of the pipeline) is actually used. Then we grab the batch data from `input_chunks[phase]` and run forward computation on it.
- The resulting `outputs` will be stored in `self.output_chunks[phase]`.
- If this is the last stage (`is_last_stage`) and a `criterion` is defined, the “loss” is placed in `self.loss_chunks`. Think of it as the final workshop outputting a “defect score” for inspection.

#### 3.2 `_backward_compute_chunk(self, phase, enable_zb: bool = False)`

    def _backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return
        phase ^= self.is_in_second_half
        chunk_id = self.current_b_chunk_id[phase]
        ...
        if is_last_stage:
            # at the last stage, directly call backward on the loss
            loss = self.loss_chunks[chunk_id]
            loss.backward()
            loss.detach_()
        else:
            # run_backward on outputs + output_grads
            outputs = self.output_chunks[phase][chunk_id]
            ...
            run_backward(outputs, output_grads)
        ...
        if enable_zb:
            WeightGradStore.flush()

        # update input_grads
        ...

- For backward, the main logic is: if you’re on the final stage, call `backward()` on the `loss`; otherwise call `run_backward()` on the intermediate outputs to pass the gradient upstream.
- `enable_zb` activates the Zero-Bubble (e.g., ZB1P) approach, where some parameter-grad computations are deferred by putting them into `WeightGradStore` and flushing them at the right moment. This aligns with our earlier explanation of **decoupling “input gradient calc” and “parameter gradient calc.”**
- Once backward finishes, the “gradients” for the upstream stage are placed into `self.input_grad_chunks[phase]`, akin to Lao Wang returning some defect report to the previous machine.

#### 3.3 `_forward_backward_compute_chunk(self, phase0, phase1)`

    def _forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        if self.forward_only:
            self._forward_compute_chunk(phase0)
            return

        if not self.overlaped_forward_backward:
            self._forward_compute_chunk(phase0)
            self._backward_compute_chunk(phase1)
            return

        # 1) pre-forward
        # 2) pre-backward
        # 3) overlaped_forward_backward(...)
        # 4) post-forward
        # 5) post-backward

The true core of DualPipe: if `overlaped_forward_backward` is `True`, the same GPU can fuse forward and backward for certain batches (like “both hands working together”).

- The function first grabs the forward data from `input_chunks` and the backward data from `output_chunks`, then calls `module0.overlaped_forward_backward(...)`.
  - In the workshop analogy, the operator gathers the new batch of parts to be processed, as well as the defect reports and adjustments from a prior batch, then uses a single “combined machine operation” to handle both forward + backward tasks.
- It finally stores the new output and gradient into `output_chunks` and `input_grad_chunks`. This means partial forward-backward steps can occur within the same stage, rather than calling them separately or waiting for another stage (as in ZB1P).

---

### 4. Communication Management: Hiding “Transport Time” Behind Computation

In large-scale model pipeline parallelism, every stage’s GPU frequently needs to communicate with upstream/downstream GPUs (like transporting half-finished parts from casting to forging). DualPipe uses several functions to break down and schedule communication so that computation and communication overlap extensively:

- **`_recv_forward(self, phase)` / `_send_forward(self, phase)`**  
  Used to receive/send “forward outputs” or “forward inputs.” In the workshop analogy, handing off the partially processed parts to the next machine.

- **`_recv_backward(self, phase)` / `_send_backward(self, phase)`**  
  For receiving/sending “backward gradients” or “backward inputs,” analogous to passing inspection reports between machines.

- **`_commit_and_wait_comm(self)`**  
  Internally calls something like `dist.batch_isend_irecv(self.comm_ops)` to launch all non-blocking communications, then `wait()`. This allows compute to proceed in parallel with communication, so the “shipping time” is “hidden” within timeslots when the machine is idle or can devote resources to transfers.

---

### 5. WeightGradStore: Delayed Gradient Update Design

During backward passes, the same weights may accumulate gradients multiple times. `WeightGradStore` uses a static queue to hold these updates, then executes them collectively at the right time. Benefits:

- **Fewer syncs and memory writes**: Instead of updating parameters on every micro-batch, you can batch them up for a single update when appropriate.
- **Pipeline integration**: Avoid interrupting pipeline concurrency, plus sync at times that overlap with other tasks.

```python3=
    class WeightGradStore:
        enabled: bool = False
        cache: List[Callable] = []
        funcs_queue = queue.Queue()

        @classmethod
        def put(cls, func: Callable) -> None:
            cls.cache.append(func)

        @classmethod
        def flush(cls) -> None:
            cls.funcs_queue.put(cls.cache)
            cls.cache = []

        @classmethod
        def pop(cls) -> None:
            funcs = cls.funcs_queue.get()
            for func in funcs:
                func()
```

> Note: The `phase ^= self.is_in_second_half` trick is a neat way of “flipping” the phase based on whether the pipeline rank is in the second half, allowing the same function to handle left-to-right and right-to-left data transfers.

---

### 6. Overall Scheduling: The `step(...)` Method’s 8 Stages

The core logic resides in `step(...)`. This function is like Lao Wang’s central command that orchestrates all machines. To achieve DualPipe’s two-way pipeline, it proceeds through these phases (in a simplified view, see code comments for details):

1. **Step 1: nF0**  
   At pipeline startup, let one end process a certain number of forward batches first. Like partially “pre-processing” raw material on the left side while the right side is idle.

2. **Step 2: nF0F1**  
   Gradually let the other end begin forward operations so that both directions are activated—calling `_forward_chunk(0)` and `_forward_chunk(1)` in alternation.

3. **Step 3: nB1W1F1**  
   When backward shows up for some batches, mix backward (`_backward_chunk(1)`) with forward (`_forward_chunk(1)`), plus `_weight_chunk()` for delayed weight updates (related to ZeroBubble).

4. **Step 4: nF0B1F1B0 (main loop)**  
   DualPipe’s main “interleaving” step: `_forward_backward_chunk(0,1)` / `_forward_backward_chunk(1,0)` for simultaneous scheduling of forward and backward.

5. **Step 5: nB1F1B0**  
   Continue driving backward + forward interweaving.

6. **Step 6: nB1B0**  
   Focus more on backward, letting `WeightGradStore` flush partial gradient data.

7. **Step 7: nWB0**  
   More “weight update + backward” cycles.

8. **Step 8: nW**  
   Final weight updates, ensuring all tasks wrap up cleanly.

In the workshop analogy, it’s a “scheduling timetable” with multiple time slots: first let the left-side machines do some initial runs, then after a certain point, the right side also feeds in raw materials; both sides keep passing along half-finished parts and defect reports until everything is done, with a final unify-and-finish step.

> The multiple steps may seem complex, but that’s because a two-way pipeline requires careful management of micro-batches at different phases to keep bubbles minimal.

---

### Summary

From the `step(...)` orchestration to `_forward_backward_compute_chunk(...)`’s “fusion of forward and backward,” you can see how DualPipe leverages **fine-grained partitioning** and **two-way injection**. On the code side, it **greatly hides communication overhead** and enables a single GPU to run forward while running backward at just the right times.

> Going back to the workshop analogy: a single time slot might see Machine A process new raw materials (forward) while Machine B handles the defect feedback from the previous batch (backward), and the “transport truck” (communication) zips around mostly during those moments of spare capacity—reducing idle time on every machine.

The benefits are evident:

1. **Minimal pipeline bubbles**: DualPipe truly has “front-back concurrency,” lowering serial wait times.  
2. **Extensive communication overlap**: Via `_commit_and_wait_comm()` and async send/receive, much cross-node all-to-all or pipeline traffic finishes while the GPU is busy computing.  
3. **Scalability**: Even if the cluster grows (more GPUs, more nodes), as long as compute-comm ratio is managed, this overlap stays effective.  
4. **Memory optimization**: Placing the shallowest layer and deepest layer on the same pipeline rank with Zero-Bubble can help reduce middle-layer storage overhead.

Such a “two-pronged” solution significantly speeds up large-model training (e.g., MoE) that demands heavy cross-node expert parallel communication. It can let you train 100-billion-parameter networks on limited GPU resources (like H800s) without grinding to a halt.

---

## Conclusion and Outlook

From a machine workshop analogy to large-language-model parallel training, we have covered **no parallelism**, **model parallelism**, **data parallelism**, and **pipeline parallelism**. In pipeline parallelism, the hardest part is minimizing “pipeline bubbles,” cutting communication wait times, and overlapping forward-backward execution. **DualPipe** is a high-level solution that cleverly arranges micro-batches and forward-backward timing to reach a “zero-bubble” ideal (or near-ideal), greatly boosting pipeline training speed and utilization.

In short, DualPipe not only conceptually overlaps front and back passes, but also **encapsulates communication and scheduling details** to let users more easily perform complex pipeline-parallel training. For those wanting to dive deeper into large-model distributed training, studying its source code (plus hands-on testing) can be highly beneficial for developing more efficient parallel solutions.

In a nutshell: **like a well-orchestrated assembly workshop, DualPipe synchronizes forward and backward processing to move in tandem, massively improving multi-GPU efficiency and laying a vital technical foundation for the era of large models.**

