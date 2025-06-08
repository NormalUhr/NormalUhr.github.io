# 机器学习中的装饰器

Python 装饰器（Decorators）可能是你见过最好用的语言特性之一。它能轻松地为你的函数或类增加新功能，而无需直接修改原始代码。在这篇博客中，我们将深入两个顶级开源机器学习框架 —— vllm 和 trl 的源码，逐步解析Python装饰器的各种用法：从常见的内置装饰器，到自定义高级装饰器，最后到标准库 functools 中那些常用的工具。

在本文中，你将了解到：
* 为什么装饰器在现代 Python 编程中如此重要，尤其是在机器学习项目中；
* 装饰器的底层工作机制；
* 了解常用的装饰器，实现缓存、上下文管理、模型包裹、配置注入等实际任务；
* 如何从零构建一个的装饰器，并集成到你的代码库中；
* 了解 `functools` 中常用的装饰器。


## 1. 机器学习项目中的常用装饰器

### 类方法装饰器：`@classmethod`

源代码：https://github.com/huggingface/trl/blob/main/trl/core.py#L91

在 HuggingFace 的 trl 库源码中，有一个很经典的例子：

```python
class PPODecorators:
    optimize_device_cache = False

    @classmethod
    @contextmanager
    def empty_device_cache(cls):
        yield
        if cls.optimize_device_cache:
            if is_torch_xpu_available():
                gc.collect()
                torch.xpu.empty_cache()
                gc.collect()
            elif is_torch_npu_available():
                gc.collect()
                torch.npu.empty_cache()
                gc.collect()
            elif torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
```

这个 `@classmethod` 的作用是在类的作用域内定义一个可以访问类变量的上下文管理器（context manager）。一个函数如果被 `@classmethod` 修饰：它不是实例方法，而是类方法，第一个参数是 cls（类本身），可访问类属性，比如 `cls.optimize_device_cache`；被 `@contextmanager` 修饰：这意味着它是一个上下文管理器，可以用 with 语句调用：
```python
with PPODecorators.empty_device_cache():
    # 做某事，比如运行强化学习优化步骤
```

如果没有 `@classmethod`，这个函数只能作为普通函数或实例方法调用，无法通过类来访问类变量。加了 `@classmethod` 后，它就可以这样访问：

```python
if cls.optimize_device_cache:
    ...
```
也就是说，它能根据类变量 `optimize_device_cache` 的值，动态控制是否执行清理缓存的操作。


**为什么使用类变量而不是成员变量？**

类变量定义在类中、所有实例共享、属于类，通过 `ClassName.var` 或 `self.__class__.var` 来访问。成员变量定义在 `__init__` 或实例方法中，属于具体实例，通过 `self.var` 访问。

**为什么这里用类变量 `optimize_device_cache`？**

因为这个变量控制的是一个全局性的行为开关 —— 是否清理 device cache，这个开关对所有实例生效；不属于某个具体对象，而是属于策略整体（PPO）或系统整体；简化状态控制逻辑（不需要实例化 `PPODecorators` 就能用）。使用类变量的典型场景包括：

* 配置选项开关（如 `optimize_device_cache`, `DEBUG = True`）；
* 缓存或注册表（如 `model_registry = {}`）；
* 计数器、共享资源（如 `instance_count = 0`）；
* 工具类、不需要实例化的场景（如静态方法、上下文管理器、装饰器类）；
在这种场景下，使用成员变量反而会增加实例化负担并带来状态一致性问题。


### 上下文管理器装饰器：`@contextmanager`

当你用 `@contextmanager` 装饰一个函数时，这个函数就变成了一个上下文管理器，可以用 `with ... as x:` 的方式调用。

`yield` 的作用相当于：把 `yield` 后面的那个值“返回”给 `with` 的 `as` 部分；是你在 `with` 语句中能操作的那个对象；而 `yield` 本身也标记了上下文的“中间点”：
* `yield` 之前执行的是进入上下文（`__enter__()`）；
* `yield` 之后执行的是退出上下文（`__exit__()`）；

上下文管理器中的代码执行顺序是怎样的？

Python 的上下文管理器执行流程（即 `with` 语句），以上边代码为例：

```python3
with PPODecorators.empty_device_cache():
    do_something()
```

其执行顺序如下：

* 调用上下文管理器（即调用 `empty_device_cache()`）；
* 进入上下文：执行 `yield` 前的代码；
* 执行 `with` 块内部的语句（如 `do_something()`）；
* `with` 块执行完后（或发生异常），继续执行 `yield` 之后的代码；
* 退出上下文。

```python  
@contextmanager
def empty_device_cache(cls):
    yield  # 此处是 with 块内部代码的插入点
    if cls.optimize_device_cache:
        ...  # 清理 cache 的代码在 with 语句执行后执行
```

`yield` 前面：没有代码（这里空着）；yield 后面：会在 with 执行结束后自动触发；所以清理操作是在 with 块结束之后进行的。类似于这样一段等价逻辑：

```python  
gen = empty_device_cache()
next(gen)           # 进入上下文，执行 yield 之前（此处是空）
do_something()      # 你在 with 里的代码
next(gen) or gen.close()   # 执行 yield 之后的逻辑
```

下面我们来看一个具体的例子：

源代码：https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L185

```python  
@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model
```

这是一个上下文管理器函数，装饰器 `@contextmanager` 来自 `contextlib`，用于简化上下文管理器的编写。它的功能是：根据是否使用 DeepSpeed Stage 3，以及是否需要 `gather` 参数，返回一个“可用于生成任务”的未包装模型。和上一个例子类似，流程如下：

```python  
with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
    # 执行这段代码时，yield 前面的代码已经执行完
    # yield 返回的对象会赋值给 unwrapped_model
```

此时：
* `yield` 之前的语句 = 上下文进入阶段（做准备工作，如参数聚合）；
* `yield` 的值 = 真正“解开包装”的模型 unwrapped_model，作为 with 语句中可用的变量；
* `yield` 之后的语句（如果有）= 上下文退出阶段（做清理工作，如重新加 hook）；


如果逐步看逻辑分支：
* 情况一：不是 DeepSpeed Stage 3
    ```python
    if accelerator.state.deepspeed_plugin is None or accelerator.state.deepspeed_plugin.zero_stage != 3:
      yield unwrapped_model
    ```
  直接返回解包的模型；没有复杂操作；对应普通 DDP 或 DeepSpeed Stage 1/2。
* 情况二：是 DeepSpeed Stage 3 且不需要 gather
    ```python
    if not gather_deepspeed3_params:
      yield accelerator.unwrap_model(model)
    ```
  如果使用 DeepSpeed ZeRO Stage 3 且不聚合参数，则跳过参数收集；更省显存，但生成速度可能变慢；这里也直接 yield 一个解包模型。
* 情况三：是 DeepSpeed Stage 3 且需要 gather
    
    ```python
    with deepspeed.zero.GatheredParameters(model.parameters()):
    remove_hooks(model)
    yield accelerator.unwrap_model(model)
    add_hooks(model)
    ```

这个写法意味着：

* 你可以用相同代码支持各种分布式训练包装器（DDP/DeepSpeed）；
* 解包逻辑自动处理，并根据条件执行必要的参数聚合和 `hook` 清理；
* 在 with 内部，你可以像对普通模型一样 `.generate(...)`；
* 并且 `with` 块结束后，它会自动清理状态（如恢复 `hook`）。

### 抽象类装饰器：`@abstractmethod` 和 函数属性化装饰器：`@property`

下面的例子，我们来看看 Python 中两个用于定义类接口的经典装饰器：`@abstractmethod` 和 `@property`。它们常常联合使用，用于构建面向对象架构中严格的接口规范。这种写法在框架设计、分布式系统和机器学习服务端代码中尤为常见。让我们从一个实际的例子出发，理解它们是如何协同工作的。

源代码：https://github.com/vllm-project/vllm/blob/main/vllm/engine/protocol.py#L27

```python
from abc import ABC, abstractmethod
class EngineClient(ABC):

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        ...

    @property
    @abstractmethod
    def errored(self) -> bool:
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...
```


**什么是 `ABC`？**

`ABC` 是` abstract base class`（抽象基类）的缩写，来自 `abc` 模块：`from abc import ABC`。使用 `ABC` 来定义一个接口或者协议类（protocol class），它规定了子类必须实现的函数或属性；可以用于设计更严谨、模块化和面向对象的代码结构；在运行时不能直接实例化含有未实现抽象方法的类，会抛出 `TypeError`。

**为什么要用 `@abstractmethod`？**

`@abstractmethod` 表示该方法或属性是抽象的，必须由子类实现。搭配 `ABC` 使用后，该类变成一个不能被直接实例化的抽象类。任何继承这个类的子类，必须实现所有标记为 `@abstractmethod` 的方法或属性，否则也不能实例化。

**为什么要用 `@property`？**

`@property `是将方法变为一个“属性”，像访问字段一样调用方法。以下面为例：
```python
@property
def is_running(self) -> bool:
    ...
```
使用 `@property` 有以下好处：
* 更直观：调用 `obj.is_running` 而不是 `obj.is_running()`，可读性更强；
* 封装内部逻辑：虽然是“属性”，但可以在方法内部执行逻辑判断；
* 统一接口：对于某些状态属性（如 `is_running`, `errored`），看起来像字段，实际由逻辑动态计算。


**总结：`EngineClient` 的设计讨论**

这个 `EngineClient` 类是一个接口定义，用于规范所有“客户端”类的结构：
* 要求客户端实现一些状态属性（如 `is_running`）；
* 要求实现一个异步生成方法 generate；
* 用 `ABC` 和 `@abstractmethod` 强制约定所有子类必须实现这些接口；
* 用 `@property` 提供更清晰的状态属性接口。

这种写法在大型工程中非常常见，是一种优雅的接口设计方式。



### 静态方法装饰器：`@staticmethod`

`@staticmethod` 是 Python 中的一种方法修饰器，表示“静态方法”：

```
class MyClass:
    @staticmethod
    def foo(x):
        ...
```

被 `@staticmethod` 修饰的方法：

* 不接收 `self` 或 `cls` 参数；
* 不能访问类的实例属性 (`self.x`) 或类变量 (`cls.y`)；
* 和普通函数几乎一样，只是放在类的命名空间中，作为类的一部分来组织逻辑。

源代码：https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L568

```python
@staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }
```

那为什么这里要用 `@staticmethod`？

```python
@staticmethod
def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
```

这个方法的特点：
* 不用 `self`（不是针对某个对象）；
* 不用 `cls`（不是访问某个类变量）；
* 只是处理输入数据的一段工具逻辑；
* 和类 `DPOTrainer` 的其他成员没有状态上的依赖；

所以它是一个纯函数，逻辑封装在类中，但不依赖类或实例状态，用 `@staticmethod` 是最合适的。使用 `@staticmethod` 的好处有
* 明确语义、表达“这个函数不依赖对象或类，只是一段逻辑”
* 不需要实例化、可以直接 `ClassName.method()` 来调用
* 更清晰的代码结构、属于类的一部分，但保持函数的独立性
* 更容易测试、不涉及类状态，测试时不需构造实例
在调用时，可以直接用`tokenized = DPOTrainer.tokenize_row(features, tokenizer, 3, 3, False)`,不需要先实例化 `DPOTrainer`。这个 `tokenize_row` 明显只是一个和 tokenization 相关的工具函数，所以用 `@staticmethod` 是非常恰当的设计。

`@staticmethod` 不需要 `self` / `cls`，不可访问类/实例状态，是工具函数，不依赖类或实例状态。@classmethod 需要 `cls`，可访问类变量，涉及类范围逻辑（如工厂方法）。

有一条很实用的工程实践经验：如果一个方法不依赖类的状态（实例或类变量），就考虑把它变成静态方法；如果它甚至不属于这个类的“概念域”，那就干脆做成独立函数。

什么时候应该写成独立的 util function？当这个函数跨多个类都有用；或者它的逻辑与当前类的语义没有强绑定关系；例如：tokenizer 的 padding 函数，字符串清洗，通用的日志格式化等。这时候写成 `utils.py` 里单独的函数会更好，易于复用、解耦和测试。



## 2. 自定义函数装饰器

### 装饰器基础回顾

当你希望对函数执行的行为进行统一、自动、可复用的修改或增强时，装饰器是最好的选择。

常见用途包括：
* 日志记录，比如打印函数名、输入参数、返回值
* 性能分析（profiling），如自动记录执行时间、内存占用
* 缓存（memoization），记住函数输出避免重复计算
* 权限控制 / 验证，检查用户权限、参数合法性
* 并发控制，比如给函数加锁、多线程保护
* 重试机制，比如函数失败后自动重试（如 API 调用）


装饰器的本质：是“语法糖”，当你看到的这个写法：

```python
@my_decorator
def foo():
    ...

# 其实 等价于：

def foo():
    ...

foo = my_decorator(foo)
```

任何函数，只要它接受另一个函数作为参数，并返回一个可调用对象（通常也是个函数），就可以作为装饰器。


```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper
```

`@my_decorator` 语法糖：会将下方的 `foo` 函数作为参数 `func` 传入 `my_decorator`；`my_decorator(func)` 返回的是一个新的函数 `foo = wrapper`，所以原来的 `foo()` 被替换成了带前后逻辑的新函数。


**带参数的装饰器怎么办？**

如果你想这样写：

```python
@my_decorator_with_args("DEBUG")
def foo(): ...
```

就需要两层函数嵌套：

```python
def my_decorator_with_args(log_level):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{log_level}] Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return real_decorator
 
```

使用时：

```python
@my_decorator_with_args("DEBUG")  # 实际执行顺序：
def foo():
    pass

# 等价于：
# foo = my_decorator_with_args("DEBUG")(foo)
```

所以你看到的 `@something(...)` 实际是：

先执行 `something(...)`  再返回一个真正的装饰器函数；

再传入 `foo` 给这个函数。

### `trl` 性能分析装饰器`@profiling_decorator`

在 `HuggingFace trl` 库中有这样一个装饰器，可以自动记录函数执行时间，非常适合大型训练库：

源代码：https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L822 

```python
class GRPOTrainer(Trainer):
...
    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

def profiling_decorator(func: callable) -> callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    Args:
        func (`callable`):
            Function to be profiled.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator

    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


@contextmanager
def profiling_context(trainer: Trainer, name: str) -> Generator[None, None, None]:
    """
    A context manager function for profiling a block of code. Results are logged to Weights & Biases or MLflow
    depending on the trainer's configuration.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object.
        name (`str`):
            Name of the block to be profiled. Used as a key in the logged dictionary.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_context

    class MyTrainer(Trainer):
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            with profiling_context(self, "matrix_multiplication"):
                # Code to profile: simulate a computationally expensive operation
                result = A @ B  # Matrix multiplication
    ```
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    profiling_metrics = {f"profiling/Time taken: {trainer.__class__.__name__}.{name}": duration}
    if "wandb" in trainer.args.report_to and wandb.run is not None and trainer.accelerator.is_main_process:
        wandb.log(profiling_metrics)

    if "mlflow" in trainer.args.report_to and mlflow.run is not None and trainer.accelerator.is_main_process:
        mlflow.log_metrics(profiling_metrics, step=trainer.state.global_step)
```

这个装饰器干了什么事？这个装饰器自动帮我们进行函数的性能分析，无需手动埋点。

```python

def profiling_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):  # ← 这里是“计时器”
            return func(self, *args, **kwargs)
    return wrapper
```

这个装饰器做了：

* 创建一个包装器函数 `wrapper`；
* 使用 `with profiling_context(...)` 包裹原函数的调用；
* `profiling_context` 是个上下文管理器，负责计时或收集 profiler 数据；
* `return func(...)` 调用原函数，让你函数本体逻辑得以正常执行；
* 用 `@functools.wraps(func)` 保留原函数的元信息（如名字、docstring）；

有些函数需要性能分析，但你不希望每个函数都手写 profiling 逻辑；有了装饰器，你可以一行代码就加入 profiling 功能；而且它自动记录函数名、作用范围清晰、非常适合大规模训练类库使用。

## 3. `functools` 模块中的常用装饰器

### 保留元信息：`@functools.wrap`

正如上边这个例子所展示的，这个装饰器的作用是在装饰器中保留原函数的元信息（比如名字、docstring、签名），通常在自定义装饰器中起到不可替代的作用。`@functools.wrap` 的用法是：

```python
@functools.wraps(orig_method)
def wrapped_method(model_self, *args, **kwargs):
    ...
```

用于保留原函数 `orig_method` 的元信息：
* 函数名 `__name__`：否则变成 `wrapped_method`；
* 函数注释 `__doc__`；
* 函数签名信息；
* 有助于调试、日志打印、文档工具、tracing 工具（比如 `TorchScript`）；

不加 `@wraps` 的话：
```python
>>> module.forward.__name__
'wrapped_method'
```
加了 `@wraps(forward)`：

```python
>>> module.forward.__name__
'forward'
```

为什么在自定义装饰器中总是会出现呢？上边在讲述装饰器这个语法糖的时候就提过，当你看到的这个写法：

```python
@my_decorator
def foo():
    ...
```

其实等价于：
```python
def foo():
    ...

foo = my_decorator(foo)
```

这里，因为有 `foo = my_decorator(foo)` 这一语句，就会导致 `foo` 本身的元信息（比如函数名、文档信息等）被改为 `my_decorator` 的，这会带来很多问题，特别是：
* 使用 IDE 查看函数，看不到原注释和签名
* 分析代码时，分析不到正确的函数结构
* 自动生成文档时，无法展示正确信息；
* 断点调试时，调试器显示的是 wrapper，不是原函数

当你按照以下方式正确定义装饰器时：

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    """This function says hello"""
    print("Hello!")

print(say_hello.__name__)   # Output: 'say_hello'
print(say_hello.__doc__)    # Output: 'This function says hello'
```
就像告诉 Python：这个 `wrapper` 函数是为 `func` 服务的，请帮我把 func 的身份和信息都转移到 `wrapper` 上。因此，只要你写了装饰器函数，就应该几乎总是加上 `@functools.wraps(func)`，除非你确实不需要保留原函数的任何信息（这几乎不可能）。

除此之外，在不涉及到装饰器的定义时，`@functools.wrap` 也是经常被用到的装饰器。我们看下边 `trl` 的另一个例子：

源代码：https://github.com/huggingface/trl/blob/main/trl/trainer/online_dpo_trainer.py#L381 

```python
@wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
```

上边的代码是对 HuggingFace Trainer 类中 `get_eval_dataloader()` 方法的重写或增强版本。

```python
@wraps(Trainer.get_eval_dataloader)
def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
    ...
```

这里 `@wraps(Trainer.get_eval_dataloader)` 其实是说：“我写了一个新的 `get_eval_dataloader()`，它逻辑增强了原方法，但我希望保留原方法的元信息（如名字、文档、签名等）。” 这里可能涉及到的重要用法是：重写父类方法时保留原方法信息。我们把上述过程抽象出来：

```python
from functools import wraps

class Base:
    def say(self):
        """Say something."""
        print("Base says")

class Sub(Base):
    @wraps(Base.say)
    def say(self):
        print("Sub overrides")

print(Sub.say.__name__)  #  'say'
print(Sub.say.__doc__)   #  'Say something.'
```

在类中重写方法时用 `@wraps(...)`，是为了让子类方法从外部看来与父类保持一致性，便于 IDE、文档、调试器、装饰链分析等工具识别。

另一方面，手动 wrap 实例方法（monkey patch），也会用到 `@functools.wrap`。Monkey patching 是指在程序运行时（而不是在源代码中），动态修改类、模块或函数的行为。通俗讲：你没有改动原代码文件，但你在代码运行时“偷偷”改写了某个函数或类的实现。这在调试过程中修改第三方库的行为等经常出现：

```python
class Greeter:
    def greet(self, name):
        """Greet someone."""
        return f"Hello {name}"

g = Greeter()
original = g.greet

@functools.wraps(original)
def new_greet(self, name):
    print("Pre-hook")
    result = original(name)
    print("Post-hook")
    return result

g.greet = new_greet.__get__(g, Greeter)

print(g.greet.__name__)  #  'greet'
print(g.greet.__doc__)   #  'Greet someone.'
```

这种 monkey-patch 场景里，`@wraps(...)` 保证替换后的方法依然像原方法一样，避免“信息污染”。


下面，我们说到这儿了，就顺便说说 `functools` 这个库里在机器学习领域常用的其他装饰器。

### 自动缓存计算结果：`@functools.cache` 和 `@functools.lru_cache`

`@functools.cache` 是标准库 functools 中的一个装饰器，作用是：自动缓存函数的返回值（基于输入参数），避免重复计算，提高效率。

源代码：https://github.com/volcengine/verl/blob/main/verl/utils/import_utils.py#L24

```python
@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None


@cache
def is_vllm_available():
    try:
        vllm_spec = importlib.util.find_spec("vllm")
    except ModuleNotFoundError:
        vllm_spec = None
    return vllm_spec is not None


@cache
def is_sglang_available():
    try:
        sglang_spec = importlib.util.find_spec("sglang")
    except ModuleNotFoundError:
        sglang_spec = None
    return sglang_spec is not None
```

这个函数的使用非常适合以下情况：
* 结果只与输入有关，且不会随时间改变，比如判断模块是否安装、计算斐波那契数、路径查找
* 函数代价高但结果稳定，比如加载模型、查找依赖、编译过程
* 不希望函数多次重复运行，比如检查、探测、初始化的函数

以上边的例子为例：

```python
@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None
```

这个函数的行为是：调用 `importlib.util.find_spec()` 判断模块是否存在；这个操作涉及搜索系统路径、加载信息，是 I/O 密集操作；结果在整个程序生命周期中是稳定不变的；所以调用一次之后缓存结果非常合理！

`@cache` 背后使用的是无上限的字典缓存，函数参数作为 key，返回值作为 value：

```python
def f(x): ...
f(1)  # → 计算并缓存
f(1)  # → 直接返回缓存的结果，不再执行函数体
```

而 `@functools.lru_cache` 是 `@functools.cache` 的“进化版”或“更灵活版”，功能更强，控制更细。`@functools.lru_cache(maxsize=N)` 缓存最近 N 个调用结果；支持“最近最少使用（Least Recently Used）”自动清理。我们看下边这个例子：

源代码：https://github.com/vllm-project/vllm/blob/main/vllm/engine/output_processor/multi_step.py#L72 

```python
@functools.lru_cache
    def _log_prompt_logprob_unsupported_warning_once():
        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        logger.warning(
            "Prompt logprob is not supported by multi step workers. "
            "(e.g., speculative decode uses multi step workers).")
```

这段代码的目的其实很明确：这个警告只想打印一次，即使函数被调用很多次。这个函数没有参数，所以：
* 第一次调用时，会打印警告并缓存返回值（即 `None`）；
* 后续调用时，由于参数相同（空），会直接返回缓存结果，不再执行函数体；
* 所以：`log` 只会执行一次。

其实是等价于：        `@functools.lru_cache(maxsize=128)  # 默认缓存上限是 128 个不同输入组合`。`@functools.lru_cache(maxsize=None)` 就是 `@cache`，无限缓存

理论上，相同的目的也可以通过 if-else 来实现：

```python
_warned = False

def _log_prompt_logprob_unsupported_warning_once():
    global _warned
    if not _warned:
        logger.warning("...")
        _warned = True
```

但是使用 `@lru_cache` 的好处是：
* 简洁，一行搞定，无需管理全局变量；
* 线程安全，内部缓存机制是线程安全的；
* 函数式风格更清晰，无副作用变量，适合多人协作的模块；
* 可拓展支持参数缓存，如果将来需要对不同参数打印不同 warning，可以直接加参数。

更复杂用法：只对某个 key 打一次日志

```python
@functools.lru_cache(maxsize=None)
def warn_once_for_key(key):
    logger.warning(f"Warning for {key}")
```

调用：
```python
warn_once_for_key("feature_a")  # 打一次
warn_once_for_key("feature_a")  # 不打了
warn_once_for_key("feature_b")  # 新 key 打一次
```
将 `@cache` 和 `@property` 可以得到将两者作用结合到一起的 `@functools.cached_property`。这里因为篇幅原因不再赘述。


## 4. 总结

从 `vllm` 和 `trl` 两个热门开源库的真实代码出发，我们剖析了机器学习项目中 Python 装饰器的各种用法，包括类方法、上下文管理器、抽象方法、静态方法、自定义装饰器以及标准库中的缓存装饰器。装饰器的巧妙使用不仅让代码更加干净整洁，更重要的是能显著提高项目的可维护性与扩展性。希望通过本文，你能对装饰器的本质和应用有更深刻的理解，甚至能在自己的机器学习项目中创造出更多优雅而强大的装饰器设计！
