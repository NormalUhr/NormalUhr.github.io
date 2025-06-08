# Decorators in Machine Learning Projects

Python decorators are arguably one of the most powerful language features you will ever encounter. They allow you to add new functionality to your functions or classes without directly modifying the original code. In this post, we will deep-dive into the source code of two top-tier open-source machine-learning frameworks—**vllm** and **trl**—and dissect how Python decorators are used in practice: from built-in decorators you see every day, to advanced custom decorators, and finally to the handy utilities hidden in the standard-library module **`functools`**.

In this article, you will learn:
* Why decorators are so important in modern Python programming, especially in machine learning projects.
* The underlying mechanics of how decorators work.
* How to use common decorators to implement practical tasks like caching, context management, model wrapping, and configuration injection.
* How to build a decorator from scratch and integrate it into your codebase.
* About the commonly used decorators in `functools`.

## 1. Common Decorators in ML Projects

### Class-method decorator: `@classmethod`

Source code: https://github.com/huggingface/trl/blob/main/trl/core.py#L91

A classic example can be found in the source code of HuggingFace's TRL library:

```python3=
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
The purpose of `@classmethod` here is to define a context manager within the class's scope that can access class variables. When a function is decorated with `@classmethod`: it becomes a class method, not an instance method. Its first parameter is `cls` (the class itself), allowing it to access class attributes, such as `cls.optimize_device_cache`. When decorated with `@contextmanager`: it becomes a context manager that can be invoked with a `with` statement:

```python3=
with PPODecorators.empty_device_cache():
    # e.g. run a PPO optimization step
```

Without `@classmethod`, this function could only be called as a regular function or an instance method, and it wouldn't be able to access class variables through the class. By adding `@classmethod`, it can access them like this:

```python3=
if cls.optimize_device_cache:
    ...
```

This means it can dynamically control whether to perform the cache clearing operation based on the value of the class variable `optimize_device_cache`.

**Why use a class variable instead of an instance variable?**

A class variable is defined within a class, shared by all instances, and belongs to the class. It is accessed via `ClassName.var` or `self.__class__.var`. An instance variable is defined in `__init__` or an instance method, belongs to a specific instance, and is accessed via `self.var`.

Here `optimize_device_cache` acts as a global behavioral switch, whether or not to clear the device cache, affecting all instances. It is not tied to a particular object but to the PPO strategy (or the whole system) as a whole, simplifying state-management logic because you don’t even need to instantiate `PPODecorators` to use it.

Why is the class variable optimize_device_cache used here?

Because this variable controls a global behavior—whether to clear the device cache. This switch should apply to all instances; it doesn't belong to a specific object but rather to the PPO strategy or the system as a whole. This simplifies the state control logic (you don't need to instantiate PPODecorators to use it). Typical scenarios for using class variables include:

* Configuration option flags (e.g., `optimize_device_cache`, `DEBUG = True`).
* Caches or registries (e.g., `model_registry = {}`).
* Counters, shared resources (e.g., `instance_count = 0`).
* Utility classes, scenarios where instantiation is unnecessary (e.g., static methods, context managers, decorator classes).

In such scenarios, using an instance variable would add the burden of instantiation and introduce state consistency problems.

### Context-Manager Decorator: `@contextmanager`

When you decorate a function with `@contextmanager`, that function becomes a context manager that can be used with the `with ... as x`: syntax.

The `yield` statement effectively "returns" the value that follows it to the `as` part of the `with` statement; this is the object you can operate on within the with block. `yield` also marks the "midpoint" of the context:

* Code before yield is executed upon entering the context (`__enter__()`).
* Code after yield is executed upon exiting the context (`__exit__()`).
* The value yielded: the object bound to the as target in the with statement.

What is the execution order in the code above?

The execution flow of a Python context manager (i.e., a `with` statement), using the code above as an example:

```python3
with PPODecorators.empty_device_cache():
    do_something()
```
The execution order is as follows:
1. The context manager is called (i.e., `empty_device_cache()` is called).
2. Enter the context: The code before `yield` is executed.
3. The statements inside the with block are executed (e.g., `do_something()`).
4. After the `with` block finishes (or an exception occurs), the code after `yield` is executed.
Exit the context.

```python3=  
@contextmanager
def empty_device_cache(cls):
    yield  # This is the insertion point for the code inside the with block
    if cls.optimize_device_cache:
        ...  # The cache clearing code executes after the with statement
```

Before `yield`: There is no code (it's empty here). After `yield`: The code is automatically triggered after the with block finishes. Therefore, the cleanup operation is performed after the with block concludes. This is equivalent to the following logic:

```python3=  
gen = empty_device_cache()
next(gen)            # Enter the context, execute code before yield (empty here)
do_something()       # Your code inside the 'with' block
try:
    next(gen)        # Execute logic after yield
except StopIteration:
    pass             # Generator is exhausted
```

Let's look at a more concrete example:


Source Code: https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L185

```python3=  
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

This is a context manager function, decorated with `@contextmanager` from `contextlib` to simplify writing context managers. Its function is to return an "unwrapped model ready for generation tasks," based on whether DeepSpeed Stage 3 is used and whether parameters need to be gathered. Similar to the previous example, the flow is as follows:


```python3=  
with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
    # When this code is executed, the code before yield has already run.
    # The object returned by yield is assigned to unwrapped_model.
```

At this point:
* Statements before `yield` = Context entry phase (doing preparatory work, like parameter aggregation).
* The value of `yield` = The truly "unwrapped" model, `unwrapped_model`, available as a variable inside the `with` statement.
* Statements after `yield` (if any) = Context exit phase (doing cleanup, like re-adding hooks).


Let's look at the logical branches step-by-step:
* Case 1: Not DeepSpeed Stage 3
    ```python3=
    if accelerator.state.deepspeed_plugin is None or accelerator.state.deepspeed_plugin.zero_stage != 3:
      yield unwrapped_model
    ```
  It directly returns the unwrapped model; there are no complex operations. This corresponds to regular DDP or DeepSpeed Stage 1/2.
* Case 2: DeepSpeed Stage 3 and no gather needed
    ```python3=
    if not gather_deepspeed3_params:
      yield accelerator.unwrap_model(model)
    ```
  If using DeepSpeed ZeRO Stage 3 without gathering parameters, it skips parameter collection. This saves VRAM but may slow down generation. Here, it also directly `yield`s an unwrapped model.
* Case 3: DeepSpeed Stage 3 and gather is needed
    ```python3=
    with deepspeed.zero.GatheredParameters(model.parameters()):
    remove_hooks(model)
    yield accelerator.unwrap_model(model)
    add_hooks(model)
    ```
This implementation means:
* You can use the same code to support various distributed training wrappers (DDP/DeepSpeed).
* The unwrapping logic is handled automatically, performing necessary parameter aggregation and hook cleanup based on conditions.
* Inside the `with` block, you can call `.generate(...)` as if it were a regular model.
* And after the `with` block finishes, it automatically cleans up the state (e.g., restores hooks).

### Abstract Class Decorator: `@abstractmethod` and Property Decorator: `@property`

In the next example, we'll look at two classic decorators in Python used for defining class interfaces: `@abstractmethod` and `@property`. They are often used together to build strict interface specifications in object-oriented architectures. This pattern is particularly common in framework design, distributed systems, and machine learning serving code. Let's start with a practical example to understand how they work together.

源代码：https://github.com/vllm-project/vllm/blob/main/vllm/engine/protocol.py#L27

```python3=
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


**What is `ABC`?**

`ABC` is short for Abstract Base Class, from the `abc` module: `from abc import ABC`. An `ABC` is used to define an interface or a protocol class. It specifies the methods or properties that subclasses must implement. It helps in designing more robust, modular, and object-oriented code structures. At runtime, you cannot directly instantiate a class that has unimplemented abstract methods; doing so will raise a `TypeError`.

**Why use `@abstractmethod?`**

`@abstractmethod` indicates that a method or property is abstract and must be implemented by subclasses. When used with `ABC`, the class becomes an abstract class that cannot be instantiated directly. Any subclass that inherits from this class must implement all methods and properties marked with `@abstractmethod`; otherwise, it cannot be instantiated either.

**Why use `@property?`**

`@property` turns a method into a "property," allowing it to be accessed like a field. Take the following example:

```python3=
@property
def is_running(self) -> bool:
    ...
```
Using `@property` has the following benefits:
* More Intuitive: You call `obj.is_running` instead of `obj.is_running()`, which is more readable.
* Encapsulates Internal Logic: Although it looks like a property, you can execute logical checks within the method.
* Uniform Interface: For certain state attributes (like `is_running`, `errored`), they appear as fields but are actually dynamically computed by logic.

**Summary: `EngineClient` Design Discussion**

The `EngineClient` class is an interface definition that standardizes the structure for all "client" classes:
* It requires client classes to implement certain state properties (like `is_running`).
* It requires the implementation of an asynchronous generate method.
* It uses `ABC` and `@abstractmethod` to enforce that all subclasses must implement these interfaces.
* It uses `@property` to provide a cleaner interface for state attributes.

This approach is very common in large-scale engineering and is an elegant way to design interfaces.



### Static Method Decorator: `@staticmethod`

`@staticmethod` is a method decorator in Python that indicates a "static method":

```
class MyClass:
    @staticmethod
    def foo(x):
        ...
```

A method decorated with `@staticmethod`:
* Does not receive `self` or `cls` as the first argument.
* Cannot access instance attributes (`self.x`) or class variables (`cls.y`).
* Is almost identical to a regular function, but it is placed within the class's namespace to logically organize it as part of the class.

Source Code: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L568

```python3=
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

**So why use `@staticmethod` here?**

```python3=
@staticmethod
def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
```

The characteristics of this method are:
* It doesn't use `self` (it's not specific to any one object).
* It doesn't use `cls` (it doesn't access any class variables).
* It's just a utility function for processing input data.
* It has no state dependency on other members of the `DPOTrainer` class.

Therefore, it is a pure function encapsulated within the class but not dependent on the class or instance state, making `@staticmethod` the most appropriate choice. The benefits of using `@staticmethod` include:
* **Semantic Clarity**: Expresses that "this function does not depend on an object or class; it's just a piece of logic."
* **No Instantiation Required**: Can be called directly via `ClassName.method()`.
* **Clearer Code Structure**: It's part of the class but maintains functional independence.
* **Easier to Test**: Since it doesn't involve class state, you don't need to construct an instance for testing.

When calling it, you can use `tokenized = DPOTrainer.tokenize_row(features, tokenizer, 3, 3, False)` directly, without first instantiating `DPOTrainer`. The `tokenize_row` method is clearly just a utility function related to tokenization, so using `@staticmethod` is a very fitting design choice.

`@staticmethod` does not require `self` or `cls`, cannot access class/instance state, and acts as a utility function independent of class or instance state. `@classmethod` requires `cls`, can access class variables, and is used for logic that operates at the class level (like factory methods).

A very practical engineering rule of thumb is: if a method does not depend on the state of the class (instance or class variables), consider making it a static method. If it doesn't even belong to the "conceptual domain" of the class, then make it an independent function altogether.

When should you write an independent utility function? When the function is useful across multiple classes, or its logic has no strong binding to the semantics of the current class. For example: a padding function for a tokenizer, string cleaning, or generic log formatting. In such cases, writing it as a separate function in utils.py is better for reusability, decoupling, and testing.

## 2. Custom Function Decorators

### A Quick Recap of Decorator Basics

When you want to modify or enhance the behavior of a function in a uniform, automatic, and reusable way, decorators are the best choice.

Common use cases include:
* **Logging:** such as printing the function name, input arguments, and return value.
* **Profiling:** such as automatically recording execution time and memory usage.
* **Caching (Memoization):** remembering function outputs to avoid redundant computations.
* **Access Control / Validation:** checking user permissions or the validity of parameters.
* **Concurrency Control:** for example, adding locks to a function for thread safety.
* **Retry Mechanisms:** for instance, automatically retrying a function after failure (e.g., an API call).

The essence of a decorator is "syntactic sugar." When you see this syntax:


```python3=
@my_decorator
def foo():
    ...

# It is actually equivalent to:

def foo():
    ...

foo = my_decorator(foo)
```

Any function that accepts another function as an argument and returns a callable object (usually also a function) can be used as a decorator.

```python3=
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper
```

The `@my_decorator` syntactic sugar passes the foo function below it as the `func` argument into `my_decorator`. The call `my_decorator(func)` returns a new function, `wrapper`, so the original `foo()` is replaced by the new function that includes the before/after logic.


**What About Decorators with Arguments?**

If you want to write something like this:

```python3=
@my_decorator_with_args("DEBUG")
def foo(): ...
```

You need two levels of function nesting:


```python3=
def my_decorator_with_args(log_level):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{log_level}] Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return real_decorator
 
```

When used:


```python3=
@my_decorator_with_args("DEBUG")  # 实际执行顺序：
def foo():
    pass

# This is equivalent to:
# foo = my_decorator_with_args("DEBUG")(foo)
```

So, what you see as `@something(...)` is actually:

* First, something(...) is executed, which returns a true decorator function.
* Then, the foo function is passed to this returned decorator.



### `trl`'s Performance Profiling Decorator: `@profiling_decorator`

In the HuggingFace TRL library, there is a decorator that can automatically record a function's execution time, which is highly suitable for large-scale training libraries.

Source code: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L822 

```python3=
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

What does this decorator do? It automatically performs performance analysis for our functions, eliminating the need for manual instrumentation.

```python3=

def profiling_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):  # ← This is the "timer"
            return func(self, *args, **kwargs)
    return wrapper
```

This decorator does the following:
* It creates a wrapper function `wrapper`.
* It wraps the original function call with with `profiling_context(...)`.
* `profiling_context` is a context manager responsible for timing and collecting profiler data.
* return `func(...)` calls the original function, allowing the function's body to execute normally.
* It uses `@functools.wraps(func)` to preserve the original function's metadata (like its name and docstring).

Some functions require performance analysis, but you don't want to write profiling logic manually for each one. With a decorator, you can add profiling functionality with a single line of code. It automatically records the function name, has a clear scope of action, and is perfectly suited for use in large-scale training libraries.

## 3. Common Decorators in the `functools` Module

### Preserving Metadata: `@functools.wraps`

As shown in the previous example, the purpose of this decorator is to preserve the original function's metadata (like its name, docstring, and signature) within a decorator. It plays an indispensable role in custom decorator definitions. The usage of `@functools.wraps` is as follows:


```python3=
@functools.wraps(orig_method)
def wrapped_method(model_self, *args, **kwargs):
    ...
```

It is used to preserve the metadata of the original function `orig_method`:
* Function name `__name__`: otherwise, it would become `wrapped_method`.
* Function docstring `__doc__`.
* Function signature information.
* This is helpful for debugging, logging, documentation tools, and tracing tools.

Without` @wraps`:
```python3=
>>> module.forward.__name__
'wrapped_method'
```
With `@wraps(forward)`：

```python3=
>>> module.forward.__name__
'forward'
```

Why does it almost always appear in custom decorators? As mentioned earlier when discussing decorator syntax, when you see this:

```python3=
@my_decorator
def foo():
    ...
```

It is actually equivalent to:

```python3=
def foo():
    ...

foo = my_decorator(foo)
```

Here, the statement `foo = my_decorator(foo)` causes `foo`'s own metadata (like its name and docstring) to be replaced by that of the `wrapper` function returned by `my_decorator`. This can lead to many problems, especially:
* When viewing the function in an IDE, you can't see the original docstring and signature.
* When analyzing code, you can't inspect the correct function structure.
* When auto-generating documentation, you can't display the correct information.
* When debugging with breakpoints, the debugger shows the `wrapper`, not the original function.

When you correctly define a decorator as follows:


```python3=
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
This essentially tells Python that the `wrapper` function is a proxy for `func`, instructing it to transfer all of `func`'s metadata to wrapper. Therefore, whenever you write a decorator function, you should almost always add `@functools.wraps(func)`, unless you truly have no need to preserve the original function's information (which is rarely the case).

Besides its use in decorator definitions, `@functools.wraps` is also frequently used on its own. Let's look at another example from trl:

Source Code: https://github.com/huggingface/trl/blob/main/trl/trainer/online_dpo_trainer.py#L381

```python3=
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

The code above is an overridden or enhanced version of the `get_eval_dataloader()` method from the HuggingFace `Trainer` class.

```python3=
@wraps(Trainer.get_eval_dataloader)
def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
    ...
```

Here, `@wraps(Trainer.get_eval_dataloader)` is essentially saying: "I've written a new `get_eval_dataloader()` that enhances the original method, but I want to preserve the original method's metadata (like its name, docstring, signature, etc.)." An important use case this might involve is preserving the original method's info when overriding a parent class method. Let's abstract this process:

```python3=
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
print(Sub.say.__doc__)   # 'Say something.'
```

Using `@wraps(...)` when overriding a method in a class ensures that the subclass method appears consistent with the parent class method from an external perspective, which facilitates recognition by tools like IDEs, documenters, debuggers, and decorator chain analyzers.

On another note, `@functools.wraps` is also used when manually wrapping instance methods (monkey patching). Monkey patching refers to dynamically modifying the behavior of a class, module, or function at runtime (rather than in the source code). In simpler terms: you don't change the original code file, but you "secretly" rewrite the implementation of a function or class while the code is running. This is common for modifying the behavior of third-party libraries during debugging:

```python3=
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

In this monkey-patching scenario, `@wraps(...)` ensures that the replaced method still looks like the original, avoiding "metadata corruption."

Now that we're on the topic, let's briefly discuss other commonly used decorators from the `functools` library in the machine learning domain.

### Automatically Caching Computation Results: `@functools.cache` and `@functools.lru_cache`

`@functools.cache` is a decorator from the standard library `functools` that automatically caches the return values of a function (based on its input arguments) to avoid redundant computation and improve efficiency.

Source Code: https://github.com/volcengine/verl/blob/main/verl/utils/import_utils.py#L24


```python3=
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

Using this decorator is ideal for situations where:

* The result depends only on the input and does not change over time, such as checking if a module is installed, calculating Fibonacci numbers, or pathfinding.
* The function is expensive to run but its result is stable, such as loading a model, finding dependencies, or a compilation process.
* You don't want the function to run repeatedly, such as checks, probes, or initialization functions.


Taking the example above:


```python3=
@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None
```

This function's behavior is to call `importlib.util.find_spec()` to determine if a module exists. This operation involves searching system paths and loading information, making it an I/O-intensive operation. The result is stable throughout the program's lifecycle, so caching the result after the first call is very reasonable!

Behind the scenes, `@cache` uses an unbounded dictionary for caching, with the function arguments as the key and the return value as the value:

```python3=
def f(x): ...
f(1)  # → computes and caches
f(1)  # → directly returns the cached result, does not execute the function body again
```

`@functools.lru_cache` is an "evolved" or "more flexible" version of `@functools.cache`, offering more powerful and finer-grained control. `@functools.lru_cache(maxsize=N)` caches the results of the last N calls and supports automatic eviction based on a Least Recently Used (LRU) policy. Let's look at the example below:

Source Code: https://github.com/vllm-project/vllm/blob/main/vllm/engine/output_processor/multi_step.py#L72

```python3=
@functools.lru_cache
    def _log_prompt_logprob_unsupported_warning_once():
        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        logger.warning(
            "Prompt logprob is not supported by multi step workers. "
            "(e.g., speculative decode uses multi step workers).")
```

The purpose of this code is clear: this warning should only be printed once, even if the function is called many times. The function has no arguments, so:

* On the first call, it prints the warning and caches the return value (which is `None`).
* On subsequent calls, since the arguments are the same (none), it will directly return the cached result without executing the function body.
* Therefore, the `logger.warning` is executed only once.

This is actually equivalent to `@functools.lru_cache(maxsize=128)` (the default cache size is 128 different input combinations). `@functools.lru_cache(maxsize=None)` is equivalent to `@cache` (unbounded cache).

Theoretically, the same goal could be achieved with an if-else statement:

```python3=
_warned = False

def _log_prompt_logprob_unsupported_warning_once():
    global _warned
    if not _warned:
        logger.warning("...")
        _warned = True
```

However, the benefits of using **@lru_cache** are:

* **Conciseness**: one line does the job, no need to manage global variables.
* **Thread Safety**: the internal caching mechanism is thread-safe.
* **Clearer Functional Style**: no side-effect variables, suitable for modules with multiple contributors.
* **Extensibility**: supports argument-based caching. If you need to print different warnings for different parameters in the future, you can simply add parameters.

A more complex use case: logging only once for a specific `key`.

```python3=
@functools.lru_cache(maxsize=None)
def warn_once_for_key(key):
    logger.warning(f"Warning for {key}")
```

Calling it:
```python3=
warn_once_for_key("feature_a")  # Logs once
warn_once_for_key("feature_a")  # Does not log again
warn_once_for_key("feature_b")  # New key, logs once
```
Combining `@cache` and `@property` gives you `@functools.cached_property`, which merges their effects. We won't elaborate on this here due to space constraints.

## 4. Conclusion

Starting from real-world code in two popular open-source libraries, `vllm` and `trl`, we have dissected the various uses of Python decorators in machine learning projects. This includes class methods, context managers, abstract methods, static methods, custom decorators, and caching decorators from the standard library. The clever use of decorators not only makes code cleaner and more organized but, more importantly, can significantly improve a project's maintainability and extensibility. Hopefully, through this article, you have gained a deeper understanding of the essence and application of decorators and will be inspired to create more elegant and powerful decorator designs in your own machine learning projects!
