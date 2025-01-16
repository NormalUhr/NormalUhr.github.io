> Spoiler alert! This is gonna be a long long blog.

## Why Write This BLOG

I’ve been meaning to write this blog post—or you could call it a summary—for a very long time. *Effective C++* is written and translated by true C++ masters, and this work has enjoyed global renown for many years. But no book is perfect, and no person is flawless. For various reasons (which I’d rather describe as “issues that make me uncomfortable”), I find it necessary to write a summary that serves more like a **reference guide version** of *Effective C++*. My hope is that when I want to revisit a particular item in the future, I can save as much time as possible. If anyone who hasn’t read the book gets impatient because of the translation or other factors, you can take a look at this article. From a Chinese (Mainland) perspective and language habit (the original translator is from Taiwan), I’ll try to cover each item’s most important knowledge points as directly and clearly as possible so you can grasp the core ideas in the shortest time possible, then tackle each problem one by one. I don’t think this article can replace *Effective C++*—it’s far from sufficient. I also won’t include too much code or too many details here. If you want to dig into every detail, pick up the original book and read it, page by page.

First, let me talk about what makes me uncomfortable about this book:

1. Some content is a bit outdated. The book doesn’t cover C++11. In other words, with more recent compilers, many items addressing solutions via C++98 feel somewhat redundant. In my summary of each item, I’ll directly point out solutions in newer C++ versions. Personally, I think some methods introduced in the book can be retired. These include but are not limited to `final, override, shared_ptr, = delete`.
2. The translation is stiff. This isn’t Hou Jie’s fault. Faced with a master’s work, we inevitably vacillate between preserving the original language style and trying to adapt to the language habits of each reader base, which leads to some rather English-flavored expressions appearing in the text, such as “在这个故事结束之前” or “那就进入某某某变奏曲了,” making readers unfamiliar with English feel baffled—“what do they mean by ‘variation’?” Honestly, even I, who know the English original, find it a bit odd. So in my summary, whenever cause and effect are involved, I’ll speak them plainly. Since I’m not a master, I’ll focus on efficiency.
3. The author’s writing style requires the reader to treat the book like a novel. In explaining each item, the author carefully prepared all kinds of jokes, famous quotes, historical references, and examples to minimize a stuffy textbook feel and to make the “lecture” less rigid (although the translation mentioned above stiffened it a bit). But on my second and even third reading of the book, I wanted it to be more like a **reference guide**. For example, if a certain item solves a certain problem, on my first reading I pay attention to **how** the problem is solved; on my second reading, I may want to know in which situations this type of problem might occur—**when to use it**—which is my main concern. Unfortunately, the three scenarios that trigger this problem may be scattered across the corners of that item, requiring me to read once again the jokes (now unfunny) and the historical references (no longer interesting) just to gather them properly. Hence this blog post organizes the points that I care about most when reviewing, aiming to let me recall an item’s outline in two minutes or less. That’s the purpose of this post.

Finally, once again, my utmost respect to Meyers and Hou Jie.

---

## II. Construction, Destruction, and Assignment

Construction and destruction, on the one hand, mark the birth and termination of objects; on the other hand, they also mean resource allocation and deallocation. Errors in these operations can lead to far-reaching consequences—you face risks for every object you create and destroy. These functions form the backbone of a custom class. Ensuring their correct behavior is a matter of life and death.

### Item 05: Know which functions C++ silently writes and calls

For any class you write, the compiler will actively declare a copy constructor, a copy assignment operator, and a destructor; at the same time, if you don’t declare any constructor, the compiler will also declare a default version of the copy constructor. These functions are all `public` and `inline`. Note that this is about declarations only. These functions are only implemented by the compiler if they are called. However, the compiler-generated functions can cause problems if there are references or pointers within the class, `const` members, or if the type is virtual.

- For the copy constructor, consider whether the class’s members require deep copying. If they do, you need to implement your own copy constructor/operator rather than relying on the compiler.
- For the copy constructor, if the class has reference members or `const` members, you need to define the copy behavior yourself, since the compiler-generated copy logic is likely to be problematic for these two cases.
- For the destructor, if the class is meant to be polymorphic, proactively declare the destructor as `virtual`. For details, see Item 07.

Aside from these special cases, if the type is anything more complex than the most trivial kind, write the constructor, destructor, copy constructor, copy assignment operator, move constructor, and move assignment operator (C++11, if necessary) yourself.

### Item 06: If you don’t want the compiler to automatically generate a function, explicitly refuse it

Continuing the previous item, if, in terms of semantics or functionality, your type must forbid certain functions (for example, if copying is disallowed), then you should forbid the compiler from automatically generating them. The author provides two ways to achieve this:

- Declare the forbidden functions as `private` and omit their implementations, preventing calls from outside the class. However, if they’re accidentally called inside the class (from a member function or a friend), you’d get a linker error.
- Move the potential linker error to compile time by designing a non-copyable utility base class, and let your truly non-copyable class privately inherit from that base. But this approach is overly complex for a type that already uses inheritance, as it introduces multiple inheritance and makes code obscure.

But with C++11, you can simply use `= delete` for the copy constructor to explicitly forbid the compiler from generating it.

### Item 07: Declare `virtual` for polymorphic base classes

The core of this item is: **A base class with polymorphic properties must declare its destructor as virtual to prevent only partial destruction of the object when deleting a derived object through a base pointer.** If a class entails polymorphism, it’s almost inevitable that a base pointer or reference will point to a derived object. Because non-virtual functions don’t have dynamic types, if the base’s destructor isn’t virtual, when a base pointer is destroyed, it will call only the base destructor, leading to partial destruction of a derived object and the risk of a memory leak. In addition:

- Note that an ordinary base class does not—and should not—have a virtual destructor, because a virtual function implies cost in both time and space. For details, see *More Effective C++*, Item 24.
- If a type isn’t designed to be a base class but might be inherited from by mistake, declare the class as `final` (C++11) to prohibit derivation and avoid the above problem.
- The compiler-generated destructor is non-virtual, so a polymorphic base class must **explicitly declare** its destructor as `virtual`.

### Item 08: Don’t let exceptions escape destructors

Generally speaking, a destructor shouldn’t throw exceptions because that can result in various undefined problems, including but not limited to memory leaks, program crashes, or resource ownership getting stuck.

A straightforward explanation is that a destructor is the last moment of an object’s lifetime and is responsible for handing back important resources such as threads, connections, and memory ownership. If an exception is thrown at some point during the destructor, it means the remaining cleanup code won’t be executed, which is extremely dangerous—because a destructor often acts as the safety net for the class object, possibly called when an exception has already occurred somewhere else. In that scenario, if the destructor is also throwing an exception during stack unwinding, the program may crash immediately, which no programmer wants to see.

That said, if certain operations within a destructor are prone to throwing exceptions (like resource release), and you don’t want to swallow them, then move them out of the destructor and offer a normal function for that cleanup. The destructor should only record some data. We must ensure the destructor can always reach the end without throwing.

### Item 09: Never call `virtual` functions during construction or destruction

As the item title states: do not call `virtual` functions in constructors or destructors.

In a polymorphic context, we need to rethink the meaning of constructors and destructors. During their execution, the object’s type transitions from a base to a derived class and then from a derived class back to a base.

When a derived object begins creation, the base constructor is called first. Until the derived constructor is invoked, **the object remains a “base object.”** Naturally, any virtual function calls in the base constructor will point to the base version. Once you enter the derived constructor, **the original base object becomes a derived object**, and calling the virtual function there invokes the derived version. Similarly, during destruction, the derived object type degenerates to the base type.

Therefore, if you’re hoping the base constructor calls a derived virtual function, forget about it. Unfortunately, you may do that unintentionally. For instance, a common practice is to abstract the constructor’s main work into an `init()` function to avoid repeating code in multiple constructors, but you need to be careful about whether `init()` calls a virtual function. The same holds for the destructor.

### Item 10: Have `operator=` return a reference to *this*

Put simply, this enables your assignment operator to support chained assignments:

```C++
x = y = z = 10;
```

When designing interfaces, an important principle is **to make them as similar as possible to the built-in types that offer the same functionality**. If there’s no special reason, let the return type of your assignment operator be `ObjectClass&` and then `return *this` in your implementation.

### Item 11: Handle “self-assignment” in `operator=`

Self-assignment refers to assigning something to itself. Although this operation may look silly and useless, it happens far more often than one might think, frequently by way of pointer manipulation:

~~~C++
*pa = *pb;		 			//pa and pb points to the same object, this is self-assignment
arr[i] = arr[j];		//if i == j, this is also self-assignment
~~~

Therefore, in the `operator=` that manages certain resources, be particularly careful to check whether it’s a self-assignment. Whether you use deep copying or resource ownership transfer, you must release the original memory or ownership before assignment. If you fail to handle self-assignment, you could release your own resource and then assign it back to yourself—an error.

One way is to check for self-assignment before the assignment, but that approach isn’t exception-safe. Imagine if an exception is thrown at any point before assigning but after freeing the original pointer, then the pointer references memory that’s already been deleted.

~~~C++
SomeClass& SomeClass::operator=(const SomeClass& rhs) {
  if (this == &rhs) return *this;
  
  delete ptr;	
  ptr = new DataBlock(*rhs.ptr);				//If an exception is thrown out here, ptr will be pointing to memory that has been deleted
  return *this;
}
~~~

If we also consider exception safety, we arrive at a method that, happily, also solves the self-assignment problem.

~~~C++
SomeClass& SomeClass::operator=(const SomeClass& rhs) {
  DataBlock* pOrg = ptr;
  ptr = new DataBlock(*rhs.ptr);				//If an exception is thrown out here, ptr will still point to the memory before
  delete pOrg;
  return *this;
}
~~~

An alternative using copy-and-swap is explained in detail in Item 29.

### Item 12: Don’t forget every component when copying objects

By “every component,” the author is reminding you of two points:

- When you add new member variables to a class, don’t forget to handle them in your copy constructor and assignment operator. If you forget, the compiler won’t warn you.
- If your class involves inheritance, when implementing a copy constructor for the derived class, be extra mindful to copy every part of the base class. These parts tend to be private, so you can’t directly access them. You should let the derived copy constructor call the base copy constructor:

  ~~~C++
  ChildClass::ChildClass(const ChildClass& rhs) : BaseClass(rhs) {		
    	// ...
  }
  ~~~

Moreover, neither your copy constructor nor your copy assignment operator should call the other. Although it may look like a good way to avoid duplication, it’s flawed. A copy constructor creates a new object (which doesn’t exist before the call), while the assignment operator modifies an existing object (which already exists). The former calling the latter is akin to assigning to an uninitialized object; the latter calling the former is like constructing an object that already exists. Don’t do it!

---

## III. Resource Management

Memory is just one of many resources we manage. The same principle applies to other common resources like mutexes, file descriptors, and database connections: if you’re no longer using them, make sure they’re returned to the system. This chapter discusses handling resource management in the context of exceptions, multiple return paths within functions, and sloppy maintenance by programmers. Besides introducing object-based resource management, it also delves into deeper suggestions for memory management.

### Item 13: Manage resources with objects

The core idea of this item is that if you manage resources (acquisition and release) in a process-oriented way, unexpected situations may cause loss of control over those resources, leading to leaks. Process-oriented resource management means encapsulating acquisition and release in separate functions, which forces the caller who acquires resources to be responsible for releasing them. We then have to consider: will the caller always remember to release them? Can they ensure they’re released properly? A design that doesn’t assign too many duties to the caller is a good design.

First, let’s see what might cause the caller’s plan to release resources to fail:

- A simple `delete` statement may not execute if there’s an early `return` or if an exception is thrown before the `delete`.
- Even if the code is carefully written at first, when the software is maintained, someone else might add a `return` or throw an exception before the `delete`, repeating the same error.

To ensure resources are acquired and released properly, we package their acquisition and release into an object. When we construct the object, it automatically acquires the resource. When we no longer need that resource, we let the object’s destructor handle it. That’s the idea behind “Resource Acquisition Is Initialization (RAII),” because we always initialize a management object in the same statement that acquires the resource. No matter how control flow exits that block, once the object is destroyed (for instance, when leaving its scope), its destructor is automatically invoked.

For a practical example in C++11, see `shared_ptr<T>`.

---

## IV. Design and Declarations

Interface design and declarations are an entire discipline in themselves. Note that I’m talking about how the interface should look, not the internal implementation. How should you choose parameter types? What about return types? Should a function go inside the class or outside of it? These decisions profoundly impact interface stability and correctness. This chapter addresses these concerns one by one.

### Item 18: Make interfaces easy to use correctly and hard to use incorrectly

This item tells you how to **help your clients avoid mistakes when using your interface**.

When designing interfaces, we often wrongly assume that interface users **possess some necessary knowledge to avoid obvious mistakes**. But the fact is, they might not be as “clever” as we are or know the “inside information” of the interface’s implementation, and that leads to instability. Such instability might be due to the caller lacking prior knowledge or simply being careless. The caller might be someone else or your future self. Therefore, a well-designed interface should, as much as possible, help its callers avoid potential hazards **at the syntax level** and **before the program runs**, i.e., at compile time.

- Use **wrapper types** to remind callers to verify their parameters, restricting additional conditions to **the type itself**

When someone tries to pass “13” as a “month,” you could check at runtime inside the function and issue a warning or throw an exception, but that’s just shifting the blame—only after calling does the user discover they mistakenly typed 13 instead of 12. If you abstract “month” into a separate type at the design level (for example, using an enum class), you can catch the problem at compile time. Restricting a parameter’s additional conditions to the type itself helps make the interface more convenient.

- Restrict what callers **cannot** do at the *syntax* level

Callers often make mistakes unknowingly. Therefore, you need to impose constraints at the syntax level. A common example is adding `const` to a function’s return type—for instance, having the return type of `operator*` be `const` to prevent an accidental assignment like `if (a * b = c)`.

- Make your interface behave consistently with built-in types

Let your custom type behave like built-in types. For example, if you design your own container, match the naming convention of the STL. Or if you need two objects to be multiplied, it’s better to overload `operator*` rather than create a member function called “multiply.”

- At the syntax level, require what callers **must** do

**Never rely on callers to remember to do certain tasks**. The interface designer should assume they’ll **forget** those requirements. For example, replacing raw pointers with smart pointers is a way of being mindful of the caller. If a core method needs setup and teardown (like acquiring a lock and releasing it) before and after use, it might be better to define pure virtual functions and force the caller to inherit from an abstract class, guaranteeing they implement those steps. The interface designer is responsible for calling those setup and teardown steps around the core method call.

The fewer responsibilities the caller (our client) has, the fewer mistakes they can make.

### Item 19: Design a class as you would design a type

This item reminds us that designing a class requires attention to detail, though it doesn’t provide solutions to all of them—it’s just a reminder. Each time you design a class, mentally walk through these questions:

- How should objects be created and destroyed? Consider constructors, destructors, and possibly overriding `new`/`delete`.
- How should the constructor differ from the assignment operator in behavior, especially regarding resource management?
- What happens if the object is copied? Consider a copy constructor.
- What are valid values for the object? Ideally, ensure this at the syntax level or at least before runtime.
- Should the new type fit into some existing inheritance system? That might involve virtual function overrides.
- What about implicit conversions between the new type and existing types? That implies thinking about conversion operators and non-`explicit` constructors.
- Should any operators be overloaded?
- Which interface should be exposed, and which techniques should be encapsulated (public vs. private)?
- What about efficiency, resource management, thread safety, and exception safety?
- Does this class have the potential to be a template? If so, consider making it a template class.

### Item 20: Prefer pass-by-reference-to-const over pass-by-value

When designing a function’s interface, you should usually take parameters by `const` reference rather than by value, or you risk the following:

- Passing by value can involve copying large amounts of data, and many of these copies are unnecessary.
- If the copy constructor is designed for deep copying rather than shallow copying, the copying cost may far exceed just copying a few pointers.
- In a polymorphic context, if you pass a derived object to a function that expects a base by value, only the base portion is copied, discarding the derived part. That can cause unpredictable errors, and virtual functions won’t be called.
- Even if a type is small, that doesn’t guarantee that pass-by-value is cheap. The type’s size depends significantly on the compiler. Furthermore, small can become large in future code reuse or refactoring.

Nonetheless, for built-in types or STL iterators and function objects, we usually stick to pass-by-value.

### Item 21: If you must return an object, don’t try returning its reference

The core message of this item is not to return the result by reference. The author analyzes the many potential errors, be it returning a stack object or a heap object. It won’t be repeated here. The author’s final conclusion: if you must return by value, just do it. That extra copy isn’t a big deal, and you can rely on compiler optimizations.

However, with C++11 or later compilers, you can write a “move constructor” for your type and use `std::move()` to **elegantly eliminate the time and space overhead caused by copying**.

### Item 22: Declare data members as private

First, the conclusion—**declare all data members in a class as `private`**. `private` implies variable encapsulation. But this item contributes more valuable insights on how different access specifiers—`public`, `private`, `protected`—**reflect design philosophies**.

Put simply, making all members private has two benefits. First, all the `public` and `protected` members are functions, so the user no longer needs to distinguish among them, ensuring syntactic consistency. Second, by encapsulating the variables, **you minimize the necessary changes in external code if you alter the class internally**.

Once all variables are encapsulated, no external entity can access them directly. If users (potentially your future self or someone else) want to do something with these private variables, they **must** do so via the interfaces you provide. These interfaces buffer outside code from the class’s internal changes—**what’s invisible causes no impact**—thus not forcing outside code to change. So if a well-designed class is changed internally, the impact on the rest of the project should merely be **a recompile rather than code modifications**.

Next, **`public` and `protected` are partially equivalent**. A custom type is offered to its “clients,” who typically use it in one of two ways—**they either instantiate it** or **inherit from it**—we’ll call these two categories “the first kind of client” and “the second kind of client.” From an encapsulation standpoint, a `public` member is one that **the class author decides not to encapsulate from the first kind of client**, and a `protected` member is one that **the class author decides not to encapsulate from the second kind of client**. In other words, if we treat both client types equally, then `public`, `protected`, and `private` reflect how the class designer handles encapsulation—full, partial, or none.

### Item 23: Prefer non-member, non-friend to member functions

I’m willing to elaborate more on this item because it’s very important, and the author didn’t explain it as clearly as one might hope.

Within a class, I would describe **those `public` or `protected` member functions that need direct access to private members** as **low-granularity functions**. They form the first line of encapsulation for private members. By contrast, **public member functions that are formed by combining several other public (or protected) functions** I’d call **high-granularity functions**. These high-granularity functions don’t need direct access to private members themselves—they merely piece together lower-level tasks. This item tells us that such functions should, whenever possible, be placed outside the class.

```C++
class WebBrowser {							//	We browser class
public:ß
  	void clearCache();					// Clear caches, this has direct access to private member
  	void clearHistory();				// Clear caches, this has direct access to private member 
  	void clearCookies();				// Clear cookies, this has direct access to private member 
  
  	void clear();								// This function has larger granularity, and calls the three functions above. Therefore, it should not have direct access to private members. This Clause tells us to move this function outside of the class.
}
```

If these high-granularity functions remain as member functions in the class, they can, on the one hand, erode encapsulation and, on the other, reduce the flexibility of how the function might be wrapped.

1. Class encapsulation

Encapsulation’s purpose is to minimize the impact of internal changes on outside code—we hope only a small number of clients are affected by changes inside the class. A simple method to measure the encapsulation quality of a member is to see how many `public` or `protected` functions directly access that member. The more that do, the weaker the member’s encapsulation—its changes could spread further. Back to our discussion: high-granularity functions, at the time of design, are **not supposed to directly access any private members** but instead use public members. That’s the best way to maintain encapsulation. Unfortunately, this intention isn’t enforced in code. A future maintainer (maybe you or someone else) might forget and directly access private members in what was supposed to be a “high-granularity” function, accidentally damaging encapsulation. By making it a non-member function, you avoid that possibility at the syntax level.

2. Wrapping flexibility and design approach

Extracting high-granularity functions to the outside allows us to **organize code from more perspectives** and **optimize compile dependencies**. For instance, if the function `clear()` is initially created to integrate low-granularity tasks from the browser’s perspective, you might reorganize those tasks from the perspectives of “cache,” “history,” “cookies,” etc. Perhaps you combine “search history” and “clear history” into “selective clear history,” or “export cache” and “clear cache” into “export and clear cache.” Doing that outside the browser class yields greater flexibility. Usually, one might use a utility class like `class CacheUtils` or `class HistoryUtils` with static functions, or place them in separate namespaces. Then if you only need certain functionality, you can include the relevant header, rather than forcibly bringing in code for cookies when you only care about cache. That’s also the benefit of namespaces being able to span multiple files.

~~~C++
// Header file webbrowser.h: For class WebBrowserStuff itself
namespace WebBrowserStuff {
class WebBrowser { ... };        // Core functionalities
}

// Header file webbrowsercookies.h: For WebBrowser and cookie-related functionalities
namespace WebBrowserStuff {
	...                          // Utility functions related to cookies
}

// Header file webbrowsercache.h: For WebBrowser and cache-related functionalities
namespace WebBrowserStuff {
	...                          // Utility functions related to cache
}
~~~

Finally, note that this item refers to functions that **do not directly access private members**. If you have a `public` (or `protected`) function that must directly touch private members, forget this item, because extracting that function to the outside would be far more involved.

### Item 24: If all parameters need type conversion, use a non-member function

This item addresses **the difference between overloading an operator as a member function versus a non-member function**. The author wants to point out that if you **expect every operand to support implicit conversions** for the operator, **then make that operator a non-member**.

First, note that if an operator is a member function, then **its first operand (the calling object) does not undergo implicit conversions**.

Let’s begin with a brief explanation: once an operator is written as a member function, it becomes less obvious from the expression which object is actually calling it. For example, if a rational number class overloads the `+` operator, and you use `Rational z = x + y;`, you don’t see which object is really calling `operator+`—does the `this` pointer refer to `x` or `y`?

~~~C++
class Rational {
public:
  //...
  Rational operator+(const Rational rhs) const; 
pricate:
  //...
}
~~~

As a member function, the operator’s invisible `this` pointer always points to the first operand, so `Rational z = x.operator+(y);` is effectively what happens. The compiler decides which operator function to call based on the static type of the first operand. Therefore, the first operand can’t be implicitly converted to the correct type. For example, if `Rational`’s constructor allows an `int` to be implicitly converted to `Rational`, then `Rational z = x + 2;` compiles because `x` is a `Rational` and `2` can be converted. But `Rational z = 2 + x;` might fail to compile because the first operand is `2` (an `int`), so the compiler tries to convert `x` to an `int`, which doesn’t work.

Hence, if you’re writing operators like addition, subtraction, multiplication, division, etc. (not limited to these) and want each operand to allow implicit conversion, **do not overload them as member functions**. If the first operand isn’t of the correct type, you’ll run into a failed call. The solution is to **declare the operator as a non-member function**; you can make it a friend if that makes the operator’s work easier, or you can wrap private members behind more public functions—whatever you choose.

Hopefully, this clarifies why operators behave differently when written as member functions versus non-members. The rule doesn’t strictly apply only to operators, but other than operators, it’s hard to think of a more suitable example.

By the way, if you want to forbid implicit conversions, mark every single-parameter constructor with the `explicit` keyword.

### Item 25: Consider writing a non-throwing `swap` function



## VI. Inheritance and Object-Oriented Design

When designing a class involving inheritance, there are many considerations:

- What type of inheritance is it?
- Are its interfaces virtual or non-virtual?
- How are default parameters handled?

Answering these questions properly demands understanding even more topics: what exactly does each type of inheritance mean? What is the true purpose of a virtual function? How does inheritance affect name lookups? Is a virtual function really necessary? Are there alternatives? These issues are all discussed in this chapter.

### Item 32: Ensure your public inheritance represents an is-a relationship

Public inheritance means: __the derived class is a specialized version of the base class__. That’s the so-called “is-a” relationship. But this item points out a deeper meaning: with public inheritance, __the derived class must encompass all features of the base class, unconditionally inheriting all of its traits and interfaces__. That’s singled out because if we rely purely on real-world intuition, we might slip up.

For example, is an ostrich a bird? If we consider flight as a feature (or interface), then the Ostrich class definitely can’t publicly inherit from Bird because ostriches can’t fly; we want to eliminate the possibility of calling a flight interface at **compile time**. But if we only care about laying eggs, then by that logic, an ostrich can indeed inherit from Bird. The same idea applies to rectangles and squares in geometry. Real-world experience says a square is a rectangle, but in code, a rectangle has length and width as two separate variables; a square cannot have two unconstrained variables—there’s no syntax-level way to ensure they’re always equal. Hence, no public inheritance. 

So before deciding on public inheritance, first ask yourself, **does the derived class need all of the base class’s features?** If not, no matter what real life might suggest, it isn’t an “is-a” relationship. __Public inheritance won’t reduce or weaken the base class’s traits or interfaces—it can only extend them.__

### Item 33: Avoid hiding inherited names

This item discusses **name hiding** among repeatedly overloaded virtual functions in inheritance. If you don’t have a design involving multiple overloads of the same virtual function, you can skip this.

Suppose the base class has two overloads for a virtual function `foo()`, maybe `foo(int)` and `foo() const`. If the derived class overrides only one of them, then the other overloads (`foo(int)` and `foo() const`) in the base class become unavailable in the derived class. This is the name hiding problem—**name hiding at the scope level is independent of parameter types and virtual-ness**. Even if the derived class overrides just one function with the same name, all the same-named functions in the base class are effectively hidden. Personally, I find that rather counterintuitive.

If you want to restore the base class’s function names, you must use `using Base::foo;` in the scope where you need it in the derived class (perhaps in a member function or under public or private). That will make `foo(int)` and `foo() const` from the base class visible again in the derived class.

If you only want to reuse one of the base class’s once-hidden overloads in the derived class, you can do so with an inline forwarding function.

### Item 34: Distinguish interface inheritance from implementation inheritance

We explored the essential meaning of public inheritance in Item 32. Now, in this item, we clarify that within a public inheritance hierarchy, different types of functions—pure virtual, virtual, and non-virtual—have **hidden design logic**.

First, you must understand that member function interfaces are always inherited. Public inheritance ensures that if you can call some function on the base class, you can call it on the derived class. Different types of functions reflect **the base class’s different expectations of how the derived class will implement them**.

- Declaring a pure virtual function in the base class **forces** the derived class to have that interface and **forces** it to provide an implementation.
- Declaring a normal virtual function in the base class **forces** the derived class to have that interface and **provides a default implementation** for it.
- Declaring a non-virtual function in the base class **forces** the derived class to accept both the interface and the provided implementation, disallowing any changes by the derived class (Item 36 requires that we not override the base’s non-virtual functions).

The potential issue arises with normal virtual functions, because the default implementation in the base class may not be suitable for all derived classes. Thus, if a derived class forgets to implement a custom version when it should, the base class lacks a mechanism to **warn** the derived class’s designer at the code level. One solution is to **provide an implementation in the base class for the pure virtual function** so that if the derived class finds it suitable, it can call that default. This makes the derived class explicitly check the default behavior’s suitability in code.

Hence, the difference between pure virtual and normal virtual functions isn’t whether the base class has an implementation—pure virtual functions can also have one. Instead, it’s that the base class’s expectations differ. The former states, “implement me explicitly,” while the latter states, “use the default or override me if needed.” Non-virtual functions don’t allow any freedom in derived classes, guaranteeing a single consistent implementation across the hierarchy.

### Item 35: Consider alternatives to virtual functions

---

### Item 36: Never redefine non-virtual functions inherited from a base class

That means if your function needs dynamic (polymorphic) dispatch, be sure to declare it virtual. Otherwise, if the base function is non-virtual and you “override” it in the derived class, any dynamic calls (through a base pointer to a derived object) won’t call your overridden function, which probably causes errors.

Conversely, if a function in the base class is non-virtual, **never** override it in the derived class. You’d face a similar problem.

Why? Because only virtual functions are dynamically bound. Non-virtual functions are statically bound at compile time based on the pointer or reference type, ignoring the object’s dynamic type.

In other words, a virtual function means “__the interface is indeed inherited, but the implementation can be changed in the derived class__,” while a non-virtual function means “__both the interface and the implementation are inherited__.” That’s the essence of “virtual.”

### Item 37: Never redefine inherited default parameter values

This item has two implications:

1. Do not alter default parameter values of a non-virtual function in the base class. Essentially, **don’t override anything about a non-virtual function** in the base. Don’t modify it!
2. Virtual functions should not have default parameter values, nor should they be changed in derived classes. **A virtual function should always avoid default parameters**.

The first point was explained in Item 36. The second is because default parameters belong to __static binding__, while virtual functions are __dynamically bound__. Virtual functions are typically used in dynamic calls, but any default parameter values changed in the derived class won’t take effect in that dynamic context, causing confusion and leading the caller to believe they’ve changed while in reality they haven’t.

Default parameter values are statically bound for efficiency reasons at runtime.

If you really want a virtual function to have default parameters in a particular class, make that virtual function `private`, and in the `public` interface create a non-virtual “wrapper” function that has default parameters. Of course, that wrapper is single-use—don’t override it after inheriting again.

### Item 38: Use composition to model has-a or “implemented in terms of”

Besides inheritance, there’s another relationship between two classes: one class’s object can serve as a member of another class. We call this relationship “class composition.” This item explains when composition is appropriate.

The first scenario is quite straightforward: it explains that one class “owns” another class object as an attribute. For instance, a student has a pencil, a citizen has an ID card, etc. No problem there.

The second scenario is more discussed: “one class is implemented in terms of another.” For example, implementing a queue using a stack, or implementing a Redcore browser using an old version of the Google Chrome kernel.

Here, it’s crucial to distinguish the second scenario from the “is-a” relationship in public inheritance. Always remember: the only test for “is-a” is whether a derived class must **fully** inherit every feature and interface of the base class. Meanwhile, “implemented in terms of another class” is about **hiding** that tool class. For example, people don’t really care whether your queue is implemented with a stack, so you hide the stack interface and only expose the queue interface. Likewise, Redcore browser developers don’t want others to see that it’s built on Chrome’s kernel, so they need “hidden” behavior.

### Item 39: Use private inheritance judiciously and cautiously

Similar to composition, private inheritance expresses “__implementing one class by means of another tool class__.” By definition, the tool class should be hidden inside the target class—no interfaces or variables are externally exposed. This is the essence of private inheritance: a __technical form of encapsulation__. Unlike public inheritance, which expresses “__both the implementation and the interface are inherited__,” private inheritance conveys “__only the implementation is inherited, but the interface is omitted__.”

Accordingly, in private inheritance, __all base class members become private to the derived class__. They’re not publicly accessible, and the outside world doesn’t care about the details of the base class that the derived class used.

When you need to “implement one class in terms of another,” how do you decide between composition and private inheritance?

- Prefer composition whenever you can. Avoid private inheritance unless necessary.
- When you need to override certain methods (virtual functions) of the tool class—methods specifically designed for inheritance or callbacks—private inheritance is a better fit, because the user can customize the implementation.

If you do use private inheritance, you can’t prevent further derived classes from overriding the same virtual functions again. If you want to block that behavior—similar to marking a function as `final`—one approach is to declare a private nested class inside the target class. That nested class publicly inherits from the tool class and overrides its methods within itself.

~~~C++
class TargetClass {              // Target class
private:
	class ToolHelperClass : public ToolClass {  // Nested class, publicly inheriting from the tool class
    public:
        void someMethod() override;            // Methods that should be overridden by TargetClass are implemented in the nested class, preventing TargetClass's subclasses from overriding them.
    };  
};
~~~

That way, subclasses of your target class can’t re-override those critical methods.

### Item 40: Use multiple inheritance judiciously and cautiously

In principle, multiple inheritance isn’t encouraged because it may lead to multiple parents sharing an ancestor, potentially causing the derived class to hold multiple copies of that ancestor. C++ addresses this with **virtual inheritance**, but that expands object size and slows member data access—both costs of virtual inheritance.

If you must use multiple inheritance, design your classes carefully, avoiding diamond patterns wherever possible (“B and C derive from A, then D derives from both B and C”). If you can’t avoid a diamond pattern, consider using virtual inheritance, but keep in mind the overhead. If the base class is virtually inherited, try to store as little data there as possible.
