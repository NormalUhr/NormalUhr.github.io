---
layout:     post
title:      "漫谈C++——你该怎么构造shared_ptr？"
subtitle:   "C++学习笔记"
date:       2020-09-26
author:     "Felix Zhang"
header-img: "img/in-post/2020-09-26-Make-Shared/bg.jpg"
catalog: true
tags:
   - C++
   - Modern C++
---

# 你该如何构造shared_ptr？

通常我们有两种方法去初始化一个std::shared_ptr：通过它自己的构造函数或者通过std::make_shared。这两种方法都有哪些不同的特性和取舍呢？

## 简单一览shared_ptr和weak_ptr

让我们快速过一遍std::shared_ptr是如何工作的。shared_ptr的背后其实是一套引用计数器机制，当我们复制一份shared_ptr时，计数器加一，当一个shared_ptr被销毁时，计数器减一，当计数器达到零时，对象被自动销毁。看似完美的shared_ptr虽然解决了内存管理的问题，也引入了引用成环的问题，这种问题是靠他自己没办法解决的。

std::weak_ptr其实可以看成shared_ptr的一个搭档，就是为了解决shared_ptr成环问题而存在的：它自己并不拥有对动态对象的管辖权，自己指向shared_ptr的目标也不会增加计数器的值，相反，它自己拥有一套不纳入计数器的指针系统。

在shared_ptr的计数器的值不为零时，换句话说被指向的内存仍然存在时，weak_ptr可以被转化成shared_ptr来访问这块内存。我们马上就会看到，其实对于weak_ptr本身来说，它内部也有一套自己的计数系统。

shared_ptr是非侵入式的，即计数器的值并不存储在shared_ptr内，它其实是存储在其他地方——在堆上的。当一个shared_ptr由一块儿内存的原生指针创建的时候（原生内存：代指这个时候还没有其他shared_ptr指向这块内存），这个计数器也就随之产生。这个计数器结构的内存会一直存在——直到所有的shared_ptr和weak_ptr都被销毁的时候。这个时候就比较巧妙了，当所有shared_ptr都被销毁时，这块儿内存就已经被释放了，但是可能还有weak_ptr存在——也就是说计数器的销毁有可能发生在内存对象销毁后很久才发生。从图上来看，整个结构看起来就像下图一样：

![结构](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io/img/in-post/2020-09-26-Make-Shared/shared_ptr_structure.png)

## std::make_shared

有了上边的背景后，当我们使用创建一个shared_ptr管理一块原生内存时，在堆上实际上发生了两次内存分配：

~~~C++
auto* ptr = new MyObject{/*args*/};   //分配内存给MyObject
std::shared_ptr<MyObject> shptr{ptr}; //分配内存给shared_ptr的计数器
~~~

当我们使用一个原生指针、一个unique_ptr或者通过一个空白shared_ptr来设置指向一块儿内存时，发生的内存分配和上边都很类似。就像大家都知道的，内存分配和回收基本上是C++中最慢的单次操作了。鉴于此，我们有一种办法来把这两种内存分配合二为一：没错就是你想的：std::make_shared。std::make_shared可以同时为计数器和原生内存分配内存空间，结果就像下图展示的：

![](/Users/normaluhr/Documents/Git/StarkSchroedinger.github.io/img/in-post/2020-09-26-Make-Shared/make_shared.png)

## std::make_shared和普通构造函数——孰是孰非？

什么都不是完美的，我们在这里也是一样。使用make_shared是可以给我们带来一些好处，但是它的另一面我们也要了解。

### make_shared的好处

就像上边提到的，使用make_shared最大的好处就是减少单次内存分配的次数，如果我们马上要提到的坏影响不是那么重要的话，这几乎就是我们使用make_shared的唯一理由。

另一个好处就是可以增大Cache局部性(Cache Locality)：使用make_shared，计数器的内存和原生内存就在堆上排排坐，这样的话我们所有要访问这两个内存的操作就会比另一种方案减少一半的cache misses。所以，如果cache miss对你来说是个问题的话，你确实要好好考虑一下make_shared。

> 引入 Cache 的理论基础是**程序局部性原理**，包括时间局部性和空间局部性。即最近被CPU访问的数据，短期内CPU 还要访问（时间）；被 CPU 访问的数据附近的数据，CPU 短期内还要访问（空间）。因此如果将刚刚访问过的数据缓存在Cache中，那下次访问时，可以直接从Cache中取，其速度可以得到数量级的提高。CPU要访问的数据在Cache中有缓存，称为“命中” (Hit)，反之则称为“缺失” (Miss)。

然后就是，执行顺序以及异常安全性也是一个应该考虑的问题。来看看下边的代码段：

~~~C++
struct MyStruct {
  int i;
};

void doSomething(std::shared_ptr<MyStruct>, double d);
//这个函数可能抛出异常
double couldThrowException();

int main() {
  doSomething(std::shared_ptr<MyStruct>(new MyStruct{512}), couldThrowException());
}
~~~

分析上边的代码，在doSomething函数被调用之前至少有三件事情被完成：构造并给MyStruct分配内存，构造shared_ptr以及调用couldThrowException()。C++17中引入了更加严格的鉴别函数参数构造顺序的方法，但是在那之前，上边三件事情的执行顺序应该是这样的：

1. new MyStruct
2. 调用couldThrowException()
3. 构造shared_ptr<MyStruct>并管理步骤1开辟的内存。

看出问题了吧？一旦步骤二抛出异常，步骤三就永远都不会发生，因此没有智能指针去管理步骤一开辟的内存——内存泄漏了，但是智能指针说它很无辜，它都还没来得及到这个世上看一眼。这也是为什么我们要尽可能地使用std::make_shared来让步骤一和步骤三紧挨在一起，因为你不知道中间可能会发生什么事。

### make_shared的坏处

大伙儿使用make_shared，首先最可能遇到的问题就是make_shared函数必须能够调用目标类型构造函数或构造方法。然而这个时候即使把make_shared设成类的友元恐怕都不够用，因为其实目标类型的构造是通过一个辅助函数调用的——不是make_shared这个函数。

另一个问题就是我们目标内存的生存周期问题（我说的不是目标对象的生存周期）。正如上边说过的，即使被shared_ptr管理的目标对象都被释放了，shared_ptr的计数器还会一直持续存在，直到最后一个指向目标内存的weak_ptr被析构。这个时候，如果我们使用make_shared函数，问题就来了，程序自动地把被管理对象占用的内存和计数器占用的堆上内存视作一个整体来管理，这就意味着，