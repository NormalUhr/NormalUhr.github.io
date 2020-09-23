---
layout:     post
title:      "override和final让你的虚函数更安全"
subtitle:   "C++学习笔记"
date:       2020-09-24
author:     "Felix Zhang"
header-img: "img/in-post/2020-02-12-Niuke-Notes/bg.jpg"
catalog: true
tags:
   - C++
   - Modern C++
---
今天我想谈谈`override`和`final`，这一对在C++11中不太引人注意的特性，相比于C++11其他特性而言也更简单。这两个特性都能让我们的程序在继承类和覆写虚函数时更安全，更清晰。

## 覆写（override）虚函数

你有没有像我一样遇到过这种情况：在子类中，本来你想覆写虚函数，结果虚函数却没有被正确地调用呢？或者更惨的是，你有时不得不去修改父类虚函数的声明。在所有的子类中查找重载的函数这件事可真的是很麻烦，而且墨菲定律告诉你：你永远会忘掉你搞错了的那一个子类。不管怎么样，看下边这个例子吧：

```C++
 struct Base {
   virtual void doSomething(int i) const {
     std::cout << "This is from Base with " << i << std::endl;
   }
 };
 
 struct Derivied : Base {
   virtual void doSomething(int i) {
     std::cout << "This is from Derived with " << i << std::endl;
   }
 };
 
 void letDoSomething(Base& base) {
   base.doSomething(419);
 }
 
 int main() {
   Derived d;
   letDoSomething(d);  //输出结果： "This is from Base with 419"
 }
```

相信我，我曾经傻傻地花了好久解决诸如上边的问题——当然是在更复杂的程序中。如果你现在还没发现问题，我提示你一下：`Derived::doSomething`函数把`Base::doSomething`的`const`给搞丢了。所以他们两者并没有相同的函数签名，前者也没有如我们预想的对后者进行覆写。确实会有编译器弹出warning提示这个问题，但是往往在我们本意就不想覆写虚函数时它也会报警——久而久之我们就麻木了，编译器并不能从根本上解决这个问题。像这样的场景，我们也没有工具来区分告诉编译器我们的本意是否向覆写这个父类的虚函数。因此，在C++11中，我们引入了`override`这个关键字。

```C++
 struct Derived : public Base {
   void doSomething(int i) override {  // ERROR,编译器报警没有正确覆写Base::doSomething
     std::cout << "This is from Derived with " << i << std::endl;
   }
 };
```

很简单吧，加个关键字，让编译器来检查我们又没有正确覆写父类的虚函数。因此，任何子类覆写虚函数后导致函数签名的变化，都会导致编译器报错。

除此之外，如果你一直使用`override`，他还会给你带来一个意想不到的收获：在C++11之前，关于子类覆写父类虚函数后，子类的虚函数还要不要加`virtual`关键字，还是个值得争论的问题。人们一开始之所以想在子类的覆写函数上也加上`virtual`，就是为了提醒读代码的人这是个覆写的虚函数。但是在有了`override`之后，这个关键字本身就起了这个作用，之前的做法也就不是必须的了。所以建议的做法是，在最顶层的虚函数上加上`virtual`关键字后，其余的子类覆写后就不再加`virtual`了，但是要统一加上`override`

> Prefer to use override whenever you are overriding a virtual function andvirtual only for the topmost declaration of that function.

## 防止覆写

针对上面特性的反面，C++11也他讨论了如何防止子类再覆写父类的虚函数了——即父类的某些特性或方法被设计成不想再被改变。在C++11出现前，比较难受的一点就在于，即使父类的虚函数是`private`的情况下，我们仍然无法阻止它的子类覆写这个方法。

```C++
 class Base {
 public:
   void doSomething() const {
     std::cout << "I did something" << dontChangeMe() << std::endl;
   }
 private:
   virtual int dontChangeMe() const = 0;
 };
 
 class ChildOfBase : public Base {
 private:
   int dontChangeMe() const override { return 419; }
 };
 
 class BadChildOfChild : public ChildOfBase {
   int dontChangeMe() const override { return 61; }
 };
 
 int main() {
   BadChildOfChild badLittleChild;
   badLittleChild.doSomething(); //结果是61
 }
```

在C++11之前，你还对上边的行为无可奈何，至少你无法在语法层面去禁止开发人员这么做。现在，我们有了`final`关键字来帮我们这个忙。

```C++
 class ChildOfBase : public Base {
 private:
   int dontChangeMe() const final { return 61; }
 };
 
 class BadChildOfChild : public ChildOfBase {
   int dontChangeMe() const override;  //ERROR
 }
```

关于`final`和`override`关键字位置的一个小提示：这两者应放在在`const`，`volatile`等其他关键字后边，但是应该在纯虚标记，也就是"`=0`"的前边。一个`final`的纯虚函数是没什么意义的，因为本身就是一个抽象函数又不让后边的子类覆写给与它实际意义。另外就是`override final`和`final override`并没有什么区别，只是后者读起来可能更顺一些吧。只写一个`final`并不会像`override`那样检查覆写类型，所以最好还是两个都写上。

现在再去覆写`ChildOfBase`类的`dontChangeMe`函数就不可能了，但是写一个新的子类继承`Base`再覆写`dontChangeMe`还是允许的。

## 被`final`修饰的类

`final`还有第二种用法，直接用在类上，紧跟着类名，表示这个类禁止任何其他类继承它，无论是`public`继承还是`private`继承。

```c++
 class DontDeriveFromMe final {
   // ...
 };
 
 class Failure : public DontDeriveFromMe { //ERROR
   // ...
 };
```

但是在你给类使用`final`之前，还是麻烦你想清楚`final`是否真的是必要的，如果不是，那它就是一个坑。

## 如何更新你的代码

如果你的代码已经非常冗长并且到处都是覆写的虚函数，这个时候你想用`final`和`override`来更新你的代码库，我会给你以下建议：

* 在处理`final`时应该慎之又慎，需要case by case地处理。
* 至于`override`，无论哪里用到了覆写，直接加上去就完事了，只会有益处而不会有害，不过看清楚你的本意到底是不是覆写哈。尤其是你加上`override`后编译器报错的，一定要先检查你的本意到底是不是覆写，确定你真的犯了错误后，再去修改源代码。
* 加上`override`后，我建议你把子类的`virtual`关键字去掉，并且以后都遵循上边提到的规范：只有顶层的虚函数加上`virtual`，其余子类在覆写时全部用`override`来标记。
* 当你看到一个函数标记了`virtual`时，弄清楚它是不是顶层虚函数其实是一件非常困难的事情，这个时候我建议你多用用你的编译器，当你使用`override`报错时，也有可能它确实就是顶层`virtual`函数了。

## 结论

`override`和`final`都能帮助我们改善虚函数相关的安全性，使用`final`需要更多的权衡但是`override`就尽管大胆地用吧。