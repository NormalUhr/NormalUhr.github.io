---
layout:     post
title:      "漫谈C++——谈谈在函数中使用auto"
subtitle:   "C++学习笔记"
date:       2020-10-30
author:     "Felix Zhang"
header-img: "img/in-post/2020-10-30-Auto-Function/bg.jpg"
catalog: true
tags:
   - C++
---
# 谈谈在函数中使用`auto`

前边我写了[一篇文章](https://starkschroedinger.github.io/2020/09/24/auto/)关于如何在变量中使用`auto`，今天我们来看看在函数中使用`auto`时的场景，以及需要注意的细节。

基本上，在函数中使用`auto`的情形大致可分为两类，在C++11中，`auto`被引入放到函数声明中返回类型的位置，用间接的方式来定义函数的返回类型，如下：

~~~C++
//等价于 std::string someFunc(int i, double j);
auto someFunc(int i, double j) -> std::string;
~~~

在C++14中，编译器可以直接推断出函数的返回类型了，所以可以写成下边的样子：

~~~C++
auto someFunc(int i, double j) {
  //自动推断返回类型为std::string
  return std::to_string(i + j);
}
~~~

## 尾置返回类型

上边C++11的例子并没有给我们带来很多直观的感受——`auto`在其中到底有什么用？既然我们还是要显示声明函数的返回类型，而且和更传统的返回类型表述相比，我们为了实现尾置还要再多写一个`auto`和`->`，更重要的是，这种声明看起来真的太丑了，我们为什么要用这种表述方式呢？

在很多返回类型要取决于参数类型的时候，比如在函数模板中，上边这种写法就会有很大作用，因为你很有可能并不知道进行某种操作后自己会得到什么类型，请看下例：

```C++
template<typename T, typename V>
auto addWithTwoTypes(T t, V v) -> decltype(t + v) {
  return t + v;
}
```

上边这个函数会返回T类型变量和V类型变量的和，如果`T`和`V`分别是`short`和`int`，那么返回类型就会自动推断为`int`，但是如果一个是`double`一个是`int`，那么返回类型就会是`double`。因此，返回值类型和两个模板类型都相关。

如果将上述例子写成如下形式，不使用`auto`，可不可行呢？

~~~C++
template<typename T, typename V>
decltype(t + v) addWithTwoTypes(T t, V v) {
  return t + v;
}
~~~

答案是否定的，因为在推导`decltype(t + v)`时`t`和`v`还没定义，因此会有类似“模板未实例化”的报错。

再举一个例子，如果有如下定义：

~~~C++
class JackRoseCreator {
public:
	Jack giveMeJack();
  Rose giveMeRose();
};

Baby operator+(Jack const& jack, Rose const& rose);

template<typename T>
auto giveMeSomething(T const& t) -> decytype(t.giveMeJack() + t.giveMeRose()) {
  return t.giveMeJack() + t.giveMeRose();
}
~~~

上边这个例子就是第一个例子的复杂版，在我们的`auto`写法中，最后`auto`会被编译器推导为一个`Baby`类型，但是其他写法可能就没这么简单了。

另外一个不常用的地方在于，使用尾置返回类型也可简化一些函数的写法，比如：

~~~C++
int (*generatorArr(int i))[10086];	//返回一个指向大小为10086的int类型的数组的指针
~~~

上边函数可以用`auto`来简化成：

~~~C++
auto generatorArr(int i) -> int (*)[10086];
~~~

是不是看起来更方便一些？

## 返回类型推导

在C++14中，编译器终于可以自己推导任何函数的返回类型了，无论多复杂的都可以。唯一的条件就是，在单一的返回语句中，返回的类型必须在编译期时确定的，其他的规则就和在变量中使用`auto`一模一样了。

因为在推导类型时，编译器必须需要知道函数的定义，也就是说，这种用法被限制在了内联函数、函数模板以及lambda表达式中。对于一个在头文件中声明、在其他文件中实现的函数来说，`auto`这样的用法是不可行的。然而，内联函数、函数模板以及lambda表达式这三种情况，也足够应付你**需要**以及**应该**使用`auto`的地方了。

我说了“应该”，因为就像变量的自动推导一样，函数返回类型的自动推导可以避免不必要的转换，以及事后修改变量类型时对代码所做的必要的改动。请看下例：

~~~C++
class Container {
  typedef std::vector<int> Container_t;
  Container_t vals;
  
public:
  auto begin() const {
    return std::begin(vals);
  }
  
  auto at(Containter_t :: size_type id) const {
    return vals[id];
  }
  //...
};
~~~

事后变量类型的更改指的是，也许你觉得`vector`不是最好的选择呢？或许你觉得`int`不够用了呢？没关系，只需要把`int`改为`long long`就好了，剩下的成员函数都可以原封不动地留在那里。

有了返回类型的推导，大多数的尾置用法都可以被替换了，比如上边的例子就可以改写为：

~~~C++
template<typename T>
auto giveMeSomething(T const& t) {
  return t.giveMeJack() + t.giveMeRose();
}
~~~

简洁明了。

作为代码的阅读者来说，他们也希望像编译器一样一眼就看到有返回类型推导的函数的`return`语句，这也就是说你的函数应该尽可能的短——当然，函数短小精悍是一个普遍的要求，只是有返回类型推导的函数更应该注意便是了。

## 结论

就以上两种用法作总结，如果条件允许，尽可能的去使用返回类型推导吧，这样做会让你的变量类型上下一致性更高。但是至于尾置返回类型，还是要尽可能的避免，因为他们的语法实在是太不美观，同样让人读起来也很困难。