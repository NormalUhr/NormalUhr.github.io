---
layout:     post
title:      "漫谈C++——工厂模式中的通行证策略"
subtitle:   "C++学习笔记"
date:       2020-10-27
author:     "Felix Zhang"
header-img: "img/in-post/2020-10-27-Pass-Key/bg.jpg"
catalog: true
tags:
   - C++
---
# 比友元更安全——工厂模式中的通行证策略

这次是个小短文，我们从友元谈起。

## 从友元谈起

在C++中，**友元`friend`**可谓是为数不多的比继承还要更强的耦合关系。耦合带来便利，但同时也不可避免得迫使我们在**访问控制**和安全上做出一些让步。这些问题甚至成为了工厂模式中普遍的缺陷，不信？你看！

~~~C++
class Secret {
friend class SecretFactory;
private:
  //SecretFactory应当对构造函数有权限，这里没问题
  explicit Secret(std::string str) : data(str) {}
  //对于其他的private成员函数，SecretFactory理应无权访问，但是在语法上却被允许。
  void addData(std::string const& otherData);
  
private:
  //对于private数据，SecretFactory在设计上应无权访问，但这里依然被允许
  std::string data;
};
~~~

在工厂模式中，目标类的构造函数由于访问控制权的收口而被设置成私有类型。对于工厂而言，它的作用是生产一个目标对象——所以对于他的访问权限，一个构造函数也许就够了。但无论如何，工厂在设计上不应被给予访问包括数据在内的被封装起来的成员。友元`friend`地问题在于，它给予了这个友元类过多的权限——实际上是所有的权限，编译期才不会管这个工厂类有没有去触碰一些构造函数之外的机密的数据。总而言之，`friend`就像一个**权限分配的总开关**——要么没权限，要么权没限。

所以当我们将一个类设计成了友元时，我们就给予了它无尽的权限，让它胡作非为。

## 通行证策略（PassToken）——给权限把关

除非我们有办法给每个我们希望工厂类能触碰到的地方加个小开关（一般是构造函数），工厂类拿着个通行证（token, key, whatever you'd like to name it \^_\^）——它可以帮助工厂类解锁相应`private`权限，这样就可以避免将工厂类直接设计成友元、过多地放开权限。像下面这样：

~~~C++
class Secret {
	class SpecialKey {
    friend class SecretFactory;
  private:
    SpecialKey() {};
    SpecialKey(SpecialKey const&) = default;
  };
  
  public:
  //只要有SpecialKey的权限就能访问，显然SecretFactory是可以的。
  explicit Secret(std::string str, SpecialKey) : data(str) {}
  
  private:
  //但此时SecretFactory无法访问私有成员了。
  void addData(std::string const& otherData);
  std::string data;
};

class SecretFactory {
public:
  Secret getSecret(std::string str) {
    return Secret{str, {}};	//没问题
  }
  
  /*
  void illegalModify(Secret& secret, std::string const& otherData) {
  	secret.addData(otherData);	//这个函数在直接友元中是行得通的，但用了PassToken就不再可能了。
  }
  */
};

int main() {
  Secret s{"Ohhh!", {}}; 	//非法操作，Secret的对象创建权限被收口到工厂类中，外界无法获取SpecialKey的权限
  
  SecretFactory sf;
  Secret s = sf.getSecret("That's my boy!");
}
~~~

解释一下上边的代码。

与直接友元的做法不同的是，我们把`SecretFactory`设置为`SpecialKey`的友元，`SecretFactory`只有对`SpecialKey`的所有成员的访问权限——然而没啥用，`SpecialKey`除了构造函数以外啥都没有。重点来了，我们把`SpecialKey`作为`Secret`的构造函数的一个必要参数，尽管`Secret`的构造函数谁都能访问，但是除了`SecretFactory`谁都没有`SpecialKey`的权限。这种操作既给予了`SecretFactory`独家的（exlusive，英语这个词形容最贴切）访问权限，又把`SecretFactory`因友元而得到的其余访问权限锁在`SpecialKey`里。

## 小注释

你的脑瓜里可能会蹦出几个问题，我先有预见性的解答一下。

* 为什么`Specialkey`要放在Secret的类中作为嵌套类，还是私有属性，放在外边不行吗？

先说答案：可以，所有作为`Key`的类不是必须作为Secret的私有成员类的，放在外边，或者设置成`public`都无可非议，特别是放在类外边的话这个`Key`类还可以被多个工厂复用。

* `SpecialKey`的构造函数设为私有，是为了让它的构造权限收口到`SecretFactory`，那为什么拷贝构造也要私有呢？

先说答案：通行证法策略的核心就在于此，`Key`类的构造和拷贝构造权限都必须收归工厂类（设为`private`），即使你把`Key`类设置成一个`Secret`类中的私有嵌套类，你还是要遵循这一点。首先，类的构造函数必须是手动定义的——不能用`default`。因为如果用了`default`，即使`Key`类的构造函数是`private`的，如果这个类没有数据成员的话，我们还是可以用**花括号的方式**（C++11推出的[uniform initialization](https://en.cppreference.com/w/cpp/language/list_initialization)）构造它，如下所示：

~~~C++
// SpecialKey的定义
// ...
private:
SpecialKey() = default;
// ...

Secret s("How could it be???", {});		//很遗憾，这是成立的
~~~

它的原因在于，统一初始化（`uniform initialization`）在没有成员数据的情况下（上边的`SpecialKey`就没有成员变量），会直接调用类的聚合初始化([Aggregate initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization))，这个时候是不管有没有默认初始化函数的。

同时，拷贝构造函数也许要是私有的，特别在`Key`类不是`Secret`类的一个私有成员类的情况下。否则的话，下边这个case就会趁虚而入：

~~~C++
SpecialKey* pSk = nullptr;
Secret s("That is not fair!!!", *pSk);
~~~

虽然对空指针`pSk`解引用会产生未定义的行为，但是呢对于上述这种情况，大多数编译期也只是产生warning而已。把拷贝构造函数设为私有的，就可以从语法层面避免上述违规行为的发生。

## 结语

今天说的这个东西可能用处不是特别大啦，但是这样的小技巧积少成多，还是能够把我们的程序变得更robust。