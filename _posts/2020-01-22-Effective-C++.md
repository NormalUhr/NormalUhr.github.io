---
layout:     post
title:      "Effective C++ 学习笔记"
subtitle:   "C++学习笔记 二"
date:       2020-01-22
author:     "Felix"
header-img: "img/in-post/2020-01-22-Effective-C++/bg.jpg"
catalog: true
tags:
   - C++
---

# Effective C++

## 条款零 术语

> 定义式：对对象而言，定义式是编译器为此对象拨发内存的地点。

* 使用explicit关键字防止构造函数定义的隐式转换。

  构造函数的隐式转换：可以用单个实参来调用的构造函数定义了从形参到该类型的隐式转换。

  例如：

  ~~~C++
  class things
  {
   		public:
    			things(const std::string& name = ""): m_name(name), height(0), weight(0) {}
    			int CompareTo(const things& other);
    	private:
    			std::string m_name;
    			int height;
    			int weight;
  };
  ~~~

  那么在things的构造函数可以只用一个实参完成初始化，所以可以完成一个隐式转化。如下：

  ~~~C++
  things a;
  /*
  ...这里是对a的初始化及相关操作...
  */
  std::string nm = "book";
  //下面使用nm完成string到things的隐式转换,该对象在函数结束后被析构。
  int result = a.CompareTo(nm);
  ~~~

  这时将析构函数声明为explicit，来防止隐式类型转换。

  ~~~C++
  class things
  {
   		public:
    			explicit things(const std::string& name = ""): m_name(name), height(0), weight(0) {}
    			int CompareTo(const things& other);
    	private:
    			std::string m_name;
    			int height;
    			int weight;
  };
  ~~~

  这时仍然可以通过显式使用构造函数完成以上类型转换。

  ~~~c++
  things a;
  /*
  ...这里是对a的初始化及相关操作...
  */
  std::string nm = "book";
  //下面使用nm完成string到things的隐式转换,该对象在函数结束后被析构。
  int result = a.CompareTo(things(nm));
  ~~~

  Google的C++规范中提到explicit的优点是可以避免不合时宜的类型转换，并无重大缺点，所以规定所有单参数的构造函数必须加上explicit，只有极少数的情况下的拷贝构造函数可以不声明为explicit，如作为其他类的透明包装容器。

  > 被声明为explicit的拷贝构造函数往往比其non-explicit兄弟更受欢迎，因为他们禁止编译器执行非预期（往往也不被期望）的类型转换。除非我有一个好的理由允许构造函数被用于隐式类型转换，否则我会把它声明为explicit。我鼓励你遵循相同的政策。





## 条款二 尽量以const, enum, inline替换#define

> 宁可以编译器替换预处理器。

1. #define不被视作语言的一部分。

~~~C++
#define ASPECT_RATIO 1.653
~~~

被#define定义的名称或记号在被编译器开始处理之前就被移除了。记号式调试器（symbolic debugger）中并不会出现相应的记号，所使用的名称可能并未进入记号表（symbol table）。解决策略是用const常量替换上述的宏。

~~~C++
const double AspectRatio = 1.653;
~~~

另外，对于浮点数而言，浮点常量被分配的空间可能比使用宏的要小。

2. enum hack

对于const static类型的类成员变量，我们可以采取以下做法

* 在类内使用声明式进行赋值并在实现文件定义。

  ~~~C++
  class GamePlayer
  {
  private:
    	static const int Num = 5; 	//常量声明式
    	int scores[Num];						//使用该常量
    ...
  };
  
  const int GamePlayer::Num;			//在实现文件中，Num的定义，此处由于声明已赋值，因而不用赋值。
  ~~~

* 在类内使用声明式并在实现文件中定义且赋值。

  ~~~C++
  class GamePlayer
  {
  private:
    	static const int Num; 			//常量声明式
    	int scores[Num];						//使用该常量
    ...
  };
  
  const int GamePlayer::Num - 5;			//在实现文件中，Num的定义，此处由于声明已赋值，因而不用赋值。
  ~~~

需要特别声明的是，#define并不支持封装，也没有域的概念。因此无法在class内使用#define定义一个常量。针对以上const static int类型的变量我们有一个替代做法。

~~~C++
class GamePlayer
{
private:
  	enum {Num = 5}; 						//常量声明式
  	int scores[Num];						//使用该常量
  ...
};
~~~

理论依据是：一个枚举类型的数值可被当作int使用。如果不想让别人活的一个指针或引用指向这个整数常量，ennum也可以实现这个约束。并且enum不会导致非必要的内存分配。

3. 不要用宏来实现函数。

使用template inline函数可以获得宏带来的效率以及一般函数的安全性。即使对每个参数加上小括号，宏的安全性也无法被保证，可由以下代码看出：

~~~C++
#define Max(a, b) f((a) > (b) ? (a) : (b))

int a = 5, b = 0;
Max(++a, b);				//a被累加2次
Max(++a, b + 10);		//a被累加1次
~~~

使用template inline可将上述函数改为：

~~~C++
template<typename T>
inline void Max(const T& a, const T& b)
{
  f(a > b ? a : b);
}
~~~



## 条款三 尽量使用const

### 1. 将const用于函数的返回值，各参数以及函数自身（如果是成员函数）。

* 将函数**返回值**设为常量可以降低客户因错误而造成的意外，且不至于放弃安全性和高效性。如：

  ~~~C++
  class Rational { ... }		//定义了一个有理数类。
  const Rational operator* (const Rational& lhs, const Rational& rhs);
  ~~~

  这里将有理数的乘法重载设置为const是为了避免以下错误：

  ~~~C++
  Rational a, b, c;
  ...
  if(a * b = c) { ... } // 如果*重载返回值未设置为const，那么此处不会报错。
  ~~~

* const成员函数

  此函数的目的是为了确定该成员函数可作用于const对象身上，此函数的重要意义在于提供了C++的class接口，告诉客户哪个函数可以改动对象而哪个不行。

  > 改善C++效率的一个根本办法是以pass by reference-to-const的方式传递对象，而此技术可行的根本前提是我们有const成员函数来处理取得的const对象。

* 常量性不同的两个成员函数可以被重载。

  这是C++的一个重要特性，由以下读取文字块类中字符的函数可以体现出：

  ~~~C++
  class TextBlock
  {
  public:
    	...
      const char& operator[] (std::size_t pos) const	//operator[]用于const对象 pass-reference-to-const实现
      { return text[pos]; }
    	char& operator[] (std::size_t pos)
      { return text[pos]; }
  private:
    	std::string text;
  }
  
  void print(TextBlock& tb)
  {
    	std::cout << tb[0];
    	tb[0] = 'x';	//正确
  }
  
  void print(const TextBlock& ctb)
  {
    	std::cout << ctb[0];	//调用const TextBlock::operator[]
  		ctb[0] = 'x'; 				//错误
  }
  ~~~

  

