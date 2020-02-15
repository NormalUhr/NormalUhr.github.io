---
layout:     post
title:      "牛客网刷题笔记"
subtitle:   "面试题汇总 五"
date:       2020-02-12
author:     "Felix"
header-img: "img/in-post/2020-02-12-Niuke-Notes/bg.jpg"
catalog: true
tags:
   - Niuke



---

# 牛客网刷题笔记

* 关于输入输出

  string类型：`string s; getline(cin, s);`

  char类型：· char c; cin >> c;·

* 不稳定的排序算法有：希快选堆，分别希尔排序，快速排序，直接选择排序，堆排序

* 'int a = 10; char c = 'a';那么表达式` a + c + 4.32`的类型为？

  double，在表达式运算类型总是低等级往高等级转换。表达式的类型等级为：**char** < **short** < **int** < unsigned int < long < unsigned long < long long < unsigned long long < **float** < **double** < long double 不同类型运算结果类型向右边靠齐。无符号比有符号等级高。

* 【算法】大数相乘

  问题描述：由两个字符串表述成的两个大数（数位很长），求这两个数的积。

  我的思路：将每个数看成这个位数上的数字和10的相应次幂的组合。那么两个位数相乘，如第四位数（10^3）和第五位（10^4）数相乘，得到的结果就是10^7上的结果。这第一个数字中的每个数字都与另一个数字中的每个数字两两组合，得到的结果加到相应的位置上，最后结果再进位。

  坑：

  * 需要对输入数字的合法性进行检测。
  * string里的输入是符合人们正常书写习惯的，即左边的是高位，那么string里的低索引是高位。但是我们保存结果的时候希望10的次幂按照位数保存，所以需要一个转换，如果有k位数（k是string的size或者length），那么第i位数就是10的（k - i - 1）次。两个组合起来就是size1 + size2 - i - j - 2，这个容易弄错。
  * 在输出时也按照读数的习惯，从高位向低位输出。有可能前几位高位都是0，这个时候应该作预判，从第一位非零最高位往下读取。
  * 理论上来说，n位数乘m位数结果位数不大于m+n位数，不小于m+n-1位数。

  代码：

  ~~~C++
  #include<iostream>
  #include<string>
  using namespace std;
  bool check(string s)
  {
      if(s.empty()) return false;
      for(int i = 0; i < s.size(); i++)
      {
          if(s[i] < '0' || s[i] > '9') return false;
      }
  }
  
  void bigDataMult(string& s1, string& s2)
  {
      if(!check(s1) || !check(s2)) return;
      //string里的索引低的是高位，如果有k位数，那么第i位就是10的（k-i-1）次。
      //size1和size2代表的位数
      int size1 = s1.length(), size2 = s2.length();
      int size = size1 + size2;
      int ans[size];
      //初始化
      for(int i = 0; i < size; i++)
      {
          ans[i] = 0;
      }
      for(int i = 0; i < size1; i++)
      {
          for(int j = 0; j < size2; j++)
          {
              //索引数字大的是高位,代表10的次幂
              ans[size - i - j - 2] += (s1[i] - '0') * (s2[j] - '0');
          }
      }
      
      int jinwei = 0;
      
      for(int i = 0; i < size; i++)
      {
          ans[i] += jinwei;
          jinwei = ans[i] / 10;
          ans[i] = ans[i] % 10;
      }
      bool flag = false;
      for(int i = size - 1; i >= 0; i--)
      {
          if(!flag)
          {
              if(i && !ans[i])
              {
                  continue;
              }
              else
              {
                  flag = true;
                  cout << ans[i];
              }
          }
          else 
              cout << ans[i];
      }
      cout << endl;
      return;
  }
  int main()
  {
      string s1, s2;
      cin >> s1;
      cin >> s2;
      bigDataMult(s1, s2);
      return 0;
  }
  ~~~

* unsigned int 与 int 比大小？

  ~~~C++
  cout << (unsigned int)(-1) > 1 ? 1 : 0;
  ~~~

  以上代码将返回1，而不是0；这与将int类型的-1转换为unsigned int有关。

  > 计算机中32位int类型变量的范围，其中int类型是带符号整数。
  >
  > 正数在计算机中表示为原码，**最高位为符号位**:
  >
  > 1的原码为0000 0000 0000 0000 0000 0000 0000 0001
  >
  > 2147483647的原码为0111 1111 1111 1111 1111 1111 1111 1111
  >
  > 所以最大的正整数是2147483647
  >
  > 负数在计算机中表示为**补码**，最高位为符号位：
  >
  > -1：
  >
  > 原码为1000 0000 0000 0000 0000 0000 0000 0001，
  >
  > 反码为1111 1111 1111 1111 1111 1111 1111 1110，
  >
  > 补码为1111 1111 1111 1111 1111 1111 1111 1111
  >
  > -2147483647：
  >
  > 原码为1111 1111 1111 1111 1111 1111 1111 1111，
  >
  > 反码为1000 0000 0000 0000 0000 0000 0000 0000，
  >
  > 补码为1000 0000 0000 0000 0000 0000 0000 0001
  >
  > 所以最小的负数是-2147483647吗？错，不是。
  >
  > 在二进制中，0有两种表方法。
  >
  > +0的原码为0000 0000 0000 0000 0000 0000 0000 0000，
  >
  > -0的原码为1000 0000 0000 0000 0000 0000 0000 0000，
  >
  > 因为0只需要一个，所以把-0拿来当做一个最小的数-2147483648。
  >
  > -2147483648的补码表示为1000 0000 0000 0000 0000 0000 0000 0000，在32位没有原码。
  >
  > 注意，这个补码并不是真正的补码，真正的补码是1 1000 0000 0000 0000 0000 0000 0000 0000，溢出。
  >
  > 所以带符号32位int类型整数为-2147483648~2147483647

  

> 最后更新日期：2020-02-12