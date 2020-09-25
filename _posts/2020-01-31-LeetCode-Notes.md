---
layout:     post
title:      "刷题笔记——异或的用法"
subtitle:   "LeetCode 刷题总结一"
date:       2020-01-31
author:     "Felix"
header-img: "img/in-post/2020-01-31-LeetCode-Notes/bg.jpg"
catalog: true
tags:
   - LeetCode 
   - Algorithm
---

# 按位异或"^"的用法 #

对于某些问题，巧用按位异或可以大大提高空间利用率，将某些算法的空间复杂度优化到O(1)。异或运算满足以下性质：

* 相同的数异或后为0，对每一位和整个数都成立。

* 符合交换律和结合律。
* 任何数与0异或结果不变。

## 231E. Power of 2 ##

判断一个数是否是2的次幂，从传统的角度可以不停的对这个数mod2，从二进制的角度，一个2的次幂的二进制表示应该只有一位是2，其余全是0，如0100（4）和01000000（64）。那么这个数减1后应是从右到左一串连续的1，如0011（3）和00111111（63）。那么这两个数按位异或，结果应该是0，如果这个数不是2的次幂，那么异或的结果不会为0。代码如下：

~~~c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n <= 0 ? false : !(bool)(n ^ (n - 1));
    }
};
~~~



## 342E. Power of 4 ##

接着上一道题的思路，4的次幂应是在2的次幂的基础上，对其二进制表示的唯一的1的位置有了要求，应该在奇数位上，如0100（4），00010000（16）和00100000（32）不是4的次幂。那么需要在检测2的次幂的基础上再此检测这个1的位置。

~~~c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        bool res = n ^ (n - 1);
        if(!res) return res;
        res = n ^ 0xAAAAAAAA;
        return !res;
    }
};
~~~



##  136E. Single Number ##

在一组数中，除了一个数以外，所有数都成对出现，找出这个数。将代码空间利用率优化到O(1)，使数组中所有数进行异或运算，因为两个相同的数异或后结果为0，所得的结果即为Single Number。

~~~c++
class Solution{
public:
    bool isPowerOfTwo(vector<int>& nums)
    {
        int res = 0;
        for(auto it = nums.begin(); it !=nums.end(); it++)
        {
            res ^= *it;
        }
        return res;
    }
};
~~~

