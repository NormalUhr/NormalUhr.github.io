---
layout:     post
title:      "LeetCode 刷题记录"
subtitle:   "思路与坑"
date:       2019-12-10
author:     "Felix Zhang"
header-img: "img/in-post/2019-12-10-LeetCode-Records/bg.jpg"
catalog: true
tags:
   - LeetCode



---

# LeetCode 刷题记录

本帖子记录刷题过程中的思路，经验以及遇到的坑。

*****
## 2M. 两数相加

**问题描述**：给定两已知链表由低到高保存两数的各位，将两数相加后返回一链表。

~~~
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
~~~

**我的思路**：由于不知道哪个链表长，所以两个链表都被在将相加后的结果所替换。为了处理最后一位是否要进位的问题，添加哨兵辅助链表的遍历。（在所有需要对链表后续进行操作的问题都建议添加哨兵并从哨兵开始用do...while{}语句）

**坑**：不用哨兵的话代码会冗杂一些。

**代码**：

~~~C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        if(!l1) return l2;
        if(!l2) return l1;
        ListNode *pt1 = l1, *pt2 = l2;
        bool flag = false;
        while(pt1 && pt2)
        {
            pt2->val += pt1->val;
            pt1->val = pt2->val;
            pt2 = pt2->next;
            pt1 = pt1->next;
        }
        ListNode *pt = new ListNode(0), *l = nullptr; //pt是哨兵
        if(pt1) 
        {
            pt->next = l1;
            l = l1;
        }
        else 
        {
            pt->next = l2;
            l = l2;
        }
        do
        {
            pt = pt->next;
            if(flag) 
            {
                pt->val++;
                flag = false;
            }
            if(pt->val >= 10)
            {
                pt->val -= 10;
                flag = true;
            }
        }while(pt->next);
        if(flag)
        {
            pt->next = new ListNode(1);
        }
        return l;
    }
};
~~~



*****

## 3M. 不含重复字母的最长子串

## 159M. 最多含有2个重复字母的最长字串

## 340H. 最多含有k个重复字母的最长字串

参见滑动窗口的应用<https://starkschroedinger.github.io/2020/02/01/LeetCode-Notes/。

*****

## 6M. ZigZag转换

**问题描述**：将给定的字符串转换为zigzag形式，并将转换后的字符串从左到右重新读出。如给定字符串"PAYPALISHIRING"，转换行数为3，那么转换后的形式为：“PAHNAPLSIIGYIR”

~~~
P   A   H   N
A P L S I I G
Y   I   R
~~~

**我的思路**：找方法时可以将每个字母的坐标写下，观察坐标的变化，如果我们给每一行创建一个string用于存储，那么其实我们关心的是每个字母的y坐标如何变化。观察后发现，如果设定行数为n，那么每个字母的转换后行数变化应该有如下规律：1--2--...--n--n-1--...-1。利用这个规律，我们可以对原string的每一个字母进行新的行数分配，再将每个新的string串起来得到答案。

**加速**：我们观察，每一行字母在原字符串中的序号是有规律的，如第0行每个序号相差2n-2，第i行相差(n-i+1)+(n-i+1)和(i)+(i)交替。得知此规律我们就可以直接对输入string的字母进行重新排列。

**代码**：

~~~C++
class Solution
{
public:
    string convert(string s, int numRows)
    {
        if(s.empty()) return "";
        if(numRows < 2) return s;
        string res = "";
        for(int i = 0; i < numRows; i++)
        {
            int cur = i;
            while(1)
            {
                if(i != numRows - 1)
                {
                    if(cur >= s.size()) break;
                    string curS1(1, s[cur]);
                    res += curS1;
                    cur += 2 * (numRows - i - 1);
                }
                if(i != 0)
                {
                    if(cur >= s.size()) break;
                    string curS2(1, s[cur]);
                    res += curS2;
                    cur += 2 * i;    
                }
            }
        }
        return res;
    }
};
~~~



*****

## 136E 孤独的数

## 231E 2的平方

## 342E 4的平方

参见异或的用法<<https://starkschroedinger.github.io/2020/01/31/LeetCode-Notes/>>

*****

## 141E. 检测链表是否有环

参见快慢指针<https://starkschroedinger.github.io/2020/02/02/LeetCode-Notes/>。

*****



