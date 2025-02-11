---
layout:     post
title:      "算法笔记——单调栈"
subtitle:   "Leetcode刷题总结 五"
date:       2020-03-02
author:     "Felix Zhang"
header-img: "img/in-post/2020-03-02-Monotonous-Stack/bg.jpg"
catalog: true
tags:
---

# 单调栈

## 单调栈

单调栈本身是栈，其存放的数据是**有序的**，是处理无序数据的一种工具，单调栈也分为单调递增栈和单调递减栈。所谓的增减都是指在**栈中存放的元素的大小**。对一个无序数据列构造一个若干个有序数据段，则需要使用单调栈。其伪代码如下：
~~~C++
stack<int> st;
for (遍历这个数组)
{
    if (栈空 || 栈顶元素大于等于当前比较元素)
    {
        入栈;
    }
    else
    {
        while (栈不为空 && 栈顶元素小于当前元素)
        {
            栈顶元素出栈;
            更新结果;
        }
        当前数据入栈;
    }
}
~~~

以上算法只经历了一次遍历，所以时间复杂度为O(n)。以下为若干应用：

### 例1: 1019M. 链表中的下一个更大的节点

**题目描述**：给定一个链表，返回一个数组，存入每个节点后第一个比他大的节点的值，如果不存在那么存入0。

**思路**：依靠单调栈的特性，我们按照上述法则依次存入节点至栈，遇到比栈顶大的，那么就在这个节点对应的数组位置处存下这个数。需要注意的是，链表并不能直接靠索引找到相应的位置，因此在未确定的时候，在数组中相应节点的位置存入的是原节点的值，这样就可以在数组中采用索引的方式加速算法。

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
    vector<int> nextLargerNodes(ListNode* head) {
        if(!head) return {};
        vector<int> res = {};
        stack<int> stk;
        ListNode* cur = head;
      	// 用size来标记索引
        int size = 0;
        while(cur)
        {
            if(stk.empty() || res[stk.top()] >= cur->val)
            {
                res.push_back(cur->val);
                stk.push(size);
                cur = cur->next;
                size++;
            }
            else
            {
                res[stk.top()] = cur->val;
                stk.pop();
            }
        }
        while(!stk.empty())
        {
            res[stk.top()] = 0;
            stk.pop();
        }
        return res;
    }
};
~~~

### 例2: 84H. 直方图中最大的矩形

**题目描述**：Given *n* non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

![](https://assets.leetcode.com/uploads/2018/10/12/histogram.png)

Above is a histogram where width of each bar is 1, given height = `[2,1,5,6,2,3]`.

![](https://assets.leetcode.com/uploads/2018/10/12/histogram_area.png)

The largest rectangle is shown in the shaded area, which has area = `10` unit.

**我的思路**：从上例开始分析，比如我们想找高度为5的矩形，那么我们就必须找到左右制约着5这个bar的bar们，即图中的1和2，如果采用平凡的思路，那么就必须从5开始分别向左右遍历，然后找到制约着5的左右bar，那么算法的时间复杂度为O(n2)。我们采用单调栈的思路，在栈内始终存放递增序列，那么如果遇到新一个值比栈顶的小，那么栈顶的右制约bar就找到了，就是这个新的值，那左制约bar呢——必然是栈内处于栈顶下边的那个数，如果栈空了，那么左制约bar就是最左边。我们套用这个思路分析上例：

> 栈空，栈压入2——1比2小，计算2的candidate面积，1是2的右制约bar，因为栈内2下边没有其他数，因此2的左制约bar为左边界，弹出2——栈空、栈压入1——栈压入5——栈压入6——2比6大，计算6的candidate面积，左右制约bar分别为当前的2和栈内的5，弹出6——2比5大，计算5的candidate面积，左右制约bar分别为栈内的1和当前的2，弹出5——栈压入2——栈压入3——计算3的candidate面积，左右制约bar分别为栈内的2和当前的右边界，弹出3——计算2的candidate面积，左右制约bar分别是栈内的1和当前的右边界，弹出2——计算1的candidate面积，左右制约bar分别是左边界(因为栈内空了)和右边界(当前指向)。

每算出一个candidate面积即和当前的最大值比较并更新。最后得到的结果一定是正确的。

**代码**：

~~~C++
class Solution {
public:
    int largestRectangleArea(vector<int> &height) 
    {
        stack<int> s;
        height.push_back(0);
        int result = 0;
        for(int i = 0; i < height.size();) 
        {
            if (s.empty() || height[i] > height[s.top()])
                s.push(i++);
            else 
            {
                int tmp = s.top();
                s.pop();
                result = max(result, height[tmp] * (s.empty() ? i : i - s.top() - 1));
            }
        }
        return result;
    }
};
~~~

### 例3: 视野总和

**问题描述**：有n个人站队，所有的人全部向右看，个子高的可以看到个子低的发型，给出每个人的身高，问所有人能看到其他人发型总和是多少。

**我的思路**：平凡的思想依然是从每个开始遍历直到找到下个比自己高的，那么采用单调栈的思想问题就回到了例1，只不过例1是每次出栈改变对应序列的数，这里是计数问题。

### 例4: 最大区间

**问题描述**：给出一组数字，求一区间，使得区间元素和乘以区间最小值最大，求出这个最大值。

**我的思路**：首先我们可以明确一个思考方向：最后这个满足要求的最大区间，他左或右边的数一定比这区间的最小值要小，反证法：如果一个区间的左或右边的数比这个区间的最小值还大，那么将这个数包含进区间可以得到一个结果更大的区间，矛盾。因此，每一个数一定对应着包含着这个数的一个区间，使得这个区间的所有数都比这个数大，但是区间左右两边的数（如果存在）比这个数小。比如`6，3，7，4，5，8，2`这个序列4对应的区间即为` 7，4，5，8`。问题就又回到了例2，即找到每个数的左右限制bar，只不过这里所谓的"Candidate面积"计算方法和上次不同。
