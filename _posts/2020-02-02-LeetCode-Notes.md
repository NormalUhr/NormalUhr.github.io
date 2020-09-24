---
layout:     post
title:      "快慢指针"
subtitle:   "LeetCode刷题总结 三"
date:       2020-02-02
author:     "Felix"
header-img: "img/in-post/2020-02-02-LeetCode-Notes/bg.jpg"
catalog: true
tags:
   - LeetCode
   - Algorithm


---

# 快慢指针

快慢指针是解决链表问题的常见工具之一，其中快指针和慢指针同时遍历链表，快指针常作为信号工具，总是在慢指针的前面。当快指针发出信号时，通常慢指针所指的就是我们想要的节点。

## 19M. 移除倒数第n个链表结点。

**问题描述**：给定链表头，移除倒数第n个链表结点。

**我的思路**：采用快慢指针，第一个指针快第二个指针n个节点遍历，当第一个指针访问到链表尾部(null)时，第二个指针刚好指向倒数第n个。

**坑：**通常删去链表结点的方法是让被删结点前一个节点指向被删结点后一个节点，但是此方法对head不起作用，需额外判断。

代码：

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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if(!head) return head;
        ListNode *front = head, *back = new ListNode(0);
        back->next = head;
        while(n--)
        {
            front = front->next;
        }
        while(front)
        {
            front = front->next;
            back = back->next;
        }
        if(back->next == head) return head->next;
        back->next = back->next->next;
        return head;
    }
};
~~~



## 141E. 链表的循环检测

**问题描述**：检测一个链表是否有环。

**思路**：暴力的方法是遍历每一个节点，并将每一个节点的地址保存在一个hash表中，每到一个节点都检测这个节点的地址是否存在再hash表中。这种方法的时间和空间复杂度均为O(N)。利用快慢指针，每次快指针移动两个节点，而慢指针移动一个节点，那么快指针相对于慢指针移动的相对速度为一个节点，所以只要有环，快指针总会追上慢指针，如果遇到了链表的结尾，那么就没有环。

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
    bool hasCycle(ListNode *head) {
        if(!head) return false;
        ListNode *fast = head, *slow = head;
        while(fast->next && fast->next->next) 
        {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow) return true;
        }
        return false;
    }
};
~~~



