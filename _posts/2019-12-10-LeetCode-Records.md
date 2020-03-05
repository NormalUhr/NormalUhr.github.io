---
layout:     post
title:      "LeetCode 刷题记录"
subtitle:   "思路与坑"
date:       2020-03-05
author:     "Felix Zhang"
header-img: "img/in-post/2019-12-10-LeetCode-Records/bg.jpg"
catalog: true
tags:
   - LeetCode
---

# LeetCode 刷题记录

本帖子分类记录刷题过程中的思路，经验以及遇到的坑。

## I. 线性表(数组+链表)

### 8M. 字符串中提取整数

**问题描述**：从一个给定的字符串中提取数字，要求从第一个非空字符开始，如果第一个非空字符不是正负符号或数字则返回0；如果超出int类型的范围（大于INT_MAX或小于INT_MIN）则返回INT_MAX或INT_MIN。

**我的思路**：很平凡的遍历思想，首先判断把空字符给跳过去，然后判断是否是数字，是数字的话记录数字。问题是坑太多，还有就是在一位一位等待数字的过程中判断是否会溢出int很重要，可以记住。

**坑**：在遍历的时候因为需要判断的条件太多，使用while循环很容易忘记判断角标i是否超出str的size。另一个坑就是判断溢出。

**优化**：判断是否溢出的代码`if((res > INT_MAX / 10) || (temp >= INT_MAX % 10 && res == INT_MAX / 10)) return INT_MAX;` 和`if((res == INT_MIN / 10 && -temp <= INT_MIN % 10) || res < INT_MIN / 10) return INT_MIN;`。

**代码**：

~~~C++
class Solution {
public:
    int myAtoi(string str) {
        if(str.empty()) return 0;
        int i = 0, res = 0;
        bool flag = true;
        
        //找到第一个空格后的字符
        while(str[i] == ' ')
        {
            i++;
        }
      	//判断是否是字符串
        if((str[i] - '0' < 0 || str[i] - '0' > 9) && (str[i] != '-') && (str[i] != '+')) return 0; 
        //判断正负，flag用于标记
        else if(str[i] == '-' || str[i] == '+') 
        {
            flag = str[i] == '-' ? false : true;
            i++;
        }
        //开始处理数字
      	//以下有重复代码段，应该避免，可以优化。
        if(flag)
        {
            while((str[i] - '0' >= 0 && str[i] - '0' < 10) && i < str.size())
            {
                int temp = str[i] - '0';
                if((res > INT_MAX / 10) || (temp >= INT_MAX % 10 && res == INT_MAX / 10)) return INT_MAX;
                res *= 10;
                res += temp;
                i++;
            }    
        }
        else
        {
            while((str[i] - '0' >= 0 && str[i] - '0' < 10) && i < str.size())
            {
                int temp = str[i] - '0';
                if((res == INT_MIN / 10 && -temp <= INT_MIN % 10) || res < INT_MIN / 10) return INT_MIN;
                res *= 10;
                res -= temp;
                i++;
            }
        }
        return res;
    }
};
~~~

### 19M. 移除倒数第n个链表结点。

**问题描述**：给定链表头，移除倒数第n个链表结点。

**我的思路**：采用快慢指针，第一个指针快第二个指针n个节点遍历，当第一个指针访问到链表尾部(null)时，第二个指针刚好指向倒数第n个。

**坑：**通常删去链表结点的方法是让被删结点前一个节点指向被删结点后一个节点，但是此方法对head不起作用，需额外判断。

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

### 21E. 合并两有序链表

**问题描述**：将两个有序链表合并成一个有序链表。

**我的思路**：思路其实很简单，从小到大两链表都有对应的指针，一个一个遍历比较大小即可，但是实现的时候把代码写的简洁却不是一件容易的事。下边放出官方给出的效率最高的代码之一。

**代码**：

~~~C++
* Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* head = NULL;
        ListNode* prev = NULL;
        ListNode* cur;
      	//将l1和l2判断为空整合到下边
        while(l1 || l2) {
            if(l1==NULL || (l2 && l2->val<l1->val)) {
                cur = l2;
                l2 = l2->next;
            } else {
                cur = l1;
                l1 = l1->next;
            }
            if(head==NULL) head = cur;
            if(prev) prev->next = cur;
            prev = cur;
        }
        return head;
    }
};
~~~

### 24M. 成对地交换链表结点

**问题描述**：给出一个链表，成对交换节点的第1、2，3、4...个节点。要求不能改变节点的值，只能改变节点的指向。

**我的思路**：很平常的思路，一道练习链表节点操作的题。

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
    ListNode* swapPairs(ListNode* head) {
        if(!head || !head->next) return head;
        ListNode *record = head->next, *pre = nullptr;
        while(head && head->next)
        {
            ListNode *temp = head->next;
            //保存2号
            temp = head->next;
            //1号指向3号
            head->next = head->next->next;
            //2号指向1号
            temp->next = head;
            //0号指向2号，如果0号为空，设置0号为1号。
            if(!pre) pre = head;
            else pre->next = temp;
            //0号更新到1号
            pre = head;
            //当前1号更新到3号
            head = head->next;
        }
        return record;
    }
};
~~~

### 31M. 下一种排序（Next Permutation）

**问题描述**：给出一个数字序列，找出这个数字序列的下一个排序，使得在这些数字所组成的所有序列按从小到大排列的话，新的序列刚好位于原序列的下一个。如果找不到的话，就把整个序列翻转顺序输出。

**我的思路**：在序列中找到一对数，在序列中构成升序。并且近可能位于序列尾端。再第一个数尽可能接近尾部的前提下找到最接近尾段的第二个数。将两个数交换，再把第一个数位置后边的所有数翻转。得到的序列就是答案。如果找不到这样一对数对，就把整个数组翻转。

**坑**：一开始没想到把第一个数翻转。另一个坑就是两个loop的次序一开始想错了。

**代码**：

~~~C++
lass Solution {
public:
    void nextPermutation(vector<int>& nums) {
        if(nums.size() < 2) return;
        int naughty1, naughty2;
        for(naughty2 = nums.size() - 2; naughty2 >= 0; naughty2--)
        {
            for(naughty1 = nums.size() - 1; naughty1 > naughty2; naughty1--)
            if(nums[naughty2] < nums[naughty1])
            {
                int temp = nums[naughty1];
                nums[naughty1] = nums[naughty2];
                nums[naughty2] = temp;
                reverse(naughty2 + nums.begin() + 1, nums.end());
                return;
            }
        }
        reverse(nums.begin(), nums.end());
        return;
    }
};
~~~

**思考**：如果问题改成上一个排序的话，应该怎么做？同样还是找到位于最后的一对数使其组成降序，将他们交换顺序，之后再把第一个数后边的所有数翻转。

### 33M. 在旋转过的有序序列寻找目标

**问题描述**：已知一个[旋转过的有序序列]()长度为n，在log(n)时间复杂度内寻找target，存在则返回数组索引，不存在则返回-1。被旋转的序列：[0,1,2,4,5,6,7]旋转后变为[5,6,7,0,1,2,4]。

**我的思路**：首先使用二分法查找pivot，即围绕哪一点旋转的。然后再从被pivot分割成的两段中的某一段中使用二分法寻找目标。

**坑**：当数组只有一个元素时，需要特别注意。

**代码**：

~~~C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.empty()) return -1;
        int l = 0, r = nums.size() - 1, m = (l + r) / 2;
        //pivot
        int key = -1;
        //result
        int res = -1;
        //二分法查找pivot
        if(nums[0] <= nums.back()) key = nums.size() - 1;
        else
        {
            while(1)
            {
                if(nums[l] <= nums[r])
                {
                    if(nums[m] > nums[m + 1])
                    {
                        key = m;
                    }
                    else if(nums[m] < nums[m - 1])
                    {
                        key = m - 1;
                    }
                    break;
                }
                m = (l + r) / 2;
                if(nums[l] > nums[m])
                {
                    r = m - 1;
                }
                else
                {
                    l = m + 1;
                }
            }
        }
        //测试用
        //return key;
        //划分目标区间
        if(target >= nums[0])
        {
            l = 0; r = key;
        }
        else
        {
            r = nums.size() - 1;
          	//防止被划分的区间只有一个元素，换言之key和r相等
            l = (key + 1) > r ? r : (key + 1);
        }
        //二分法查找目标
        do
        {
            m = (l + r) / 2;
            if(nums[m] == target)
            {
                res = m;
                break;
            }
            else if(nums[m] > target)
            {
                r = m - 1;
            }
            else
            {
                l = m + 1;
            }
        }while(l <= r);
        return res;
    }
};
~~~

### 34M. 在有序序列中找到元素的第一个和最后一个位置

**问题描述**：给出一个有序数组和目标元素，找出数组中该元素出现的第一个和最后一个位置。要求对数级时间复杂度。

**我的思路**：首先通过二分法，找出该元素的任意一个位置。再从该位置和二分法的左右端的最后位置分别用二分法查找该元素的左、右起始位置。

**坑**：对于常见的二分法查找，在每次迭代的过程中为了防止最后left = right + 1时的死循环，使left = middle + 1和right = middle - 1是有效的方法。但是对于寻找该元素的第一个和最后一个位置，这样的做法不合适。因为循环的终止条件并不是middle等于或不等target，而是middle等于target但是middle的下一个不是target。所以在之后的二分法中，迭代只能是left = middle 和 right = middle 而不再加减1。这样需要额外的防止死循环的方法—当middle等于left时，判断并终止。

**代码**：

~~~C++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> res = {};
        if(nums.empty() || nums[0] > target || nums.back() < target) 
            return {-1, -1};
        int middle, left = 0, right = nums.size() - 1;
        while(left <= right)
        {
            middle = (left + right) / 2;
            if(nums[middle] == target)
            {
                int l, r, m = middle;
                while(1)
                {
                    if(nums[left] == target) 
                    {
                        l = left;
                        break;
                    }
                    l = (m + left) / 2;
                    //m = left + 1;防止死循环
                    if(l == left) 
                    {
                            l = m;
                    }
                    if(l == m) break;
                    if(nums[l] == target && nums[l - 1] != target) break;
                    else if(nums[l] == target) 
                    {
                        //这个保证了nums[m]永远是target
                        m = l;
                    }
                    else
                    {
                        left = l;
                    }
                }
                m = middle;
                while(1)
                {
                    if(nums[right] == target)
                    {
                        r = right;
                        break;
                    }
                    r = (m + right) / 2;
                    if(r == right)
                    {
                        r = m;
                    }
                    if(r == middle || r == m) break;
                    if(nums[r] == target && nums[r + 1] != target) break;
                    else if(nums[r] == target)
                    {
                        m = r;
                    }
                    else
                    {
                        right = r;
                    }
                }
                res.push_back(l); res.push_back(r);
                return res;
            } 
            else if(nums[middle] > target)
            {
                right = middle - 1;
            }
            else
            {
                left = middle + 1;
            }
        }
        return {-1, -1};
    }
};
~~~

### 56M. 融合区间

**问题描述**：给出若干个小区间，如果其中任意两个有重叠则二者可以融合，返回所有融合操作过的区间。

**我的思路**：按照区间头从小到大排序，比较好分析一些。对每对相邻区间排序，如果后者的头小于前者的尾，则可以混合，所有相邻的混合过的区间可用start和end统一表示成一个区间。这道题可分为允许和不允许对原输入进行操作的，如果允许改变输入，那么混合过后可以直接删除多余的那个，改变另一个的尾部，而剩去start和end进行计数，当然从提交结果来看，vector的erase操作是相当费时的。

**代码**：

不改动输入：

~~~C++
class Solution {
public:
    static bool compFirst(const vector<int>& a, const vector<int>& b)
    {
        return a[0] < b[0] ? true : false;
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.size() < 2) return intervals;
        sort(intervals.begin(), intervals.end(), compFirst);
        vector<vector<int>> res = {};
        //注意处理最后一个更新了的问题。
        int start = 0;
        int end = start;
        for(int i = 0; i < intervals.size();i++)
        {
            //更新end
            if(start == i) end = intervals[start][1];
            //如果涉及到要合并，注意每次合并的是i的下一个，即i+1。
            if(i != intervals.size() - 1 && end >= intervals[i + 1][0])
            {
                end = max(end, intervals[i + 1][1]);
            }
            else 
            {
                res.push_back({intervals[start][0], end});
                //start重新开始
                start = i + 1;                
            }
        }
        return res;
    }
};
~~~

改动输入：

~~~C++
class Solution {
public:
    static bool compFirst(const vector<int>& a, const vector<int>& b)
    {
        return a[0] < b[0] ? true : false;
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.size() < 2) return intervals;
        sort(intervals.begin(), intervals.end(), compFirst);
        //由于有删除操作，所以intervals的大小可能会更改，需要注意。
        for(int i = 0; i < intervals.size();)
        {
            if(i != intervals.size() - 1 && intervals[i][1] >= intervals[i + 1][0])
            {
                intervals[i][1] = max(intervals[i][1], intervals[i + 1][1]);
                auto it = intervals.begin() + i + 1;
                intervals.erase(it);
            }
            else i++;
        }
        return intervals;
    }
};
~~~

### 61M. 轮换链表

**问题描述**：给出一个链表和一个非负整数k，旋转链表，将链表每个节点向右移动k个位置，链表尾移动后变为链表头。

**我的思路**：思路可分为寻找支点(pivot)和寻找移动过后的链表头。支点即位倒数第k个数的前一个数。

**代码**：

1. 寻找支点：

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
       ListNode* rotateRight(ListNode* head, int k) {
           if(k < 0) return {};
           else if(!head || k == 0) return head;
           //链表的个数
           int size = 1;
           ListNode *last = head;
           while(last->next)
           {
               size++;
               last = last->next;
           }
           k = k % size;
           if(k == 0) return head;
           //找到倒数第k个数,这时k大于0小于n
           ListNode *pivot = head;
           while(--size != k)
           {
               pivot = pivot->next;
           }
           //现在pivot指向的就是倒数第k个，本身是最后一个，指向NULL
           last->next = head;
           head = pivot->next;
           pivot->next =nullptr;
           return head;
       }
   };
   ~~~

2. 寻找链表头 略

### 86M. 分割链表

**问题描述**：给定一个数组和一个分割数，输出数组满足比分割数小的数在前，其余的数在后，且与输入的相对顺序不能乱。

**我的思路**：采用双指针方法，一个指针负责遍历，另一个指针负责指向已处理好的小于分割数的节点的最前端。为了方便处理head的问题，添加了哨兵便于操作（反而使占用内存变大了）。

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
    ListNode* partition(ListNode* head, int x) {
        //覆盖了head是空集和x不在链表中的情况
        ListNode* here = head;
        if(!here) return head;
        //Pivot
        ListNode *less = new ListNode(0), *last = less;
        less->next = head;
        //特殊情况 如果头是小于x和大于等于x成立不成立呢？
        //需要注意的是，如果head之前需要加上一个数，那么head需要前一一位，否则会造成节点丢失
        while(here)
        {
            if(here->val < x)
            {
                if(here == less->next)
                {
                    less = less->next;
                    here = here->next;
                    last = last->next;
                }
                else
                {
                    last->next = here->next;
                    here->next = less->next;
                    less->next = here;
                    less = less->next;
                    here = last->next;
                    if(head->val >= x) 
                        head = less;
                }
            }
            else
            {
                here = here->next;
                last = last->next;
            }
        }
        return head;
    }
};
~~~

### 141E. 检测链表是否有环

参见快慢指针<https://starkschroedinger.github.io/2020/02/02/LeetCode-Notes/>。

### 189E. 翻转数组

**问题描述**：将数组的后k位顺序不变的移动到数组的前部。

**我的思路**：在数组后边重复一个数组，同时选取符合要求的部分，重新组成数组。

**优化**：将前k位翻转，将剩下的再翻转一次，之后整个数组再翻转一次。

**代码**：

~~~C++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        if(nums.size() == 0 || k == 0) return;
        k = k % nums.size();
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
        return;
    }
};
~~~

### 977E. 有序数组的平方

**问题描述**：一个有序数组包含所有正负数和0，将每个数的平方输出并按大小排列成为新的有序数组，

**我的思路**：和上边一题很相似，一开始的做法是找到第一个正数，然后就和上题很相似，从左右分别检测绝对值大小并延伸下去，后来发现其实可以直接从最大的开始找，因为最大的在最左边或最右边，效率会提升很多。

**代码**：

~~~C++
//可优化，我的方法是先循环找到第一个大于0的数，然后从中间往两边双指针遍历，再比较大小。、
//比较高效的做法是首先将res的size固定好。vector<int> res(A.size(), 0)然后再从两边往中间遍历，
//直接把平方结果从res的最后往前边放。之前做法还有一个坏处是，由于最后不知道正数部分还是负数部分剩的
//多，所以之后的遍历方向不能确定。因此不可避免的会造成代码重复（两个while循环）。    
class Solution {
public:
    vector<int> sortedSquares(vector<int>& A) {
        if(A.empty()) return {};
        vector<int> res = {};
        int m = 0;
        while(m < A.size() && A[m] <= 0)
            m++;
        int k = m - 1;
        while(m < A.size() && k >= 0)
        {
            if(A[m] > (-A[k]))
            {
                res.push_back(A[k] * A[k]);
                k--;
            }
            else
            {
                res.push_back(A[m] * A[m]);
                m++;
            }
        }
        while(k >= 0)
        {
            res.push_back(A[k] * A[k]);
            k--;
        }
        while(m < A.size())
        {
            res.push_back(A[m] * A[m]);
            m++;
        }
        return res;
    }
};
~~~

### 1365E. 数组中有多少数小于当前数

**问题描述**：给出一个数组，返回大小相同的数组，每个位置保存着原数组中有多少数小于对应位置的数。其中每个数不大于100且不小于0，总数量不大于100。

**我的思路**：建立一个大小为101的数组，每个数组都代表给定序列中这个数的个数，计数后相加。

**代码**：

~~~C++
static auto magic = []() {ios_base::sync_with_stdio(false); cin.tie(nullptr); return false; }();

class Solution {
public:
	int n;

	vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
		vector<int> count(105, 0);
		n = static_cast<int>(nums.size());
		for (int  i = 0; i < n; i++)
		{
			count[nums[i]]++;
		}
		for (int i = 1; i < 105; i++)
		{
			count[i] += count[i - 1];
		}
		vector<int> ans;
		for (int i = 0; i < n; i++)
		{
			if (nums[i] == 0)
			{
				ans.push_back(0);
			}
			else
			{
				ans.push_back(count[nums[i] - 1]);
			}
		}
		return ans;
	}
};
~~~



## II. 字符串

### 12M. 阿拉伯数字到罗马数字转换

**问题描述**：将1～3999范围内的整数转换为罗马数字。

**我的思路**：无脑转换，写的复杂一点，算起来很快。

**代码**：

~~~C++
class Solution {
public:
    string intToRoman(int num) {
        if(num > 3999) return "";
        
        int ge = num % 10;
        num /= 10;
        int shi = num % 10;
        num /= 10;
        int bai = num % 10;
        num /= 10;
        int qian = num;
        
        string res = "";
        while(qian--)
        {
            res += "M";
        }
        
        if(bai < 4)
        {
            while(bai--)
            {
                res += "C";
            }
        }
        else if(bai < 9)
        {
            if(bai == 4)
                res += "CD";
            else
            {
                res += "D";
                while(bai - 5)
                {
                    res += "C";
                    bai--;
                }
            }
        }
        else res += "CM";
        
        if(shi < 4)
        {
            while(shi--)
            {
                res += "X";
            }
        }
        else if(shi < 9)
        {
            if(shi == 4)
                res += "XL";
            else
            {
                res += "L";
                while(shi - 5)
                {
                    res += "X";
                    shi--;
                }
            }        
        }
        else res += "XC";
        
        if(ge < 4)
        {
            while(ge--)
            {
                res += "I";
            }
        }
        else if(ge < 9)
        {
            if(ge == 4)
                res += "IV";
            else
            {
                res += "V";
                while(ge - 5)
                {
                    res += "I";
                    ge--;
                }
            }
        }
        else res += "IX";
        
        return res;
    }
};
~~~

### 151M.翻转字符串中的单词

**问题描述**：给一字符串，将所有单词的顺序颠倒并隔空格组成新的字符串，其中单词本身不改变。所有非空格字符都看成单词的一部分，字符串收尾的空格在输出时要忽略。

**思路**：遍历后组成新的字符串，可用vector和stack来实现；也可直接在原字符串上进行更改，下边用这三种方法实现，其中因为在原字符串上涉及到很多erase操作，所有效率最低，vector效率最高。

**代码**：

stack实现：

~~~c++
class Solution {
public:
    string reverseWords(string s) {
        stack<string> stk;
        int pos = 0;
        string res;

        while (pos < s.length()) {
            int end, start;
            
            while(s[pos] == ' ' && pos < s.length())
                pos++;
            
            start = pos;
            
            while (s[pos] != ' ' && pos < s.length())
                pos++;
            
            end = pos;
            //s末尾的一段空格
            if (end == start)
                break;
            
            stk.push(s.substr(start, end - start));
        }

        while (!stk.empty()) {
            res += stk.top();
            stk.pop();
            if (!stk.empty())
                res += " ";
        }

        return res;
    }
};
~~~

vector实现：

~~~c++
class Solution {
public:
    string reverseWords(string s) {
        vector<string> vkt;
        int pos = 0;
        string res = "";

        while (pos < s.length()) {
            int end, start;
            
            while(s[pos] == ' ' && pos < s.length())
                pos++;
            
            start = pos;
            
            while (s[pos] != ' ' && pos < s.length())
                pos++;
            
            end = pos;
            //s末尾的一段空格
            if (end == start)
                break;
            
            vkt.push_back(s.substr(start, end - start));
        }
        for(int i = vkt.size() - 1; i >= 0; i--)
        {
            res += vkt[i];
            if(i != 0)
                res += " ";
        }
        return res;
    }
};
~~~

在原字符串上更改：

~~~C++
class Solution
{
public:
    string reverseWords(string s) {
        if(s.empty()) return s;
        bool flag = false;
        unsigned start = s.size() - 1;
        for(int i = s.size() - 1; i >= 0; i--)
        {
            //当前是空格分为是空格的开头和已经有一串空格在其身后
            if(isspace(s[i]))
            {
                if(flag)
                {
                    s += s.substr(i + 1, start - i) + " ";
                    //i不删start要删
                    s.erase(s.begin() + i + 1, s.begin() + start + 1);
                    //这里的start记录的是空格的开始。
                    start = i;
                    flag = false;
                }
                continue;
            }
            else
            {
                if(!flag)
                {
                    //i不删start要删
                    s.erase(s.begin() + i + 1, s.begin() + start + 1);
                    //这里start记录的是单词的开始  
                    start = i;
                    flag = true;
                }
            }
        }
        //处理最前边的单词
        if(flag)
        {
            s += s.substr(0, start + 1);
            s.erase(s.begin(), s.begin() + start + 1);
        }
        else
        {
            //开头的删掉空格
            s.erase(s.begin(), s.begin() + start + 1);
            int i = s.size() - 1;
            while(i >=0 && isspace(s[i]))
            {
                i--;
            }
            //删除最后的一个空格，或者全是空格的时候删除全部
            s.erase(s.begin() + i + 1, s.end());
        }
        return s;
    }
};
~~~

## III. 栈和队列

### 20E. 有效的括号对

**问题描述**：判断一个给定字符串是否是有效的括号对。有效的括号对只任何两个配对的括号对中间都必须是完整的配对括号对。如：` "{[]}"`、`"[[{}{()}]]"`等。

**我的思路**：一个自然的思路自然是stack，如果当前的是左括号就压入stack，如果是右括号就把stack顶部元素弹出，符合要求的一定会与当前括号匹配。否则返回无效。

**坑**：注意判断stack的时候需要注意没有右括号的case，如`"(("`，在判断的时候由于没有弹出stack判断的操作自然会没有因此判断为无效，因此在末尾要判断stack内是否还有元素。

**优化**：如果本身括号的个数是奇数，那么一定不会是有效的。

**代码**：

~~~C++
#define XIAO    0
#define ZHONG   1
#define DA      2

class Solution {
public:
    bool isValid(string s) {
        if(s.size() % 2) return false;
        stack<int> myStk;
        
        for(int i = 0; i < s.size(); i++)
        {
            if(s[i] == '(') myStk.push(XIAO);
            else if(s[i] == '[') myStk.push(ZHONG);
            else if(s[i] == '{') myStk.push(DA);
            else if(s[i] == ')')
            {
              	//判断是否越界一定要在数组值引用之前
                if(myStk.empty() || myStk.top() != XIAO) return false;
                myStk.pop();
            }
            else if(s[i] == ']')
            {
                if(myStk.empty() || myStk.top() != ZHONG) return false;
                myStk.pop();
            }
            else if(s[i] == '}')
            {
                if(myStk.empty() || myStk.top() != DA) return false;
                myStk.pop();
            }
        }
        if(myStk.empty()) return true;
        else return false;
    }
};
~~~

### 84H. 直方图中最大的长方形

**问题描述**：给出一个直方图（宽为1，直方图由数组的形式给出），找出这个直方图中最大的长方形。如下图最大长方形面积即为10。

![avatar](https://assets.leetcode.com/uploads/2018/10/12/histogram_area.png)

**我的思路**：

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

### 150M. 逆波兰表达式

**题目描述**：给出一个字符串序列，分别表示逆波兰表达式的各项，求出这个表达式的运算结果。

**我的思路**：首先弄清楚逆波兰表达式的原则就简单了，使用栈来保存数，遇到运算符号取栈内前两个数运算，运算结果再压入栈。

**坑**：减法和除法中减数和除数都是位于栈顶的；注意判断是否是字符时负数的符号也位于字符串首，不要误判为减号。

**代码**：

~~~C++
class Solution {
public:
    void calc(stack<int>& nums, char sign)
    {
        int num1 = nums.top();
        nums.pop();
        int num2 = nums.top();
        nums.pop();
        int res;
        if(sign == '+')
            res = num1 + num2;
        else if(sign == '-')
            res = num2 - num1;
        else if(sign == '*')
            res = num1 * num2;
        else
            res = num2 / num1;
        nums.push(res);
        
        return;
        
    }
    int evalRPN(vector<string>& tokens) {
        stack<int> nums;
        for(int i = 0; i< tokens.size(); i++)
        {
            if(isdigit(tokens[i][0]) || (tokens[i][0] == '-' && tokens[i].size() > 1))
            {
                stringstream ss(tokens[i]);
                int num;
                ss >> num;
                nums.push(num);
            }
            else
            {
                char sign = tokens[i][0];
                calc(nums, sign);
            }
        }
        return nums.top();
    }
};
~~~



### 1019M. 链表中下一个更大的节点

**题目描述**：给定一个链表，返回一个数组，存入每个节点后第一个比他大的节点的值，如果不存在那么存入0。

**思路**：依靠单调栈的特性，我们按照上述法则依次存入节点至栈，遇到比栈顶大的，那么就在这个节点对应的数组位置处存下这个数。需要注意的是，链表并不能直接靠索引找到相应的位置，因此在未确定的时候，在数组中相应节点的位置存入的是原节点的值，这样就可以在数组中采用索引的方式加速算法。

**注**：参考单调栈 [Leetcode刷题总结 五](https://starkschroedinger.github.io/2020/03/02/Monotonous-Stack/)。

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

## IV. 树

### 101E. 对称树

**问题描述**：检查一棵树是不是对称树。

**我的思路**：分为用队列迭代和递归的方法，前者用队列实现，后者递归的时候左树和右树传进去的子节点应刚好左右相反。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
#define Recursion

//The first method is recursion.
#ifdef Recursion
class Solution 
{
public:
    bool isSymmetric(TreeNode* root) 
    {
        if(!root) return true;
        return isMirror(root->left, root->right);
    }
    bool isMirror(TreeNode* left, TreeNode* right)
    {
        if(!left && !right) return true;
        else if(!left || !right) return false;
        
        if(left->val != right->val) return false;
        bool b1 = isMirror(left->left, right->right);
        bool b2 = isMirror(left->right, right->left);
        if(!b1 || !b2) return false;
        return true;
    }
};
#endif

#ifdef Iteration
class Solution
{
public:
    bool isSymmetric(TreeNode* root)
    {
        if(!root) return true;
        if(!root->left && !root->right) return true;
        else if(!root->left || !root->right) return false;
        
        std::queue<TreeNode*> treeQ;
        treeQ.push(root->left); treeQ.push(root->right);
        
        while(!treeQ.empty())
        {
            TreeNode *l, *r;
            l = treeQ.front(); treeQ.pop();
            r = treeQ.front(); treeQ.pop();
            if(!l && !r) continue;
            else if(!l || !r) return false;
            
            if(l->val != r->val) return false;
            
            treeQ.push(l->left);
            treeQ.push(r->right);
            treeQ.push(l->right);
            treeQ.push(r->left);
        }
        return true;
    }
};
#endif
~~~

### 102M. 二叉树的层序遍历

**问题描述**：完成对二叉树的层序遍历。

**我的思路**：一般层序遍历都是用队列实现，每次压入一层，弹出时将弹出节点的非空左右孩子再压入队列，直到队列为空，这里需要计数来区分不同层。这里也可以用迭代遍历，迭代函数中有一引用形参表示当前的层数，在对应层数的vector中压入val。

**代码**：

* 队列：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        
        queue<TreeNode*> treeQ;
        treeQ.push(root);
        
        while(!treeQ.empty())
        {
            vector<int> cur;
            int sizeQ = treeQ.size();
            for(int i = 0; i < sizeQ; i++)
            {
                TreeNode* nD = treeQ.front();
                treeQ.pop();
                cur.push_back(nD->val);
                if(nD->left) treeQ.push(nD->left);
                if(nD->right) treeQ.push(nD->right);
            }
            res.push_back(cur);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
~~~

* 递归：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(!root) return {};
        vector<vector<int>> res = {};
        int depth = 0;
        recurHelper(root, depth, res);
        return res;
    }
    void recurHelper(TreeNode* root, int depth, vector<vector<int>>& res)
    {
        if(!root) return;
        if(res.size() < depth + 1)
        {
            vector<int> temp = {root->val};
            res.push_back(temp);
        }
        else
        {
            res[depth].push_back(root->val);
        }
        recurHelper(root->left, depth + 1, res);
        recurHelper(root->right, depth + 1, res);
        return;
    }
};
~~~

### 103M. 二叉树之字形层序遍历

**问题描述**：之字形层序遍历二叉树，即第一层从左往右遍历，第二层从右往左遍历。

**我的思路**：这种循环往复的自然想到用栈来实现，但是和队列有所不同的是，这里是用两个栈实现，并且从左到右和从右到左的时先压入左孩子还是先压入右孩子也不同，需要具体事例具体分析。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if(!root) return {};
        stack<TreeNode*> left, right;
        left.push(root);
        //order向左为true向右为false
        bool order = true;
        vector<vector<int>> res = {};
        while(1)
        {
            vector<int> temp = {};
            stack<TreeNode*> *out = nullptr, *in = nullptr;
            if(order)
            {
                out = &left;
                in = &right;
            }
            else
            {
                out = &right;
                in = &left;
            }
            while(!out->empty())
            {
                if(out->top() == nullptr) 
                {
                    out->pop();
                    continue;
                }
                temp.push_back(out->top()->val);
                if(order)//这次向左就先加左
                {
                    in->push(out->top()->left);
                    in->push(out->top()->right);
                }
                else//这次向右就先加右
                {
                    in->push(out->top()->right);
                    in->push(out->top()->left);
                }
                out->pop();
            }
            //遍历次序翻转
            order = order ? false : true;
            if(!temp.empty()) res.push_back(temp);
            else break;
        }
        return res;
    }
};
~~~

### 104E. 二叉树的最大深度

**问题描述**：求出二叉树的最大深度

**我的思路**：递归没话说

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
~~~

### 105M. 由先序遍历和中序遍历恢复二叉树

### 107M. 二叉树的层序遍历 II

**问题描述**：从最底层到最顶层层序遍历二叉树。

**我的思路**：从最顶层开始遍历然后再reverse。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        
        queue<TreeNode*> treeQ;
        treeQ.push(root);
        
        while(!treeQ.empty())
        {
            vector<int> cur;
            int sizeQ = treeQ.size();
            for(int i = 0; i < sizeQ; i++)
            {
                TreeNode* nD = treeQ.front();
                treeQ.pop();
                cur.push_back(nD->val);
                if(nD->left) treeQ.push(nD->left);
                if(nD->right) treeQ.push(nD->right);
            }
            res.push_back(cur);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
~~~

### 108E. 将有序链表转化为二叉搜索树

**问题描述**：将有序链表转化为二叉搜索树，要求树是平衡的。

**我的思路**：因为平衡二叉树的左右子树深度相差最多为1，所以从中间开始加入树，然后将原列表分成两半，分别对应左右子树进行迭代。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if(nums.size() == 0) return NULL;
        TreeNode* root;
        return helper(nums, 0, nums.size() - 1, root);
    }
    TreeNode* helper(vector<int>& nums, int left, int right, TreeNode* current)
    {
        
        if(left > right) return NULL;
        if(left == right) return new TreeNode(nums[left]);
        
        int middle = (left + right) / 2;
        current = new TreeNode(nums[middle]);
        current->left = helper(nums, left, middle - 1, current->left);
        current->right = helper(nums, middle + 1, right, current->right);
        return current;
    }
};
~~~

### 110E. 平衡二叉树

**问题描述**：检查一棵树是不是平衡二叉树。

**我的思路**：形参返回左该树的高度，然后一层一层比较，遇到不一样的返回false。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        return search(root)>=0;
    }
    
    int search(TreeNode* root){
        if (root==NULL) return 0;
        
        int left = search(root->left);
        int right = search(root->right);
        
        if (right<0 || left<0 || abs(right-left)>1) return -1;
        else return max(left,right)+1;
    }
};
~~~

### 111E. 二叉树的最小深度

问题描述：返回二叉树的最小深度。

我的思路：递归，每次比较左子树和右子树的最小深度，遇到叶子返回。

代码：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    int minDepth(TreeNode* root) {
        if(!root)
            return 0;
        if(!root->left)
            return 1 + minDepth(root->right);
        if(!root->right)
            return 1 + minDepth(root->left);
        
        return 1 + min(minDepth(root->left), minDepth(root->right));
    }
};
~~~

### 112E. 路径之和

**问题描述**：给出一个二叉树和一个target，问是否有一个从根到叶子的路径，使得路径上叶子的node之和等于target。

**我的思路**：递归，递归参数中有一个当前之和是多少，遇到叶子检查是否等于target，如果是的就返回true，否则返回到上个节点重新迭代。

坑：注意因为树中节点可能含有负数，所以并不能用当前和与

**代码**：

~~~C++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (root == NULL) {
            return false;
        }
        return helper2(root, sum);
    }
    
    bool helper2(TreeNode* root, int sum) {
        // std::cout << root->val << std::endl;

        if (root->right == NULL && root->left == NULL) {
            return (sum - root->val) == 0;
        }
        if (root->right == NULL) {
            return helper2(root->left, sum - root->val);
        }
        if (root->left == NULL) {
            return helper2(root->right, sum - root->val);
        }
        return helper2(root->right, sum - root->val) || helper2(root->left, sum - root->val);
    }
~~~

### 113M. 路径之和 II

问题描述：同112E，只不过这道题目要求将所有满足条件的路径找出并输出。



### 114M. 展开树为链表

**问题描述**：给出一棵二叉树，将它展成如下左子树均为空的“链表”。

~~~
    1
   / \
  2   5
 / \   \
3   4   6
~~~

将上树展成如下形式：

~~~
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
~~~

**我的思路**：如何选择合适的递归函数是关键，假设如果对于一棵树的左子树和右子树都成了链表形式，那么只需要找到左子树最右下的节点，将这个节点的下一个指向root的右节点，再将root的右节点指向root的左节点，root的左节点置空，就成功地将左子树插入到了右子树当中且不破坏形式，需要注意的是对左右节点为空的情况的判断。

**代码**：

~~~C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        recurHelper(root);
        return;
    }
    
private:
    void recurHelper(TreeNode* cur)
    {            
        if(!cur) return;
        recurHelper(cur->right);
        recurHelper(cur->left);
        TreeNode* last = cur->left;
      	//如果左子树为空
        if(!last) return;
      	//寻找左子树的最右下节点
        while(last->right)
            last = last->right;
      	//将左子树插入到root和右子树中间
        last->right = cur->right;
        cur->right = cur->left;
        cur->left = NULL;
        return;
    }
};
~~~

## V. 排序

### 75M. 颜色排序

**问题描述**：一个数组只含有0，1和2，分别代表三种颜色，给这个数组排序。要求仅一次遍历完成排序。

**我的思路**：从头开始遍历，使用三指针，一个指针用于遍历，两个指针分别位于首尾，中间指针每指向一个，就将其归位。每归位一个，相应的前或后指针就像相应的前方移动一位。

**代码**：

~~~C++
class Solution {
public:
    void swap(vector<int>& nums, int first, int second)
    {
        int temp = nums[first];
        nums[first] = nums[second];
        nums[second] = temp;
    }
    void sortColors(vector<int>& nums) {
        if(nums.empty()) return;
        int zero = 0, one = 0, two = nums.size() - 1;
        //因为这里的two和zero每次交换后都会向后顺延，所以two和zero指向的都是待处理的数据，而非处理过后的。所以one == two的情况是一定要考虑的，否则可能会少处理最后一个数据。
        while(one <= two)
        {
            if(nums[one] == 0)
            {
                //交换结果：one处可能为1和0,zero处一定是0
                swap(nums, one, zero);
                one++;
                zero++;
            }
            else if(nums[one] == 1)
            {
                one++;
            }
            else
            {
                //交换结果，one处可能为0，1和2，所以不再one++，但two处一定是2
                swap(nums, one, two);
                two--;
            }
        }
        
        return;
    }
};
~~~

### 1366M. 计数投票给team排序 周赛178

**问题描述**：n a special ranking system, each voter gives a rank from highest to lowest to all teams participated in the competition. The ordering of teams is decided by who received the most position-one votes. If two or more teams tie in the first position, we consider the second position to resolve the conflict, if they tie again, we continue this process until the ties are resolved. If two or more teams are still tied after considering all positions, we rank them alphabetically based on their team letter. Given an array of strings `votes` which is the votes of all voters in the ranking systems. Sort all teams according to the ranking system described above.

Return *a string of all teams* **sorted** by the ranking system.

Example:

~~~
Input: votes = ["ABC","ACB","ABC","ACB","ACB"]
Output: "ACB"
Explanation: Team A was ranked first place by 5 voters. No other team was voted as first place so team A is the first team.
Team B was ranked second by 2 voters and was ranked third by 3 voters.
Team C was ranked second by 3 voters and was ranked third by 2 voters.
As most of the voters ranked C second, team C is the second team and team B is the third.
~~~

**我的思路**：给每一支队伍创建一个特殊的记分数据结构，如果有n支队伍参与排名，这个积分结构则是大小为n+1的`vector<int>`,其中前n项的第i项为这个队伍被当作第i名投了多少票，最后一项是字母顺序，a是0，z是25。给这个vector重新拍一下序，利用`<algorithm>`头文件中的sort函数的第三项自定义排序函数，两个队伍在比较时从第一项开始比，如果前n项都一样则比较最后一项字母大小，自动完成排序。

**代码**：

~~~C++
class Solution {
public:
    string rankTeams(vector<string>& votes) {
        if(votes.size() == 1) return votes[0];
        int num = votes[0].size();
        unordered_map<char, int> hash;
        //在统计的最后加上他们的ASCII码
        for(int i = 0; i < num; i++)
        {
            hash[votes[0][i]] = i;
        }
        vector<int> single(num, 0);
        vector<vector<int>> all(num, single);
        for(int i = 0; i < votes.size(); i++)
        {
            for(int j = 0; j < num; j++)
            {
                char cur = votes[i][j];
                all[hash[cur]][j]++;
            }
        }
        for(int i = 0; i < num; i++)
        {
            all[i].push_back((int)(votes[0][i] - 'A'));
        }
        //排序
        sort(all.begin(), all.end(), voteTurn);
        string res = "";
        for(int i = 0; i < num; i++)
        {
            char cur = all[i].back() + 'A';
            string s(1, cur);
            res += s;
        }
        return res;
    }
    
private:
    static bool voteTurn(const vector<int>& a, const vector<int>& b)
    {
        for(int i = 0; i < a.size() - 1; i++)
        {
            if(a[i] > b[i]) return true;
            else if(a[i] < b[i]) return false;
            else continue;
        }
        if(a.back() < b.back())
            return true;
        else return false;
    }
};
~~~

## VI. 查找

### 2M. 两数相加。

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

### 15M. 三数之和

**问题描述**：给定n个整数，返回所有满足条件的三元组(a, b, c)，使得a + b + c = 0，并且这三元组不能重复。

**我的思路**：

回忆2Sum问题，用了O(n)的时间和空间，使用哈希表，将每一个已经遍历了的数对应的“余数”存入哈希表，看接下来的数是否满足条件，思想是用空间换时间。另一种方法是首先将给定数组排序（最快的快排也需要O(nlogn)的时间，因此对于2Sum并不是很划算），然后用双指针分别指向首尾，求和，根据求和的结果和target进行比较来相应地移动首或尾指针，直到求和结果等于target。

将3Sum的第一个数固定，然后问题就变成了2Sum的问题，此时target就是-a，我们可以在2Sum算法的基础上额外增加一次遍历，使得算法的复杂度处于O(n^2)的水平。剩下还需要解决一些细枝末节的问题。

**坑**:

* 为了排除因为第一个数字重复而导致的三元组重复，需要在第一个数字遍历时，遇到重复的直接跳过本次循环。

  `if(i > 0 && nums[i] == nums[i - 1]) continue;`

* 2Sum使用哈希表的思路在3Sum问题中当输入数组特别大的时候会导致Time Limit Exceeded。Test case: <https://leetcode.com/submissions/detail/300659782/testcase/>。

**加速**:

* 排序过后，当第一个数是正数时，后边两个数也都是正数，那么绝不可能相加为0，大循环可终止。

  `for(int i = 0; i < nums.size() - 2 && nums[i] <= 0; i++)`

**代码**

~~~C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        if(nums.size() < 3) return {};
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());    //为了排除重复三元组
        for(int i = 0; i < nums.size() - 2 && nums[i] <= 0; i++)
        {
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = nums.size() - 1;
            while(j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];
                if(sum < 0) j++;
                else if(sum > 0) k--;
              	//防止第二个和第三个数重复，下边的原则是，第二个数碰到一连串一样的时候，跳到最右边的那个压入res，而第三个数碰到一连串一样的时候，跳到最左边的那个。但是防止j和k是紧邻的，也就是说nums[j]和nums[k]一样，这时还是要压入res的。
                else if(nums[j] == nums[j + 1] && j != k - 1) j++;
                else if(nums[k] == nums[k - 1] && j != k - 1) k--;
                else
                {
                    res.push_back({nums[i], nums[j], nums[k]});
                    j++;
                    k--;
                }
            }
        }
        return res;
    }  
};

//使用hash表，以下算法会在某些test case运算超时。
/*
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        if(nums.size() < 3) return {};
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());    //为了排除重复三元组
        for(int i = 0; i < nums.size() - 2 && nums[i] < 0; i++)
        {
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            unordered_map<int, int> hash;
            for(int j = i + 1; j < nums.size(); j++)
            {
                if(nums[j] > 0)
                {
                    auto it = hash.find(nums[j]);
                    if(it != hash.end())
                    {
                        if(!res.empty() && nums[i] == res.back()[0] && nums[j] == res.back()[2])
                        continue;
                        else res.push_back({nums[i], nums[it->second], nums[j]});
                    }
                }
                hash.insert((pair<int, int>(0 - nums[i] - nums[j], j)));
            }
        }
        return res;
    }  
};

*/
~~~

### 16M. 最接近的三数之和

**问题描述**：给出一个数列和一个目标target，找出其中的三数，这三数之和最接近target，返回这三数之和。

**我的思路**：这道题看似和15M的三数之和很相似，但有本质的区别，实际上要更简单一些，因为遇到相等的就直接返回了。大循环第一个数，从开始到允许访问的结束，之后两个从两端到中间收缩，根据当前和与target的大小关系移动第二个数和第三个数，每次计算最新的和与target之差和最小的差比较，并更新，遇到和target一样的直接返回target，如果没有最后再返回最新的和。

**代码**：

~~~C++
//代码是leetcode给出的0ms的答案，思路和我的是一样的。
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        if (nums.size() < 3) return {};
        sort(nums.begin(), nums.end());      
        int n=nums.size();
        int j, k, sum;
        int t=nums[0]+nums[1]+nums[2];
        int u=nums[n-1]+nums[n-2]+nums[n-3];
        int clsum=(abs(target-t)>abs(target-u))?t:u;
        for (int i = 0; i < nums.size()-1; i++){
            j = i+1;
            k = nums.size()-1;
            sum = -1;
            while (j < k){
                sum = nums[i] + nums[j] + nums[k];
                if(abs(sum-target)<abs(clsum-target))clsum=sum;
                if(sum-target>0)k--;
                else if(sum-target<0)j++;
                else
                    return clsum;
            }
       }
        return clsum; 
    }
};
~~~



### 18M. 四数之和

**问题描述**：给出一个数列和一个目标target，找出其中所有的四元数组，其四数之和等于target。

**我的思路**：这道题可以在三数之和的基础上，增加一轮循环，重点在于如何加速循环。首先对数组进行排序，在最外层循环，循环不用持续到最后，当第一个数大于target的四分之一时，就停止循环，因为剩下的四数之和只能大于target；同理对第二个数，当第一个数和第二个数之和大于target的一半时，停止循环；第三个数和第四个数从两边向中间靠拢。对于第三个数，当第三个数大于target减去第一个数和第二个数剩下的一半时，停止循环，第四个数从最大开始，小于target的四分之一时，停止循环。

**坑**：为了防止重复，计策和三数之和相同。对于第一个数和第二个数都采取如果前一个数和当前的数相同，那么跳过本次大循环。对于第三个和第四个数而言，如果找到了符合的四元数组，当第三个数发现下一个数和当前数一样时，直接跳到相同数的最后一个，第四个数与之相应的，遇到重复的直接跳到相同数的最前端，当然不要忘记第三个数和第四个数的情况，不符合上述的跳跃规则。

**代码**：

~~~C++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res = {};
        if(nums.size() < 4) return res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i <= nums.size() - 4 && nums[i] <= target / 4; i++)
        {
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            int stop_j = target / 2 - nums[i];
            for(int j = i + 1; j <= nums.size() - 3 && nums[j] <= stop_j; j++)
            {
                if(j > i + 1 && nums[j] == nums[j - 1]) continue;
                int m = j + 1, n = nums.size() - 1;
                int stop_m = (target - nums[i] - nums[j]) / 2;
                int stop_n = target / 4;
                while(m < n && nums[m] <= stop_m && nums[n] >= stop_n)
                {
                    int sum = nums[m] + nums[n] + nums[i] + nums[j];
                    if(sum < target) m++;
                    else if(sum > target) n--;
                    else if(nums[m] == nums[m + 1] && m != n - 1) m++;
                    else if(nums[n] == nums[n - 1] && m != n - 1) n--;
                    else
                    {
                        vector<int> resTemp = {nums[i], nums[j], nums[m], nums[n]};
                        res.push_back(resTemp);
                        m++;
                        n--;
                    }
                }
            }
        }
        return res;
    }
};
~~~



### 49M. 同构异形字符串

**问题描述**：给出一组字符串，找出其中的同构异形字符串并输出，同构异形字符串指那些字符串字母组成和数量都相等的单排列不同的字符串。

**我的思路**：很平凡的思路是：首先对每个字符串按字母顺序先排序，然后将排序后的存入一个hash表，最后同构异形的都被存入一个hash表中。

**优化**：由于unordered_map内置的对string类型的key 计算hash值速度较慢，我们可以采用质数相乘的方法手动取hash值。方法是：从2开始，给26个字母每个字母分配一个质数，然后一个字符串的的hash值就是他们字母对应的质数的乘积。由于质数没有其他质因子，且乘法有交换律，因此同构异形字符串对应的hash值都相同。

**代码**：

1. 采用unordered_map的内置hash

   ~~~C++
   class Solution {
   public:
       vector<vector<string>> groupAnagrams(vector<string>& strs) {
           if(strs.empty()) return {};
           vector<vector<string>> res;
           unordered_map<string, vector<string>> hash;
           for(int i = 0; i < strs.size(); i++)
           {
               string temp = strs[i];
               sort(temp.begin(), temp.end());
               hash[temp].push_back(strs[i]);
           }
           for(auto it = hash.begin(); it != hash.end(); it++)
           {
               res.push_back(it->second);
           }
           return res;
       }
       
   };
   ~~~

2. 采用质数手动计算hash值

   ~~~C++
   class Solution {
   private:
       int primeNum[26];
   public:
       void initializePrime()
       {
           bool flag[150];
           for(int i = 0; i < sizeof(flag); i++)
           {
               flag[i] = true;
           }
           for(int i = 2; i < sizeof(flag); i++)
           {
               if(flag[i])
               {
                   int k = 2;
                   while(i * k < sizeof(flag))
                   {
                       flag[i * k] = false;
                       k++;
                   }
               }
           }
           int i = 0;
           int j = 2;
           while(i < 26)
           {
               while(!flag[j]) j++;
               primeNum[i++] = j++;
           }
           return;
       }
       unsigned long calcHash(const string& str)
       {
           unsigned long res = 1;
           for(int i = 0; i < str.size(); i++)
           {
               res *= primeNum[str[i] - 'a'];
           }
           return res;
       }
       vector<vector<string>> groupAnagrams(vector<string>& strs) {
           if(strs.empty()) return {};
           vector<vector<string>> res;
           unordered_map<unsigned long, vector<string>> hash;
           initializePrime();
           for(int i = 0; i < strs.size(); i++)
           {
               unsigned long cur = calcHash(strs[i]);
               auto it = hash.find(cur);
               if(it != hash.end())
               {
                   it->second.push_back(strs[i]);
               }
               else
               {
                   hash[cur] = {strs[i]};
               }
           }
           for(auto it = hash.begin(); it != hash.end(); it++)
           {
               res.push_back(it->second);
           }
           return res;
       }
       
   };
   ~~~

## VII. 策略性枚举

### 17M. 电话号码的字母组合

**问题描述**：给定一个字母串，每一个字母代表九宫格输入法上若干个字母，问可能组成的字母组合有哪些，并输出所有的字母组合。

**我的思路**：这个问题逻辑上非常简单，就是最基本的排列组合问题。但是在实施的过程中有一点难度。我的思路是采用迭代，每考虑后一个数字都在前一个数字对应字母组合的基础上再增加相应的几倍。如“23”，其中2对应的是“a, b, c”，在这之后增加3对应的“d，e，f”后变为"a, b, c, bd, be, bf, cd, ce, cf"，再把初始的字母组合后边添加变为"ad, ae, af, bd, be, bf, cd, ce, cf"。

**优化**：待定。

**代码**：

~~~C++
class Solution {
public:
	vector<vector<string>> dict = {
	{},{},
	{"a", "b", "c"}, 
	{"d", "e", "f"}, 
	{"g", "h", "i"}, 
	{"j", "k", "l"}, 
	{"m", "n", "o"}, 
	{"p", "q", "r", "s"}, 
	{"t", "u", "v"}, 
	{"w", "x", "y", "z"}
	};

    vector<string> letterCombinations(string digits) 
	{
        if(digits.empty()) return {};
        vector<string> res = {""};
        int digit = digits.size();
        for(int i = 0; i < digit; i++)
        {
            int cur = digits[i] - '0';
            if(cur < 2) return {};
            forward(res, cur);
        }
        if(res.size() < 3) return {};
        return res;
    }

    void forward(vector<string>& res, int n)
    {
        int size = res.size();
        for(int i = 1; i < dict[n].size(); i++)
        {
            for(int j = 0; j < size; j++)
            {
                res.push_back(res[j] + dict[n][i]);
            }
        }
        for(int j = 0; j < size; j++)
        {
            res[j] += dict[n][0];
        }
        return;
    }
    
};
~~~

### 46M. 排列组合

**问题描述**：给出一个字母彼此互不相同的字符串，输出这些字母的所有可能的排列方式。

**我的思路**：考虑递推，假设前边n个字母的所有排列方式都已经举出，那么对于每一种**排列方式**，第n+1个字母都有n+1个位置放置（每两个字母之间以及第一个字母之前和最后一个字母之后），依此递推。

**坑**：

**代码**；

~~~C++
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res = {};
        if(nums.size() == 0) return res;
        res = {{nums[0]}};
        for(int i = 1; i < nums.size(); i++)
        {
            int size = res.size();
            for(int j = 0; j < size; j++)
            {
                for(int k = 0; k < i; k++)
                {
                    vector<int> temp(res[j]);
                    temp.insert(temp.begin() + k, nums[i]);
                    res.push_back(temp);
                }
                res[j].push_back(nums[i]);
            }
        }
        return res;
    }
};
~~~

### 48M. 旋转图像

**问题描述**：给出一个方阵，方针每个数字代表图片的每个像素，将这个图片顺时针旋转90度。

**我的思路**：在旋转的过程中，一定是四个像素对应，互相轮换。如果一个像素的行列坐标为(col, row)，那么与之对应的顺时针旋转的位置应该是(col, row)—(col, size - row)—(size - row, size - col)—(size - col, row)。将其封装为函数`rotateInGroup4`。现在只需找到哪些坐标时第一次旋转的坐标且不会重复。从第0行开始一直到第n/2行中，第i行从第i个像素一直到第size-i个像素是不重复的第一旋转像素。

**代码**：

~~~C++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        if(matrix.size() < 2) return;
        size = matrix.size() - 1;
        
        for(int layer = 0; layer <= size / 2; layer++)
        {
            for(int start = layer; start < size - layer; start++)
            {
                rotateInGroup4(matrix, start, layer);
            }
        }
        
        return;
    }
    void rotateInGroup4(vector<vector<int>>& matrix, int row, int col)
    {
        int temp1 = matrix[row][size - col];
        matrix[row][size - col] = matrix[col][row];
        int temp2 = matrix[size - col][size - row];
        matrix[size - col][size - row] = temp1;
        temp1 = matrix[size - row][col];
        matrix[size - row][col] = temp2;
        matrix[col][row] = temp1;
    }
private:
    int size;
};
~~~

### 78M. 子集

问题描述：给出一个包含不同的元素的序列，求出他的所有子集并输出。

我的思路：采用递推的形式，由于包含n个元素的集合就是在包含n-1个元素的集合的所有子集上添上这个元素并和原来的子集合并，因此可以一步一步推出所有子集。

代码：

~~~C++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        if(nums.empty()) return {};
        vector<vector<int>> res = { {}, {nums[0]} };
        for(int i = 1; i < nums.size(); i++)
        {
            int size = res.size();
            for(int j = 0; j < size; j++)
            {
                vector<int> temp = res[j];
                temp.push_back(nums[i]);
                res.push_back(temp);
            }
        }
        return res;
    }
};
~~~

### 264M. 丑数II

**问题描述**：丑数是质因子只有2、5和8的数，规定1是第一个丑数。求第n个丑数。

**我的思路**：采用三指针法。因为每一个丑数都能看作从前边某个丑数与2、5或8相乘得到。因此，我们从1开始创建三个附属于丑数序列的虚序列。每个指针都指向当前2、3或5作为比较的对象，当当前对象乘以相应因子后仍小于最大的丑数时，则不断循环前进。这样不会漏掉任何一个丑数。

**代码**：

~~~C++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> uglyNum = {1};
        int it2 = 0,it3 = 0,it5 = 0; 
        while(uglyNum.size() < n)
        {            
            while((uglyNum[it2] * 2) <= uglyNum.back())
                it2++;            
            while((uglyNum[it3] * 3) <= uglyNum.back())
                it3++;
            while((uglyNum[it5] * 5) <= uglyNum.back())
                it5++;
            
            int min2 = uglyNum[it2] * 2, min3 = uglyNum[it3] * 3, min5 = uglyNum[it5] * 5;
            if(min2 < min3)
            {
                if(min2 < min5)
                {
                    uglyNum.push_back(min2);
                    it2++;
                }
                else
                {
                    uglyNum.push_back(min5);
                    it5++;
                }
            }
            else
            {
                if(min3 < min5)
                {
                    uglyNum.push_back(min3);
                    it3++;
                }
                else
                {
                    uglyNum.push_back(min5);
                    it5++;
                }
            }                
        }
        return uglyNum.back();
    }
};
~~~

## VIII. 广度优先搜索

## IX. 深度优先搜索(DFS)

### 22M. 产生括号对

**问题描述**：给出应产生的括号对数n，生成所有符合条件的括号对。其中任何一对括号对中不能包含未闭合的单个括号，如`(()`是错误的，而`((()()))`等是正确的。

**思路**：这个问题属于最基本的卡特兰数问题，但是弄清一共有多少个还不够，把每一种情况确定下来需要用到递归。我们知道左括号和右括号的初始数量都是n，每使用一个左括号或右括号都把变量left或right减去1，递归终止条件是左括号和右括号的数量都为0。

**代码**：

~~~C++
class Solution{
public:
    vector<string> generateParenthesis(int n)
    {
        if(n == 0) return {""};
        vector<string> res = {};
        helper(n, n, "", res);
        
        return res;  
    }
    void helper(int left, int right, string s, vector<string>& res)
    {
        if(left == 0 && right == 0)
        {
            res.push_back(s);
            return;
        }
        if(left > 0)
            helper(left - 1, right, s + "(", res);
        if(left < right)
            helper(left, right - 1, s + ")", res);
        return;
    }
};
~~~

### 36M. 有效的数独*

**问题描述**：判定一个完成/未完成的数独是否有效。

**我的思路**：遍历三遍，分别判定数独有效的三个条件。即横竖和九宫格不出现相同的数字。对于每次遍历，都采用一组hash来保存。缺点：效率低，遍历次数多。

**我的代码**：

~~~C++
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for(int i = 0; i < board.size(); i++)
        {
            unordered_set<char> hash;
            for(int j = 0; j < board.size(); j++)
            {
                if(board[i][j] == '.') continue;
                if(hash.find(board[i][j]) != hash.end()) return false;
                else hash.insert(board[i][j]);
            }
        }
        for(int j = 0; j < board.size(); j++)
        {
            unordered_set<char> hash;
            for(int i = 0; i < board.size(); i++)
            {
                if(board[i][j] == '.') continue;
                if(hash.find(board[i][j]) != hash.end()) return false;
                else hash.insert(board[i][j]);
            }
        }
        for(int k = 0; k < 3; k++)
        {
            for(int p = 0; p < 3; p++)
            {
                unordered_set<char> hash;
                for(int i = k * 3; i < (k + 1) * 3; i++)
                {
                    for(int j = p * 3; j < p * 3 + 3; j++)
                    {
                        if(board[i][j] == '.') continue;
                        if(hash.find(board[i][j]) != hash.end()) return false;
                        else hash.insert(board[i][j]);
                    }
                }
            }
        }
        return true;
    }
};
~~~

**优化**：最好能一次遍历完，需要更改数据结构，使用bitmap效率会很高。

### 39M. 组合之和

**问题描述**：给出一组数据candidates，其中无重复数据，另外给出一个目标target，求所有可能组成和为target的组合，不能有重复的解，并且**candidates中每个元素都可以无限制次数地使用**。

**我的思路**：按照深度优先搜索的思路，从后往前遍历，对每个数，判断当前remain是否为0（每次加入解集一个数，remain即减去这个数），接下来可以用两种方法实现这道题，第一种：可以分为将这个数加入当前解集和不加入当前解集，然后氛围两个支路继续递归。需要注意的是，对于加入这个数到当前解集这个递归支路，由于每个数可用的次数不限制，那么这个支路相当于又会产生若干次递归，一直加到remain小于当前这个数为止，因此递归的次数非常之多。第二种方法：是正统的深度优先搜索，对于每一次递归，如果遍历到了当前序号为i的元素，那么都从i开始依次将剩下的元素加入解集，每加入一次都将产生一次递归。

**代码**：

方法一（效率低）：

~~~C++
class Solution {
public:
    void recurHelper(const vector<int>& candidates, vector<vector<int>>& res, vector<int> curVec, int remain, int pointer)
    {
        if(remain == 0)
        {
            res.push_back(curVec);
            return;
        }
        int current = candidates[pointer];
        if(pointer == 0)
        {
            if(remain < current || remain % current != 0)
                return;
            else
            {
                while(remain != 0)
                {
                    remain -= current;
                    curVec.push_back(current);
                }
                res.push_back(curVec);
                return;
            }
        }
        else
        {
            recurHelper(candidates, res, curVec, remain, pointer - 1);
            while(remain >= current)
            {
                remain -= current;
                curVec.push_back(current);
                recurHelper(candidates, res, curVec, remain, pointer - 1);
            }
            return;
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res = {};
        if(candidates.empty()) return res;
        sort(candidates.begin(), candidates.end());
        if(target == 0)
        {
            res = {{}};
            return res;
        }
        else if(candidates[0] > target) return res;
        vector<int> curVec = {};
        recurHelper(candidates, res, curVec, target, candidates.size() - 1);
        return res;
    }
};
~~~

方法二（效率高）：

~~~C++
class Solution {
public:
    void recurHelper(const vector<int>& candidates, vector<vector<int>>& res, vector<int>& curVec, const int remain, const int pointer)
    {
#ifdef DEBUG
        res.push_back({0,0,0,remain,pointer,0,0,0});
#endif
        if(remain < 0) return;
        //找到合法解
        if(remain == 0)
        {
            res.push_back(curVec);
            return;
        }
        for(int i = pointer; i >= 0; i--)
        {
            curVec.push_back(candidates[i]);
            recurHelper(candidates, res, curVec, remain - candidates[i], i);
            curVec.pop_back();
            
        }
        return;
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res = {};
        sort(candidates.begin(), candidates.end());
        if(candidates.empty() || candidates[0] > target) return res;
        vector<int> curVec = {};
        recurHelper(candidates, res, curVec, target, candidates.size() - 1);
        return res;
    }
};
~~~

### 40M. 组合之和II

**问题描述**：给出一组数据candidates，其中可能有重复的数；另外给出一个目标target，求所有可能组成和为target的组合，并且不能有重复的解。

**我的思路**：按照深度优先搜索的思路，从后往前遍历，对每个数，判断当前remain是否为0（每次加入解集一个数，remain即减去这个数），可以分为将这个数加入当前解集合和不加入当前解集合，然后分为两个支路继续递归。需要注意的是如何排除重复的解：出现重复解的原因在于，给出的candidates中本身有重复的元素，那么则需要在上边两个递归前加入条件判断语句——**如果前后两个相同的元素，前者没有被加入到本次解集中，那么在前者所产生的支路，后者也不加入解集。**举例：

```C++
candidates: [5,4,4,3,2,1]
target: 10
```

可能重复的解是[5,4,1]，原因是有两个4。在循环中，如果第一个4没有被加入到当前解集中，那么第二个4也不加入到当前解集，否则会和之前加入第一个4的解集重复。这样的判断不会漏掉任何一个解集，比用哈希表每次加入前判断是否已存在当前解效率要高很多。

**代码**：

```C++
class Solution {
public:
    void recurHelper(vector<vector<int>>& res, vector<int>& candidates, vector<int>& curSol, int remain, int index, vector<bool>& flag)
    {
        if(remain == 0)
        {
            res.push_back(curSol);
            return;
        }
        else if(remain < 0 || index < 0)
            return;
        recurHelper(res, candidates, curSol, remain, index - 1, flag);
        
        //如果前一个数和当前数相等且前一个数没被选中，那么这个数页不选，直接跳过本次迭代，为了避免重复的情况。
        if((index < candidates.size() - 1) && candidates[index] == candidates[index + 1] && !flag[index + 1])  
        {
            return;
        }
        flag[index] = true;
        curSol.push_back(candidates[index]);
        recurHelper(res, candidates, curSol, remain - candidates[index], index - 1, flag);
        curSol.pop_back();
        flag[index] = false;
        
        return;
    }
    
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        if(candidates.empty() || target == 0) return {{}};
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res = {};
        vector<bool> flag(candidates.size(), false);
        vector<int> curSol = {};
        recurHelper(res, candidates, curSol, target, candidates.size() - 1, flag);
        
        return res;
    }
};
```



**优化后的代码**：

~~~C++
//用来加速，详情<https://blog.csdn.net/huanucas/article/details/88981012>
static const auto __ = []() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    return nullptr;
}();

class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        const int rows = board.size();		//使用const好习惯
        const int cols = board[0].size();
        
        vector<bitset<9>> rowValid(rows);	//用来保存记录	
        vector<bitset<9>> colValid(rows);	
        vector<bitset<9>> boxValid(rows);
        
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                if (board[i][j] == '.')
                    continue;
                
                int val = board[i][j] - '0' -  1;
                int box = (i / 3) * 3 + (j / 3);	//这个方法好，如果从上倒下从左到右分别把九宫格编号编为0～8，那么这种方法刚好能使每一个数字都赋予相应的九宫格编号。
                
                // 在一次遍历中统统检查完，效率高。
                if (rowValid[i].test(val) || colValid[j].test(val) || boxValid[box].test(val))
                    return false;
                
                rowValid[i].set(val);
                colValid[j].set(val);
                boxValid[box].set(val); 
            }
        }
        
        return true;
    }
};
~~~

## X. 分治法

## XI. 贪心法

### 3M. 不含重复字母的最长子串

**问题描述**：寻找一个字符串中不含重复字母的最长子串

**我的思路**：由于要检测重复，所以选择hash map来实现滑动窗口。

~~~c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        int res = 0;
        int left = 0; // 记录滑动窗口的开始
        set<int> slidingWin;
        
        for(auto it = s.begin(); it != s.end(); it++)
        {
            //检测是否有重复字符，如果有，将窗口的开始向右移动直到重复字符被排除在外。
            while(slidingWin.find(*it) != slidingWin.end())	
            {
                slidingWin.erase(s[left]);
                left++;
            }
            res = max(res, (int)(it - s.begin()) - left + 1);
            slidingWin.insert(*it);
        }
        
        return res;
    }
};
~~~

对于上述解法，复杂度达到了O(2n)，即只有最后两个字母重复的情况，我们可以继续改进

> The reason is that if *s*[*j*] have a duplicate in the range [*i*,*j*) with index '*j*′, we don't need to increase *i* little by little. We can skip all the elements in the range [*i*,*j*′] and let *i* to be *j*′+1 directly. 

因为j只从1遍历到n，所以复杂度降为O(n)。

~~~c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        int n = s.size();
        int[] index = new int[128]; //for all ASCII characters.
        
        for(int left = 0, j = 0; j < n; j++)
        {
            //如果s[j]没有出现过，那么index[s[j]]应该是0，否则应该是比现有left大的一个数，这时更新left。
            left = max(index[s[j]], left);
			res = max(res, j + 1 - left);
            index[s[j]] = j + 1;
        }
        return res;
    }
};
~~~

### 11M. 能盛最多水的容器

**问题描述**·；有若干相距为1的立起来的板子，他们的高度依次被存在给定的数组中。现在需要找到个板子，使得这两个板子之间能盛的水最多。

![avatar](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

**我的思路**：这道题如果暴力求解需要找出两两配对的情况，复杂度在O(n^2)。现在比较巧妙的方法是，首先取首尾两个板子，然后逐渐向中间移动，直到碰头，规则是：左右两边较低一侧的指针往中间移动。这样能保证最大的情况一定能被遍历到，且只用O(n)的时间。（证明略，用反证法比较容易想清楚。）

**代码**：

~~~C++
class Solution {
public:
    int maxArea(vector<int>& height) {
        if(height.size() == 2) return height[0] < height[1] ? height[0] : height[1];
        int areaMax = 0;
        int left = 0, right = height.size() - 1;
        while(left < right)
        {
            areaMax = max(areaMax, (right - left) * min(height[left], height[right]));
            if(height[left] < height[right])    left++;
            else right--;
        }
        
        return areaMax;
    }
};
~~~

### 159M. 最多包含两个不同字母的子串 ###

**问题描述**；选择一个子串，使得这个子串最多包含两个不同的字母。

**我的思路**：滑动窗口的判断标准为判断字串内不同字母的个数。使用变量counter记录重复的字母数，index记录每个字母出现在滑动窗口内的次数。

~~~C++
class Solution{
public:
    int lengthOfLongestSubstringTwoDistinct(string s)
    {
        if(s.empty()) return 0;
        int n = s.size();
        int[] index = new int[128];
        //记录不同字母的个数
        int counter = 0;
        int res = 0;
        
        for(int left = 0, j = 0; j < n; j++)
        {
           	if(index[s[j]] == 0)
            {
                counter++;
            }
            index[s[j]]++;
            while(counter > 2)
            {
                index[s[left]]--;
                if(index[s[left]] == 0)
                    counter--;
                left++;
            }
            res = max(res, j + 1 - left);
        }
        return res;
    }
};
~~~



## XII. 动态规划

### 53E. 最大的子序列

**问题描述**：给出一个序列，寻找其中一个连续的子序列，使得这个子序列的所有元素和最大。

**我的思路**：

* 动态规划：假设我们从头到尾遍历，那么对于当前的这个数有加入和不加入子序列的两种选择，如果前边的子序列的和为正数，那么我们可以选择加入这个子序列，这时我们称前面这个子序列是有贡献的，我们将前边的子序列加入candidates；反之如果前边子序列已经为负了，那么前边的子序列就可以抛弃了，我们从当前这个数再起一个新的子序列，最后在candidate不断比较的过程中选出最优解。
* 分治法：对于每一个序列，我们都可将其分为如下序列：`[l, m]`和` [m + 1, r]`，我们对这两个序列分别计算五个数: lmax, rmax, lbmax, rbmax, max，其中前两个指的是左半子序列和右半子序列的最大子序列和，中间两个是指包含靠中间边界的最大子序列和，最后一个是整个序列的最大子序列和，即` max(lmax, rmax, max(rbmax, lbmax, rbmax + lbmax))`。

**我的代码**：

~~~C++
#define DC

#ifdef DP
//Dynamic programming
//Note that if we find the contiguous subarray (i, j) which contains the largest sum, then for each k(i < k < j), the sum of subarray (i, k) should not be negative, otherwise we can simply remove the (i, k) and get a subarray (k+1, j) with a larger sum.
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.empty()) return 0;
        int partialSum = 0;
        int maxSum = INT_MIN;
        for(int i = 0; i < nums.size(); i++){
            partialSum += nums[i];
            maxSum = (maxSum > partialSum)?maxSum:partialSum;
            if(partialSum < 0) partialSum = 0;
        }
        return maxSum;
    }
};
#endif

#ifdef DC
/*
Devide and Conquer
For each array (l, r), we devide it into the left(l, m) and right(m + 1, r) subarrays and calculate five values:
lmax: the max sum in the left subarray
rmax: the max sum in the right subarray
lbmax: the max sum of the boundary subarray ending at m,
rbmax: the max sum of the boundary subarray starting at m+1,
max: the max sum of the subarrays of this array, which is max(lmax, rmax, lbmax+rbmax).
*/
class Solution{
public:
    int maxSubArray(vector<int>& nums){
        if(nums.size() == 0) return 0;
        else if(nums.size() == 1) return nums[0];
        
        int res = maxSubArrayDC(nums, 0, nums.size() - 1);
        return res;
    }
    
    int maxSubArrayDC(vector<int>& nums, int l, int r){
        int lmax = 0, rmax = 0;
        int lbmax, rbmax;
        int bmax, max;
        if(nums.size() == 2){
            max = nums[0] > nums[1] ? nums[0] : nums[1];
            max = max > nums[0] + nums[1] ? max : nums[0] + nums[1];
            return max;
        }
        int m = (l + r) >> 1;
        if(l < m - 1){
            lmax = maxSubArrayDC(nums, l, m);
            int temp = 0;
            lbmax = nums[m];
            for(int i = m; i > -1; i--){
            temp += nums[i];
            lbmax = lbmax>temp?lbmax:temp;
            }
        }   
        else {
            if(l == m){
                lmax = nums[m];
                lbmax = nums[m];
            } else {
                lmax = nums[l] > nums[m] ? nums[l] : nums[m];
                lmax = lmax > nums[l] + nums[m] ? lmax : nums[l] + nums[m];
                if(nums[l] > 0) lbmax = nums[l] + nums[m];
                else lbmax = nums[m];
            }
        }
        if(m + 1 < r - 1){
            rmax = maxSubArrayDC(nums, m + 1, r);
            int temp = 0;
            rbmax = nums[m + 1];
            for(int i = m + 1; i < nums.size(); i++){
            temp += nums[i];
            rbmax = rbmax>temp?rbmax:temp;
            }
        } 
        else{
            if(r == m + 1){
                rmax = nums[m + 1];
                rbmax = nums[m + 1];
            } else {
                rmax = nums[r] > nums[m + 1] ? nums[r] : nums[m + 1];
                rmax = rmax > nums[r] + nums[m + 1] ? rmax : nums[r] + nums[m + 1];
                if(nums[r] > 0) rbmax = nums[m + 1] + nums[r];
                else rbmax = nums[m + 1];
            }
        }
        
        
        bmax = rbmax + lbmax;
        bmax = bmax > rbmax ? bmax : rbmax;
        bmax = bmax > lbmax ? bmax : lbmax;
        
        max = lmax>rmax?lmax:rmax;
        max = max>bmax?max:bmax;
        return max;
    }
};
#endif
~~~

### 120M. 三角形的最短路径

**问题描述**：给出一个从上到下排列的二维数组组成的三角阵，找出一个从最上层移动到最下层的路径使得途径点的值的和最小。移动的规则是当前点只能向下或向右下移动一格。能否用O(n)的空间复杂度解决问题呢？

~~~
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
~~~

**我的思路**：这是个动态规划问题，我们找出每一个位置对应的值，即从(i, j)这一点移动到最下层的最小路径，那么我们知道当前点只能从正上方的点和左上方的点移动过来，因此` d[i][j] = min(d[i - 1][j], d[i - 1][j - 1]) + triangle[i][j]`。为了使用最少的空间，我们只分配一个大小为n的数组，从头到尾只利用这一个数组的空间，同时利用一些临时变量做一些记录（代码中的temp用来记录被覆盖的左上方数据，t用来记录即将被覆盖的正上方数据）。

**代码**：

~~~C++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        if(triangle.size() == 0) return 0;
        else if(triangle.size() == 1) return triangle[0][0];
        int size = triangle.size();
        vector<int> res(size, INT_MAX);
        res[0] = triangle[1][0] + triangle[0][0];
        res[1] = triangle[1][1] + triangle[0][0];
        for(int i = 2; i < triangle.size(); i++)
        {
          	//用来记录左上方的点的结果，即将被覆盖
            int temp = res[0];
            res[0] = res[0] + triangle[i][0];
            res[i] = res[i - 1] + triangle[i][i];
            for(int j = 1; j < i; j++)
            {
              	//用来计算正上方点的结果，即将被覆盖
                int t = res[j];
                res[j] = min(temp, t) + triangle[i][j];
                temp = t;
            }
        }
        int min = *min_element(res.begin(), res.end());
        return min;
    }
};
~~~

## XIII. 图

## XIV. 细节实现题

### 6M. ZigZag转换

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

### 29M. 两数相除 *

**问题描述**：不用乘除法和取模运算，完成两int相除。

**我的思路**：很简单也很低效的思路：被除数一直减除数，看能减多少次。下边有几个优化的思路：

**坑**：问题永远是如何处理正负数以及负数最小是-2847483648，其绝对值比正数最大的2847483647还要大，所以如果算法把负数变成正数处理最后再处理符号问题的话，应该注意int向其他类型转换的问题。

**优化**：

**代码**：

~~~C++
class Solution {
public:
    int divide(long long int dividend, long long int divisor) {
        
        int sign = ((dividend<0)^(divisor<0))?-1:1;
        
        dividend = abs(dividend);
        divisor = abs(divisor);
        long long int quotient, temp,i;
        quotient = 0;
        temp = 0;
        for(i = 31; i >= 0; i--){
            if((temp + (divisor << i)) <= dividend){
                temp += divisor << i;
                quotient |= 1LL << i;
                
            }
        }
        
        quotient *= sign;
        return quotient >= INT_MAX ? INT_MAX : quotient <= INT_MIN ? INT_MIN : quotient;
        
    }
};
~~~

### 43M. 字符串相乘

**问题描述**：给出两字符串，分别代表两个数字，返回一个字符串作为其乘积。

**我的思路**：将每个数看成这个位数上的数字和10的相应次幂的组合。那么两个位数相乘，如第四位数（10^3）和第五位（10^4）数相乘，得到的结果就是10^7上的结果。这第一个数字中的每个数字都与另一个数字中的每个数字两两组合，得到的结果加到相应的位置上，最后结果再进位。

**坑**：

* 需要对输入数字的合法性进行检测。
* string里的输入是符合人们正常书写习惯的，即左边的是高位，那么string里的低索引是高位。但是我们保存结果的时候希望10的次幂按照位数保存，所以需要一个转换，如果有k位数（k是string的size或者length），那么第i位数就是10的（k - i - 1）次。两个组合起来就是size1 + size2 - i - j - 2，这个容易弄错。
* 在输出时也按照读数的习惯，从高位向低位输出。有可能前几位高位都是0，这个时候应该作预判，从第一位非零最高位往下读取。
* 理论上来说，n位数乘m位数结果位数不大于m+n位数，不小于m+n-1位数。

**代码**：

~~~C++
class Solution {
public:
    string multiply(string num1, string num2) {
        if(num1.empty() || num2.empty()) return "";
        if(num1[0] == '0' || num2[0] == '0') return "0";
        //低索引是高位，
        int resInt[num1.size() + num2.size()] = {0};
        for(int i = 0; i < num1.size(); i++)
            for(int j = 0; j < num2.size(); j++)
                resInt[num1.size() + num2.size() - i - j - 2] += (num1[i] - '0') * (num2[j] - '0');
        
        for(int i = 1; i < num1.size() + num2.size(); i++)
        {
            resInt[i] += resInt[i - 1] / 10;
            resInt[i - 1] = resInt[i - 1] % 10;
        }
        bool flag = false;
        string res = "";
        for(int i = num1.size() + num2.size() - 1; i >= 0; i--)
        {
            if(!flag)
                if(resInt[i] != 0) flag = true;
            if(flag) res = res += to_string(resInt[i]);
        }
        return res;
    }
};
~~~

### 50M. 实现幂函数

问题描述：实现幂函数pow(double x, int n)的功能，其中x在[-100, 100]范围内，n是int类型的整数。

坑：这道题看似简单有无数个坑，其中第一个，如果首先把n是负数转化为正数来做，当n为-2147483648的时候，他的相反数无法用int来表示，会报错。同时，如果只是用单纯的循环n次来做这道题，当n取为2147483647时，runtime会超时，因此必须采用效率更高的做法。这里是用的是对n的折半算法。

我的思路：当n是偶数时，n折半，这时对应的是**当前的x**平方运算，如果n是奇数，将当前的x乘到res上作为折半后的补偿。最后如果n是负数才将res取倒数。

代码：

~~~C++
class Solution {
public:
    double myPow(double x, int n) {
        double res = 1.0;
        for(int i = n; i != 0; i /= 2)
        {
            if(i % 2 == 1 || i % 2 == -1)
                res *= x;
            x *= x;
        }
        return n < 0 ? 1.0 / res : res;
    }
};
~~~

### 55M. 跳跃游戏

**问题描述**：给出一个数组，每个数代表从当前位置能够跳跃的最大距离，问是否存在一条路径从头跳到尾。

**我的思路**：一开始想递归，后来发现有更简单的方法。极端情况：如果数组里边都是正数，那么一定可以，比如每个只走一步就行。发现：不能到达最后的情况一定是由数组中的0导致的，加入所有的0都能被逾越，那么一定可以完成。检查每个0，从0开始往前回溯，看是否有一个位置m能够从这里跳过这个0，并且之后从m继续往后检查。时间复杂度O(n)。

**代码**：宏定义中RECUR代表使用递归的方法，因为给出的测试用例太大，导致runtime limit exceeded，无法通过。

~~~C++
#define ERASE0

#ifdef ERASE0
class Solution
{
public:
    bool canJump(vector<int>& nums) {
        if(nums.size() < 2) return true;
        for(int i = nums.size() - 2; i >= 0;)
        {
            int k = i - 1;
            if(nums[i] == 0)
            {
                int flag = false;
                while(k >= 0)
                {
                    if(nums[k] > i - k)
                    {
                        flag = true;
                        break;
                    }
                    k--;
                }
                if(!flag) return false;
            }
            i = k;
        }
        return true;
    }
};
#endif

#ifdef RECUR
class Solution {
public:
    void recurHelper(vector<int>& nums, int layer, bool& flag)
    {
        if(layer == 0)
        {
            flag = true;
            return;
        }
        for(int i = layer - 1; i >= 0; i--)
        {
            if(nums[i] >= layer - i)
            {
                recurHelper(nums, i, flag);
                if(flag) return;
            }
        }
        return;
    }
    bool canJump(vector<int>& nums) {
        if(nums.empty()) return false;
        auto it = min_element(nums.begin(), nums.end());
        if(*it > 0) return true;
        int layer = nums.size() - 1;
        bool flag = false;
        recurHelper(nums, layer, flag);
        return flag;
    }
};
#endif
~~~

### 60M. 组合的顺序

**问题描述**：给出一组数{1, 2, ... , n}, 他们的排列方式有n!种，问第k种排序方式是什么？例如：n = 3时，

~~~
1."123"
2."132"
3."213"
4."231"
5."312"
6."321"
~~~

**我的思路**：观察上边的排序原则，发现排序总是从第一位选，将选出来的数放到第一位，然后剩下的依次按原顺序再接着选，因此把所有的排序看成若干分组，所有排序可以分成n组，然后每一组可以再分为n - 1组。那么对k而言，k能直接决定它位于每个组的位置，方法就是取余数实现。

坑：由于这里k是从1开始的，而非从0开始，取余数运算过程中往往是从0开始的，因此k带入运算时最好将范围从1~n!变为0~n!-1。

**代码**：

~~~C++
class Solution {
public:
    int getFac(int n)
    {
        int res = 1;
        while(n != 1)
        {
            res *= n;
            n--;
        }
        return res;
    }
    string getPermutation(int n, int k) {
        if(n > 9 || k <= 0) return "";
        //结果
        string res = "";
        //用于保存每次挑剩下的字符串
        vector<int> num(n, 0);
        for(int i = 0; i < n; i++)
        {
            num[i] = i + 1;
        }
        //保存阶乘
        int fac[n + 1];
        for(int i = 0; i < n + 1; i++)
        {
            fac[i] = 1;
        }
        //用于遍历产生阶乘和确定字符串的顺序
        int tra = 1;
        //产生阶乘
        while(tra <= n)
        {
            fac[tra] = fac[tra - 1] * tra;
            tra++;
        }
        //确定顺序
        if(k > fac[n]) return "";
        //从n-1的阶乘开始找自己的分组
        tra = n;
      	//将k的范围前移一个，使其从0开始计数
        k = k-1;
        while((--tra) >= 0)
        {
            int cur = k / fac[tra];            
            res += to_string(num[cur]);
            num.erase(num.begin() + cur);
            k = k % fac[tra];
        }
        return res;
    }
};
~~~

### 136E. 孤独的数

**问题描述**：给出一数组，其中每个数都出现了两次，只有一个数出现了一次，找出这个数。

**我的思路**：将代码空间利用率优化到O(1)，使数组中所有数进行异或运算，因为两个相同的数异或后结果为0，所得的结果即为Single Number。

**代码**：

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

### 231E. 2的平方

**问题描述**：判断一个数是不是二的平方

**我的思路**：判断一个数是否是2的次幂，从传统的角度可以不停的对这个数mod2，从二进制的角度，一个2的次幂的二进制表示应该只有一位是2，其余全是0，如0100（4）和01000000（64）。那么这个数减1后应是从右到左一串连续的1，如0011（3）和00111111（63）。那么这两个数按位异或，结果应该是0，如果这个数不是2的次幂，那么异或的结果不会为0。

**代码**：

~~~c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n <= 0 ? false : !(bool)(n ^ (n - 1));
    }
};
~~~

### 342E. 4的平方

**问题描述**：判断一个数是否是4的平方。

**我的思路**：在2的平方问题的基础上，4的次幂应是在2的次幂的基础上，对其二进制表示的唯一的1的位置有了要求，应该在奇数位上，如0100（4），00010000（16）和00100000（32）不是4的次幂。那么需要在检测2的次幂的基础上再此检测这个1的位置。

**代码**：

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

参见异或的用法<https://starkschroedinger.github.io/2020/01/31/LeetCode-Notes/>

最后编辑时间 2020-02-20

