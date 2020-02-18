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

本帖子分类记录刷题过程中的思路，经验以及遇到的坑。

## 1. 线性表(数组+链表)

### 2M. 两数相加。

问题描述：给定两已知链表由低到高保存两数的各位，将两数相加后返回一链表。

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

### 8M. 字符串中提取整数

**问题描述**：从一个给定的字符串中提取数字，要求从第一个非空字符开始，如果第一个非空字符不是正负符号或数字则返回0；如果超出int类型的范围（大于INT_MAX或小于INT_MIN）则返回INT_MAX或INT_MIN。

**我的思路**：很平凡的遍历思想，首先判断把空字符给跳过去，然后判断是否是数字，是数字的话记录数字。问题是坑太多，还有就是在一位一位等待数字的过程中判断是否会溢出int很重要，可以记住。

**坑**：在遍历的时候因为需要判断的条件太多，使用while循环很容易忘记判断角标i是否超出str的size。另一个坑就是判断溢出。

优化：判断是否溢出的代码`if((res > INT_MAX / 10) || (temp >= INT_MAX % 10 && res == INT_MAX / 10)) return INT_MAX;` 和`if((res == INT_MIN / 10 && -temp <= INT_MIN % 10) || res < INT_MIN / 10) return INT_MIN;`。

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

### 15M. 三数之和

**问题描述**：给定n个整数，返回所有满足条件的三元组(a, b, c)，使得a + b + c = 0，并且这三元组不能重复。

**我的思路**：

回忆2Sum问题，用了O(n)的时间和空间，使用哈希表，将每一个已经遍历了的数对应的“余数”存入哈希表，看接下来的数是否满足条件，思想是用空间换时间。另一种方法是首先将给定数组排序（最快的快排也需要O(nlogn)的时间，因此对于2Sum并不是很划算），然后用双指针分别指向首尾，求和，根据求和的结果和target进行比较来相应地移动首或尾指针，直到求和结果等于target。

将3Sum的第一个数固定，然后问题就变成了2Sum的问题，此时target就是-a，我们可以在2Sum算法的基础上额外增加一次遍历，使得算法的复杂度处于O(n^2)的水平。剩下还需要解决一些细枝末节的问题。

**坑**

* 为了排除因为第一个数字重复而导致的三元组重复，需要在第一个数字遍历时，遇到重复的直接跳过本次循环。

  `if(i > 0 && nums[i] == nums[i - 1]) continue;`

* 2Sum使用哈希表的思路在3Sum问题中当输入数组特别大的时候会导致Time Limit Exceeded。Test case: <https://leetcode.com/submissions/detail/300659782/testcase/>。

**加速**

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

### 19M. 移除倒数第n个链表结点。

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

### 21E. 合并两有序链表

问题描述：将两个有序链表合并成一个有序链表。

我的思路：思路其实很简单，从小到大两链表都有对应的指针，一个一个遍历比较大小即可，但是实现的时候把代码写的简洁却不是一件容易的事。下边放出官方给出的效率最高的代码之一。

代码：

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

问题描述：给出一个链表，成对交换节点的第1、2，3、4...个节点。要求不能改变节点的值，只能改变节点的指向。

我的思路：很平常的思路，一道练习链表节点操作的题。

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

## 2. 字符串

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

## 3. 栈和队列

### 20E. 有效的括号对

问题描述：判断一个给定字符串是否是有效的括号对。有效的括号对只任何两个配对的括号对中间都必须是完整的配对括号对。如：` "{[]}"`、`"[[{}{()}]]"`等。

我的思路：一个自然的思路自然是stack，如果当前的是左括号就压入stack，如果是右括号就把stack顶部元素弹出，符合要求的一定会与当前括号匹配。否则返回无效。

坑：注意判断stack的时候需要注意没有右括号的case，如`"(("`，在判断的时候由于没有弹出stack判断的操作自然会没有因此判断为无效，因此在末尾要判断stack内是否还有元素。

优化：如果本身括号的个数是奇数，那么一定不会是有效的。

代码：

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

### 

## 4. 树

## 5. 排序

## 6. 查找

## 7. 策略性枚举

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

### 264M. 丑数II

问题描述：丑数是质因子只有2、5和8的数，规定1是第一个丑数。求第n个丑数。

我的思路：采用三指针法。因为每一个丑数都能看作从前边某个丑数与2、5或8相乘得到。因此，我们从1开始创建三个附属于丑数序列的虚序列。每个指针都指向当前2、3或5作为比较的对象，当当前对象乘以相应因子后仍小于最大的丑数时，则不断循环前进。这样不会漏掉任何一个丑数。

代码：

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

## 8. 广度优先搜索

## 9. 深度优先搜索

### 36M. 有效的数独

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

## 10. 分治法

## 11. 贪心法

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

因为j只从1遍历到n，所以复杂度降为O(n)。

### 11M. 能盛最多水的容器

**问题描述**·；有若干相距为1的立起来的板子，他们的高度依次被存在给定的数组中。现在需要找到个板子，使得这两个板子之间能盛的水最多。

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



## 12. 动态规划

## 13. 图

## 14. 细节实现题

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

