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


## 8M. 字符串中提取整数

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

******

## 11M. 能盛最多水的容器

问题描述·；有若干相距为1的立起来的板子，他们的高度依次被存在给定的数组中。现在需要找到个板子，使得这两个板子之间能盛的水最多。

我的思路：这道题如果暴力求解需要找出两两配对的情况，复杂度在O(n^2)。现在比较巧妙的方法是，首先取首尾两个板子，然后逐渐向中间移动，直到碰头，规则是：左右两边较低一侧的指针往中间移动。这样能保证最大的情况一定能被遍历到，且只用O(n)的时间。（证明略，用反证法比较容易想清楚。）

代码：

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

*****

## 12M. 阿拉伯数字到罗马数字转换

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




*****
## 15M. 3Sum

**问题描述**：给定n个整数，返回所有满足条件的三元组(a, b, c)，使得a + b + c = 0，并且这三元组不能重复。

**我的思路**：

回忆2Sum问题，用了O(n)的时间和空间，使用哈希表，将每一个已经遍历了的数对应的“余数”存入哈希表，看接下来的数是否满足条件，思想是用空间换时间。另一种方法是首先将给定数组排序（最快的快排也需要O(nlogn)的时间，因此对于2Sum并不是很划算），然后用双指针分别指向收尾，求和，根据求和的结果和target进行比较来相应地移动首或尾指针，直到求和结果等于target。

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



*****

## 17M. 电话号码的字母组合

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


*****

## 20E. 有效的括号对

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



*****

## 136E 孤独的数

## 231E 2的平方

## 342E 4的平方

参见异或的用法<<https://starkschroedinger.github.io/2020/01/31/LeetCode-Notes/>>

*****

## 141E. 检测链表是否有环

参见快慢指针<https://starkschroedinger.github.io/2020/02/02/LeetCode-Notes/>。

*****



