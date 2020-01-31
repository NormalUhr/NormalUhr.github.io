---
layout:     post
title:      "LeetCode 刷题总结 二"
subtitle:   "滑动窗口"
date:       2020-02-01
author:     "Felix"
header-img: "img\in-post\2020-02-01-LeetCode-Notes.jpg"
catalog: true
tags:
    - Markdown
---

# 滑动窗口 #



滑动窗口的实现方式有很多种，可以用double pointer, Hash Map 和队列。其本质是用队列实现头和尾可移动的窗口，这一类问题总是选择出满足条件的某个最值区间，符合条件的区间始终在窗口区域内。

## 3. Longest Substring Without Repeating Characters ##

寻找一个字符串中不含重复字母的最长子串，由于要检测重复，所以选择hash map来实现滑动窗口。

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

