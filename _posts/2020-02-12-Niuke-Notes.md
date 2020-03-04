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
   - C++
   - Algorithm
---

# 牛客网刷题笔记

* 常用函数记录

  * 关于输入输出

    string类型：`string s; getline(cin, s);`

    char类型：· char c; cin >> c;

    C++ 64位输出请用printf("%lld")

    OJ一次处理多个case，所以代码需要循环处理，一般通过while循环来出来多个case。

    对于数字类型的循环输入，直接while(cin >> num)就可以了。

    对于string 类型对象，cin对空格敏感，getline对空格不敏感。

  * 数组的初始化

    \1. memset()函数原型是extern void *memset(void *buffer, int c, int count)     buffer：为指针或是数组，c：是赋给buffer的值，count：是buffer的长度。这个函数在socket中多用于清空数组.如:原型是memset(buffer, 0, sizeof(buffer))。Memset 用来对一段内存空间全部设置为某个字符，一般用在对定义的字符串进行初始化为‘ ’或‘/0’。

  * vector

    vector中的最小值：int min = *min_element(v.begin(),v.end()); 

    vector中的最大值：int max = *max_element(v.begin(),v.end()); 

    vector插入

    insert(it, n, a); 在it这个迭代器位置插入n个a

    构造函数：

    vector<string> v(n); v中n个空字符串。

    vector<string>  v(n, s); v中初始化为n个s

    排序算法sort在牛客网中必须包含<algorithm>文件

  * unordered_map 常用函数

    clear();erase();find();

  * <cctype>常用函数

    isalnum(); isalpha(); isdigit(); islower(); isupper(); isspace(); tolower(); toupper(); 

  * string记不住的函数

    string s(n, a); s是n个char a

    substr函数的形式为s.substr(pos, n)

    to_string();

    int idx = s.find('a');

    int c = count(s.begin(), s.end(), ' ');

    string s = s.substr(0, s.find(' '));		截取第一个空格前的字字符串

  * sstream

    将字符串转化为数字：stringstream ss;

  * algorithm头文件

    sort函数在此文件中

    reverse函数在此文件中

    stable_sort在此文件中，有可能会用到。

* 不稳定的排序算法有：希快选堆，分别希尔排序，快速排序，直接选择排序，堆排序

* 'int a = 10; char c = 'a';那么表达式` a + c + 4.32`的类型为？

  double，在表达式运算类型总是低等级往高等级转换。表达式的类型等级为：**char** < **short** < **int** < unsigned int < long < unsigned long < long long < unsigned long long < **float** < **double** < long double 不同类型运算结果类型向右边靠齐。无符号比有符号等级高。

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

* 理解以下数组声明的意思：
  ` int *ptrs[10]` 

  ` int &refs[10]`

  ` int (*Parray)[10] = &arr;`

  ` int (&arrRef)[10] = arr;` 

  ` int arr[10]; int *e = &arr[10];`

  解答：第一个：声明一个包含十个对象的数组，数组内对象是int类型的指针；第二个：是错误的，无法声明对象为引用的数组；第三个：声明了一个指针，指向一个包含10个int对象的数组；第四个：声明了一个包含着10个int类型对象的数组的引用； 第五个：创建一个指针指向arr最后一个元素后的位置，类似于迭代器里的end()。

* 解释以下声明:

  ~~~C++
  bool (*pf)(const string&, const string&);
  bool *pf(const string&, const string&);
  int (*func(int *a)) [10];
  
  int *matrix[10];
  int (*matrix)[10]；
  ~~~

  答：第一个声明了一个函数指针，其返回值类型为bool，第二个声明了一个函数，返回值是bool类型的指针；第三个声明了一个函数，其返回值是一个大小为10的int类型的数组。

  第四个声明了一个大小为10类型为int*的数组，即数组中存放着int指针；第五个声明了一个指针，他指向一个存放着10个int的数组。

1. ### **报数游戏**

输入：第一行输入一个整数m，表示有m组数据，输入m行数据，每行一个整数，代表着报数圈的人数，报数出去的人的key是个定值3，要求输出m行数据，每行是依次离开的人的次序。

我的代码：

~~~C++
#include <iostream>
#include <vector>
using namespace std;


class Node
{
public:
    Node() = default;
    Node(int val):value(val), next(nullptr){}
    int getValue()
    {
        return this->value;
    }
    Node* getNext()
    {
        return this->next;
    }
    void setValue(int val)
    {
        value = val;
    }
    void setNext(Node* node)
    {
        next = node;
    }
public:
    int value;
    Node* next;
};

int main()
{
    int size;
    cin >> size;
    int Key = 3;
    while(size != 0)
    {
        int cur;
        Node *head = new Node(0), *trav = head; // Used as pivot.
        cin >> cur;
        for(int i = 1; i <= cur; i++)
        {
            trav -> next = new Node(i);
            trav = trav->next;
        }
        //Form a loop
        trav->next = head->next;
        //trav goes to the start.
      	//这里一开始写成了trav = head->next，是容易出错的地方，因为在循环的时候是先输出，后将trav向后移
        trav = head;
        int num = 0;
        while(trav->next != trav)
        {
            num++;
            if(num == Key)
            {
                cout << trav->next->value << " ";
                Node *temp = trav->next;
                trav ->next = trav->next->next;
                delete temp;
                num = 0;
            }
            else trav = trav->next;
        }
        cout << trav->value << endl;
        delete trav;
        delete head;
        size--;
    }
    return 0;
}
~~~

2. ### 删除字符串中出现次数最少的字符

题目描述：实现删除字符串中出现次数最少的字符，若多个字符出现次数一样，则都删除。输出删除这些单词后的字符串，字符串中其它字符保持原来的顺序。**注意每个输入文件有多组输入，即多个字符串用回车隔开**

坑：坑就在于这个循环输入的问题，要有while循环输入，否则是错误，通过率0.00%。注意memset的用法以及头文件<memory.h>

实现方法：使用数组从0到26来记录每个字母出现的次数。然后用hash表unordered_set来从前往后遍历看有没有最少的，每次碰到最少的就更新unordered_set，最后再遍历字符串删除。

代码：

~~~C++
#include <iostream>
#include <string>
#include <unordered_set>
#include <memory.h>
using namespace std;

int main()
{
    string input = "";
    while(getline(cin, input))
    {
        int numRecord[26];
        memset(numRecord, 0, sizeof(numRecord));
        for(auto it = input.begin(); it != input.end(); it++)
        {
            numRecord[*it - 'a']++;
        }
        unordered_set<int> dispose;
        int min = 0;
        for(int i = 0; i < 26; i++)
        {
            if(numRecord[i] == 0) continue;
            if(min == 0)
            {
                min = numRecord[i];
                dispose.insert(i);
            }
            else
            {
                if(numRecord[i] < min)
                {
                    min = numRecord[i];
                    dispose.clear();
                    dispose.insert(i);
                }
                else if(numRecord[i] == min)
                {
                    dispose.insert(i);
                }
            }
        }
        for(auto it = input.begin(); it != input.end();)
        {
            if(dispose.find(*it - 'a') != dispose.end())
            {
                it = input.erase(it);
            }
            else it++;
        }
        cout << input << endl;
    }
    return 0;
}
~~~

3. ### 情报翻译

   题目描述：在情报传递过程中，为了防止情报被截获，往往需要对情报用一定的方式加密，简单的加密算法虽然不足以完全避免情报被破译，但仍然能防止情报被轻易的识别。我们给出一种最简的的加密方法，对给定的一个字符串，把其中从a-y,A-Y的字母用其后继字母替代，把z和Z用a和A替代，则可得到一个简单的加密字符串。

   坑：除了字母之外的其他符号都不处理。

4. ### **找出字符串中所有连续最长的数字串**

   题目描述：输入一个字符串，输出字符串中最长的数字字符串和它的长度。如果有相同长度的串，则要一块儿输出，但是长度还是一串的长度。

   坑：首先是输入输出的坑，其中如果有多个相同长度的最长数字串，这些数字串的输出中间不用逗号或空格隔开。其次从头到尾遍历时，不要忘记最后一个字母可能是符合条件的串，这时应单独处理。

   代码：

   ~~~C++
   #include <iostream>
   #include <cctype>
   #include <string>
   #include <vector>
   using namespace std;
   
   int main()
   {
       string s;
       while(getline(cin, s))
       {
           vector<string> res = {};
           int start = -1, end = -1;
           int max = 0, cur = 0;
           for(int i = 0; i < s.size(); i++)
           {
               if(isdigit(s[i]))
               {
                   //说明是新数字串的第一个
                   if(cur == 0)
                   {
                       cur++;
                       start = i;
                   }
                   else
                   {
                       cur++;
                   }
                   if(i == s.size() - 1)
                   {
                       if(cur < max) continue;
                       else if(cur == max)
                       {
                           res.push_back(s.substr(start, i - start + 1));
                       }
                       else
                       {
                           res = {s.substr(start, i - start + 1)};
                           max = cur;
                       }
                   }
               }
               else
               {
                   if(cur != 0)
                   {
                       if(cur < max)
                       {
                           cur = 0;
                           continue;
                       }
                       else if(cur == max)
                       {
                           end = i - 1;
                           res.push_back(s.substr(start, end - start + 1));
                           cur = 0;
                       }
                       else
                       {
                           end = i - 1;
                           max = cur;
                           res.clear();
                           res.push_back(s.substr(start, end - start + 1));
                           cur = 0;
                       }
                   }
               }
           }
           if(!res.empty())
           {
               for(int i = 0; i < res.size(); i++)
                   cout << res[i];
           }
           cout << "," << max << endl;
       }
       return 0;
   }
   ~~~

5. ### 句子逆序

   题目描述：将一个英文语句以单词为单位逆序排放。得到逆序的句子。如输入` I am a boy`输出 ` boy a am I`。

   坑：如上一题，不要忘记从前往后遍历的得到的位于句首的单词。

   代码：

   ~~~C++
   #include <iostream>
   #include <string>
   #include <stack>
   #include <cctype>
   
   using namespace std;
   
   int main()
   {
       string s;
       while(getline(cin, s))
       {
           int start = s.size(), end = s.size();
           //oh代表第一个检查出的单词，其之前不需要输出空格
           bool flag = false, oh = false;
           for(int i = s.size() - 1; i >= 0; i--)
           {
               if(isspace(s[i]))
               {
                   if(!flag) continue;
                   else
                   {
                       flag = false;
                       start = i + 1;
                       cout << s.substr(start, end - start + 1);
                   }
               }
               else
               {
                   if(flag) continue;
                   else
                   {
                       flag = true;
                       if(oh) cout << " ";
                       else oh = true;
                       end = i;
                   }
               }
           }
           if(flag) cout << s.substr(0, end + 1) << endl;
           else cout << endl;
       }
       return 0;
   }
   ~~~

6. ### 字符个数统计

   问题描述：编写一个函数，计算字符串中含有的不同字符的个数。字符在ACSII码范围内(0~127)，换行表示结束符，不算在字符里。不在范围内的不作统计。

   坑：关于换行符LF的转义字符需要判断'\n'。用一个hash就能实现了。

   代码：

   ~~~C++
   #include <iostream>
   #include <string>
   #include <unordered_set>
   using namespace std;
   
   int main()
   {    
       string s;
       while(getline(cin, s))
       {
           unordered_set<char> hash;
           int count = 0;
           for(int i = 0; i < s.size(); i++)
           {
               if(s[i] == '\n') break;
               if(hash.find(s[i]) == hash.end())
               {
                   count++;
                   hash.insert(s[i]);
               }
           }
           cout << count << endl;
       }
       return 0;
   }
   ~~~

7. 德州扑克

   题目描述：

   坑：坑太多了，直接采取下边的最佳做法，find等函数用的炉火纯青;

   代码：

   ~~~C++
   链接：https://www.nowcoder.com/questionTerminal/d290db02bacc4c40965ac31d16b1c3eb?answerType=1&f=discussion
   来源：牛客网
   
   #include <iostream>
   #include <algorithm>
   using namespace std;
    
   int main()
   {
       string tb = "345678910JQKA2jokerJOKER";
       string s;
       while(getline(cin, s))
       {
           int idx = s.find('-');
           string t1 = s.substr(0,idx);
           string t2 = s.substr(idx+1);
           int c1 = count(t1.begin(), t1.end(), ' ');
           int c2 = count(t2.begin(), t2.end(), ' ');
           if(c1 != c2) {
               if(t1 == "joker JOKER" || t2 == "joker JOKER") {
                   cout << "joker JOKER" << endl;
               }else if(c1 == 3 ){
                   cout << t1 << endl;
               }else if(c2 == 3){
                   cout << t2 << endl;
               }else {
                   cout << "ERROR" << endl;
               }
           } else {
               string s1 = t1 + ' ', s2 = t2 + ' ';
               s1 = s1.substr(0, s1.find(' '));
               s2 = s2.substr(0, s2.find(' '));
               int i1 = tb.find(s1);
               int i2 = tb.find(s2);
               if(i1 > i2) {
                   cout << t1 << endl;
               } else {
                   cout << t2 << endl;
               }
           }
       }
       return 0;
   }
   ~~~

   