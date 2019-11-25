---
layout:     post
title:      "Hello World by Markdown"
subtitle:   "A start."
date:       2019-11-25
author:     "Felix"
header-img: "img/post-bg-C++.jpg"
catalog: true
tags:
    - Markdown
---

# Hello World #

这是第一个Markdown文档，用来测试各种字体功能，以及写出这个hello world.

# 常用语法 #

* *，-，+这三个符号效果都一样，这3个符号被称为Markdown列表符号。而有序列表则使用数字接着一个英文句点（数字大小并不会影响输出序列）。 

* 使用成对的星号*和底号_表示标签，效果为输出字体为斜体。

* “>” 可表示印用，可见写于第一行，也可以每一行都添加；区块的引用可嵌套。

  > 这句话不是我说的。
  >
  > > 这句话也不是我说的。

* 链接的实现：方括号里面输入文字，接着一个圆括号里边输入网址即可。例如[Felix的主页](https://starkschroedinger.github.io)

* 图片的插入：和链接非常相似，只是多了一个感叹号!。

  <img src="D:\Github\StarkSchroedinger\huxpro.github.io\img\post-bg-C++.jpg" style="zoom:50%;" />

* 分割线：可以使用三个以上的*或者-连建立分割线。例如连续三个 * 可产生如下效果

  上文

  ***

  下文

* 自动链接，将简短的网址实用尖括号括起来可实现短连接的自动生成。例如感谢<https://www.jianshu.com/p/d7867cb330ec>的技术支持。

* 表格制作。使用\|和\-来绘制表格。:可控制左对齐右对齐和剧中。例如

  ```markdown
  | Title | Description|
  |:-----|:---------------:|
  |Version|0.0.1|
  |Editor|[Felix](StarkSchroedinger.github.com)|
  ```

  其效果如下：

  | Title   |              Description              |
  | :------ | :-----------------------------------: |
  | Version |                 0.0.1                 |
  | Editor  | [Felix](StarkSchroedinger.github.com) |
