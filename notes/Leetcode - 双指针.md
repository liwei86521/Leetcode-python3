# Leetcode 题解 - 双指针
<!-- GFM-TOC -->
* [Leetcode 题解 - 双指针](#leetcode-题解---双指针)
    * [1. 有序数组的 Two Sum](#1-有序数组的-two-sum)
    * [2. 两数平方和](#2-两数平方和)
    * [3. 反转字符串中的元音字符](#3-反转字符串中的元音字符)
    * [4. 回文字符串](#4-回文字符串)
    * [5. 归并两个有序数组](#5-归并两个有序数组)
    * [6. 判断链表是否存在环](#6-判断链表是否存在环)
    * [7. 最长子序列](#7-最长子序列)
<!-- GFM-TOC -->


双指针主要用于遍历数组，两个指针指向不同的元素，从而协同完成任务。

## 1. 有序数组的 Two Sum

167\. Two Sum II - Input array is sorted (Easy)

[Leetcode](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) / [力扣](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/description/)

```html
Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2
```

题目描述：在有序数组中找出两个数，使它们的和为 target。

使用双指针，一个指针指向值较小的元素，一个指针指向值较大的元素。指向较小元素的指针从头向尾遍历，指向较大元素的指针从尾向头遍历。

- 如果两个指针指向元素的和 sum == target，那么得到要求的结果；
- 如果 sum \> target，移动较大的元素，使 sum 变小一些；
- 如果 sum \< target，移动较小的元素，使 sum 变大一些。

数组中的元素最多遍历一次，时间复杂度为 O(N)。只使用了两个额外变量，空间复杂度为  O(1)。

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/437cb54c-5970-4ba9-b2ef-2541f7d6c81e.gif" width="200px"> </div><br>

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        #双指针， left指针从左往右扫，right指针从右往左扫
        left=0
        right=len(numbers)-1
        while left<right:
            total = numbers[left]+numbers[right]
            if total==target:
                return [left+1,right+1]
            elif total<target:
                left+=1
            else:
                right-=1

        return [-1, -1]
}
``` 

## 2. 两数平方和

633\. Sum of Square Numbers (Easy)

[力扣](https://leetcode-cn.com/problems/sum-of-square-numbers/description/)

```html
Input: 5
Output: True
Explanation: 1 * 1 + 2 * 2 = 5
```

题目描述：判断一个非负整数是否为两个整数的平方和。

可以看成是在元素为 0\~target 的有序数组中查找两个数，使得这两个数的平方和为 target，如果能找到，则返回 true，表示 target 是两个整数的平方和。

本题的关键是右指针的初始化，实现剪枝，从而降低时间复杂度。设右指针为 x，左指针固定为 0，为了使 0<sup>2</sup> + x<sup>2</sup> 的值尽可能接近 target，我们可以将 x 取为 sqrt(target)。

因为最多只需要遍历一次 0\~sqrt(target)，所以时间复杂度为 O(sqrt(target))。又因为只使用了两个额外的变量，因此空间复杂度为 O(1)。

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        j = int(math.sqrt(c))
        i = 0
        while i <= j:
            total = i * i + j * j
            if total > c:
                j = j - 1
            elif total < c:
                i = i + 1
            else:
                return True
        return False
``` 

## 3. 判断子序列
给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

392\. 判断子序列（easy） [力扣](https://leetcode-cn.com/problems/is-subsequence/description/)

示例 1:
s = "abc", t = "ahbgdc"   返回 true.

示例 2:
s = "axc", t = "ahbgdc"   返回 false


```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        #双指针
        m, n = len(s), len(t)
        i, j = 0, 0 # i为慢指针，j为快指针
        while i < m and j < n:
            if s[i] == t[j]:
                i += 1 # 单个字符成功匹配 i就加1
                j += 1
            else:
                j += 1

        return i == m  # 相当于如果字符串s变量完了，即是 s 是否为 t 的子序列
        
``` 

## 4. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。


344\. 反转字符串（easy） [力扣](https://leetcode-cn.com/problems/reverse-string/description/)

示例 1:
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]


示例 2:
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]


```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
		i, j = 0, len(s)- 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i, j = i+1, j-1
        
``` 


## 5. 反转字符串中的元音字母
给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

345\. 反转字符串中的元音字母（easy） [力扣](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/description/)

示例 1:
输入："hello"  输出："holle"


示例 2:
输入："leetcode"   输出："leotcede"


```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        s = list(s)
        n = len(s)
        ls = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        pre = 0 # 双指针
        last = n - 1
        while pre < last:
            if s[pre] in ls and s[last] in ls:
                s[pre], s[last] = s[last], s[pre] 
                pre += 1
                last -= 1

            if s[pre] not in ls or s[last] not in ls:
                if s[pre] not in ls:
                    pre += 1
                if s[last] not in ls:
                    last -= 1
                    
        return ''.join(s)
        
``` 

## 6. 移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

283\. 移动零（easy） [力扣](https://leetcode-cn.com/problems/move-zeroes/description/)

示例 1:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]


```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        #Do not return anything, modify nums in-place instead.
        # ps: 与 27. 移除元素 基本是一样的思想

        j = 0 # 双指针

        for i in range(len(nums)):
            if (nums[i] != 0):
                nums[j] = nums[i]
                j = j + 1

        while(j < len(nums)):
            nums[j] = 0 # 把最后几位修改为0
            j = j + 1
        
``` 


## 7. 链表的中间结点
给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点

876\. 链表的中间结点（easy） [力扣](https://leetcode-cn.com/problems/middle-of-the-linked-list/description/)

示例 1:
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 

示例 1:
输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。


```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        # 典型的 双指针问题
  
        slow, fast = head, head
        while (fast and fast.next): 
            fast = fast.next.next
            slow = slow.next
                
        return slow
        
``` 

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

141\. 环形链表（easy） [力扣](https://leetcode-cn.com/problems/linked-list-cycle/description/)

示例 1:

<img style="height: 97px; width: 300px;" src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png" alt="">

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。


```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        # 经典双指针问题，如果又环，则快慢指针必然会相遇,反之没有环
        slow = head
        fast = head
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False
        
```

