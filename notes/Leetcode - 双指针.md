# Leetcode 题解 - 双指针
<!-- GFM-TOC -->
* [Leetcode 题解 - 双指针](#leetcode-题解---双指针)
    * [1. 有序数组的 Two Sum](#1-有序数组的-two-sum)
    * [2. 两数平方和](#2-两数平方和)
    * [3. 判断子序列](#3-判断子序列)
    * [4. 反转字符串](#4-反转字符串)
    * [5. 反转字符串中的元音字母](#5-反转字符串中的元音字母)
    * [6. 移动零](#6-移动零)
    * [7. 链表的中间结点](#7-链表的中间结点)
    * [8. 环形链表](#8-环形链表)
    * [9. 删除链表的倒数第N个节点](#9-删除链表的倒数第N个节点)
    * [10. 盛最多水的容器](#10-盛最多水的容器)
    * [11. 环形链表 II](#11-环形链表-II)
    * [12. 螺旋矩阵 II](#12-螺旋矩阵-II)
    * [13. 螺旋矩阵](#13-螺旋矩阵)
    * [14. 三数之和](#14-三数之和)
    * [15. 最短无序连续子数组](#15-最短无序连续子数组)
    * [16. 长度最小的子数组](#16-长度最小的子数组)
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
        i = 0 # 双指针
        n = len(s)
        for j in range(len(t)):
            if i == n: return True # 已经提前找到

            if s[i] == t[j]:
                i += 1

            #前面2个if可以合并为1个 if
            # if i < n and s[i] == t[j]:
            #     i += 1

        return i == n

    def isSubsequence_v1(self, s: str, t: str) -> bool:
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

## 8. 环形链表
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

## 9. 删除链表的倒数第N个节点
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。给定的 n 保证是有效的

19\. 删除链表的倒数第N个节点（middle） [力扣](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/description/)

示例 1:

给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.


```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        
        dummy = ListNode(-1) # 哨兵节点避免方便边界条件处理
        dummy.next = head

        # 给定的 n 保证是有效的，n 最大为链表的长度 且n一定是大于0的，所以不需要考虑链表是否为空
        fast, slow = dummy, dummy

        while n >= 0: # fast 指针先走
            fast = fast.next
            n = n -1

        while fast != None: # slow和fast再同时走
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next # 删除倒数节点

        return dummy.next
        
``` 

## 10. 盛最多水的容器
给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

11\. 盛最多水的容器（middle） [力扣](https://leetcode-cn.com/problems/container-with-most-water/description/)

示例 1:

<img style="height: 287px; width: 600px;" src="https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg" alt="">

输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。


```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        #双指针法
        left = 0
        right = len(height) - 1
        area = 0
        
        while left < right:
            cur = min(height[left], height[right]) * (right - left)
            area = max(area, cur)
            # 将容量小的指针向数组内部移动，下一个矩阵面积才有可能比当前面积大
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            
        return area     

``` 

## 11. 环形链表 II
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

142\. 环形链表 II（middle） [力扣](https://leetcode-cn.com/problems/linked-list-cycle-ii/description/)

示例 1:

<img style="height: 97px; width: 300px;" src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png" alt="">

输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。


```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head == None:
            return None

        p, q = head, head #步骤一：使用快慢指针判断链表是否有环，找到快慢指针的相遇节点(meet = p)
        flagCycle = 0
        #while q.next != None and q.next.next != None:# q为快指针（ps： 因为and的性质）OK
        while q != None and q.next != None:# q为快指针（ps： 因为and的性质）推荐
            p = p.next
            q = q.next.next
            if p == q:# 有环的话一定会在某个位置相等
                flagCycle = 1
                break # 退出while循环

        # 步骤二：若有环，找到入环开始的节点，(从head和meet同时出发，两指针速度一样，相遇时的节点就是入环口) 
        if flagCycle:
            cur = head
            while p != cur:
                p = p.next
                cur = cur.next
            return cur
        else:
            return None    

``` 

## 12. 螺旋矩阵 II
给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵

59\. 螺旋矩阵 II（middle） [力扣](https://leetcode-cn.com/problems/spiral-matrix-ii/description/)

示例 1:

输入: 3
输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]


```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        #还是模拟过程,控制好边界,行的上下边界, 列的左右边界, 很经典
        
        res = [[0] * n for _ in range(n)] # 把结果矩阵全置0,然后下面按规则填充值
        
        above_row = 0 #上边界index
        below_row = n-1 #下边界index
        left_col = 0 #左边界index
        right_col = n - 1 #右边界index
        
        num = 1 #填充的初始值设置为1

        while(above_row <= below_row and left_col <= right_col):
            # 从左到右循环
            for i in range(left_col, right_col+1):
                res[above_row][i] = num #ps: 这里用above_row，是因为马上要更新它
                num = num + 1
            #更新above_row，上边界index加1
            above_row = above_row + 1

            # 从上到下循环
            for i in range(above_row, below_row + 1):
                res[i][right_col] = num  # ps: 这里用right_col，是因为马上要更新它
                num = num + 1
            # 更新right_col，右边界index减1
            right_col = right_col - 1

            # 从右到左循环
            for i in range(right_col, left_col-1, -1):
                res[below_row][i] = num  # ps: 这里用below_row，是因为马上要更新它
                num = num + 1
            # 更新below_row，下边界index减1
            below_row = below_row - 1

            # 从下到上循环
            for i in range(below_row, above_row-1, -1):
                res[i][left_col] = num  # ps: 这里用left_col，是因为马上要更新它
                num = num + 1
            # 更新left_col，右边界index加1
            left_col = left_col + 1

        return res   

```

## 13. 螺旋矩阵
给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

54\. 螺旋矩阵（middle） [力扣](https://leetcode-cn.com/problems/spiral-matrix/description/)

示例 1:

输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]


```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        #与 59. 螺旋矩阵 II 是一样的意思,但是由于不是方形的，边界不能套用
        if len(matrix) == 0:
            return []

        rows = len(matrix)
        cols = len(matrix[0])
        left, right, low, high = 0, cols - 1, 0, rows - 1

        res = []

        while left <= right and low <= high:
            # left ---> right
            for i in range(left, right+1):
                res.append(matrix[low][i])
            low = low + 1
            if low > high:
                break  # 退出while 循环

            # low ---> high
            for i in range(low, high+1):
                res.append(matrix[i][right])
            right = right - 1
            if left > right:
                break  # 退出while 循环

            # right ---> left
            for j in range(right, left-1, -1):
                res.append(matrix[high][j])
            high = high - 1
            if low > high:
                break  # 退出while 循环

            # high ---> low
            for j in range(high, low-1, -1):
                res.append(matrix[j][left])
            left = left + 1
            if left > right:
                break  # 退出while 循环

        return res 

``` 

## 14. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。


15\. 三数之和（middle） [力扣](https://leetcode-cn.com/problems/3sum/description/)

示例 1:

给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]



```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 经典双指针问题,注意先排序，还有去重
        # 注意与 16. 最接近的三数之和 比较不同
        nums.sort() # 排序是为了 方便双指针移动
        n = len(nums)
        res = []
        for i in range(n-2):
            # [-1,0,1,2,-1,-4] --->如果注释这句 结果为 [[-1,-1,2],[-1,0,1],[-1,0,1]]
            if i > 0 and nums[i] == nums[i - 1]: 
                continue #去重,[-2,-2, 0,1,1,2]即第二个 -2就不需要进行下面的循环了，否则会产生重复元素

            left = i + 1
            right = n - 1
            while left < right:
                add = nums[i] + nums[left] + nums[right] # 固定一个数求和
                if add == 0:
                    tmp = [nums[i], nums[left], nums[right]]
                    res.append(tmp) # nums = [0,0,0,0,0], 这里2个while循环就是去重的
                    #[[0,0,0],[0,0,0]] ---> [[0,0,0]]
                    while left+1 < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right -1 and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1 # add等于0是 left += 1，right -= 1， 否则进入死循环
                    right -= 1

                elif add > 0:
                    right -= 1 ## 否则就是大了 right左移
                else:
                    left += 1 # 小了就 left 右移
        return res

``` 

## 15. 最短无序连续子数组
给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

你找到的子数组应是最短的，请输出它的长度。


581\. 最短无序连续子数组（middle） [力扣](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/description/)

示例 1:

输入: [2, 6, 4, 8, 10, 9, 15]
输出: 5
解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # 先排序，再双指针
        sorted_nums = sorted(nums)
        i = 0 # 左指针
        j = len(nums) - 1 # 右指针
        while i <= j and sorted_nums[i] == nums[i]: # 比较右边2列表的相等值得个数
            i += 1
            
        while j >= i and sorted_nums[j] == nums[j]:
            j -= 1
        return j - i + 1

``` 

## 16. 长度最小的子数组
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

209\. 长度最小的子数组（middle） [力扣](https://leetcode-cn.com/problems/minimum-size-subarray-sum/description/)

示例 1:

输入：s = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # 双指针
        if len(nums) == 0:
            return 0

        min_len = len(nums) + 1 # 定义最小值，因为最小值不可能大于它
        j = 0 
        sums = 0 # 求和
        for i in range(len(nums)):
            sums += nums[i]
            while (sums >= s):
                min_len = min(min_len, i - j+1)
                #print("ok ---> ", nums[j:i+1])
                sums = sums - nums[j]
                j += 1 # 用于下次比较

        #print(nums[j: j + (i - j) + 1]) # 这个就是满足条件的 子数组
        return 0 if min_len == len(nums) + 1 else min_len

``` 
