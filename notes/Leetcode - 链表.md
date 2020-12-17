# Leetcode - 链表
<!-- GFM-TOC -->
* [Leetcode - 链表](#leetcode---链表)
    * [1. 删除链表中的节点](#1-删除链表中的节点)
    * [2. 反转链表](#2-反转链表)
    * [3. 相交链表](#3-相交链表)
    * [4. 环形链表](#4-环形链表)
    * [5. 回文链表](#5-回文链表)
    * [6. 删除排序链表中的重复元素](#6-删除排序链表中的重复元素)
    * [7. 链表的中间结点](#7-链表的中间结点)
    * [8. 奇偶链表](#8-奇偶链表)
    * [9. 旋转链表](#9-旋转链表)
    * [10. 分隔链表](#10-分隔链表)
    * [11. 删除链表的倒数第N个节点](#11-删除链表的倒数第N个节点)
    * [12. 两两交换链表中的节点](#12-两两交换链表中的节点)
    * [13. 环形链表 II](#13-环形链表-II)
    * [14. 排序链表](#14-排序链表)
    * [15. 两数相加](#15-两数相加)
<!-- GFM-TOC -->

链表（Linked list）是一种常见的基础数据结构，是一种线性表，是一种物理存储单元上非连续、非顺序的存储结构。
链表由一系列结点（链表中每一个元素称为结点）组成，结点可以在运行时动态生成。很多链表问题可以用递归来处理。

## 1. 删除链表中的节点

请编写一个函数，使其可以删除某个链表中给定的（**非末尾**）节点。传入函数的唯一参数为 **要被删除的节点** 。

237\. 删除链表中的节点（middle） [力扣](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/description/)

示例 1:

```html
输入：head = [4,5,1,9], node = 5
输出：[4,1,9]
解释：给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

输入：head = [4,5,1,9], node = 1
输出：[4,5,9]
解释：给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val # 因为node 非尾节点,所以node.next一定不为空

        #ps 不需要考虑 node.next.next 是否为None
        node.next = node.next.next

``` 

## 2. 反转链表

反转一个单链表

206\. 反转链表（middle） [力扣](https://leetcode-cn.com/problems/reverse-linked-list/description/)

示例 1:

```html
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 方法1 迭代
        prev = None # 前指针节点
        cur = head # 当前指针节点
        while (cur != None) : #每次循环，都将当前节点指向它前面的节点，然后当前节点和前节点后移
            temp = cur.next #存储, 临时节点，暂存当前节点的下一节点（不然链表就会丢失），用于后移
            cur.next = prev # 将当前节点指向它前面的节点
            
            # ps: 下面2句顺序不能乱
            prev = cur #前指针后移
            cur = temp # 当前指针后移

            #cur.next, prev, cur = prev, cur,  cur.next 等效上面4句
            
        return prev


    #递归表示有点难理解    
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:return head # 递归终止条件

        cur=self.reverseList(head.next)
        head.next.next=head
        head.next=None
        return cur

``` 

## 3. 相交链表

编写一个程序，找到两个单链表相交的起始节点。

160\. 相交链表（middle） [力扣](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/description/)

示例 1:
<img style="height: 130px; width: 400px;" src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png" alt="">

```html
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # two pointer
        """
        len(A) + len(B) == len(B) + len(A)
        #两条路线同时走: A, B 相交，最后 8 == 8 结束循环
            4-->1-->8-->4-->5 -->5-->0-->1-->8
            5-->0-->1-->8-->4-->5 -->4-->1-->8 
        
         # 如果AB不相交，while最多在 len(p)+len(q)后结束
            A  [1, 2, 3], B = [4,5,6,7]  A, B 不相交，最后 None == None 结束循环
            路线1: 1, 2, 3, 4,5,6,7, None
            路线2: 4,5,6,7, 1, 2, 3, None
        """

        p, q = headA, headB
        while p!=q: # p q最后比相等 要么是有相交节点要么就 都等于None 
            #if p:
            #    p = p.next
            #else:
            #    p = headB
            
            p = p.next if p else headB  # 这个厉害  
            q = q.next if q else headA
            
        return p 

``` 

## 4. 环形链表

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。**注意：pos 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

141\. 环形链表（easy） [力扣](https://leetcode-cn.com/problems/linked-list-cycle/description/)

示例 1:
<img style="height: 97px; width: 300px;" src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png" alt="">

```html
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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

## 5. 回文链表

请判断一个链表是否为回文链表

234\. 回文链表（easy） [力扣](https://leetcode-cn.com/problems/palindrome-linked-list/description/)

示例 1:

```html
输入: 1->2
输出: false

输入: 1->2->2->1
输出: true

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        res = []
        while(head):
            res.append(head.val)
            head = head.next

        if len(res) < 1: # ps: 空链表是回文链表
            return True
        else:
            return res == res[::-1]

``` 

## 6. 删除排序链表中的重复元素

给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次

83\. 删除排序链表中的重复元素（easy） [力扣](https://leetcode-cn.com/problems/palindrome-linked-list/description/)

示例 1:

```html
输入: 1->1->2
输出: 1->2

输入: 1->1->2->3->3
输出: 1->2->3

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None or head.next == None: # bad case
            return head

        cur = head # cur 用来遍历

        while (cur and cur.next):
            if cur.val == cur.next.val: #当前节点的值和下一个节点的值相等
                cur.next = cur.next.next
            else:
                cur = cur.next

        return head # 返回头指针

``` 

## 7. 链表的中间结点

给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点

876\. 链表的中间结点（easy） [力扣](https://leetcode-cn.com/problems/middle-of-the-linked-list/description/)

示例 1:

```html
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.

输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        # 典型的 双指针问题
  
        slow, fast = head, head
        while (fast and fast.next): 
            fast = fast.next.next
            slow = slow.next
                
        return slow

``` 

## 8. 奇偶链表

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推

328\. 奇偶链表（middle） [力扣](https://leetcode-cn.com/problems/odd-even-linked-list/description/)

示例 1:

```html
输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
示例 2:

输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        """ 把原链表分成奇偶两个链表，再把奇数聊表尾部与偶数链表头部连接起来
        空间复杂度应为 O(1) 即只能用常数个变量，时间复杂度应为 O(n)
        """
        if not head or not head.next:
            return head

        odd = head # 奇数链表头指针,ps 头指针始终没有变
        even, evenHead  = head.next, head.next #偶数链表头指针
        while(even and even.next):
            odd.next = even.next #连接奇节点
            odd = odd.next # 更新odd节点游标
            
            even.next = odd.next # 这里 odd.next是否为None不影响
            even = even.next #更新even节点游标

        odd.next = evenHead #把偶链表的接在 奇链表后面
        return head

``` 

## 9. 旋转链表

给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数

61\. 旋转链表（middle） [力扣](https://leetcode-cn.com/problems/rotate-list/description/)

示例 1:

```html
输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL

输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL
```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        # base cases
        if not head or not head.next:
            return head
        
        # 第一步:先将链表闭合成环， 先找到尾节点和统计链表的长度
        rear = head
        n = 1 # n表示链表中节点的个数
        while rear.next: #ps : 这里必须这样做，不然无法进行闭环
            rear = rear.next
            n += 1
            
        rear.next = head #进行闭环
        
        #第二步:找到相应的位置断开这个环，确定新的链表尾和链表头
        new_tail = head #new_tail 用来移动

        # 头结点的位置在 n-k处，尾节点在头结点前面 故为n-k-1 因为k有可能大于n为了
        # 避免出现负数 尾节点 n-k%n-1
        for i in range(0, n-k%n-1):
            new_tail = new_tail.next
            
        new_head = new_tail.next # 得到新头结点
        
        new_tail.next = None # 断开环
        
        return new_head

``` 

## 10. 分隔链表

给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。

你应当保留两个分区中每个节点的初始相对位置

86\. 分隔链表（middle） [力扣](https://leetcode-cn.com/problems/partition-list/description/)

示例 1:

```html
输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:

        before = before_head = ListNode(0) # 建立dummy变量方便操作
        after = after_head = ListNode(0)
        
        while(head):
            if head.val < x:
                before.next = head
                before =before.next
            else:
                after.next = head
                after = after.next
            
            head = head.next # 更新头节点
        
        #循环结束后拼接链表
        before.next = after_head.next
        after.next = None # 尾节点next置空
            
        return before_head.next

``` 

## 11. 删除链表的倒数第N个节点

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

给定的 n 保证是有效的

19\. 删除链表的倒数第N个节点（middle） [力扣](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/description/)

示例 1:

```html
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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

## 12. 两两交换链表中的节点

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

24\. 两两交换链表中的节点（middle） [力扣](https://leetcode-cn.com/problems/swap-nodes-in-pairs/description/)

示例 1:

```html
输入：head = [1,2,3,4]
输出：[2,1,4,3]

输入：head = []
输出：[]

输入：head = [1]
输出：[1]

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(-1) # 定义哨兵节点 方便遍历
        dummy.next = head
        cur = dummy # cur 游标用来移动

        while head and head.next:
            # Nodes to be swapped
            first_node = head
            sec_node = head.next

            # Swapping
            cur.next = sec_node
            first_node.next = sec_node.next
            sec_node.next = first_node

            # update the cur and head for next swap
            cur = first_node
            head = first_node.next

        # Return the new head node.
        return dummy.next


    def swapPairs(self, head: ListNode) -> ListNode:
        # 递归 
        if not head or not head.next:  #递归终止条件
            return head
        
        tmp = head.next
        head.next = self.swapPairs(head.next.next)
        tmp.next = head
        
        return tmp

``` 

## 13. 环形链表 II

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。**注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。**

**说明：不允许修改给定的链表**

142\. 环形链表 II（middle） [力扣](https://leetcode-cn.com/problems/linked-list-cycle-ii/description/)

示例 1:

```html
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。

输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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

## 14. 排序链表

给你链表的头结点 head ，请将其按 **升序** 排列并返回 **排序后的链表**

148\. 排序链表（middle） [力扣](https://leetcode-cn.com/problems/sort-list/description/)

示例 1:

```html
输入：head = [4,2,1,3]
输出：[1,2,3,4]

输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]

输入：head = []
输出：[]

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:  # 递归结束条件
            return head

        # cut the LinkedList at the mid index.
        slow, fast = head, head.next  # 链表长度为6， index就为3
        while (fast and fast.next):  # 找到中间节点
            fast, slow = fast.next.next, slow.next

        mid, slow.next = slow.next, None  # 切分为2个新链表

        # 分治思想的分，一个大链表从中间索引切分成2个，然后递归调用
        left, right = self.sortList(head), self.sortList(mid)

        # 分治思想的治 (merge) merge `left` and `right` linked list
        return self.merge(left, right)

    def merge(self, left, right):
        cur = res = ListNode(0)  # 哨兵节点
        while left and right:
            if left.val < right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next

            cur = cur.next  # 更新 cur

        cur.next = left if left else right
        return res.next

``` 

## 15. 两数相加

给出两个 **非空** 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 **逆序** 的方式存储的，并且它们的每个节点只能存储 **一位** 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

2\. 两数相加（middle） [力扣](https://leetcode-cn.com/problems/add-two-numbers/description/)

示例 1:

```html
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807

```

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        re = ListNode(0) # re 相当于 头结点，永远不变
        r=re # r 是用来 移动为节点指针的
        carry=0 # 用来看是否要进位
        while(l1 or l2):
            x= l1.val if l1 else 0
            y= l2.val if l2 else 0
            
            s=carry+x+y # 加上一位的进位 carry
            carry=s//10 # 更新进位，下次循环用
            
            r.next=ListNode(s%10) #添加节点
            r=r.next              # 更新 r 的指针位置， 非常重要
            
            #更新 l1,l2
            if(l1!=None):l1=l1.next  # 非常重要
            if(l2!=None):l2=l2.next
            # ps 下面 l1和l2的更新是错误的， 因为 None是没有.next方法的
            #l1 = l1.next if l1.next else None
            #l2 = l2.next if l2.next else None

        if carry == 1:# 这里是判断最后1位是否需要进位
            r.next=ListNode( carry )
            
        return re.next

``` 
