<!-- GFM-TOC -->
* [Leetcode - 分治](#leetcode-题解---分治)
    * [1. 二叉树的最小深度](#1-二叉树的最小深度)
    * [2. Pow(x,n)](#2-Pow(x,n))
    * [3. 首个共同祖先](#3-首个共同祖先)
    * [4. 数组中的第K个最大元素](#4-数组中的第K个最大元素)
    * [5. 排序链表 I](#5-排序链表)
<!-- GFM-TOC -->

## 1. 二叉树的最小深度
给定一个二叉树，找出其最小深度。
最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
说明：叶子节点是指没有子节点的节点。

111\. 二叉树的最小深度（easy） [力扣](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/description/)

示例 1:
<img style="width: 432px; height: 302px;" src="https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg" alt="">

输入：root = [3,9,20,null,null,15,7]    
输出：2

输入：root = [2,null,3,null,4,null,5,null,6]   
输出：5


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        #  分治（递归的高级应用）
        if not root: return 0

        # root左子树为None，而右子树不为None，说明得在右子树中找叶子结点
        if root.left is None and root.right is not None:
            return self.minDepth(root.right) + 1

        # root左子树不为None，而右子树为None，说明得在左子树中找叶子结点
        if root.left is not None and root.right is None:
            return self.minDepth(root.left) + 1

        #divide and conquer
        leftMinDepth = self.minDepth(root.left)
        rightMinDepth = self.minDepth(root.right)

        # process subproblems' results
        results = 1 + min(leftMinDepth, rightMinDepth)

        return results

``` 

## 2. Pow(x,n)
实现 pow(x, n) ，即计算 x 的 n 次幂函数。

50\. Pow(x, n)（easy） [力扣](https://leetcode-cn.com/problems/powx-n/description/)

示例 1:

输入: 2.00000, 10      输出: 1024.00000

输入: 2.00000, -2      输出: 0.25000

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 暴力法时间复杂度 为O(n)
        # 典型分治法(递归的高级应用) 时间复杂度为 O(logN)
        if n == 0: # 递归结束条件
            return 1

        if n < 0: # 这个和上面一样
            x = 1 / x
            n = -n

        if n % 2: #判断是否为奇数
            return x * self.myPow(x, n-1)
        else:
            # 否则就为偶数
            return self.myPow(x*x, n/2) # 分治法

``` 

## 3. 首个共同祖先
设计并实现一个算法，找出二叉树中某两个节点的第一个共同祖先。不得将其他的节点存储在另外的数据结构中。注意：这不一定是二叉搜索树。

121\. 首个共同祖先（middle） [力扣](https://leetcode-cn.com/problems/first-common-ancestor-lcci/description/)

例如，给定如下二叉树: root = [3,5,1,6,2,0,8,null,null,7,4]

示例 1:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        #所有的结点都是唯一的，p、q 不同且均存在于给定的二叉树中
        if not root or root == p or root == q:
            return root

        # 典型分治法
        left = self.lowestCommonAncestor(root.left, p, q) # 结果只有3种，None， p, q
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right: # p,q分别位于 左右子树
            return root
        
        return left if left else right

``` 

## 4. 数组中的第K个最大元素
在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

215\. 数组中的第K个最大元素（middle） [力扣](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/description/)

示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_sort(alist, start, end):
            """快速排序"""
            if start >= end: # 递归结束条件
                return

            # pivot = alist[0] # 这里别写死了，特别注意
            pivot = alist[start] # 假设基准值
            low = start
            high = end

            while (low < high):
                # 必须 high先左移
                while (low < high) and alist[high] >= pivot:
                    high =high -1
                alist[low] = alist[high] # 改变low的值

                while (low < high) and alist[low] < pivot:
                    low =low + 1
                alist[high] = alist[low] # 改变high的值

            # 从循环退出时，low==high, 这样就找到了pivot应该在的位置
            alist[low] = pivot

            #再 利用分治法分别对 小于pivot的数组 和大于pivot的数组 分别进行递归调用
            # 对小于pivot的 左边的列表执行快速排序
            quick_sort(alist, start, low - 1)

            # 对大于pivot的右边的列表递归调用
            quick_sort(alist, low + 1, end)
        
        quick_sort(nums,0, len(nums)-1) # 时间复杂度为 Nlog(N)

        return nums[-k]

``` 

## 5. 排序链表
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

148\. 排序链表（middle） [力扣](https://leetcode-cn.com/problems/sort-list/description/)

示例 1:

<img src="https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg" alt="">

输入：head = [4,2,1,3]
输出：[1,2,3,4]


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
