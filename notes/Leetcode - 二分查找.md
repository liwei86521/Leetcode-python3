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


## 1. 二分查找
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。你可以假设 nums 中的所有元素是不重复的。

704\. 二分查找（easy） [力扣](https://leetcode-cn.com/problems/binary-search/description/)

示例 1:

输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1

        # 二分查找模板 好好理解，避免了加1减1
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                right = mid # 注意这里没有return
            elif nums[mid] < target:
                left = mid
            elif nums[mid] > target:
                right = mid
        
        # 2种情况:情况1: 数组只有1或者2个数据（是因为没有进入while循环）
        #情况2： 数组长度>=3 进入while循环后，正常退出后还是2个元素
        if nums[left] == target:
            return left
        if nums[right] == target:
            return right
        
        return -1
    
    '''
    def search(self, nums: List[int], target: int) -> int:
        #  普通二分法
        left,right=0,len(nums)-1        
        while(left<=right):
            mid=(left+right)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]<target:
                left=mid+1
            else:
                right=mid-1
        return -1
    
    '''

``` 

## 2. 有效的完全平方数
给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。不要使用任何内置的库函数，如  sqrt

704\. 有效的完全平方数（easy） [力扣](https://leetcode-cn.com/problems/valid-perfect-square/description/)

示例 1:

输入：16  输出：True 
输入：14  输出：False


```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # 二分法
        if num <= 1:
            return True

        left, right = 0, num
        while (left + 1 < right):
            mid = left + (right - left) // 2

            if mid ** 2 == num:
                return True
            elif mid ** 2 < num:
                left = mid
            else:
                right = mid

        # 循环退出的时候left, right是2个相邻的数，且left < right
        return False

``` 

## 3. x 的平方根
实现 int sqrt(int x) 函数。 计算并返回 x 的平方根，其中 x 是非负整数。

69\. x 的平方根（easy） [力扣](https://leetcode-cn.com/problems/sqrtx/description/)

示例 1:

输入: 4   输出: 2
输入：8  输出：2


```python
class Solution:
    def mySqrt(self, x: int) -> int:
        #计算并返回 x 的平方根，其中 x 是非负整数。由于返回类型是整数，结果只保留整数的部分
        if x <= 1:
            return x

        left, right = 0, x
        while (left + 1 < right):
            mid = left + (right - left) // 2
            
            if mid ** 2 == x:
                return mid
            elif mid ** 2 < x:
                left = mid
            else:
                right = mid
        
        # 循环退出的时候left, right是2个相邻的数，且left < right
        return left

``` 

## 4. 搜索插入位置
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

35\. 搜索插入位置（easy） [力扣](https://leetcode-cn.com/problems/search-insert-position/description/)

示例 1:

输入: [1,3,5,6], 5
输出: 2

输入: [1,3,5,6], 2
输出: 1

输入: [1,3,5,6], 7
输出: 4

输入: [1,3,5,6], 0
输出: 0


```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return 0

        left, right = 0, len(nums) - 1
        # ps 当nums中只有1个或者2个数据都不会进入循环
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid
            else: #nums[mid] > target:
                right = mid

        # 2种情况:情况1: 数组只有1或者2个数据（是因为没有进入while循环）
            #情况2： 数组长度>=3 进入while循环后，正常退出后还是2个元素
        if nums[left] >= target:
            return left
        if nums[right] >= target:
            return right

        # 如果 target > nums[-1]
        return right + 1

``` 

## 5. 剑指 Offer 53 - I. 在排序数组中查找数字 I
统计一个数字在排序数组中出现的次数。

剑指 Offer 53 - I\. 在排序数组中查找数字 I（easy） [力扣](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/description/)

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0


```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 因为是 排序数组 故想到 二分法 时间复杂度为 O(logN)
        left_ind = self.find_target_left_ind(nums, target)
        if left_ind == -1:
            return 0

        right_ind = self.find_target_right_ind(nums, target)
        
        return right_ind - left_ind + 1
    
    def find_target_left_ind(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1

        i, j = 0, n -1
        while (i + 1 < j):
            mid = i + (j - i) // 2
            if nums[mid] < target:
                i = mid + 1
            elif nums[mid] > target:
                j = mid - 1
            else:
                j = mid # 最后找到的是第一次出现target的 ind

        #执行到这 2种情况: 1:未进入while 2:成while正常退出
        if nums[i] == target:
            return i
        if nums[j] == target:
            return j

        return -1
    
    def find_target_right_ind(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1

        i, j = 0, n -1
        while (i + 1 < j):
            mid = i + (j - i) // 2
            if nums[mid] < target:
                i = mid + 1
            elif nums[mid] > target:
                j = mid - 1
            else:
                i = mid # 最后找到的是第后次出现target的 ind

        #执行到这 2种情况: 1:未进入while 2:成while正常退出
        # ps: 这里一定要先从右边边界开始判断
        if nums[j] == target:
            return j

        if nums[i] == target:
            return i

        return -1

``` 

## 6. 在排序数组中查找元素的第一个和最后一个位置
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1]

34\. 在排序数组中查找元素的第一个和最后一个位置（middle） [力扣](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

示例 1:

输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

输入：nums = [], target = 0
输出：[-1,-1]


```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # ps： 时间复杂度必须是 O(log n) 级别 ---> 很明显的二分查找
        if len(nums) == 0:
            return (-1, -1)

        left_ind = self.find_target_left_ind(nums, target)
        if left_ind == -1: # 数组中不存在目标值
            return [-1, -1]

        right_ind = self.find_target_right_ind(nums, target)
        
        return [left_ind, right_ind]
    
    def find_target_left_ind(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1

        i, j = 0, n -1
        while (i + 1 < j):
            mid = i + (j - i) // 2
            if nums[mid] < target:
                i = mid + 1
            elif nums[mid] > target:
                j = mid - 1
            else:
                j = mid # 最后找到的是第一次出现target的 ind

        #执行到这 2种情况: 1:未进入while 2:成while正常退出
        if nums[i] == target:
            return i
        if nums[j] == target:
            return j

        return -1
    
    def find_target_right_ind(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1

        i, j = 0, n -1
        while (i + 1 < j):
            mid = i + (j - i) // 2
            if nums[mid] < target:
                i = mid + 1
            elif nums[mid] > target:
                j = mid - 1
            else:
                i = mid # 最后找到的是第后次出现target的 ind

        #执行到这 2种情况: 1:未进入while 2:成while正常退出
        # ps: 这里一定要先从右边边界开始判断
        if nums[j] == target:
            return j

        if nums[i] == target:
            return i

        return -1

``` 

## 7. 寻找旋转排序数组中的最小值
假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。

请找出其中最小的元素。

153\. 寻找旋转排序数组中的最小值（middle） [力扣](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/description/)

示例 1:

输入：nums = [3,4,5,1,2]
输出：1

输入：nums = [4,5,6,7,0,1,2]
输出：0

输入：nums = [1]
输出：1


```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        #排序 时间复杂度 NlogN
        #for 遍历 时间复杂度 N
        # 二分法 时间复杂度 logN

        i, j = 0, len(nums) - 1
        while i < j:
            m = (i + j) // 2
            if nums[m] > nums[j]: 
                i = m + 1
            elif nums[m] < nums[j]: 
                j = m
            else: 
                j -= 1 # j = m 对 [3,3,1,3]无效，[1,3,3,3] 
        return nums[i]
``` 

## 8. 搜索旋转排序数组
给你一个整数数组 nums ，和一个整数 target 。

该整数数组原本是按升序排列，但输入时在预先未知的某个点上进行了旋转。（例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ）。

请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。ps:nums 中的每个值都 独一无二, 1 <= nums.length <= 5000

33\. 搜索旋转排序数组（middle） [力扣](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/)

示例 1:

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

输入：nums = [1], target = 0
输出：-1


```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0: #bad case
            return -1

        left, right = 0, len(nums) - 1
        # ps 当nums中只有1个或者2个数据都不会进入循环
        while (left+1 < right):
            mid = left + (right -left) // 2
            if nums[mid] == target:
                return mid

            if (nums[left] < nums[mid]):# 代表left到mid是有序的
                #代表 target 在 [left, mid] 中
                #if nums[left] <= target and target <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid
                else:
                    left = mid
            else: # 代表mid到right是有序的
                ##代表 target 在 [mid, right] 中
                if nums[mid] <= target and target <= nums[right]:
                    left = mid
                else:
                    right = mid

        #循环结束条件2种:最后只剩2个数据，最后只剩1个数据
        if nums[left] == target:
            return left

        if nums[right] == target:
            return right

        return -1  

``` 
