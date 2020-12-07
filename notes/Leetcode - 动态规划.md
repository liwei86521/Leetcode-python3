# Leetcode 题解 - 动态规划
<!-- GFM-TOC -->
* [Leetcode - 动态规划](#leetcode---动态规划)
    * [斐波那契数列](#斐波那契数列)
        * [1. 爬楼梯](#1-爬楼梯)
        * [2. 跳水板](#2-跳水板)
        * [3. 最大连续1的个数](#3-最大连续1的个数)
        * [4. 打家劫舍](#4-打家劫舍)
        * [5. 打家劫舍 II](#5-打家劫舍-II)
    * [矩阵路径](#矩阵路径)
        * [1. 矩阵的最小路径和](#1-矩阵的最小路径和)
        * [2. 矩阵的总路径数](#2-矩阵的总路径数)
        
        
<!-- GFM-TOC -->


## 1. 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢

70\. 爬楼梯（easy） [力扣](https://leetcode-cn.com/problems/climbing-stairs/description/)

示例 1:

输入： 2
输出： 2

解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶

输入： 3
输出： 3

解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

```python
class Solution:  
    def climbStairs(self, n: int) -> int:
        #基本思路：状态转移方程：dp[i] = d[i-2]+dp[i-1]，先求出初始状态的dp[0]和dp[1]
        
        #DP动态规划解决（其实就是斐波那契数列）
        dp = [0] * (n+1) # dp[i] 表示爬到i层楼，有的不同方法数
        dp[0] = 1
        dp[1] = 1 # 初始状态
        for i in range(2, n+1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[-1]   #这里 dp[-1] == dp[n]
        
    """
    def climbStairs(self, n: int) -> int: # 推荐用这个
        if n <= 2:
            return n

        a = 1 
        b = 2
        for i in range(3, n+1):
            a, b = b, a+b # swap 
        return b
    """

``` 

## 2. 跳水板
你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。

返回的长度需要从小到大排列。

18\. 跳水板（easy） [力扣](https://leetcode-cn.com/problems/diving-board-lcci/description/)

示例 1:

输入：
shorter = 1
longer = 2
k = 3
输出： [3,4,5,6]

解释：
可以使用 3 次 shorter，得到结果 3；使用 2 次 shorter 和 1 次 longer，得到结果 4 。以此类推，得到最终结果。

```python
class Solution:  
    def climbStairs(self, n: int) -> int:
        #基本思路：状态转移方程：dp[i] = d[i-2]+dp[i-1]，先求出初始状态的dp[0]和dp[1]
        
        #DP动态规划解决（其实就是斐波那契数列）
        dp = [0] * (n+1) # dp[i] 表示爬到i层楼，有的不同方法数
        dp[0] = 1
        dp[1] = 1 # 初始状态
        for i in range(2, n+1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[-1]   #这里 dp[-1] == dp[n]
        
    """
    def climbStairs(self, n: int) -> int: # 推荐用这个
        if n <= 2:
            return n

        a = 1 
        b = 2
        for i in range(3, n+1):
            a, b = b, a+b # swap 
        return b
    """

``` 

## 3. 最大连续1的个数
给定一个二进制数组， 计算其中最大连续1的个数。ps 输入的数组只包含 0 和1

485\. 最大连续1的个数（easy） [力扣](https://leetcode-cn.com/problems/max-consecutive-ones/description/)

示例 1:

输入: [1,1,0,1,1,1]
输出: 3

解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        #典型dp 时间复杂度 O(n), 空间复杂度为 O(n)
        dp = [0] * len(nums)
        dp[0] = nums[0] #dp[i]表示以i结尾最长连续1的个数
        for i in range(1, len(nums)):
            if nums[i]==1:
                dp[i] = dp[i-1] + 1
                
            #else: # 可以省略
                #dp[i] = dp[i]

        return max(dp)
        
    '''
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        #典型dp 时间复杂度 O(n), 空间复杂度为 O(1)
        
        dp, max_count = 0, 0
        for num in nums:
            if num == 1:
                count += 1
            else:
                # Find the maximum till now.
                max_count = max(max_count, dp)
                # Reset dp of 1.
                dp = 0
        return max(max_count, dp)

    '''
 
 ```
 
 ## 4. 打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

198\. 打家劫舍（easy） [力扣](https://leetcode-cn.com/problems/house-robber/description/)

示例 1:

输入：[1,2,3,1]
输出：4

解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。


输入：[2,7,9,3,1]
输出：12

解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 动态规划
        n = len(nums)
        if n == 0:
            return 0
        
        #dp_rob[i] 第i个房子 抢获得最大金额
        dp_rob = [0] * len(nums) # DP 维护2个数组
        dp_no_rob = [0] * len(nums) # dp_no_rob[i] 第i个房子不抢获得最大金额

        dp_rob[0] = nums[0] # 初始值
        dp_no_rob[0] = 0 # 第一家不抢
        for i in range(1, len(nums)):
            dp_rob[i] = dp_no_rob[i-1]+nums[i] #第i家抢
            #[i房子不抢 ---> i-1房子 可以不抢，也可以抢
            dp_no_rob[i] = max(dp_rob[i - 1], dp_no_rob[i-1])

        # nums = [2,7,9,3,1]
        #print(dp_rob) # [2, 7, 11, 10, 12]
        #print(dp_no_rob) # [0, 2, 7, 11, 11]
        return max(dp_no_rob[-1], dp_rob[-1])
 
 ```
 
 ## 5. 打家劫舍 II
你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额

198\. 打家劫舍 II（middle） [力扣](https://leetcode-cn.com/problems/house-robber-ii/description/)

示例 1:

输入：nums = [2,3,2]
输出：3

解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。

输入：nums = [1,2,3,1]
输出：4

解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。

输入：nums = [0]
输出：0


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # rob_helper 是打家劫舍 I的代码
        def rob_helper(nums: List[int]) -> int:
            if not nums:return 0
            n = len(nums)

            dp_rob = [0] * n
            dp_no_rob = [0] * n
            dp_rob[0], dp_no_rob[0] = nums[0], 0
            # dp_rob[i] #代表 第i天rob
            for i in range(1, n):
                dp_rob[i] = dp_no_rob[i-1]+nums[i] # 第i家抢获得的最大值
                dp_no_rob[i] = max(dp_rob[i-1], dp_no_rob[i-1]) # 第i家不抢获得的最大值

            return max(dp_no_rob[-1],dp_rob[-1])

        if len(nums)==0:return 0
        if len(nums)==1:return nums[0]  # 答案是这样设置的
        # if len(nums)==1:return 0 # 个人觉得应该返回0

        # [1,2] --- 返回2
        # 因为是环形，故如果抢了第一家就不能抢最后一家，如果不抢第一家则可以抢最后一家
        return max(rob_helper(nums[0:-1]), rob_helper(nums[1:]))
 
 ```
