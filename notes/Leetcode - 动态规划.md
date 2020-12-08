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
        * [1. 最小路径和](#1-最小路径和)
        * [2. 不同路径](#2-不同路径)
        * [3. 不同路径 II](#2-不同路径-II)
        * [4. 三角形最小路径和](#2-三角形最小路径和)
        
        
<!-- GFM-TOC -->

## 斐波那契数列

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

## 矩阵路径
## 1. 最小路径和
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。 说明：每次只能向下或者向右移动一步。

64\. 最小路径和（middle） [力扣](https://leetcode-cn.com/problems/minimum-path-sum/description/)

示例 1:

<img style="width: 242px; height: 242px;" src="https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg" alt="">

输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7

解释：因为路径 1→3→1→1→1 的总和最小。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # 二维动态规划
        m = len(grid)
        if m < 1: return 0

        n = len(grid[0])
        if n < 1: return 0

        dp = [[0] * n for _ in range(m)] # ***
        dp[0][0] = grid[0][0]

        # 第一列和第一行 dp初始值设置
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]

        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n): # 状态转移方程
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

        return dp[m-1][n-1]

``` 

## 2. 不同路径
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？


62\. 不同路径（middle） [力扣](https://leetcode-cn.com/problems/unique-paths/description/)

示例 1:

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png">

例如，上图是一个7 x 3 的网格。有多少可能的路径？

输入: m = 7, n = 3     
输出: 28


```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 动态转移方程： dp[i][j] = dp[i-1][j] + dp[i][j-1]
        #到达当前位置[i, j]的路径数 = 从上边来的[i, j-1] + 从左边来的[i-1， j]
        
        #注意这里的初始值设定
        dp = [[1]*n for _ in range(m)] # 初始化dp二维数组
        
        for i in range(1, m):
            for j in range(1, n):
                #dp[i][j] 为后一个状态，dp[i-1][j]+dp[i][j-1]都以前一个状态
                dp[i][j] = dp[i-1][j]+dp[i][j-1] 
                
        return dp[m-1][n-1]

``` 

## 3. 不同路径 II
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

63\. 不同路径 II（middle） [力扣](https://leetcode-cn.com/problems/unique-paths-ii/description/)

示例 1:

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png">

网格中的障碍物和空位置分别用 1 和 0 来表示。

<img style="width: 242px; height: 242px;" src="https://assets.leetcode.com/uploads/2020/11/04/robot1.jpg" alt="">

输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2

解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # 典型 dp 问题
        rows = len(obstacleGrid)
        if rows < 1:return 0

        cols = len(obstacleGrid[0])
        if cols < 1:return 0

        if obstacleGrid[0][0] == 1:return 0
        #dp = [[1 for _ in range(cols)] for _ in range(rows)]
        dp = [[1]*cols for _ in range(rows)]

        #  第一列， 第一行 初始化
        for i in range(1, rows): # 必须从1开始
            if obstacleGrid[i][0] == 1:
                dp[i][0] = 0
            else:
                if dp[i-1][0] == 0: # 前面有障碍物
                    dp[i][0] = 0

        for j in range(1, cols): # 必须从1开始
            if obstacleGrid[0][j] == 1:
                dp[0][j] = 0
            else:
                if dp[0][j-1] == 0:
                    dp[0][j] = 0

        # 状态转移 dp[i][j] = dp[i-1][j] + dp[i][j-1] (前一位置转移而来)
        for row in range(1, rows):
            for col in range(1, cols):
                if obstacleGrid[row][col] == 0:
                    print(i, j, dp[i-1][j], dp[i][j-1])
                    dp[row][col] = dp[row-1][col] + dp[row][col-1]
                else:
                    dp[row][col] = 0

        return dp[rows-1][cols-1]

``` 

## 4. 三角形最小路径和
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

120\. 三角形最小路径和（middle） [力扣](https://leetcode-cn.com/problems/triangle/description/)

例如，给定三角形：
```html

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]

```

自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 倒着看
        # triangle[i, j] --> 当前元素的最短路径
        # triangle[i][j]=min(triangle[i+1][j],triangle[i+1][j+1])+triangle[i][j]
        if not triangle or len(triangle[0]) == 0: return 0
        rows = len(triangle)

        for row in range(rows-2,-1,-1):
            for col in range(len(triangle[row]) - 1, -1, -1):#保持美感
            #for col in range(len(triangle[row])): #也是OK的
                triangle[row][col] += min(triangle[row+1][col], triangle[row+1][col+1])

        return triangle[0][0]

``` 
