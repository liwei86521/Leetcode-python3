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
        * [0. 杨辉三角](#0-杨辉三角)
        * [1. 最小路径和](#1-最小路径和)
        * [2. 不同路径](#2-不同路径)
        * [3. 不同路径 II](#3-不同路径-II)
        * [4. 三角形最小路径和](#4-三角形最小路径和)
    * [分割整数](#分割整数)
        * [1. 完全平方数](#1-完全平方数)
        * [2. 解码方法](#2-解码方法)
        * [3. 丑数](#3-丑数)
    * [子序列问题](#子序列问题)
        * [1. 子数组最大平均数 I](#1-子数组最大平均数-I)
        * [2. 最大子序和](#2-最大子序和)
        * [3. 最长连续递增序列](#3-最长连续递增序列)
        * [4. 乘积最大子数组](#4-乘积最大子数组)
        * [5. 最长上升子序列](#5-最长上升子序列)
    * [股票交易](#股票交易)
        * [1. 买卖股票的最佳时机](#1-买卖股票的最佳时机)
        * [2. 买卖股票的最佳时机 II](#2-买卖股票的最佳时机-II)
        * [3. 买卖股票的最佳时机含手续费](#3-买卖股票的最佳时机含手续费)
    * [0-1背包问题](#0-1背包问题)
        * [1. 分割等和子集](#1-分割等和子集)
        * [2. 零钱兑换](#2-零钱兑换)
        * [3. 零钱兑换 II](#3-零钱兑换-II)
        * [4. 单词拆分](#4-单词拆分)
    * [字符串问题](#字符串问题)
        * [1. 最长回文子串](#1-最长回文子串)
        * [2. 回文子串](#1-回文子串)
<!-- GFM-TOC -->


<h2>动态规划原理</h2>
<ul>
<li><strong>基本思想：</strong>问题的最优解如果可以由子问题的最优解推导得到，则可以先求解子问题的最优解，再构造原问题的最优解；若子问题<strong>有较多的重复出现</strong>，则可以<strong>自底向上</strong>从最终子问题向原问题逐步求解。</li>
<li><strong>使用条件：<strong>可分为多个相关子问题，子问题的解被重复使用</strong></strong>
<ul>
<li>Optimal substructure（优化子结构）：
<ul>
<li>一个问题的优化解包含了子问题的优化解，反过来说就是，可以通过子问题的最优解，推导出问题的最优解。</li>
<li>缩小子问题集合，只需那些优化问题中包含的子问题，降低实现复杂性</li>
<li>我们可以自下而上的</li>
</ul>
</li>
<li>Subteties（重叠子问题）：在问题的求解过程中，很多子问题的解将被多次使用。</li>
</ul>
</li>
<li><strong>动态规划算法的设计步骤：</strong>
<ul>
<li>分析优化解的结构</li>
<li>递归地定义最优解的代价</li>
<li>自底向上地计算优化解的代价保存之，并获取构造最优解的信息</li>
<li>根据构造最优解的信息构造优化解</li>
</ul>
</li>
<li><strong>动态规划特点：</strong>
<ul>
<li>把原始问题划分成一系列子问题；</li>
<li>求解每个子问题仅一次，并将其结果保存在一个表中，以后用到时直接存取，不重复计算，节省计算时间</li>
<li>自底向上地计算。</li>
<li>整体问题最优解取决于子问题的最优解（状态转移方程）（将子问题称为状态，最终状态的求解归结为其他状态的求解）</li>
</ul>
</li>
</ul>
<p>&nbsp;</p>


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
## 0. 杨辉三角
给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。

<img src="https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif" alt="">

在杨辉三角中，每个数是它左上方和右上方的数的和。

118\. 杨辉三角（easy） [力扣](https://leetcode-cn.com/problems/pascals-triangle/description/)

示例 1:

```html

输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

```

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 0:
            return []

        # dp dp[i][j]=dp[i−1][j−1]+dp[i−1][j] 状态转移方程
        dp = [[1] * (i+1) for i in range(numRows)]
        for row in range(2, numRows):
            for j in range(1, row):
                dp[row][j] = dp[row-1][j-1] + dp[row-1][j]

        return dp

``` 

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

```html

   提示：
   m == obstacleGrid.length
   n == obstacleGrid[i].length
   1 <= m, n <= 100
   obstacleGrid[i][j] 为 0 或 1
```

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
        # 典型 dp 问题  提示了 1 <= rows, cols <= 100
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])

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

## 分割整数
## 1. 完全平方数
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

279\. 完全平方数（middle） [力扣](https://leetcode-cn.com/problems/perfect-squares/description/description/)

示例 1:

```html

输入: n = 12
输出: 3 
解释: 12 = 4 + 4 + 4.

输入: n = 13
输出: 2
解释: 13 = 4 + 9.

```

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # 典型的DP问题
        dp = [i for i in range(n + 1)] # 初始值设置，理解
        for i in range(2, n + 1):
            for j in range(1, int(i ** (0.5)) + 1):#range(1, i) 会超时
                # 这里就不用 if和break了，这样大大降低了运算时间
                dp[i] = min(dp[i], dp[i - j * j] + 1) # 重点理解

        return dp[n]

``` 


## 2. 解码方法
一条包含字母 A-Z 的消息通过以下方式进行了编码：

```html

'A' -> 1
'B' -> 2
...
'Z' -> 26

```

给定一个只包含数字的非空字符串，请计算解码方法的总数。

题目数据保证答案肯定是一个 32 位的整数。

91\. 解码方法（middle） [力扣](https://leetcode-cn.com/problems/decode-ways/description/)

示例 1:

```html

输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。

输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

输入：s = "0"
输出：0

输入：s = "1"
输出：1

输入：s = "2"
输出：1

```


```python
class Solution:
    def numDecodings(self, s: str) -> int:
        #给定一个只包含数字的 非空 字符串，请计算解码方法的总数
        #如果包含连续的0两个及以上,一定无解码方法,直接返回0
        #dp[i]表示长度为i的s的解码总数
        # 该题为带条件的 dp[i] = dp[i-2] + dp[i-1]， 画图

        n = len(s)
        if n==0 or s[0] =="0": # 注意是 s[0]  eg: "01"
            return 0
        
        dp =[0] * (n+1) # 要处理 "10"，"20" 
        dp[0], dp[1] = 1, 1 # 初始值设置
        
        # len(s)>=2 case
        for i in range(2, n + 1):
            tmp = int(s[i-2:i]) # eg：s[1:3] --> 有2位数
            if tmp == 0 : # 连续2个0 一定没有解码的方法
                return 0
            
            # 最后剩2位数字  则：dp[i] += dp[i-2]， 这样就有 f(n-2) 成立
            if 10 <= tmp <= 26: # tmp至少第一位数不为0，最后2位能一起翻译
                dp[i] += dp[i-2]
            
            # 最后剩1位数字，则 dp[i] += dp[i-1]，最后一位不为0 f(n-1) 成立
            #if s[i-1] != '0': # tmp第二位数不为0，最后1位能一起翻译
            if tmp % 10 != 0: # tmp第二位数不为0，最后1位能一起翻译
                dp[i] += dp[i-1]  # "12340"  --> 结果为 解码方法总数 == 0
                
        return dp[n] #dp[n] 即 dp[-1]

``` 

## 3. 丑数
我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
ps: 1 是丑数


119\. 丑数（middle） [力扣](https://leetcode-cn.com/problems/chou-shu-lcof/description/)

示例 1:

```html

输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

```

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 典型 DP 
        dp = [1] * n
        a, b, c = 0, 0, 0

        for i in range(1, n):
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(n2, n3, n5)

            # if dp[i] == n2: a += 1
            # elif dp[i] == n3: b += 1
            # else: c += 1 # ps 上面是错的，用下面的替代
            #因为 a和c可能同时 +1  5*2， 2*5

            if dp[i] == n2: a += 1
            if dp[i] == n3: b += 1
            if dp[i] == n5: c += 1

        return dp[-1]

``` 

## 子序列问题
## 1. 子数组最大平均数 I
给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。 1 <= k <= n <= 30,000

643\. 子数组最大平均数 I（easy） [力扣](https://leetcode-cn.com/problems/maximum-average-subarray-i/description/)

示例 1:

```html
输入: [1,12,-5,-6,50,3], k = 4
输出: 12.75
解释: 最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
```

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 条件 1 <= k <= n <= 30,000
        moving_sum = sum(nums[:k])

        res = moving_sum
        for j in range(k, len(nums)):
            moving_sum = moving_sum - nums[j-k] + nums[j]
            if moving_sum > res :
                res = moving_sum
                
        return  res/k

``` 

## 2. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

53\. 最大子序和（easy） [力扣](https://leetcode-cn.com/problems/maximum-subarray/description/)

示例 1:

```html
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 定义状态：dp[i] 表示以 nums[i] 结尾的连续子数组的最大和
        #状态转移方程：dp[i] = max{num[i], dp[i - 1] + num[i]}

        n = len(nums) # 时间复杂度：O(N)， 空间复杂度：O(N)
        if n == 1:
            return nums[0]

        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])

        # 最后不要忘记 求一遍最大值，或者在上面遍历的时候，就保存最大值
        return max(dp) #max(dp)

    """# 改进版 既然当前状态只与上一个状态有关，时间复杂度：O(N)，空间复杂度O(1)
    def maxSubArray(self, nums):
        ret = dp = nums[0]
        for i in range(1, len(nums)):
            dp = max(dp + nums[i], nums[i])
            ret = max(dp, ret)
        return ret
    
    """

``` 

## 3. 最长连续递增序列
给定一个未经排序的整数数组，找到最长且 **连续递增的子序列**，并返回该序列的长度。

674\. 最长连续递增序列（easy） [力扣](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/description/)

示例 1:

```html
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 

输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。

```

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        # 动态规划的思想 压缩空间复杂度为 O(1)
        if not nums: return 0
        # dp 表示以i结尾最长连续1的个数
        dp, res = 1, 1 # 注意初始值设置
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                dp = dp + 1
                res = max(dp, res)
            else:
                dp = 1 # 重置 dp

        return res

    '''
    def findLengthOfLCIS(self, nums: List[int]) -> int:

        if not nums: return 0
        # dp[i] 表示以i结尾最长连续1的个数
        dp = [1] * len(nums) # 注意初始值设置
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1] + 1

        return max(dp)

    '''

``` 

## 4. 乘积最大子数组

给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

152\. 乘积最大子数组（middle） [力扣](https://leetcode-cn.com/problems/maximum-product-subarray/description/)

示例 1:

```html
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:return 0
        
        # 维护2个dp 一个存最大值，一个存最小值
        dp_max = [0] * len(nums)
        dp_min = [0] * len(nums)
        dp_max[0] = dp_min[0] = nums[0]
        
        # 如果数组的数是负数，那么会导致最大的变最小的，最小的变最大的。
        # dp_max[i]表示以i位置结尾的最大乘积
        for i in range(1, len(nums)):
            dp_max[i] = max(dp_max[i - 1] * nums[i], dp_min[i - 1] * nums[i], nums[i])
            dp_min[i] = min(dp_min[i - 1] * nums[i], dp_max[i - 1] * nums[i], nums[i])
        
        return max(dp_max)

``` 

## 5. 最长上升子序列

给定一个无序的整数数组，找到其中最长上升子序列的长度
可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可

300\. 最长上升子序列（middle） [力扣](https://leetcode-cn.com/problems/longest-increasing-subsequence/description/)

示例 1:

```html
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。

```

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        #定义状态：dp[i] 表示以第 i 个数字为结尾的最长上升子序列的长度。即在 [0, ..., i] 的范围内，
        #选择 以数字 nums[i] 结尾 可以获得的最长上升子序列的长度
        """
        转移方程： 设 j∈[0,i)，考虑每轮计算新 dp[i]时，遍历 [0,i)列表区间，做以下判断：
        当 nums[i] > nums[j]时: nums[i]可以接在 nums[j] 之后（此题要求严格递增），此情况下最长上升子序列长度为 dp[j] + 1
        当 nums[i] <= nums[j]时: nums[i] 无法接在 nums[j] 之后，此情况上升子序列不成立，跳过。
        转移方程： dp[i] = max(dp[i], dp[j] + 1) for j in [0, i)
        """
        if not nums:return 0
            
        n =len(nums)
        dp = [1] * n  # 初始赋值
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    # dp[j]表示 nums[0…j] 中以nums[j] 结尾的最长上升子序列
                    dp[i] = max(dp[j] + 1, dp[i]) # 好巧妙， dp[i]可能会改变多次

        print('dp --> ',dp)
        return max(dp)
        # 为什么要max（dp[j]+1 ,dp[i]）,解释如下
        # [1,3,6,7,9,4,10,5,6] 对于4，如果不用max(dp[j]+1,dp[i]),那 
        # 么dp['4'] 等于3，当求10，dp['10']时候，10>4,则dp['10'] = 
        # dp['4']+1=4,所以最后max(dp),答案为5，而不是6

``` 

## 股票交易
## 1. 买卖股票的最佳时机

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

121\. 买卖股票的最佳时机（easy） [力扣](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/description/)

示例 1:

```html
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。


输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        # dp[i][0] 下标为 i 这天结束的时候，不持股，手上拥有的现金数
        # dp[i][1] 下标为 i 这天结束的时候，持股，手上拥有的现金数

        n = len(prices)
        dp = [[None, None] for _ in range(n)] # 二维 DP

        dp[0][0] = 0        # 第一天没有股票，说明没买没卖，获利为0
        dp[0][1] = -prices[0]   # 第一天持有股票，说明买入了，花掉一笔钱

        for i in range(1, n):
            # 前一天持有 或者 今天买进持有
            dp[i][1] = max(dp[i - 1][1], - prices[i])
            # 前一天卖掉 或者 今天卖掉
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])

        return dp[n - 1][0]


    """
    def maxProfit(prices: List[int]) -> int:
        if not prices:return 0
        #hold[i] #代表 第i天持有可获得的最大收益
        #sell[i] #代表 第i天卖掉可获得的最大收益
        n = len(prices)
        hold = [-float("inf")] * n
        sell = [0] * n
        hold[0], sell[0] = -prices[0], 0
        for i in range(1, n):
            # 前一天持有 或者 今天买进持有
            hold[i] = max(hold[i-1], -prices[i])
            # 前一天卖掉 或者 今天卖掉
            sell[i] = max(sell[i-1], prices[i]+hold[i-1])

        return sell[-1]
        
    def maxProfit(self, prices: List[int]) -> int:
        """ leetcode 超时 """
        res = 0
        n = len(prices)
        for i in range(1, n):
            for j in range(i):
                res = max(res, prices[i]-prices[j])

        return res
    """

``` 

## 2. 买卖股票的最佳时机 II

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）

122\. 买卖股票的最佳时机 II（easy） [力扣](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/description/)

示例 1:

```html
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。


输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        # dp[i][0] 下标为 i 这天结束的时候，不持股，手上拥有的现金数
        # dp[i][1] 下标为 i 这天结束的时候，持股，手上拥有的现金数
    
        n = len(prices)
        if n<=1: return 0
        dp = [[None, None] for _ in range(n)]
    
        dp[0][0] = 0        # 第一天没有股票，说明没买没卖，获利为0
        dp[0][1] = -prices[0]   # 第一天持有股票，说明买入了，花掉一笔钱
    
        for i in range(1, n):
            # 前一天卖掉 或者 今天卖掉
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
    
            # 前一天持有 或者 今天买进持有
            #dp[i][1] = max(dp[i - 1][1],  - prices[i]) # 这是只能一次交易
            dp[i][1] = max(dp[i - 1][1], dp[i-1][0] - prices[i]) # 允许多次交易
    
        return dp[-1][0]
    

    def maxProfit_v2(self, prices: List[int]) -> int:
        # dp_hold[i] 为 第i天持有可获得的最大收益
        # dp_cash[i] 为 第i天卖掉可获得的最大收益
        dp_hold = [float("-inf")] * len(prices)
        dp_cash = [0] * len(prices)

        # 初始值 设定
        dp_hold[0] = -prices[0]
        dp_cash[0] = 0

        for i in range(1, len(prices)):
            #第i天持有股票有2种情况 前一天持有 或者 今天买进持有
            dp_hold[i] = max(dp_hold[i-1], dp_cash[i-1]-prices[i])

            #第i天持有Cash有2种情况 前一天卖掉 或者 今天卖掉
            dp_cash[i] = max(dp_cash[i-1], dp_hold[i-1]+prices[i])

        return dp_cash[-1]
        
    """
    def maxProfit(self, prices: List[int]) -> int:
        # 典型贪心算法, dp
        if not prices: return 0
        n = len(prices)
        dp = [0] * n
        for i in range(1, n):
            dp[i] = dp[i-1] + max(0, prices[i]-prices[i-1])

        return dp[-1]

    """

``` 

## 3. 买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

122\. 买卖股票的最佳时机含手续费（middle） [力扣](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/)

示例 1:

```html
输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
输出: 8
解释: 能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

```

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        
        # dp[i][0] 下标为 i 这天结束的时候，不持股，手上拥有的现金数
        # dp[i][1] 下标为 i 这天结束的时候，持股，手上拥有的现金数
    
        n = len(prices)
        if n<=1: return 0
        dp = [[None, None] for _ in range(n)]
    
        dp[0][0] = 0        # 第一天没有股票，说明没买没卖，获利为0
        dp[0][1] = -prices[0] - fee  # 第一天持有股票，说明买入了，并且规定在买入股票的时候，扣除手续费
    
        for i in range(1, n):
            # 前一天卖掉 或者 今天卖掉 ---> cash
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]) #
    
            # 前一天持有 或者 今天买进持有 ---> hold
            dp[i][1] = max(dp[i - 1][1], dp[i-1][0] - prices[i] - fee) # 允许多次交易
    
        return dp[-1][0]
    
    """
    def maxProfit(prices: List[int], fee) -> int:
        n = len(prices)
        if n < 2: return 0

        # dp[i][0] 第i天 不持有股份的最大收益
        # dp[i][1] 第i天 持有股份的最大收益
        dp =[ [None, None] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, n):
            # 定义卖出时 --> 扣手续费
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i]-fee)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])

        #print(dp)
        return dp[-1][0]

    def maxProfit(self, prices: List[int], fee: int) -> int:
        #条件:你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了

        cash, hold = 0, -prices[0] #初始值设定
        #cash 表示在当前没有持有股票的状态的获取最大利润值
        #hold 表示在当前持有股票的状态的获取最大利润值
        for i in range(1, len(prices)):
            tmp_cash = cash # 临时变量，tmp_cash 代表前一天的cash的最大值
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, tmp_cash - prices[i]) #买入不要fee

        return cash
    """

``` 

## 0-1背包问题
三种背包问题:
1.组合问题公式
    dp[i] += dp[i-num]
    
377\. 组合总和 Ⅳ  [组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/description/)
    
518\. 零钱兑换 II  [零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/description/)

2.True、False问题公式
    dp[i] = dp[i] or dp[i-num]
    
139\. 单词拆分  [组合总和 Ⅳ](https://leetcode-cn.com/problems/word-break/description/)
    
416\. 分割等和子集  [组合总和 Ⅳ](https://leetcode-cn.com/problems/partition-equal-subset-sum/description/)

3.最大最小问题公式
    dp[i] = min(dp[i], dp[i-num]+1)或者dp[i] = max(dp[i], dp[i-num]+1)
    
322\. 零钱兑换  [零钱兑换](https://leetcode-cn.com/problems/coin-change/description/)
    
474\. 一和零  [一和零](https://leetcode-cn.com/problems/ones-and-zeroes/description/)  

当然拿到问题后，需要做到以下几个步骤：
1.分析是否为背包问题。
2.是以上三种背包问题中的哪一种。
3.是0-1背包问题还是完全背包问题。即nums数组中的元素是否可以重复使用。
4.如果是组合问题，是否需要考虑元素之间的顺序。需要考虑顺序有顺序的解法，不需要考虑顺序又有对应的解法。

背包问题的判定：
背包问题具备的特征：给定一个target，target可以是数字也可以是字符串，再给定一个数组nums，nums中装的可能是数字，
也可能是字符串，问：能否使用nums中的元素做各种排列组合得到target

背包问题技巧：
1.如果是0-1背包，即数组中的元素不可重复使用，nums放在外循环，target在内循环，且内循环倒序；
```python
for num in nums:
    for i in range(target, nums-1, -1):
```

2.如果是完全背包，即数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环正序。
```python
for num in nums:
    for i in range(nums, target+1):
```

3.如果组合问题需考虑元素之间的顺序，需将target放在外循环，将nums放在内循环。
```python
for i in range(1, target+1):
    for num in nums:
```


0-1背包问题:有N件物品和一个容量为V的背包,第i件物品的体积是vi，价值是wi,求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量量且总价值最大。
作为「0-1 背包问题」，它的特点是：「每个数只能用一次」。解决的基本思路是：物品一个一个选，容量也一点一点增加去考虑，这一点是「动态规划」的思想，特别重要。
![avatar](https://github.com/liwei86521/Leetcode-python3/blob/main/pics/bag_01.png?raw=true)

![avatar2](https://github.com/liwei86521/Leetcode-python3/blob/main/pics/bag_02.png?raw=true)

```python
def knapsack_01(weight, value, c):
    """
    测试数据：
    c = 8 书包能承受的重量， capacity
    weight = [2, 3, 4, 5] 每个物品的重量，共 4个物品
    value = [3, 4, 5, 6] 每个物品的价值
    """
    n = len(weight)
    dp = [[0 for j in range(c + 1)] for i in range(n + 1)]
    #dp[i][j]: 当前背包容量 j, 前 i 个物品最佳组合可获得得最大价值

    #ps 为了方便下面的操作
    weight = [0] + weight
    value = [0] + value

    #初始状态设置  dp(0,j)=V(i,0)=0  已经在dp创建的时候处理好了
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            if j < weight[i]: # 装不下第i个物体
                dp[i][j] = dp[i - 1][j]
            else:
                # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    #print(dp)
    return dp[-1][-1]


def knapsack_01_v2(weight, value, c):
    """
    测试数据：
    c = 8 书包能承受的重量， capacity
    weight = [2, 3, 4, 5] 每个物品的重量，共 4个物品
    value = [3, 4, 5, 6] 每个物品的价值
    """
    n = len(weight)
    dp = [[0 for j in range(c + 1)] for i in range(n + 1)]
    # dp[i][j]: 当前背包容量 j, 前 i 个物品最佳组合可获得得最大价值

    # 初始状态设置  dp(0,j)=V(i,0)=0  已经在dp创建的时候处理好了

    for i in range(1, n + 1):
        for j in range(1, c + 1):
            if j < weight[i - 1]:  # 装不下第i个物体 数组的索引是从0开始的
                dp[i][j] = dp[i - 1][j]
            else:
                # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value[i - 1])

    print(dp)
    return dp[-1][-1]

def knapsack_01_v3(weight, value, c):
    """
    测试数据：
    c = 8 书包能承受的重量， capacity
    weight = [2, 3, 4, 5] 每个物品的重量，共 4个物品
    value = [3, 4, 5, 6] 每个物品的价值
    """
    n = len(weight)
    dp = [[0 for j in range(c + 1)] for i in range(n)]
    # dp[i][j]: 当前背包容量 j, 前 i 个物品最佳组合可获得得最大价值

    # 初始状态设置 weight[0] 代表第一个物体， 这里已经处理好了第一个物体了
    for j in range(c + 1):  # 初始化第一行
        if (weight[0] <= j):
            dp[0][j] = value[0]

    for i in range(1, n):
        for j in range(1, c + 1):
            if j < weight[i]:  # 装不下第i个物体
                dp[i][j] = dp[i - 1][j]
            else:
                # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    #print(dp)
    return dp[-1][-1]

``` 

**空间优化**  

在程序实现时可以对 0-1 背包做优化。观察状态转移方程可以知道，前 i 件物品的状态仅与前 i-1 件物品的状态有关，因此可以将 dp 定义为一维数组，其中 dp[j] 既可以表示 dp[i-1][j] 也可以表示 dp[i][j]。此时，

<!--<div align="center"><img src="https://latex.codecogs.com/gif.latex?dp[j]=max(dp[j],dp[j-w]+v)" class="mathjax-pic"/></div> <br>-->

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/9ae89f16-7905-4a6f-88a2-874b4cac91f4.jpg" width="300px"> </div><br>

因为 dp[j-w] 表示 dp[i-1][j-w]，因此不能先求 dp[i][j-w]，防止将 dp[i-1][j-w] 覆盖。也就是说要先计算 dp[i][j] 再计算 dp[i][j-w]，在程序实现时需要按倒序来循环求解。

```python
def knapsack_01_space(weight, value, c):
    """
    测试数据：
    c = 8 书包能承受的重量， capacity
    weight = [2, 3, 4, 5] 每个物品的重量，共 4个物品
    value = [3, 4, 5, 6] 每个物品的价值
    """
    n = len(weight)
    #对dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])  空间优化
    # dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    # 其中 dp[j] 既可以表示 dp[i-1][j] 也可以表示 dp[i][j]
    dp = [0] * (c +1)

    for i in range(0, n):
        for j in range(c, 0, -1): # 防止覆盖，因为每个物品只能用一次，以免之前的结果影响其他的结果
            if j < weight[i]:  # 装不下第i个物体 数组索引从0开始
                pass
            else:
                # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp) # [0, 0, 3, 4, 5, 7, 8, 9, 10]
    return dp[-1]
``` 

## 1. 分割等和子集

给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等
ps: 每个元素只能使用一次

122\. 分割等和子集（middle） [力扣](https://leetcode-cn.com/problems/partition-equal-subset-sum/description/)

示例 1:

```html
输入: [1, 5, 11, 5]
输出: true

解释: 数组可以分割成 [1, 5, 5] 和 [11].

输入: [1, 2, 3, 5]
输出: false

解释: 数组不能分割成两个元素和相等的子集.

```

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2:
            return False

        total = sum(nums)
        maxNum = max(nums)

        if total & 1: # 等效于 if total % 2 表示为奇数时
            return False

        target = total // 2
        if maxNum > target:
            return False

        #dp[i][j]表示考虑下标[0, i]这个双闭区间里的所有整数，在它们当中是否能够选出一些数，使得这些数之和恰好为整数j
        #dp[i][j] = x 表示，对于前 i 个物品，当前背包的容量为 j 时，若 x 为 true，
        # 则说明可以恰好将背包装满，若 x 为 false，则说明不能恰好将背包装满
        dp = [[False] * (target + 1) for _ in range(n+1)]

        # 设置初始条件, 背包没有空间的时候，就相当于装满了
        for i in range(n+1):
            dp[i][0] = True

        for i in range(1, n+1):
            for j in range(1, target + 1):
                if j < nums[i-1]: # 容量不够, 数组索引从0开始
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j-nums[i-1]] or dp[i-1][j]
        #print(dp)
        return dp[n][target] # dp[-1][-1]
        

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2: return False

        total = sum(nums)
        maxNum = max(nums)

        if total & 1: # 等效于 if total % 2 表示为奇数时
            return False

        target = total // 2
        if maxNum > target:
            return False

        #dp[i][j] = dp[i-1][j-nums[i-1]] or dp[i-1][j]
        #发现 dp[i][j] 都是通过上一行 dp[i-1][..] 转移过来的，我们进行状态压缩，将二维 dp 数组压缩为一维，节约空间复杂度
        #ps:唯一需要注意的是 j 应该从 后往前反向遍历，因为每个物品（或者说数字）只能用一次，以免之前的结果影响其他的结果
        dp = [False] * (target + 1)

        # 设置初始条件, 背包没有空间的时候，就相当于装满了
        dp[0] = True

        for i in range(0, n):
            for j in range(target, 0, -1): # j 应该从 后往前反向遍历，因为每个物品（或者说数字）只能用一次
                if j >= nums[i]: # 容量够, 数组索引从0开始
                    dp[j] = dp[j] or dp[j - nums[i]]

        print(dp) # [True, True, False, False, False, True, True, False, False, False, True, True]
        return dp[target]

``` 



## 2. 零钱兑换

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

322\. 零钱兑换（middle） [力扣](https://leetcode-cn.com/problems/coin-change/description/)

示例 1:

```html
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1

输入：coins = [2], amount = 3
输出：-1

输入：coins = [1], amount = 0
输出：0

输入：coins = [1], amount = 1
输出：1

输入：coins = [1], amount = 2
输出：2

```

![image](https://github.com/liwei86521/Leetcode-python3/blob/main/pics/bag_03.png?raw=true)

![image](https://github.com/liwei86521/Leetcode-python3/blob/main/pics/bag_04.png?raw=true)

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 完全背包 问题
        n = len(coins)
        dp = [[float("inf")]*(amount+1) for _ in range(n+1)]
        #dp[i][j] 表示考虑物品区间 [0, i] 里，可重复 容量为j 的最少硬币个数
        for i in range(n+1):
            dp[i][0] = 0

        for i in range(1, n+1):
            for j in range(1, amount+1):
                if j >= coins[i-1]: #能装下， 注意下面的状态转移方程与01背包的区别
                    dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]]+1)
                else:
                    dp[i][j] = dp[i-1][j]

        #print(dp)
        return dp[-1][-1] if dp[-1][-1] != float("inf") else -1


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 完全背包 问题
        n = len(coins)

        # dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]]+1)
        # 发现前面的 dp 数组的转移只和 dp[i][..] 和 dp[i-1][..]
        dp = [float("inf")]*(amount+1) # 进行状态压缩
        # 初始值
        dp[0] = 0 

        for i in range(0, n):
            for j in range(1, amount + 1):
                if j >= coins[i]: #能装下，因为可以多次使用，所以不用倒序
                    dp[j] = min(dp[j], dp[j-coins[i]]+1)

        # for i in range(amount + 1): # 和上面是等效的
        #     for coin in coins:
        #         if (i >= coin):  # 能够装下 coin
        #             # i>=coin时 选择拿coin硬币 这个时候 硬币数 = dp[i - coin] + 1
        #             dp[i] = min(dp[i], dp[i - coin] + 1)

        print(dp) # [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3] # 时间复杂度：O(amonut * len(coins))
        return dp[-1] if dp[-1] != float("inf") else -1

```

## 3. 零钱兑换 II

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

518\. 零钱兑换（middle） [力扣](https://leetcode-cn.com/problems/coin-change-2/description/)

示例 1:

```html
输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

输入: amount = 3, coins = [2]
输出: 0
解释: 只用面额2的硬币不能凑成总金额3。

输入: amount = 10, coins = [10] 
输出: 1

输入: amount = 0, coins = [10] 
输出: 1

```

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        #完全背包问题
        n = len(coins)
        #dp[i][j], 硬币列表的前缀子区间 [0, i] 能够凑成总金额为 j 的组合数
        dp = [[0]*(amount+1) for _ in range(n+1)]

        #初始条件 dp[i][0] = 1  都不选即可满足条件
        for i in range(n+1):
            dp[i][0] = 1

        for i in range(1, n+1):
            for j in range(1, amount+1):
                if j >= coins[i-1]: #能够装下第i个硬币
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
                else:
                    dp[i][j] = dp[i - 1][j]

        #print(dp)
        return dp[-1][-1]


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        #完全背包问题
        n = len(coins)

        # dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
        # 发现前面的 dp 数组的转移只和 dp[i][..] 和 dp[i-1][..]
        dp = [0]*(amount+1)

        #初始条件 dp[0] = 1  都不选即可满足条件
        dp[0] = 1

        for i in range(0, n): # 跟顺序不相关
            for j in range(1, amount+1):
                if j >= coins[i]: #能够装下第i个硬币
                    dp[j] = dp[j] + dp[j-coins[i]]
                # else:
                #     dp[j] = [j]


        # for j in range(1, amount+1): # 跟顺序相关
        #     for i in range(0, n):
        #         if j >= coins[i]: #能够装下第i个硬币
        #             dp[j] = dp[j] + dp[j-coins[i]]
        #         # else:
        #         #     dp[j] = [j]

        #amount = 5, coins = [1, 2, 5]
        print(dp) # [1, 1, 2, 2, 3, 4]
        return dp[-1]

``` 


## 4. 单词拆分

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词 

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词

139\. 单词拆分（middle） [力扣](https://leetcode-cn.com/problems/word-break/description/)

示例 1:

```html
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。


输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 动态规划， 感觉还有好好理解一下
        n = len(s)

        # 状态：dp[i] 表示子串 s[0:i + 1] （即长度为 i 的子串，其实就是前缀）可以被空格拆分
        dp = [False] * (n + 1)

        dp[0] = True # 这个初始状态的设置

        for i in range(1, n + 1):
            # j 其实就是前缀的长度，注意这个细节
            for j in range(0, i):
                if s[j:i] in wordDict and dp[j]:
                    dp[i] = True # 确认了 dp[i] = True 下面的可以直接跳出，减少运算次数
                    # j 无须再遍历下去，退出当前循环，进行最上面的for
                    break

        # s = "leetcode", wordDict = ["leet", "code"]  
        #print('dp --> ',dp)# [True, False, False, False, True, False, False, False, True]
        return dp[n]


# 套模板
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False]*(n+1)
        dp[0] = True

        for i in range(1, n+1):
            for word in wordDict:
                length = len(word)
                if(length<=i and word==s[i-length:i]):
                    dp[i] = dp[i] or dp[i-length]
                    # 其实 就是下面的意思 or 前面的 dp[i] 永远起不了作用
                    #dp[i] = dp[i-length]

        print(dp)
        return dp[-1]

``` 

##字符串问题
## 1. 最长回文子串

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

5\. 最长回文子串（middle） [力扣](https://leetcode-cn.com/problems/longest-palindromic-substring/description/)

示例 1:

```html
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。

输入: "cbbd"
输出: "bb"

输入: "babba"
输出: "abba"

```

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n <= 1: return s

        # 状态 dp[i][j] 表示子串 s[i:j+1]是否为回文子串
        # 状态转移方程 dp[i][j] = (s[i] == s[j]) and (dp[i+1][j-1])
        # dp[i+1][j-1]边界条件前提是 j-1-(i+1)+1 < 2  --->   j - i < 3
        dp = [[False]*n for _ in range(n)] # 二维dp

        for i in range(n): # 初始化，单个字符也是回文
            dp[i][i] = True

        longest_l = 0  # 用于判断是否更新res
        res = ""  # 作为返回值进行return

        for j in range(1, n):
            for i in range(0, j):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 在头尾字符相等的前提下，如果收缩以后不构成区间（最多只有 1 个元素），直接返回 True
                # 否则要继续看收缩以后的区间的回文性  重点理解 or 的短路性质在这里的作用
                # #查看 dp[l+1,r-1] 收缩以后的区间的回文性 eg:s="bababd"
                if s[i] == s[j] and (j-i <= 2 or dp[i+1][j-1]):
                    dp[i][j] = True
                
                # if s[i] == s[j]: # 下面7行是对上面2行的理解
                #     if j-i <= 2:
                #         dp[i][j] = True
                #     else:
                #         dp[i][j] = dp[i+1][j-1]
                # else:
                #     dp[i][j] = False
                
                    cur_len = j - i + 1
                    if cur_len > longest_l:
                        longest_l = cur_len
                        res = s[i:j+1]

        return res
        
    def longestPalindrome_v2(self, s: str) -> str:
        n = len(s)
        if n <= 1: return s

        # 状态 dp[i][j] 表示子串 s[i:j+1]是否为回文子串
        # 状态转移方程 dp[i][j] = (s[i] == s[j]) and (dp[i+1][j-1])
        # dp[i+1][j-1]边界条件前提是 j-1-(i+1)+1 < 2  --->   j - i < 3
        dp = [[False] * n for _ in range(n)]  # 二维dp

        for i in range(n):  # 初始化，单个字符也是回文
            dp[i][i] = True

        longest_l = 1  # 用于判断是否更新res  初始值不能设为0 eg: "ab"
        res = s[0]  # 作为返回值进行return

        for j in range(1, n):
            for i in range(0, j):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 在头尾字符相等的前提下，如果收缩以后不构成区间（最多只有 1 个元素），直接返回 True
                # 否则要继续看收缩以后的区间的回文性  重点理解 or 的短路性质在这里的作用
                # #查看 dp[l+1,r-1] 收缩以后的区间的回文性 eg:s="bababd"
                # if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                #     dp[i][j] = True

                if s[i] == s[j]: # 下面7行是对上面2行的理解
                    if j-i <= 2:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]

                # 如果 dp[i][j] 表示子串 s[i:j+1]是否为回文子串 回文的话
                if dp[i][j] and j - i + 1 > longest_l:
                    longest_l = j - i + 1
                    res = s[i:j + 1]

        return res

``` 

## 2. 回文子串

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

647\. 回文子串 [力扣](https://leetcode-cn.com/problems/palindromic-substrings/description/)

示例 1:

```html
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"

输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"

```

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        if n <= 1: return n

        # 状态 dp[i][j] 表示子串 s[i:j+1]是否为回文子串
        # 状态转移方程 dp[i][j] = (s[i] == s[j]) and (dp[i+1][j-1])
        # dp[i+1][j-1]边界条件前提是 j-1-(i+1)+1 < 2  --->   j - i < 3
        dp = [[False]*n for _ in range(n)] # 二维dp

        for i in range(n): # 初始化，单个字符也是回文
            dp[i][i] = True

        count = n #  单个字符都算回文, 下面回文的长度至少为2
        for j in range(1, n):
            for i in range(0, j):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 在头尾字符相等的前提下，如果收缩以后不构成区间（最多只有 1 个元素），直接返回 True
                # 否则要继续看收缩以后的区间的回文性  重点理解 or 的短路性质在这里的作用
                # #查看 dp[l+1,r-1] 收缩以后的区间的回文性 eg:s="bababd"
                if s[i] == s[j] and (j-i <= 2 or dp[i+1][j-1]):
                    dp[i][j] = True
                    
                # if s[i] == s[j]: # 下面7行是对上面2行的理解
                #     if j-i <= 2:
                #         dp[i][j] = True
                #     else:
                #         dp[i][j] = dp[i+1][j-1]
                # else:
                #     dp[i][j] = False
                
                    count += 1

        return count
``` 
