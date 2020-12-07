<!-- GFM-TOC -->
* [Leetcode - 贪心](#leetcode-题解---贪心)
    * [1. 分发饼干](#1-分发饼干)
    * [2. 剪绳子](#2-剪绳子)
    * [3. 两地调度](#3-两地调度)
    * [4. 零钱兑换](#4-零钱兑换)
    * [5. 买卖股票的最佳时机 II](#5-买卖股票的最佳时机-II)
<!-- GFM-TOC -->

## 1. 分发饼干
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

455\. 分发饼干（easy） [力扣](https://leetcode-cn.com/problems/assign-cookies/description/)

示例 1:

输入: g = [1,2,3], s = [1,1]
输出: 1

解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。

输入: g = [1,2], s = [1,2,3]
输出: 2

解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        #典型 贪心算法 总是考虑当前最有策略（局部最优解）
        #贪心策略适用的前提是：局部最优策略能导致产生全局最优解
        g.sort() # 孩子数组
        s.sort() # 饼干数组
        i, j, res = 0, 0, 0
        while i<len(g) and j<len(s):
            if s[j] >= g[i]:
                res += 1
                i += 1  # 因为i孩子饱了，下次给下一个孩子进行分配

            j += 1
        
        return res

``` 

## 2. 剪绳子
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

124\. 剪绳子（middle） [力扣](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/description/)

示例 1:

输入: 2
输出: 1

解释: 2 = 1 + 1, 1 × 1 = 1

输入: 10
输出: 36

解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        #典型 贪心算法 总是考虑当前最有策略（局部最优解）
        #贪心策略适用的前提是：局部最优策略能导致产生全局最优解
        g.sort() # 孩子数组
        s.sort() # 饼干数组
        i, j, res = 0, 0, 0
        while i<len(g) and j<len(s):
            if s[j] >= g[i]:
                res += 1
                i += 1  # 因为i孩子饱了，下次给下一个孩子进行分配

            j += 1
        
        return res

``` 

## 3. 两地调度
公司计划面试 2N 人。第 i 人飞往 A 市的费用为 costs[i][0]，飞往 B 市的费用为 costs[i][1]。
返回将每个人都飞到某座城市的最低费用，要求每个城市都有 N 人抵达。

1029\. 两地调度（middle） [力扣](https://leetcode-cn.com/problems/two-city-scheduling/description/)

示例 1:

输入：[[10,20],[30,200],[400,50],[30,20]]
输出：110
解释：
第一个人去 A 市，费用为 10。
第二个人去 A 市，费用为 30。
第三个人去 B 市，费用为 50。
第四个人去 B 市，费用为 20。

最低总费用为 10 + 30 + 50 + 20 = 110，每个城市都有一半的人在面试。


```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        """
        在对问题求解时，总是做出在当前看来是最好的选择
        • 基本思路：
            • 建立数学模型来描述问题。
            • 把求解的问题分成若干个子问题。
            • 对每一子问题求解，得到子问题的局部最优解。
            • 把子问题的解局部最优解合成原来解问题的一个解。
            • 贪心策略适用的前提是：局部最优策略能导致产生全局最优解
        """

        # 注意一定是 一半人去 A地， 一半人去 B地
        costs.sort(key=lambda x: (x[0] - x[1]))  # 计算去A地和去B低的费用差，然后按照费用差排序
        n = len(costs) # 长度
        result = 0
        result += sum([i[0] for i in costs[:n//2]])  # 前半部分去A地
        result += sum([i[1] for i in costs[n//2:]])  # 后半部分去B地

        return result

``` 

## 4. 零钱兑换
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

322\. 零钱兑换（middle） [力扣](https://leetcode-cn.com/problems/coin-change/description/)

示例 1:

输入：coins = [1, 2, 5], amount = 11
输出：3 

解释：11 = 5 + 5 + 1

输入：coins = [2], amount = 3
输出：-1

输入：coins = [1], amount = 0
输出：0

输入：coins = [1], amount = 2
输出：2

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        #dp[i] 表示凑成总金额i所需的最少的硬币个数
        dp = [float("inf")]*(amount+1)
        dp[0] = 0 #初始值设置

        for i in range(amount+1):
            for coin in coins: # 这里做一个循环
                if (i>=coin):
                    #i>=coin时 选择拿coin硬币 这个时候 硬币数 = dp[i - coin] + 1
                    dp[i] = min(dp[i], dp[i-coin]+1)

                #else: # i < coin #选择 不拿  这个时候， 硬币数 = dp[i]
                    #dp[i] = dp[i]

        return dp[-1] if (dp[-1] != float("inf")) else -1

``` 

## 5. 买卖股票的最佳时机 II
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）

122\. 买卖股票的最佳时机 II（easy） [力扣](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/description/)

示例 1:

输入: [7,1,5,3,6,4]
输出: 7

解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 典型贪心算法, dp
        if not prices: return 0

        n = len(prices)
        dp = [0] * n
        for i in range(1, n):
            dp[i] = dp[i-1] + max(0, prices[i]-prices[i-1])

        return dp[-1]

``` 
