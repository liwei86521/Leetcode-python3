# Leetcode 题解 - 位运算
<!-- GFM-TOC -->
* [Leetcode - 位运算](#leetcode---位运算)
    * [0. 原理](#0-原理)
    * [1. 只出现一次的数字](#1-只出现一次的数字)
    * [2. 找不同](#2-找不同)
    * [3. 汉明距离](#3-汉明距离)
    * [4. 位1的个数](#4-位1的个数)
    * [5. 数字的补数](#5-数字的补数)
    * [6. 交替位二进制数](#6-交替位二进制数)
    * [7. 比特位计数](#7-比特位计数)
    * [8.  数字范围按位与](#8--数字范围按位与)
    * [9. 交换数字](#9-交换数字)
    * [10. 字母大小写全排列](#10-字母大小写全排列)
    * [11. 整数转换](#11-整数转换)
    * [12. 只出现一次的数字 III](#12-只出现一次的数字-III)
<!-- GFM-TOC -->


## 0. 原理

**基本原理** 

```html
&：按位与操作，只有 1 &1 为1，其他情况为0。可用于进位运算。

|：按位或操作，只有 0|0为0，其他情况为1。

~：逐位取反。

^：异或，相同为0，相异为1。可用于加操作（不包括进位项）。

<<：左移操作，2的幂相关

>>：右移操作，2的幂相关

5的取反(翻转)为-6。那么为什么会是-6呢 ?

首先要明确几点（原码、反码、补码）：

1.正数：
       原码=反码=补码
2.负数
       反码：符号位不变，其他位取反
       补码 = 反码 + 1
3.负数补码转换为原码的规则：
       原码=补码的符号位不变，其他位取反，再加1
       
一个十进制的5，用一个字节的二进制表示为：0000 0101，因为5是正数，因此，原码=反码=补码，
现代计算机都是 使用二进制**补码**进行运算，对5的补码进行取反操作：
  得到：1111 1010（结果即为对5进行了取反之后的补码）
  
得到补码之后，接下来只需转换为人能识别的原码即可。符号位不变，
其他位取反得到：1000 0101，然后加1，得到原码即为：

1000 0101
+       1
-----------------------------
1000 0110
（其中，第一位为符号位，后面用二进制进行表示为6）因此，得到的结果为-6。
即 ~5 = -6


n = -7  # 针对的都是 有符号数 最高位为符合为
print( ~n) # 6
print(-n == (~n +1)) # True

```


0s 表示一串 0，1s 表示一串 1。

```
x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x = 0       x & x = x       x | x = x
```

利用 x ^ 1s = \~x 的特点，可以将一个数的位级表示翻转；利用 x ^ x = 0 的特点，可以将三个数中重复的两个数去除，只留下另一个数。

```
1^1^2 = 2
```

利用 x & 0s = 0 和 x & 1s = x 的特点，可以实现掩码操作。一个数 num 与 mask：00111100 进行位与操作，只保留 num 中与 mask 的 1 部分相对应的位。

```
01011011 &
00111100
--------
00011000
```

利用 x | 0s = x 和 x | 1s = 1s 的特点，可以实现设值操作。一个数 num 与 mask：00111100 进行位或操作，将 num 中与 mask 的 1 部分相对应的位都设置为 1。

```
01011011 |
00111100
--------
01111111
```

**位与运算技巧** 

n&(n-1) 去除 n 的位级表示中最低的那一位 1。例如对于二进制表示 01011011，减去 1 得到 01011010，这两个数相与得到 01011010。

```
01011011 &
01011010
--------
01011010
```

n&(-n) 得到 n 的位级表示中最低的那一位 1。-n 得到 n 的反码加 1，也就是 -n=\~n+1。例如对于二进制表示 10110100，-n 得到 01001100，相与得到 00000100。

```
10110100 &
01001100
--------
00000100
```

n-(n&(-n)) 则可以去除 n 的位级表示中最低的那一位 1，和 n&(n-1) 效果一样。

**移位运算** 

\>\> n 为算术右移，相当于除以 2**n，例如 -7 >> 2 = -2。 print(-7 >> 1) # -4      print(-7 >> 2) # -2


```
11111111111111111111111111111001  >> 2
--------
11111111111111111111111111111110
```

<< n 为算术左移，相当于乘以 2n。  -7 << 2 = -28。

```
11111111111111111111111111111001  << 2
--------
11111111111111111111111111100100
```

**mask 计算** 

要获取 111111111，将 0 取反即可，\~0。

要得到只有第 i 位为 1 的 mask，将 1 向左移动 i-1 位即可，1<<(i-1) 。例如 1<<4 得到只有第 5 位为 1 的 mask ：00010000。

要得到 1 到 i 位为 1 的 mask，(1<<i)-1 即可，例如将 (1<<4)-1 = 00010000-1 = 00001111。

要得到 1 到 i 位为 0 的 mask，只需将 1 到 i 位为 1 的 mask 取反，即 ~((1<<i)-1)。

## 1. 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。


136\. 只出现一次的数字（easy） [力扣](https://leetcode-cn.com/problems/single-number/description/)

示例 1:

```html
输入: [2,2,1]
输出: 1

输入: [4,1,2,1,2]
输出: 4

```

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0 # 与零一直异或就可以了
        for i in nums:
            res = res ^ i
        
        return res
        
``` 

## 2. 找不同

给定两个字符串 s 和 t，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。

389\. 找不同（easy） [力扣](https://leetcode-cn.com/problems/find-the-difference/description/)

示例 1:

```html
输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。

输入：s = "", t = "y"
输出："y"

输入：s = "a", t = "aa"
输出："a"

输入：s = "ae", t = "aea"
输出："a"

```

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        ret = 0  # python3 异或方法 推荐
        for c in s + t:
            ret = ret ^ ord(c) #异或只针对 int类型
        return chr(ret)
        
``` 

## 3. 汉明距离

两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。

给出两个整数 x 和 y，计算它们之间的汉明距离。

注意：
0 ≤ x, y < 2**31.

461\. 汉明距离（easy） [力扣](https://leetcode-cn.com/problems/hamming-distance/description/)

示例 1:

```html
输入: x = 1, y = 4

输出: 2

解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

上面的箭头指出了对应二进制位不同的位置。

```

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        
        # 两个二进制中 位数不同的个数，--->其实就是求 2个数异或后 二进制的 1的个数
        res = x ^ y
        
        distance = 0
        while res:
            
            if res & 1:
                distance += 1

            res = res >> 1
        return distance

print(Solution().hammingDistance(28, 3)) # 5
        
``` 

## 4. 位1的个数

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）

注意： 输入必须是长度为 32 的 二进制串

191\. 位1的个数（easy） [力扣](https://leetcode-cn.com/problems/number-of-1-bits/description/)

示例 1:

```html
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。

输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。

输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。

```

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        #时间复杂度：O(1)，n为32位的数，操作次数为二进制的位数
        res=0
        while n:
            res+=n&1
            n>>=1
        return res


    def hammingWeight(self, n: int) -> int: # 更高级
        #时间复杂度：O(1)，n为32位的数，操作次数为二进制中1的个数
        res=0
        while n:
            n&=n-1 # 消除最后一位 1
            res+=1

        return res
``` 

## 5. 数字的补数

给定一个正整数，输出它的补数。补数是对该数的二进制表示取反。

476\. 数字的补数（easy） [力扣](https://leetcode-cn.com/problems/number-complement/description/)

示例 1:

```html
输入: 5
输出: 2
解释: 5 的二进制表示为 101（没有前导零位），其补数为 010。所以你需要输出 2 。

输入: 1
输出: 0
解释: 1 的二进制表示为 1（没有前导零位），其补数为 0。所以你需要输出 0 。

```

```python
class Solution:
    def findComplement(self, num: int) -> int:
        #找到一个二进制位数与num相同但每一位都为1的数，然后用这个数 减去 num。
        #例如 0b111-0b101=0b10,7-5=2,这里7就是我们要找的数
        i = 1
        # 最高位为1，其余为0，刚好比num大然后用这个数减去1就是我们要找的数
        while num >= i: 
            i = i << 1 # 每次向左移1位 i=0b1000
            
        return i-1-num
``` 

## 6. 交替位二进制数

给定一个正整数，检查它的二进制表示是否总是 0、1 交替出现：换句话说，就是二进制表示中相邻两位的数字永不相同。

693\. 交替位二进制数（easy） [力扣](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/description/)

示例 1:

```html
输入：n = 5
输出：true
解释：5 的二进制表示是：101

输入：n = 7
输出：false
解释：7 的二进制表示是：111.

输入：n = 11
输出：false
解释：11 的二进制表示是：1011.

输入：n = 10
输出：true
解释：10 的二进制表示是：1010.

输入：n = 3
输出：false

```

```python
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        # temp = n // 2,   n & temp == 0 # 这是错误的  eg: 4
        
        tmp = n^(n>>1)
        return tmp&(tmp+1) == 0

``` 

## 7. 比特位计数

给定一个非负整数 num。对于 **0 ≤ i ≤ num** 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回

338\. 比特位计数（middle） [力扣](https://leetcode-cn.com/problems/counting-bits/description/)

示例 1:

```html
输入: 2
输出: [0,1,1]

输入: 5
输出: [0,1,1,2,1,2]

```

```python
class Solution:
    def countBits(self, num: int) -> List[int]:

        def bin_count(n):
            count=0
            while n:
                n&=n-1
                count+=1
            return count


        res=[]

        for i in range(num+1):
            res.append(bin_count(i))
        return res

``` 

## 8. 数字范围按位与

给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）

201\. 数字范围按位与（middle） [力扣](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/description/)

示例 1:

```html
输入: [5,7]
输出: 4

输入: [0,1]
输出: 0

```

```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        
        while(n>m): # ps 这样会减少很多运算次数eg: [m,n] ---> [5, 20] 2次循环就得到答案为0了
                n = n & (n-1)
        return n
    
    
    '''
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        if n-m == 0: # 超出时间限制  [m,n] ---> [5, 20] 循环次数15，其实没必要,因为中间可能会出现0
            return n

        res = m
        for i in range(m+1, n+1):
            if res == 0: #减少循环次数 还是超时了
                return 0
            res = res & i

        return res
    
    '''

``` 

## 9. 交换数字

编写一个函数，不用临时变量，直接交换numbers = [a, b]中a与b的值。

numbers.length == 2

202\. 交换数字（middle） [力扣](https://leetcode-cn.com/problems/swap-numbers-lcci/description/)

示例 1:

```html
输入: numbers = [1,2]
输出: [2,1]

```

```python
class Solution:
    def swapNumbers(self, numbers: List[int]) -> List[int]:
        # a = a ^ b, b = a ^ b, a = a ^ b
        numbers[0] = numbers[0] ^ numbers[1]
        numbers[1] = numbers[1] ^ numbers[0]
        numbers[0] = numbers[0] ^ numbers[1]
        
        return numbers


    """
    def swapNumbers(self, numbers: List[int]) -> List[int]:
        numbers[0],numbers[1] = numbers[1], numbers[0]

        return numbers
    """

``` 

## 10. 字母大小写全排列

给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

784\. 字母大小写全排列（middle） [力扣](https://leetcode-cn.com/problems/letter-case-permutation/description/)

示例 1:

```html
示例：
输入：S = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]

输入：S = "3z4"
输出：["3z4", "3Z4"]

输入：S = "12345"
输出：["12345"]

```

```python
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:

        res = [""] # 设置一个初始值，就可以减少大量code
        for i in S:
            t_lis = []
            for j in res:
                if i.isalpha():
                    t_lis.append(j+i.lower())
                    t_lis.append(j+i.upper())
                else:
                    t_lis.append(j+i)

            res = t_lis

        return res
    
    
    def letterCasePermutation_v1(self, S: str) -> List[str]:

        res = [] # 没有设置初始值，写起来很尴尬
        for i in S:
            t_lis = []
            if not res:
                if i.isalpha():
                    res.append(i.lower())
                    res.append(i.upper())
                else:
                    res.append(i)
            else:
                for j in res:
                    if i.isalpha():
                        t_lis.append(j+i.lower())
                        t_lis.append(j+i.upper())
                    else:
                        t_lis.append(j+i)
                res = t_lis

        return res
``` 

## 11. 整数转换

整数转换。编写一个函数，确定需要改变几个位才能将整数A转成整数B。

A，B范围在[-2147483648, 2147483647]之间  

784\. 整数转换（middle） [力扣](https://leetcode-cn.com/problems/convert-integer-lcci/description/)

示例 1:

```html
输入：A = 29 （或者0b11101）, B = 15（或者0b01111）
 输出：2

 输入：A = 1，B = 2
 输出：2

```

```python
class Solution:
    def convertInteger(self, A: int, B: int) -> int:
        #整形数在内存中是以 补码 的形式存放的，输出的时候同样也是按照 补码 输出的
        #print(bin(-1)) # -0b1
        #print(bin(-1 & 0xffffffff)) #-1的补码 0b11111111111111111111111111111111
        # n &= (n - 1) 能够将n最右侧为1的位变0 
        
        #  将A,B转化为无符号数
        C = (A & 0xffffffff) ^ (B & 0xffffffff)  # 32 位 考虑负数
        #C = A  ^ B   # 如果这样 遇到负数eg  (-1, 1)  会成为死循环
        #print("C ---> ", C)
        cnt = 0
        while C != 0:
            C = C & (C - 1)  # 清除最低位1
            #print("jjj --> ", C)
            cnt += 1

        return cnt

#print(Solution().convertInteger(2, -1)) #答案为 31

"""
为什么要和 oxffffffff 作与运算
一般来讲，整形数在内存中是以 补码 的形式存放的，输出的时候同样也是按照 补码 输出的。

但是在 Python 中，情况是这样的：

整型是以 **补码** 形式存放的，输出的时候是按照 **二进制** 表示输出的；
对于 bin(x)（x 为 十进制负数），输出的是它的原码的二进制表示加上一个负号（ print(bin(-2)) # -0b10 ）
对于 bin(x)（x 为 十六进制负数），输出的是对应的二进制表示。
所以为了获得十进制负数的补码，我们需要手动将其和 0xffffffff 进行与操作，得到一个十六进制数，
再交给 bin() 转化，这时内存中得到的才是你想要的补码。

"""
``` 

## 12. 只出现一次的数字 III

给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

260\. 只出现一次的数字 III（middle） [力扣](https://leetcode-cn.com/problems/single-number-iii/description/)

示例 1:

```html
输入: [1,2,1,3,2,5]
输出: [3,5] 或 [5, 3]

```

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
    
        res, first, second = 0, 0, 0
        # 相同元素的异或为0 --> 最终结果 是两个不同元素异或的结果
        for val in nums:
            res = res ^ val
        
        # 求出res中低位第一个为1的位置(1100 --> 4)
        temp = res & (-res)
        
        # 遍历元素, 通过指定位是0还是1,分为2组
        # 此时不同的元素一定在不同的组中
        # 对每组求异或即可分别求出两个不同的元素
        for ele in nums:
            if ele & temp: # 求出 任意一个个数为1 的数
                first ^= ele

        # 方法2: 通过first和res求出second的值
        second = res ^ first
        return [first, second]

"""
1. 先熟悉一下位运算相关的概念:
    1. num ^ num = 0
    2. num ^ (-num) 可以使num中低位第一个1为1, 并使其他位全为0
2. 通过 num ^ num = 0: 
    我们可以排除掉列表中重复的元素, 最终结果只是两个只出现一次的元素求"异或"的结果
3. 通过 num ^ (-num) 
    因为剩下的两个元素值不相等, 所以他们之中一定有一个位上的值不相等, 
    那么我们可以通过"异或"找到第一个不相等的位即可区分两个元素
    
4. 然后根据指定位数值位0 或1将原列表中的元素分为2组(此时不同的两个元素一定在不同的组中)
5. 然后再对每组进行取"异或", 每组的最终结果就是那个不同的元素

    def singleNumber(nums: List[int]) -> List[int]:
        # 列表生成式
        #res= [i for i in nums if nums.count(i) == 1] # OK

        from collections import Counter
         # 这里相比上面 不用每次都 count
        res = [key for key, val in Counter(nums).items() if val==1]

        return res
"""
``` 
