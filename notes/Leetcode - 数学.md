# Leetcode 题解 - 数学
<!-- GFM-TOC -->
* [Leetcode 题解 - 数学](#leetcode-题解---数学)
    * [最大公约数最小公倍数](#最大公约数最小公倍数)
        * [1. 计数质数](#1-计数质数)
        * [2. 最大公约数](#2-最大公约数)
    * [进制转换](#进制转换)
        * [1. 七进制数](#1-七进制数)
        * [2. 数字转换为十六进制数](#2-数字转换为十六进制数)
        * [3. 二进制求和](#3-二进制求和)
    * [3. 3的幂](#3-3的幂)
    * [4. 4的幂](#4-4的幂)
    * [5. 丑数](#5-丑数)
    * [6. 三个数的最大乘积](#6-三个数的最大乘积)
    * [7. 加一](#7-加一)
    * [8. 完美数](#8-完美数)
    * [9. 阶乘后的零](#9-阶乘后的零)
    * [10. 十进制整数的反码](#10-十进制整数的反码)
    * [11. 圆圈中最后剩下的数字](#11-圆圈中最后剩下的数字)
    
<!-- GFM-TOC -->

## 最大公约数最小公倍数

### 1. 计数质数

统计所有小于非负整数 n 的质数的数量   

204\. 计数质数（easy） [力扣](https://leetcode-cn.com/problems/count-primes/description/)

示例 1:

```html
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。

输入：n = 0
输出：0

输入：n = 1
输出：0

```

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        #厄拉多塞筛法( 统计所有小于非负整数 n 的质数的数量 )
        #给出要筛数值的范围n，找出以内的素数(不包括n)。先用2去筛，即把2留下，把2的倍数剔除掉；
        #再用下一个质数，也就是3筛，把3留下，把3的倍数剔除掉，... 不断重复下去

        if n < 3: # 即使 n=2 则小于n的质素为 0
            return 0     
        
        res = [1] * n # 因为不包括n 
        res[0], res[1] = 0, 0 # 初始值设定
        for i in range(2, int(n**0.5)+1):# 注意循环结束条件 
            if res[i] == 1:
                #res[i:n:i] = [0] * len(res[i:n:i]) #这个包括了i是错误的
                res[i*2:n:i] = [0] * len(res[i*2:n:i]) # i的倍数剔除掉(不包括i) eg: [6, 9, 12, 15, 18] 3的倍数
            
                # 下面是优化
                #res[i*i:n:i] = [0] * len(res[i*i:n:i]) # i的倍数 从 i*i 开始 不包括i  eg：[9, 12, 15, 18] 3的倍数
        
        #如果想输出这些 质素的话， 因为 索引0 对应0，索引2 对应2  ---> list(range(n)) 不包括n
        #primes = [i for (i, v) in enumerate(res) if v == 1] 
        #print('primes ------>  ', primes)

        return sum(res)

``` 

### 2. 最大公约数
```python
def gcd(a, b):
    """ 求最大公约数 """
    if a >= b: # 保证 a 是最小的那个数
        a, b = b, a

    while (a):
        a, b = b%a, a

    return b

``` 

最小公倍数为两数的乘积除以最大公约数

```python
def lcm(a, b):
    """ 求最小公倍数 """
    
    return a * b // gcd(a, b)

``` 

## 进制转换

### 1. 七进制数

给定一个整数，将其转化为7进制，并以字符串形式输出。

504\. 七进制数（easy） [力扣](https://leetcode-cn.com/problems/base-7/description/)

示例 1:

```html
输入: 100
输出: "202"

输入: -7
输出: "-10"

```

```python
class Solution:
    def convertToBase7(self, num: int) -> str:
        pre = "-" if num < 0 else ""
        num = abs(num)
        result = ""

        while num != 0:
            # divmod => (x//y, x%y)
            num, temp = divmod(num, 7)
            result = str(temp) + result

        return pre + result if result else "0"

``` 

### 2. 数字转换为十六进制数

```html
给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。

注意:

十六进制中所有字母(a-f)都必须是小写。
十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；
对于其他情况，十六进制字符串中的第一个字符将不会是0字符。 

给定的数确保在32位有符号整数范围内。
不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。
```

405\. 数字转换为十六进制数（easy） [力扣](https://leetcode-cn.com/problems/base-7/description/)

示例 1:

```html
输入:
26

输出:
"1a"


输入:
-1

输出:
"ffffffff"
```

```python
class Solution:
    def toHex(self, num: int) -> str:
        #数值一律用补码来表示和存储： 正数的源码最高位是0，正数的源码和反码补码都是一样的， 
        #负数的补码是在原码的基础上除符号位外其余位取反后+1
        
        res = "" # 用来返回结果
        # generate num to cha dic
        num_dic = {}
        for i in range(10):
            num_dic[i] = str(i)
        for i in range(10, 16):
            num_dic[i] = chr(i + 87) # chr(97) ---> 'a'

        # process non-positive num
        if num == 0:
            return "0"
        elif num < 0: # 这里是关键, 把负数当成补码来算，正数的源码就是补码
            num += 2 ** 32 # -1就变成了 补码为 32位1 然后求和了

        while num:
            item = (num & 15)
            res = num_dic[item] + res  # 注意这里顺序不能变
            num = (num >> 4) #num =(num >> 4) 相等于 num = num // 16
            
        return res

       
    '''
        def toHex(self, num: int) -> str:
            print(hex(num=26)) # 0x1a

    '''

``` 

### 3. 二进制求和

给你两个二进制字符串，返回它们的和（用二进制表示）。

输入为 非空 字符串且只包含数字 1 和 0。


67\. 二进制求和（easy） [力扣](https://leetcode-cn.com/problems/add-binary/description/)

示例 1:

```html
输入: a = "11", b = "1"
输出: "100"

输入: a = "1010", b = "1011"
输出: "10101"

```

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:

        def bin2ten(num):
            n = len(num)
            s = 0
            for k, v in enumerate(num):
                s += int(v) * pow(2, n-k-1)
            return s

        res = bin2ten(a) + bin2ten(b)
        

        ''' bin(int(a,2)+int(b,2))[2:]  
            int(a,base) #  base默认为0， base取值2-36， 当base=2，int(a="100",2)即
            把字符串a以2进制转10进制 int(a="100",2) ==4
        '''

        return bin(res)[2:] # 把前面的 0b100  0b去掉

``` 




## 3. 3的幂

给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3**x


326\. 3的幂（easy） [力扣](https://leetcode-cn.com/problems/power-of-three/description/)

示例 1:

```html
输入：n = 27
输出：true

输入：n = 0
输出：false

输入：n = 9
输出：true

输入：n = 45
输出：false
```

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False

        while (n % 3 == 0): # 更高级的
            n = n // 3
            
        return n == 1

``` 

## 4. 4的幂

给定一个整数，写一个函数来判断它是否是 4 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 4 的幂次方需满足：存在整数 x 使得 n == 4**x

342\. 4的幂（easy） [力扣](https://leetcode-cn.com/problems/power-of-four/description/)

示例 1:

```html
输入：n = 16
输出：true

输入：n = 5
输出：false

输入：n = 1
输出：true
```

```python
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        if(num<=0): # 因为 4的幂必大于0
            return False
        
        # 下面的 num > 0 的情况
        while(num%4==0):
            num= num // 4
            
        return num==1
            
``` 

## 5. 丑数

编写一个程序判断给定的数是否为丑数。

丑数就是只包含质因数 2, 3, 5 的正整数。

263\. 丑数（easy） [力扣](https://leetcode-cn.com/problems/ugly-number/description/)

示例 1:

```html
输入: 6
输出: true
解释: 6 = 2 × 3

输入: 8
输出: true
解释: 8 = 2 × 2 × 2

输入: 14
输出: false 
解释: 14 不是丑数，因为它包含了另外一个质因数 7。
```

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        
        if num <=0:
            return False
        if num == 1: # 题目说了，1 是丑数
            return True

        while(num > 1):
            if num % 2 ==0:
                num = num // 2
            elif num % 3 ==0:
                num = num // 3
            elif num % 5 ==0:
                num = num // 5
            else:
                return False

        return True
``` 

## 6. 三个数的最大乘积

给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

628\. 三个数的最大乘积（easy） [力扣](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/description/)

示例 1:

```html
输入: [1,2,3]
输出: 6

输入: [1,2,3,4]
输出: 24
```

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        if len(nums)<3:
            return 0
        
        # 下面len(nums) 至少是3了
        nums.sort() # 排序很重要,首先排序
        # 分两种情况：1、三个正数 2、一个最大正数，两个最大负数
        res = max(nums[0]*nums[1]*nums[-1], nums[-3]*nums[-2]*nums[-1])
        
        return res
``` 

## 7. 加一

给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头

66\. 加一（easy） [力扣](https://leetcode-cn.com/problems/plus-one/description/)

示例 1:

```html
输入：digits = [1,2,3]
输出：[1,2,4]
解释：输入数组表示数字 123。

输入：digits = [4,3,2,1]
输出：[4,3,2,2]
解释：输入数组表示数字 4321。

输入：digits = [0]
输出：[1]
```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        carry = 0
        for i, v in enumerate(digits[::-1]):
            # 最后 1 位数字 加1
            if i == 0:
                digits[n-i-1] = (v+1 + carry)%10
                carry = (v + 1 + carry) // 10
            else:
                digits[n - i - 1] = (v + carry) % 10
                carry = (v+carry) // 10

            if carry != 1: 
                break
                
        else: # 如果for循环正常退出 则判断最后 carry是否为1
            # digits = [9]
            if carry==1:
                digits.insert(0, carry)

        return digits
``` 

## 8. 完美数

对于一个 **正整数**，如果它和除了它自身以外的所有 **正因子** 之和相等，我们称它为 「**完美数**」。

给定一个 **整数** n， 如果是完美数，返回 true，否则返回 false

507\. 完美数（easy） [力扣](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/description/)

示例 1:

```html
输入：28
输出：True

解释：28 = 1 + 2 + 4 + 7 + 14
1, 2, 4, 7, 和 14 是 28 的所有正因子。


输入：num = 6
输出：true

输入：num = 496
输出：true


输入：num = 8128
输出：true

输入：num = 2
输出：false
```

```python
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        # step 1 求num除自身外的正因子
        if num <= 1: # bad case
            return False
         
        res =set() # 避免重复，比如100，如果用list的话会有2个10  如果用 num//2 + 1 会超时
        res.add(1)
        for i in range(2, int(math.sqrt(num)) + 1):# 这里从2开始是为了不把自身加进来
            if num % i == 0:
                res.add(i)
                res.add(num//i) # 聪明, 这样可以减少运算次数
        
        #print(res) #num=400 --->{1, 2, 4, 100, 5, 200, 8, 10, 40, 80, 16, 50, 20, 25}
        if sum(res) == num:
            return True
        else:
            return False
``` 

## 9. 阶乘后的零

给定一个整数 n，返回 n! 结果尾数中零的数量。

172\. 阶乘后的零（easy） [力扣](https://leetcode-cn.com/problems/factorial-trailing-zeroes/description/)

示例 1:

```html
输入: 3
输出: 0
解释: 3! = 6, 尾数中没有零。

输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.
```

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        #题目的意思是末尾有几个0， 比如6! = 【1* 2* 3* 4* 5* 6】， 只有2*5末尾才有0，所以就可以抛去其他数据
        #一个2和一个5配对 就产生一个0 所以6！末尾1个0，2肯定比5多 所以只数5的个数就行 ！！！
        '''
        需要注意的是，像 25，125 这样的不只含有一个 5 的数字的情况需要考虑进去。比如 n = 15。那么在 15! 中 有 3 个 5 (来自其中的5, 10, 15)， 所以计算 n/5 就可以 。
        但是比如 n=25，依旧计算 n/5 ，可以得到 5 个5，分别来自其中的5, 10, 15, 20, 25，但是在 25 中其实是包含 2个 5 的，这一点需要注意。
        所以除了计算 n/5 ， 还要计算 n/5/5 , n/5/5/5 , n/5/5/5/5 , ..., n/5/5/5,,,/5直到商为0，然后求和即可
        '''
        #统计5的个数就可以了
        res = 0  # 这个有点高级没怎么看懂
        while(n >=5):
            res = res + n // 5
            n =  n // 5

        return res
``` 

## 10. 十进制整数的反码

```html
每个非负整数 N 都有其二进制表示。例如， 5 可以被表示为二进制 "101"，11 可以用二进制 "1011" 表示，依此类推。注意，除 N = 0 外，任何二进制表示中都不含前导零。

二进制的反码表示是将每个 1 改为 0 且每个 0 变为 1。例如，二进制数 "101" 的二进制反码为 "010"。

给你一个十进制数 N，请你返回其二进制表示的反码所对应的十进制整数。

```

1009\. 十进制整数的反码（easy） [力扣](https://leetcode-cn.com/problems/complement-of-base-10-integer/description/)

示例 1:

```html
输入：5
输出：2
解释：5 的二进制表示为 "101"，其二进制反码为 "010"，也就是十进制中的 2 。

输入：7
输出：0
解释：7 的二进制表示为 "111"，其二进制反码为 "000"，也就是十进制中的 0 。

输入：10
输出：5
解释：10 的二进制表示为 "1010"，其二进制反码为 "0101"，也就是十进制中的 5 。

```

```python
class Solution:
    def bitwiseComplement(self, N: int) -> int:
        # 0 <= N < 10^9
        #对于一个正整数， 比如5 （101）， 其反码就是其所有二进制位上为1的数（111）
        #                              减去这个正整数= 111 - 101 = 010 （2）

        if(N == 0): return 1
        
        num = 1 # 重点
        while(num <= N):
            num <<= 1
        
        return num - N - 1
    
    '''
        def bitwiseComplement(self, N: int) -> int:
            # 0 <= N < 10^9
            return int(''.join(['0' if i== '1' else '1' for i in bin(N)[2:]]),2)

    '''
``` 

## 11. 圆圈中最后剩下的数字

```html
0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，
因此最后剩下的数字是3。

```

19\. 圆圈中最后剩下的数字（easy） [力扣](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/description/)

示例 1:

```html
输入: n = 5, m = 3
输出: 3

输入: n = 10, m = 17
输出: 2

```

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        #约瑟夫环 问题, n >= 1
        lst = list(range(n))
        begin = 0 # 从索引为0 开始数数
        while(len(lst) !=1 ):
            ind = (begin + m - 1) % len(lst) # ind >= 0

            lst.pop(ind) # 删除指定索引数字
            begin = ind # 更新 开始数数的位置

        return lst[0]

``` 
