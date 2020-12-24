# Leetcode 题解 - 数学
<!-- GFM-TOC -->
* [Leetcode 题解 - 数学](#leetcode-题解---数学)
    * [最大公约数最小公倍数](#最大公约数最小公倍数)
        * [1. 计数质数](#1-计数质数)
        * [2. 最大公约数](#2-最大公约数)
    * [进制转换](#进制转换)
        * [1. 七进制数](#1-七进制数)
        * [2. 数字转换为十六进制数](#2-数字转换为十六进制数)
    * [阶乘](#阶乘)
        * [1. 统计阶乘尾部有多少个 0](#1-统计阶乘尾部有多少个-0)
    * [字符串加法减法](#字符串加法减法)
        * [1. 二进制加法](#1-二进制加法)
        * [2. 字符串加法](#2-字符串加法)
    * [相遇问题](#相遇问题)
        * [1. 改变数组元素使所有的数组元素都相等](#1-改变数组元素使所有的数组元素都相等)
    * [多数投票问题](#多数投票问题)
        * [1. 数组中出现次数多于 n / 2 的元素](#1-数组中出现次数多于-n--2-的元素)
    * [其它](#其它)
        * [1. 平方数](#1-平方数)
        * [2. 3 的 n 次方](#2-3-的-n-次方)
        * [3. 乘积数组](#3-乘积数组)
        * [4. 找出数组中的乘积最大的三个数](#4-找出数组中的乘积最大的三个数)
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
