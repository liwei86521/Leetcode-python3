# Leetcode 题解 - 哈希表
<!-- GFM-TOC -->
* [Leetcode 题解 - 哈希表](#leetcode-题解---哈希表)
    * [1. 两数之和](#1-两数之和)
    * [2. 找不同](#2-找不同)
    * [3. Excel表列序号](#3-Excel表列序号)
    * [4. 两句话中的不常见单词](#4-两句话中的不常见单词)
    * [5. 最长回文串](#5-最长回文串)
    * [6. 回文排列](#6-回文排列)
    * [7. 键盘行](#7-键盘行)
    * [8. 存在重复元素 II](#8-存在重复元素-II)
    * [9. 扑克牌中的顺子](#9-扑克牌中的顺子)
    * [10. 字母异位词分组](#10-字母异位词分组)
    * [11. 找到字符串中所有字母异位词](#11-找到字符串中所有字母异位词)
    * [12. 和可被K整除的子数组](#12-和可被K整除的子数组)
<!-- GFM-TOC -->


哈希表使用 O(N) 空间复杂度存储数据，并且以 O(1) 时间复杂度求解问题。


## 1. 两数之和

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 **两个 整数，并返回他们的数组下标**。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。


1\. 两数之和（middle） [力扣](https://leetcode-cn.com/problems/two-sum/description/)

示例 1：

```html
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # ps: 不能用2 层for 循环，会超时，所以改成下面的
        # 不能用双指针，因为不是排序的
        dic = {}
        for key, val in enumerate(nums):
            if target-val in dic:
                return [dic[target-val], key]
            else:
                dic[val] = key
``` 

## 2. 找不同

给定两个字符串 s 和 t，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。


389\. 找不同（easy） [力扣](https://leetcode-cn.com/problems/find-the-difference/description/)

示例 1：

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
        from collections import Counter
        #print(Counter(t)) # Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1})
        #print(Counter(s)) # Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1})
        #print((Counter(t) - Counter(s))) # Counter({'e': 1})
        #print(list(Counter(t) - Counter(s))) # ['e']
        #print(list({'a': 2, 'e': 1})) # ['a', 'e']

        return list(Counter(t) - Counter(s))[0]


    def findTheDifference(self, s: str, t: str) -> str:
        ret = 0  # python3 异或方法 推荐
        for c in s + t:
            ret = ret ^ ord(c) #异或只针对 int类型
        return chr(ret)
``` 

## 3. Excel表列序号

给定一个Excel表格中的列名称，返回其相应的列序号。 例如，
```html
    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...

```

171\. Excel表列序号（easy） [力扣](https://leetcode-cn.com/problems/excel-sheet-column-number/description/)

示例 1：

```html
输入: "A"
输出: 1

输入: "AB"
输出: 28
示例 3:

输入: "ZY"
输出: 701

```

```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        ## 相当于26进制 求和
        dic = {chr(64+i):i for i in range(1, 27)} # 字典生成式
        #print(dic) # {'A': 1, 'B': 2, 'C': 3, ... 'Z': 26}

        res = 0
        n = len(s)
        for i, ch in enumerate(s):
            res += dic.get(ch) * (26**(n-i-1))

        return res
``` 

## 4. 两句话中的不常见单词

```html
给定两个句子 A 和 B 。 （句子是一串由空格分隔的单词。每个单词仅由小写字母组成。）

如果一个单词在其中一个句子中只出现一次，在另一个句子中却没有出现，那么这个单词就是不常见的。

返回所有不常用单词的列表。

您可以按任何顺序返回列表。
```

884\. Excel表列序号（easy） [力扣](https://leetcode-cn.com/problems/uncommon-words-from-two-sentences/description/)

示例 1：

```html
输入：A = "this apple is sweet", B = "this apple is sour"
输出：["sweet","sour"]

输入：A = "apple apple", B = "banana"
输出：["banana"]

```

```python
class Solution:
    def uncommonFromSentences(self, A: str, B: str) -> List[str]:
        
        from collections import  Counter
        A_list = A.split(' ')
        B_list = B.split(' ')
        A_list.extend(B_list)

        temp = Counter(A_list).items() # 转为dict_items
        #print(temp) # dict_items([('this', 2), ('apple', 2), ('is', 2), ('sweet', 1), ('sour', 1)])
        res = [k for k, v in temp if v == 1]

        return res

    def uncommonFromSentences(self, A: str, B: str) -> List[str]:
        A_set = set(A.split(" "))
        B_set = set(B.split(" "))
        
        intersect_AB = A_set.intersection(B_set) 
        union_AB = A_set.union(B_set) 
        
        return union_AB - intersect_AB
``` 

## 5. 最长回文串

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

409\. 最长回文串（easy） [力扣](https://leetcode-cn.com/problems/longest-palindrome/description/)

示例 1：

```html
输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

```

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        '''
        1:找出字符串中每个字符出现的次数
        2:若字符出现次数为奇数次，出现的次数减去1 可用于构造回文串的字符数。
          若出现的次数为偶数次，出现次数 可用于构造回文串的字符数
        3:若存在奇数次字符，则最终结果再＋1，因为奇数字符可放一个在最中间
        '''
        
        Flag = False
        res = 0
        dic = {}
        for i in s:
            dic[i] = dic.get(i, 0) + 1

        for k in dic:
            if dic[k] % 2 == 0:
                res += dic[k]
            else:
                res += dic[k] - 1
                Flag = True # 若存在奇数次字符，则最终结果再＋1, 因为奇数字符可放一个在最中间
                
        if Flag:
            return res + 1
        else:
            return res
``` 

## 6. 回文排列

给定一个字符串，编写一个函数判定其是否为某个回文串的排列之一。

回文串是指正反两个方向都一样的单词或短语。排列是指字母的重新排列。

回文串不一定是字典当中的单词

402\. 最长回文串（easy） [力扣](https://leetcode-cn.com/problems/palindrome-permutation-lcci/description/)

示例 1：

```html

输入："tactcoa"
输出：true（排列有"tacocat"、"atcocta"，等等）
```

```python
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        #每个字符出现的次数为偶数, 或者有且只有一个字符
        # 出现的次数为奇数时, 是回文的排列; 否则不是
        dic = {}
        for ch in s:
            dic[ch] = dic.get(ch, 0) + 1

        cnts = dic.values() # cnts = list(dic.values())
        #print(cnts) # <class 'dict_values'> dict_values([2, 2, 2, 1])
        res = [v for v in cnts if v % 2]

        # res = []
        # for v in cnts:
        #     if v % 2 == 1:
        #         res.append(v)

        return len(res) <= 1
``` 

## 7. 键盘行

给定一个单词列表，只返回可以使用在键盘同一行的字母打印出来的单词。键盘如下图所示
<img style="width: 100%; max-width: 600px" src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/keyboard.png" alt="American keyboard">


500\. 键盘行（easy） [力扣](https://leetcode-cn.com/problems/keyboard-row/description/)

示例 1：

```html
输入: ["Hello", "Alaska", "Dad", "Peace"]
输出: ["Alaska", "Dad"]
```

```python
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        
        line_1 = {"q","w","e","r","t","y","u","i","o","p"} # 集合
        line_2 = {"a","s","d","f","g","h","j","k","l"}
        line_3 = {"z","x","c","v","b","n","m"}
        
        res = []
        for word in words: # 判断一个单词是否可以在同一行键盘打出来
            if set(word.lower()).issubset(line_1) or 
               set(word.lower()).issubset(line_2) or 
               set(word.lower()).issubset(line_3):

                res.append(word)
                
        return res
``` 

## 8. 存在重复元素 II

给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 **nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k**。


219\. 存在重复元素 II（easy） [力扣](https://leetcode-cn.com/problems/contains-duplicate-ii/description/)

示例 1：

```html
输入: nums = [1,2,3,1], k = 3
输出: true

输入: nums = [1,0,1,1], k = 1
输出: true

输入: nums = [1,2,3,1,2,3], k = 2
输出: false

```

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        #首先是判断是存在num[i]==num[j]，存在的情况下才判断|i-j|<=k
        n = len(nums)
        if n <= 1: return False
            
        dic = {}
        for ind, val in enumerate(nums):
            if val not in dic:
                dic[val] = ind # 添加字典
            else:
                diff = ind - dic[val]
                if diff <= k:
                    return True

                dic[val] = ind # 更新索引

        return False
``` 

## 9. 扑克牌中的顺子

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

222\. 扑克牌中的顺子 II（easy） [力扣](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/description/)

示例 1：

```html
输入: [1,2,3,4,5]
输出: True

输入: [0,0,1,2,5]
输出: True

```

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        # 时间复杂度为 O(NlogN)
        joker = 0
        nums.sort() # 数组排序
        for i in range(4):
            if nums[i] == 0: joker += 1 # 统计大小王数量
            elif nums[i] == nums[i + 1]: 
                return False # 若有重复，提前返回 false
        return nums[4] - nums[joker] < 5 # 最大牌 - 最小牌 < 5 则可构成顺子
``` 

## 10. 字母异位词分组

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

222\. 字母异位词分组（middle） [力扣](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/description/)

示例 1：

```html
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

```

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # python3 常规思想
        dic = {}
        for s in strs:
            keys = "".join(sorted(s)) #排序的时间复杂度为 NlogN
            if keys not in dic:
                dic[keys] = [s]
            else:
                dic[keys].append(s)

        return list(dic.values())
``` 

## 11. 找到字符串中所有字母异位词

```html
给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：
  字母异位词指字母相同，但排列不同的字符串。
  不考虑答案输出的顺序。
```

438\. 找到字符串中所有字母异位词（middle） [力扣](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/description/)

示例 1：

```html
输入:
s: "cbaebabacd" p: "abc"

输出:
[0, 6]

解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。

输入:
s: "abab" p: "ab"

输出:
[0, 1, 2]

解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的字母异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的字母异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的字母异位词。
```

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]: 
        # 很经典的 哈希表题目
        dict_p = {}# 如果涉及到对顺序无要求的题目，可以考虑dict
        for i in p:
            dict_p[i] = dict_p.get(i, 0) + 1

        dict_s = {}
        res = []
        len_p = len(p) # 后面要重复用到，提高效率

        #print(dict_p) # {'a': 1, 'b': 1, 'c': 1}

        for k, v in enumerate(s):
            dict_s[v] = dict_s.get(v, 0) + 1 # 0是默认值

            if dict_s == dict_p:# 至少长度要一致，所有必须先循环几次
                res.append( k - len_p + 1) # 注意这里

            if (k - len_p + 1) >= 0: #当循环次数超过len_p后，就对dict_s进行更新或删除操作
                # 每往后循环一次index，就减去前面掉出去的那个字符一次(eg: 第1个字符)
                dict_s[s[k - len_p + 1]] =  dict_s.get(s[k - len_p + 1]) - 1
                # 如果被-1的那个字符是单个字符，删除掉，保证dict_s在进入下次循环前，所有k对应的v的和为2
                if dict_s[s[k - len_p + 1]] == 0: # 删除相应的 key
                    del dict_s[s[k - len_p + 1]]

        return res
``` 

## 12. 和可被 K 整除的子数组

给定一个整数数组 A，返回其中元素之和可被 K 整除的（连续、非空）子数组的数目。

974\. 和可被 K 整除的子数组（middle） [力扣](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/description/)

示例 1：

```html
输入:
输入：A = [4,5,0,-2,-3,1], K = 5
输出：7
解释：
有 7 个子数组满足其元素之和可被 K = 5 整除：
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
```

```python
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        record = {0: 1}
        total, ans = 0, 0
        for elem in A:
            total += elem
            modulus = total % K
            same = record.get(modulus, 0) #经典
            ans += same
            record[modulus] = same + 1 # #经典

        return ans
``` 

