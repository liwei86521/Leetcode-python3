# Leetcode - 栈和队列和堆
<!-- GFM-TOC -->
* [Leetcode 题解 - 栈和队列和堆](#leetcode---栈和队列和堆)
    * [1. 用栈实现队列](#1-用栈实现队列)
    * [2. 用队列实现栈](#2-用队列实现栈)
    * [3. 最小栈](#3-最小栈)
    * [4. 有效的括号](#4-有效的括号)
    * [5. 比较含退格的字符串](#5-比较含退格的字符串)
    * [6. 删除字符串中的所有相邻重复项](#6-删除字符串中的所有相邻重复项)
    * [7. 每日温度](#7-每日温度)
    * [8. 字符串解码](#8-字符串解码)
    * [9. 栈的压入弹出序列](#9-栈的压入弹出序列)
    * [10. 逆波兰表达式求值](#10-逆波兰表达式求值)
    
    * [11. 数据流中的第K大元素](#11-数据流中的第K大元素)
    * [12. 最后一块石头的重量](#12-最后一块石头的重量)
    * [13. 最小K个数](#13-最小K个数)
    * [14. 前K个高频元素](#14-前K个高频元素)
    * [15. 数组中的第K个最大元素](#15-数组中的第K个最大元素)
    * [16. 查找和最小的K对数字](#16-查找和最小的K对数字)
<!-- GFM-TOC -->

## 1. 用栈实现队列

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（push、pop、peek、empty）

实现 MyQueue 类：
<ul>
	<li><code>void push(int x)</code> 将元素 x 推到队列的末尾</li>
	<li><code>int pop()</code> 从队列的开头移除并返回元素</li>
	<li><code>int peek()</code> 返回队列开头的元素</li>
	<li><code>boolean empty()</code> 如果队列为空，返回 <code>true</code> ；否则，返回 <code>false</code></li>
</ul>

232\. 用栈实现队列（easy） [力扣](https://leetcode-cn.com/problems/implement-queue-using-stacks/description/)

示例 1:

```html
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false

```

```python
class MyQueue:

    def __init__(self):
        self.stackin = []  # 输入栈
        self.stackout = []  # 输出栈
        

    def push(self, x: int) -> None:
        self.stackin.append(x)
        

    def pop(self) -> int:
        # 从输出栈取走元素，输出栈没有元素时，将输入栈元素依次出栈压入输出栈，再从输出栈取出；
        if not self.stackout:
            while self.stackin:
                a = self.stackin.pop()
                self.stackout.append(a)
        return self.stackout.pop()
        

    def peek(self) -> int: # peek() -- 返回队列首部的元素
        if not self.stackout:
            while self.stackin:
                a = self.stackin.pop()
                self.stackout.append(a)
                
        return self.stackout[-1]
        

    def empty(self) -> bool:
        if not self.stackin and not self.stackout:
            return True
        else:
            return False


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

``` 

## 2. 用队列实现栈

使用队列实现栈的下列操作：

<ul>
	<li>push(x) -- 元素 x 入栈</li>
	<li>pop() -- 移除栈顶元素</li>
	<li>top() -- 获取栈顶元素</li>
	<li>empty() -- 返回栈是否为空</li>
</ul>

225\. 用队列实现栈（easy） [力扣](https://leetcode-cn.com/problems/implement-stack-using-queues/description/)

```python
class MyStack:

    def __init__(self):
        self.q = []
    
    # 时间复杂度 O(n)
    def push(self, x: int) -> None:
        self.q.append(x)
        q_length = len(self.q)
        while q_length > 1:
            #反转前n-1个元素，栈顶元素始终保留在队首
            self.q.append(self.q.pop(0)) 
            q_length -= 1

    # 时间复杂度 O(n)
    def pop(self) -> int:
        return self.q.pop(0)

    # 时间复杂度 O(1)
    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return len(self.q) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

``` 

## 3. 最小栈

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈

<ul>
	<li><code>push(x)</code> —— 将元素 x 推入栈中。</li>
	<li><code>pop()</code>&nbsp;—— 删除栈顶的元素。</li>
	<li><code>top()</code>&nbsp;—— 获取栈顶元素。</li>
	<li><code>getMin()</code> —— 检索栈中的最小元素。</li>
</ul>

155\. 最小栈（easy） [力扣](https://leetcode-cn.com/problems/min-stack/description/)

示例 1:

```html
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

```

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [] # 辅助栈

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack:
            self.min_stack.append(x)
        else:
            if x <= self.min_stack[-1]: # 等于不能省略
                self.min_stack.append(x)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

``` 

## 4. 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
```html
有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。
```

20\. 有效的括号（easy） [力扣](https://leetcode-cn.com/problems/valid-parentheses/description/)

示例 1:

```html
输入: "()"
输出: true

输入: "()[]{}"
输出: true

输入: "(]"
输出: false

输入: "([)]"
输出: false

输入: "{[]}"
输出: true

```

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        dic = {"(":")", "[":"]", "{":"}"}
        for ch in s:
            if ch in dic:
                stack.append(ch)
            else:
                if not stack or ch != dic[stack.pop()]:
                    return False

        return len(stack) == 0

``` 

## 5. 比较含退格的字符串

给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

844\. 比较含退格的字符串（easy） [力扣](https://leetcode-cn.com/problems/backspace-string-compare/description/)

示例 1:

```html
输入：S = "ab#c", T = "ad#c"
输出：true
解释：S 和 T 都会变成 “ac”。

输入：S = "ab##", T = "c#d#"
输出：true
解释：S 和 T 都会变成 “”。

输入：S = "a##c", T = "#a#c"
输出：true
解释：S 和 T 都会变成 “c”。

输入：S = "a#c", T = "b"
输出：false
解释：S 会变成 “c”，但 T 仍然是 “b”。

```

```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def check(s):
            stack = []
            for ch in s:
                if ch == "#":
                    if stack:
                        stack.pop()
                else:
                    stack.append(ch)

            return stack

        return check(S) == check(T)


    def backspaceCompare_v2(self, S: str, T: str) -> bool:
        # 经典栈问题
        stack_s =[] #创建2个栈
        stack_t =[]

        for ch_s in S:
            if ch_s == "#": #遇到 "#" 就pop
                if stack_s:#如果栈不为空则 pop
                    stack_s.pop()
                else: #如果栈为空 什么都不做
                    pass #空文本输入退格字符，文本继续为空
            else:
                stack_s.append(ch_s)

        for ch_t in T:
            if ch_t == "#":
                if stack_t:#如果栈不为空则 pop
                    stack_t.pop()
                else: #如果栈为空 什么都不做
                    pass
            else:
                stack_t.append(ch_t)

        return stack_s==stack_t

``` 

## 6. 删除字符串中的所有相邻重复项

给出由小写字母组成的字符串 S，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

844\. 删除字符串中的所有相邻重复项（easy） [力扣](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/description/)

示例 1:

```html
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。
之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。

```

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        #给出的是小写字母组成的字符串S
        stack = []
        stack.append("#") #类似于链表的dummy操作，方便边界操作
        for ch in S:
            if ch == stack[-1]:
                stack.pop()
            else:
                stack.append(ch)

        return "".join(stack[1:])

``` 

## 7. 每日温度

```html
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。
如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，
你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
```

739\. 每日温度（easy） [力扣](https://leetcode-cn.com/problems/daily-temperatures/description/)

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = list()
        t_length = len(T)
        res_list = [0 for _ in range(t_length)]
        #print(res_list) #[0, 0, 0, 0, 0, 0, 0, 0]

        for key, value in enumerate(T):
            if stack: # ps: stack 在整个循环完毕后, 还可能有多个值
                while stack and T[stack[-1]] < value:
                    res_list[stack[-1]] = key - stack[-1]
                    stack.pop()
            stack.append(key)
        #若 T = [73, 74, 75, 71, 69, 72, 76, 73]
        #print('最终stack剩的key ---> ',stack) # 最终stack剩的key --->  [6, 7]
        return res_list
``` 

## 8. 字符串解码

```html
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
```

739\. 字符串解码（middle） [力扣](https://leetcode-cn.com/problems/daily-temperatures/description/)

示例 1：

```html
输入：s = "3[a]2[bc]"
输出："aaabcbc"

输入：s = "3[a2[c]]"
输出："accaccacc"

输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"

```

```python
class Solution:
    def decodeString(self, s: str) -> str:
        # s = "3[a2[c]]"
        # 1. 若遇到数字，则获取对应数字，num_str
        # 2. 若遇到'[' , 将res(已解码字符串)入栈，置res为空(用来获取'[]'内的字符串)
        # 3. 若遇到']', 出栈resStack，出栈num_stack,获取重复次数k，重复res k次，添加到出栈字符串上
        # 4. 若为字母，添加到res
        
       num_stack = [] #定义2个栈，1个装数字，1个装前面处理的解码结果
       res_stack = [] # len([""]) == 1
       res = ''
       num_str = ''
       for i in s:
           if i.isdigit():
               num_str+=i
           elif i == '[':
               res_stack.append(res)
               num_stack.append(num_str)
               res = '' # 重置为下一次计算
               num_str = ''
           elif i == ']':
               res = res_stack.pop() + int(num_stack.pop()) * res
           else:
               res += i

       return res
``` 

## 9. 栈的压入弹出序列

```html
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，
序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 
就不可能是该压栈序列的弹出序列。
```

9\. 栈的压入弹出序列（middle） [力扣](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/description/)

示例 1：

```html
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1


输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。

```

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:

        #题目指出 pushed 是 popped 的排列, 无需考虑 pushed 和 popped 长度不同
        # 或 包含元素不同 的情况
        # 时间复杂度：O(N)
        stack= [] # 辅助栈 stack
        i = 0
        for num in pushed:
            stack.append(num)  # num 入栈
            while stack and stack[-1] == popped[i]:  # 循环判断与出栈
                stack.pop()
                i += 1

        return not stack
``` 

## 10. 逆波兰表达式求值

```html
根据 逆波兰表示法，求表达式的值。

有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。

给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

```

150\. 逆波兰表达式求值（middle） [力扣](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/description/)

示例 1：

```html
输入: ["2", "1", "+", "3", "*"]
输出: 9
解释: 该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9

输入: ["4", "13", "5", "/", "+"]
输出: 6
解释: 该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6

输入: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
输出: 22
解释: 
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22

```

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = [] #栈用来装数字
        # 一遇到 符号("+", "-", "*", "/") 栈就pop 2次进行运算，并把结果压栈
        for elem in tokens:
            #ps: "-11".isdigit() --> false
            if elem not in ["+", "-", "*", "/"]:
                stack.append(elem)
            else:
                num1 = stack.pop() # 这是栈最后一个数
                num2 = stack.pop() # 这是倒数第二个数
                str_eval = num2+elem+num1
                res = str(int(eval(str_eval)))
                stack.append(res)

        # 返回 是 int类型
        return int(stack[0])
``` 


## 11. 数据流中的第K大元素

```html
设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。

请实现 KthLargest 类：
  KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。

  int add(int val) 返回当前数据流中第 k 大的元素。

```

703\. 数据流中的第 K 大元素（easy） [力扣](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/description/)

示例 1:

```html
输入：
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：
[null, 4, 5, 5, 8, 8]

解释：
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8

```

```python
class KthLargest(object):
    import heapq
    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        # self k
        self.k = k
        self.heap = nums # heap其实就是个list
        heapq.heapify(self.heap) # 原地把一个list调整成堆

        # 减小到k
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        
        if len(self.heap) < self.k: # 小于堆长度，则直接添加进去
            heapq.heappush(self.heap, val)
            
        elif self.heap[0] < val: # 新的值更大，更新
            heapq.heapreplace(self.heap, val)

        return self.heap[0]

    """
    1、heapq.heapify可以原地把一个list调整成堆
    2、heapq.heappop可以弹出堆顶，并重新调整
    3、heapq.heappush 可以新增元素到堆中，并重新调整
    4、heapq.heapreplace可以替换堆顶元素，并重新调整（如果有必要）
    5、为了维持为K的大小，初始化可能需要删减，后面处理:不满K个就新增，否则做替换；
    6、heapq其实是对一个list做原地的处理，第一个元素就是最小的，直接返回就是最小的值

    """

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
``` 

## 12. 最后一块石头的重量

```html
有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 **最重的** 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，
且 x <= y。那么粉碎的可能结果如下：

  如果 x == y，那么两块石头都会被完全粉碎；
  如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。

最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。

```

1046\. 最后一块石头的重量（easy） [力扣](https://leetcode-cn.com/problems/last-stone-weight/description/)

示例 1:

```html
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。

```

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        from heapq import heapify, heappush, heappop

        # 这里注意去负号，因为下面想取最大值，而python heapq 为 小根堆 
        stones = [-i for i in stones] 
        heapify(stones) # 建堆
        while len(stones) > 0:
            y = -heappop(stones) # 堆中取最大值
            if len(stones) == 0: # 说明y为前面堆中的最后一个元素
                return y

            x = -heappop(stones) # 取堆中最大值
            if x != y:
                heappush(stones, x - y) # x - y 为一个负值

        #最后如果没有石头剩下，就返回 0
        return 0 
``` 

## 13. 最小K个数

设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可

123\. 最小K个数（easy） [力扣](https://leetcode-cn.com/problems/smallest-k-lcci/description/)

示例 1:

```html
输入： arr = [1,3,5,7,2,4,6,8], k = 4
输出： [1,2,3,4]
```

```python
class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        import heapq
        # heapq默认生成小顶堆。若要使用大顶堆，则向堆中插入负值

        if k == 0:
            return []

        heap = [] # 维护一个最大堆
        for i in arr[:k]:
            heapq.heappush(heap, -i)  # heappush(heap, val) 向堆中插入指定值并维护。

        for i in arr[k:]:
            if i < -heap[0]:  # heap[0] 返回堆中的最小值
                # heapreplace(heap, val) 将堆中的最小值替换为指定值并重新维护
                heapq.heapreplace(heap, -i)

        #任意顺序返回这k个数均可  [4,3,2,1],heappop(heap) 弹出堆中的最小值并重新维护
        #return [-x for x in heap] # 也 OK 

        return [-heapq.heappop(heap) for _ in range(k)]
``` 

## 14. 前K个高频元素

给定一个非空的整数数组，返回其中出现频率前 k 高的元素

ps: 1.你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小, 你可以按任意顺序返回答案

347\. 前K个高频元素（easy） [力扣](https://leetcode-cn.com/problems/top-k-frequent-elements/description/)

示例 1:

```html
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

输入: nums = [1], k = 1
输出: [1]
```

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic = {} # dic = Counter(nums) call func
        for num in nums:  # 统计个数
            dic[num] = dic.get(num, 0) + 1

        heap, res = [], []
        for i in dic:
            heapq.heappush(heap, (-dic[i], i)) # 建堆

        for i in range(k):
            tmp = heapq.heappop(heap)
            res.append(tmp[1])

        return res 
``` 

## 15. 数组中的第K个最大元素

在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素.

k 总是有效的，且 1 ≤ k ≤ 数组的长度

215\. 数组中的第K个最大元素（middle） [力扣](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/description/)

示例 1:

```html
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

```python
class Solution:
    import heapq
    def findKthLargest(self, nums: List[int], k: int) -> int:
        h = nums[: k] # k 总是有效
        heapq.heapify(h) # 列表原地间堆

        for i in range(k, len(nums)):
            if nums[i] > h[0]:
                #heapreplace(heap,x) 替换堆中的最小元素，并用x代替，会自动调整堆结构
                #heapreplace， This is more efficient than heappop() followed by heappush()
                #heapq.heapreplace(h, nums[i]) #OK的
                
                #heapreplace 比 下面的高效
                heapq.heappop(h) #弹出堆中的第一个元素，会自动调整堆结构
                heapq.heappush(h,nums[i]) #把元素x压到堆中，会自动调整堆结构
                
        return h[0]
``` 

## 16. 查找和最小的K对数字

给定两个以升序排列的整形数组 nums1 和 nums2, 以及一个整数 k。

定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2。

找到和最小的 k 对数字 (u1,v1), (u2,v2) ... (uk,vk)

373\. 查找和最小的K对数字（middle） [力扣](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/description/)

示例 1:

```html
输入: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
输出: [1,2],[1,4],[1,6]
解释: 返回序列中的前 3 对数：
     [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
输出: [1,1],[1,1]
解释: 返回序列中的前 2 对数：
     [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]

输入: nums1 = [1,2], nums2 = [3], k = 3 
输出: [1,3],[2,3]
解释: 也可能序列中所有的数对都被返回:[1,3],[2,3]
```

```python

class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        import heapq

        heap = []
        for num1 in nums1:
            for num2 in nums2:
                if len(heap) < k:
                    heapq.heappush(heap, (-(num1 + num2) , [num1, num2]))
                else:
                    if num1 + num2 < -heap[0][0]:
                        # heapq.heappop(heap) #分解动作
                        # heapq.heappush(heap, (-(num1 + num2), [num1, num2]))
                        
                        #heapreplace， This is more efficient than heappop() followed by heappush()
                        heapq.heapreplace(heap, (-(num1 + num2), [num1, num2]))

        return [item[1] for item in heap]
``` 
