# Leetcode - 栈和队列
<!-- GFM-TOC -->
* [Leetcode 题解 - 栈和队列](#leetcode---栈和队列)
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

##3. 最小栈

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

##4. 有效的括号

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

##5. 比较含退格的字符串

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

##6. 删除字符串中的所有相邻重复项

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

##7. 每日温度

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

##8. 字符串解码

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

##9. 栈的压入弹出序列

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

##10. 逆波兰表达式求值

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

