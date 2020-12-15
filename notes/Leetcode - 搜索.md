# Leetcode 题解 - 搜索
<!-- GFM-TOC -->
* [Leetcode 题解 - 搜索](#leetcode-题解---搜索)
    * [BFS](#bfs)
        * [1. 二叉树的最大深度](#1-二叉树的最大深度)
        * [2. 二叉树的层次遍历 II](#2-二叉树的层次遍历-II)
        * [3. 员工的重要性](#3-员工的重要性)
        * [4. 二叉树的堂兄弟节点](#4-二叉树的堂兄弟节点)
        * [5. 二叉树的最小深度](#5-二叉树的最小深度)
        * [6. 二进制矩阵中的最短路径](#6-二进制矩阵中的最短路径)
    * [DFS](#dfs)
        * [1. 二叉树的最大深度](#1-二叉树的最大深度)
        * [2. 二叉树的最小深度](#2-二叉树的最小深度)
        * [3. 相同的树](#3-相同的树)
        * [4. 平衡二叉树](#4-平衡二叉树)
        * [5. 叶子相似的树](#5-叶子相似的树)
        * [6. 路径总和](#6-路径总和)
        * [7. 二叉树的所有路径](#7-二叉树的所有路径)
    * [Backtracking](#backtracking)
        * [1. 数字键盘组合](#1-数字键盘组合)
        * [2. IP 地址划分](#2-ip-地址划分)
        * [3. 在矩阵中寻找字符串](#3-在矩阵中寻找字符串)
        * [4. 输出二叉树中所有从根到叶子的路径](#4-输出二叉树中所有从根到叶子的路径)
        * [5. 排列](#5-排列)
<!-- GFM-TOC -->

## BFS

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/95903878-725b-4ed9-bded-bc4aae0792a9.jpg"/> </div><br>

广度优先搜索一层一层地进行遍历，每层遍历都是以上一层遍历的结果作为起点，遍历一个距离能访问到的所有节点。需要注意的是，遍历过的节点不能再次被遍历。

第一层：

- 0 -\> {6,2,1,5}

第二层：

- 6 -\> {4}
- 2 -\> {}
- 1 -\> {}
- 5 -\> {3}

第三层：

- 4 -\> {}
- 3 -\> {}

每一层遍历的节点都与根节点距离相同。设 d<sub>i</sub> 表示第 i 个节点与根节点的距离，推导出一个结论：对于先遍历的节点 i 与后遍历的节点 j，有 d<sub>i</sub> <= d<sub>j</sub>。利用这个结论，可以求解最短路径等   **最优解**   问题：第一次遍历到目的节点，其所经过的路径为最短路径。应该注意的是，使用 BFS 只能求解无权图的最短路径，无权图是指从一个节点到另一个节点的代价都记为 1。

在程序实现 BFS 时需要考虑以下问题：

- 队列：用来存储每一轮遍历得到的节点；
- 标记：对于遍历过的节点，应该将它标记，防止重复遍历。

## 1. 二叉树的最大深度

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。 说明: 叶子节点是指没有子节点的节点

139\. 二叉树的最大深度（easy） [力扣](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/description/)

示例 1:

```html
给定二叉树 [3,9,20,null,null,15,7], 返回它的最大深度 3

    3
   / \
  9  20
    /  \
   15   7

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # BFS 广度优先搜索
        if not root:#根节点判空返0
            return 0
        
        queue=[root] # 把根节点看成第0层
        dpath=0 # 二叉树最大深度
        while queue:
            temp=[]#记录下一层的节点
            dpath+=1 # 只要一进循环就更新dpath
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)

            queue=temp
            
        return dpath

``` 

## 2. 二叉树的层次遍历 II

给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

107\. 二叉树的层次遍历 II（easy） [力扣](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/description/)

示例 1:

```html
给定二叉树 [3,9,20,null,null,15,7], 返回它的最大深度 3

    3
   / \
  9  20
    /  \
   15   7

返回其自底向上的层次遍历为：
[
  [15,7],
  [9,20],
  [3]
]

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        # 使用 deque 比较好
        from collections import deque
        levels = deque() # [] 用来装每层的结果的
        queue = [root] # 把根节点看成第0层
        
        while(queue):
            next_level_nodes = []  # 记录下一次层的节点
            cur_level_res = [] # 记录当前层的所有节点 val值
            for node in queue:
                cur_level_res.append(node.val) # 把该节点值加到res中
                if node.left:
                    next_level_nodes.append(node.left)
                if node.right:
                    next_level_nodes.append(node.right)
                    
            levels.appendleft(cur_level_res)#因为 自底向上 所以用deque好
            queue = next_level_nodes # 用于下次循环
            
        return list(levels)

``` 


## 3. 员工的重要性

给定一个保存员工信息的数据结构，它包含了员工唯一的id，重要度 和 直系下属的id。

比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，员工2的数据结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于并不是直系下属，因此没有体现在员工1的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。

690\. 员工的重要性（easy） [力扣](https://leetcode-cn.com/problems/employee-importance/description/)

示例 1:

```html
输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出: 11
解释:
员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。
因此员工1的总重要度是 5 + 3 + 3 = 11。

```

```python
class Employee:
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates # 直系下属id 为list类型

class Solution:
    def getImportance(self, employees, id):
        # 典型的BFS搜索
        id_2_Employee = {e.id: e for e in employees} # 构建 员工id 与 员工Employee的映射关系       
        # eg: [[1,2,[2]], [2,3,[]]]
        #print(res) # {1: <__main__.Employee object at 0x7f6>, 2: <__main__.Employee object at 0x7f>}
        
        # 当前员工(id)的信息----> 穿过来的id
        queue = [id_2_Employee[id]]
        res = 0
        while queue:
            temp = [] # 用来装当前员工所有的下属（非下属id）
            for emp in queue:
                res += emp.importance
                temp.extend(emp.subordinates) #加入 直系下属id
                    
            queue = [id_2_Employee[eid] for eid in temp] # 把下属id 映射为 下属对象， 用于下次循环
        
        return res

``` 

## 4. 二叉树的堂兄弟节点

在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。

如果二叉树的两个节点深度相同，但父节点不同，则它们是一对堂兄弟节点。

我们给出了具有唯一值的二叉树的根节点 root，以及树中两个不同节点的值 x 和 y。

只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true。否则，返回 false。

ps：二叉树的节点数介于 2 到 100 之间。  每个节点的值都是唯一的、范围为 1 到 100 的整数。
 
993\. 二叉树的堂兄弟节点（easy） [力扣](https://leetcode-cn.com/problems/cousins-in-binary-tree/description/)

示例 1:


```html
输入：root = [1,2,3,4], x = 4, y = 3
输出：false

输入：root = [1,2,3,null,4,null,5], x = 5, y = 4
输出：true

输入：root = [1,2,3,null,4], x = 2, y = 3
输出：false
```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        #因为堂兄弟节点必须要在同一深度，因此可以用层次遍历。但是还要保证是不是同一父节点，因此还要判断一下
        #典型的 BFS 搜索
        
        if not root: # bad case
            return False
        
        queue = [root] # 第0层
        while queue:
            size = len(queue) #循环开始时，队列中的元素个数就是当前层结点的个数
            #  把一层的结点值都放进来，如果遇到空结点，放置 0
            # 因为题目说了 每个节点的值都是唯一的、范围为 1 到 100 的整数
            temp = [] # 下一层所有节点，ps 包括 None节点
            cur_level = []
            for node in queue:
                if node: # 如果该节点不为 None
                    cur_level.append(node.val)
                    temp.append(node.left) # 可能包括None节点
                    temp.append(node.right)
                else:
                    cur_level.append(0)#加一个0,是为了方便判断是否为同一父节点
                    
            
            # 如果这两个索引都在一层，只有一种情况需要排除
            # 那就是两个结点挨着，并且索引小的结点的索引是偶数
            if x in cur_level and y in cur_level:
                index1 = cur_level.index(x)
                index2 = cur_level.index(y)
                if index1 > index2: # 交换位置 保证index1 小
                    index1, index2 = index2, index1

                if index1 + 1 == index2 and index1 % 2 == 0:# index1不能为偶数
                    return False

                return True
            
            # 如果索引不在同一层，直接就可以返回不是堂兄弟结点了
            if x in cur_level or y in cur_level:
                return False
            
            queue = temp # 进行下一次循环
            
        return False # ps 这里是防止 x = 4, y = 3的值都不在二叉树中的情况

``` 

## 5. 二叉树的最小深度

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。
 
993\. 二叉树的最小深度（easy） [力扣](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/description/)

示例 1:

```html
输入：root = [3,9,20,null,null,15,7]
输出：2
示例 2：

输入：root = [2,null,3,null,4,null,5,null,6]
输出：5

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        #广度优先BFS（人类思维）: 层次遍历(列队实现) 这个好理解一点
        if not root:# 根节点判空返 0
            return 0
                
        queue=[root] # 把根节点看成第0层
        ans=0
        while queue:
            ans += 1
            temp=[]#记录下一层的节点
            for cur in queue:
                #只要当前层有叶子节点则返回
                if cur.left is None and cur.right is None:
                    return ans

                if cur.left: temp.append(cur.left)
                if cur.right: temp.append(cur.right)

            queue=temp

``` 

## 6. 二进制矩阵中的最短路径

在一个 N × N 的方形网格中，每个单元格有两种状态：空（0）或者阻塞（1）。

一条从左上角到右下角、长度为 k 的畅通路径，由满足下述条件的单元格 C_1, C_2, ..., C_k 组成：
<ul>
	<li>相邻单元格&nbsp;<code>C_i</code> 和&nbsp;<code>C_{i+1}</code>&nbsp;在八个方向之一上连通（此时，<code>C_i</code> 和&nbsp;<code>C_{i+1}</code>&nbsp;不同且共享边或角）</li>
	<li><code>C_1</code> 位于&nbsp;<code>(0, 0)</code>（即，值为&nbsp;<code>grid[0][0]</code>）</li>
	<li><code>C_k</code>&nbsp;位于&nbsp;<code>(N-1, N-1)</code>（即，值为&nbsp;<code>grid[N-1][N-1]</code>）</li>
	<li>如果 <code>C_i</code> 位于&nbsp;<code>(r, c)</code>，则 <code>grid[r][c]</code>&nbsp;为空（即，<code>grid[r][c] ==&nbsp;0</code>）</li>
</ul>

返回这条从左上角到右下角的最短畅通路径的长度。如果不存在这样的路径，返回 -1 。
 
1091\. 二进制矩阵中的最短路径（middle） [力扣](https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/description/)

示例 1:

![image](https://github.com/liwei86521/Leetcode-python3/blob/main/pics/example2_2.png?raw=true)

<pre><strong>输入：</strong>[[0,1],[1,0]]
<strong>输出：</strong>2
</pre>

<pre><strong>输入：</strong>[[0,0,0],[1,1,0],[1,1,0]]
<strong>输出：</strong>4
</pre>

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # 若起始点或终点堵塞，则不可能有这样的路径
        if not grid or grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        elif n <= 2:
            return n

        queue = [(0, 0, 1)] # 先压入起点，元组最后一位表示到达这里最短畅通路径的长度
        grid[0][0] = 1 # mark as visited
        while queue:
            i, j, step = queue.pop(0)# 元组解包
            for dx, dy in [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1),(0,-1), (1,-1)]: #8个方向
                if i+dx == n-1 and j+dy == n-1: #到达右下角
                    return step + 1

                # 处理边界及是否能够访问
                if 0 <= i+dx < n and 0 <= j+dy < n and grid[i+dx][j+dy] == 0:
                    queue.append((i+dx, j+dy, step+1))
                    grid[i+dx][j+dy] = 1  # mark as visited

        return -1

``` 


## DFS

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/74dc31eb-6baa-47ea-ab1c-d27a0ca35093.png"/> </div><br>

广度优先搜索一层一层遍历，每一层得到的所有新节点，要用队列存储起来以备下一层遍历的时候再遍历。

而深度优先搜索在得到一个新节点时立即对新节点进行遍历：从节点 0 出发开始遍历，得到到新节点 6 时，立马对新节点 6 进行遍历，得到新节点 4；如此反复以这种方式遍历新节点，直到没有新节点了，此时返回。返回到根节点 0 的情况是，继续对根节点 0 进行遍历，得到新节点 2，然后继续以上步骤。

从一个节点出发，使用 DFS 对一个图进行遍历时，能够遍历到的节点都是从初始节点可达的，DFS 常用来求解这种   **可达性**   问题。

在程序实现 DFS 时需要考虑以下问题：

- 栈：用栈来保存当前节点信息，当遍历新节点返回时能够继续遍历当前节点。可以使用递归栈。
- 标记：和 BFS 一样同样需要对已经遍历过的节点进行标记。


## DFS

## 1. 二叉树的最大深度

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
 
104\. 二叉树的最大深度（easy） [力扣](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/description/)

示例 1:

```html
给定二叉树 [3,9,20,null,null,15,7], 返回它的最大深度 3 

    3
   / \
  9  20
    /  \
   15   7

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # DFS 递归   分治法
        if not root: return 0
            
        left_maxDepth = self.maxDepth(root.left)
        right_maxDepth = self.maxDepth(root.right)
        return max(left_maxDepth, right_maxDepth) + 1

``` 

## 2. 二叉树的最小深度

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量
 
111\. 二叉树的最小深度（easy） [力扣](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/description/)

示例 1:

```html
输入：root = [3,9,20,null,null,15,7]
输出：2

输入：root = [2,null,3,null,4,null,5,null,6]
输出：5

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        #  分治（递归的高级应用）
        if not root: return 0

        # root左子树为None，而右子树不为None，说明得在右子树中找叶子结点
        if root.left is None and root.right is not None:
            return self.minDepth(root.right) + 1

        # root左子树不为None，而右子树为None，说明得在左子树中找叶子结点
        if root.left is not None and root.right is None:
            return self.minDepth(root.left) + 1

        #divide and conquer
        leftMinDepth = self.minDepth(root.left)
        rightMinDepth = self.minDepth(root.right)
	
	results = 1 + min(leftMinDepth, rightMinDepth)

        return results

``` 

## 3. 相同的树

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
 
100\. 相同的树（easy） [力扣](https://leetcode-cn.com/problems/same-tree/description/)

示例 1:

```html
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true

输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false


输入:       1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

输出: false

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # 只要是树，DFS
        if not p and not q: # 2树都为空
            return True
        elif p is not None and q is not None:#2树都不为空
            if p.val == q.val:
                return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            else:
                return False
        else:
            return False

``` 

## 4. 平衡二叉树
给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
 
110\. 平衡二叉树（easy） [力扣](https://leetcode-cn.com/problems/balanced-binary-tree/description/)

示例 1:

```html
输入：root = [3,9,20,null,null,15,7]
输出：true

输入：root = [1,2,2,3,3,null,null,4,4]
输出：false

输入：root = []
输出：true

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 遍历每一节点的左右子树的高度，判定其是否符合条件；
        # 只要发现其不符合，立即退出，判定其不是平衡二叉树
        if root is None:
            return True
        
        # 分别定义左右子树的高度
        left_depth = 0
        right_depth = 0
        if root.left:
            left_depth = self.get_max_depth(root.left)
        if root.right:
            right_depth = self.get_max_depth(root.right)
            
        if abs(left_depth - right_depth) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)

    # 获取某一节点对应树的最大高度 
    def get_max_depth(self, root):
        if root is None:
            return 0
        else:
            left_cnt = self.get_max_depth(root.left)
            right_cnt= self.get_max_depth(root.right)
            return max(left_cnt, right_cnt)+1

``` 

## 5. 叶子相似的树

请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。

如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。

如果给定的两个头结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 
 
872\. 叶子相似的树（easy） [力扣](https://leetcode-cn.com/problems/leaf-similar-trees/description/)

示例 1:

```html
输入：root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
输出：true

输入：root1 = [1], root2 = [1]
输出：true

输入：root1 = [1,2], root2 = [2,2]
输出：true

输入：root1 = [1,2,3], root2 = [1,3,2]
输出：false
```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        # 定义 添加叶子节点值的函数， DFS递归 好好理解一下
        # def foo(root, res=[])  XXX
        def dfs(root): # 不要用 可变变量进行初始化 XXX
            if root is None:
                return 

            #左右子树都为空，该节点才是叶子节点
            if (root.left is None) and (root.right is None):
                temp.append(root.val)

            dfs(root.left)
            dfs(root.right)
        
        temp = [] # free 变量
        dfs(root1)
        temp1 = temp # 浅copy 

        temp = [] # 提前置空防止
        dfs(root2)
        return temp1 == temp

``` 

## 6. 路径总和

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
 
872\. 路径总和（easy） [力扣](https://leetcode-cn.com/problems/path-sum/description/)

示例 1:

```html
给定如下二叉树，以及目标和 sum = 22，

              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1

返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right: # 叶子节点
            return sum == root.val

        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

``` 

## 7. 二叉树的所有路径

给定一个二叉树，返回所有从根节点到叶子节点的路径
 
257\. 二叉树的所有路径（easy） [力扣](https://leetcode-cn.com/problems/binary-tree-paths/description/)

示例 1:

```html
   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def binaryTreePaths(self, root):

        def dfs(root, path):
            if root:
                path += str(root.val)
                if not root.left and not root.right:  # 当前节点是叶子节点
                    paths.append(path)  # 把路径加入到答案中
                else:
                    path += '->'  # 当前节点不是叶子节点，继续递归遍历
                    dfs(root.left, path)
                    dfs(root.right, path)

        paths = [] # free 变量（闭包）
        dfs(root, '')
        return paths

``` 
