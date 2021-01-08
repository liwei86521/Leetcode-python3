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
        * [8. 岛屿的最大面积](#8-岛屿的最大面积) 
        * [9. 岛屿数量](#9-岛屿数量) 
    * [Backtracking](#backtracking)
        * [1. 电话号码的字母组合](#1-电话号码的字母组合)
        * [2. 括号生成](#2-括号生成)
        * [3. 组合总和](#3-组合总和)
        * [4. 组合总和 II](#4-组合总和-II)
        * [5. 组合总和 III](#5-组合总和-III)
        * [6. 路径总和 II](#6-路径总和-II)
        * [7. 路径总和 III](#7-路径总和-III)
        * [8. 全排列](#8-全排列)
        * [9. 全排列 II](#9-全排列-II)
        * [10. 子集](#10-子集)
        * [11. 子集 II](#11-子集-II)
        * [12. 递增子序列](#12-递增子序列)
        * [13. 无重复字符串的排列组合](#13-无重复字符串的排列组合)
        * [14. 矩阵中的路径](#14-矩阵中的路径)
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
        """
        left_depth = 0 # 可以直接用下面的2句替代
        right_depth = 0
        if root.left:
            left_depth = self.get_max_depth(root.left)
        if root.right:
            right_depth = self.get_max_depth(root.right)
        """
        
        left_depth = self.get_max_depth(root.left)
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

## 8. 岛屿的最大面积

给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 )
 
695\. 岛屿的最大面积 [力扣](https://leetcode-cn.com/problems/max-area-of-island/description/)

示例 1:

```html
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]

对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1

[[0,0,0,0,0,0,0,0]]

对于上面这个给定的矩阵, 返回 0

```

```python
class Solution:
    def dfs(self, grid, cur_i, cur_j):
        if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) 
                             or grid[cur_i][cur_j] != 1:

            return 0

        grid[cur_i][cur_j] = 0 # 防止重复计算面积
        ans = 1 # 岛屿初始值设为1
        for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]: # 返回4个方向的和
            next_i, next_j = cur_i + di, cur_j + dj
            ans += self.dfs(grid, next_i, next_j)
        return ans

    # 找出某一岛屿面积思路: 1.从某位置出发，向4个方向探寻相连的土地
    # 2. 每探寻到一块土地，计数加一  3.确保每块土地只被探寻一次
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ans = 0
        for i, l in enumerate(grid):
            for j, n in enumerate(l):
                ans = max(self.dfs(grid, i, j), ans)
        return ans

``` 

## 9. 岛屿数量

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 
200\. 岛屿数量（middle） [力扣](https://leetcode-cn.com/problems/number-of-islands/description/)

示例 1:

```html
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

```

```python
class Solution:
    def dfs(self, grid, r, c):
        grid[r][c] = "0" # 将当期格的值设为0，表示已经遍历过
        nr, nc = len(grid), len(grid[0]) # 数组行和列
        # 从上下左右4个方向进行dfs
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    self.dfs(grid, r, c)
        
        return num_islands

``` 


## Backtracking

Backtracking（回溯）属于 DFS。

- 普通 DFS 主要用在   **可达性问题**  ，这种问题只需要执行到特点的位置然后返回即可。
- 而 Backtracking 主要用于求解   **排列组合**   问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，这种问题在执行到特定的位置返回之后还会继续执行求解过程。

因为 Backtracking 不是立即返回，而要继续求解，因此在程序实现时，需要注意对元素的标记问题：

- 在访问一个新元素进入新的递归调用时，需要将新元素标记为已经访问，这样才能在继续递归调用时不用重复访问该元素；
- 但是在递归返回时，需要将元素标记为未访问，因为只需要保证在一个递归链中不同时访问一个元素，可以访问已经访问过但是不在当前递归链中的元素。

[详解可变 不可变数据类型 引用 深浅拷贝](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/solution/yu-dao-jiu-shen-jiu-xiang-jie-ke-bian-bu-ke-bian-s/)

```html
回溯法写法思路：
    1. 定义全局结果数组
    2. 调用递归函数
    3. 返回全局结果数组
    4. 定义递归函数
      1) 参数，动态变化，一般为分支结果、限制条件等
      2) 终止条件，将分支结果添加到全局数组
      3) 剪枝条件
      4) 调用递归逐步产生结果，回溯搜索下一结果

total = 0 # 1716039872
print(id(total))
total -= 2 # 1716039808
print(id(total))

p = [] # 可变类型
print(id(p)) # 2305522011464
p.append(3) # 如果是 p= p + [3]  则会开辟新的内存空间, 就达不到节约内存的目的
print(id(p)) # 2305522011464

#结论是 如果使用可变类型变量(eg: [], dict, {}集合)就必须选择和撤销，好处就是可以节省空间
#      如果使用不可变类型变量(eg: int, str, tuple)就不需要 选择和撤销， 因为操作的不是同一块内存空间

详解可变 不可变数据类型 引用 深浅拷贝
############# A ##############
c = [1,2]
d = c
print(id(c),id(d))
# 2302138284424 2302138284424
############# B ##############
c.append(-1)
print(c,d)
# [1, 2, -1] [1, 2, -1]
############# C ##############


############# A ##############
a = [1,2]
b = a
print(a,id(a),b,id(b))
# [1, 2] 2849693365896
# [1, 2] 2849693365896 

############# B ##############
b+=[3]
# 等价于 b.__iadd__([3]) 
print(a,id(a),b,id(b))
# [1, 2, 3] 2849693365896 
# [1, 2, 3] 2849693365896 

############# C ##############
b = b + [4]
# 等价于 b = b.__add__([3])
print(a,id(a),b,id(b))
# [1, 2, 3] 2849693365896
# [1, 2, 3, 4] 2849694074760

首先说明浅拷贝方式：
1、完全切片操作[:]
2、list()，dict()
3、copy模块的copy()方法

############# A ##############
a = ['a', 'b', ['c', 'd', 'e']]
b = a.copy()
print(id(a),id(b))
# 2302137454088 2302138282184
############# B ##############
a.append('e')
print(a,b)
# a: ['a', 'b', ['c', 'd', 'e'], 'e'] 
# b: ['a', 'b', ['c', 'd', 'e']]
a[0] = 'g'
print(a,b)
# a: ['g', 'b', ['c', 'd', 'e'], 'e'] 
# b: ['a', 'b', ['c', 'd', 'e']]
############# C ##############
a[2].append('f')
print(a,b)
# a: ['g', 'b', ['c', 'd', 'e', 'f'], 'e']
# b: ['a', 'b', ['c', 'd', 'e', 'f']]
print(id(a[2]),id(b[2]))
# 2302137664264 2302137664264

深拷贝
首先说明深拷贝方式：
1、copy.deepcopy()   就是彻头彻尾的全部复制，没有一点引用

import copy
first = {'name':'rocky','lanaguage':['python','c++','java']}
second = copy.deepcopy(first)
>>> second
{'name': 'rocky', 'lanaguage': ['python', 'c++', 'java']}
second['lanaguage'].remove('java')
>>> second
{'name': 'rocky', 'lanaguage': ['python', 'c++']}
>>> first
{'name': 'rocky', 'lanaguage': ['python', 'c++', 'java']}
```

## 1. 电话号码的字母组合

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/9823768c-212b-4b1a-b69a-b3f59e07b977.jpg"/> </div><br>

17\. Letter Combinations of a Phone Number (Medium)  [力扣](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/description/)

```html
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
```

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone = {
            '2': list('abc'),
            '3': list('def'),
            '4': list('ghi'),
            '5': list('jkl'),
            '6': list('mno'),
            '7': list('pqrs'),
            '8': list('tuv'),
            '9': list('wxyz'),
        }

        res = []  # step1: 定义全局结果数组

        # ps: backtrack(digits, path=[])  错误的 --> 不要使用可变类型作为参数的默认值
        # step4: 定义递归函数(1:参数定义，2：终止条件，3：剪枝条件，4：调用递归逐步产生结果，回溯搜索下一结果
        def backtrack(path, digits): # 回溯
            if len(digits) == 0:
                res.append("".join(path[:]))  # path[:] == path.copy()
                return

            for c in phone[digits[0]]:  # 选择列表
                path.append(c)  # 选择
                backtrack(path, digits[1:])
                # path是一个可变类型变量 内存地址一直都不会变，所以需要 撤销
                path.pop()  # 撤销选择

        # step2: 调用递归函数
        if digits:  # 放在digits为空
            backtrack([], digits)

        return res  # step3:返回全局结果数组

    def letterCombinations_v1(self, digits: str) -> List[str]:
        phone = {
            '2': list('abc'),
            '3': list('def'),
            '4': list('ghi'),
            '5': list('jkl'),
            '6': list('mno'),
            '7': list('pqrs'),
            '8': list('tuv'),
            '9': list('wxyz'),
        }

        res = []  # step1: 定义全局结果数组

        # step4: 定义递归函数(1:参数定义，2：终止条件，3：剪枝条件，4：调用递归逐步产生结果，回溯搜索下一结果
        def backtrack(path, digits):
            if len(digits) == 0:
                res.append("".join(path[:]))  # path[:] == path.copy()
                return

            for c in phone[digits[0]]:  # 选择列表
                # path.append(c) #选择
                # backtrack(path, digits[1:]) # path+[c] 会产生新的内存空间 所以不用撤销和选择
                backtrack(path + [c], digits[1:])  # 这样比较浪费空间,最后不要这样写
                # path.pop() # 撤销选择

        # step2: 调用递归函数
        if digits:  # 放在digits为空
            backtrack([], digits)

        return res  # step3:返回全局结果数组

    def letterCombinations_v2(self, digits: str) -> List[str]:
            if not digits: return []

            phone = {'2': ['a', 'b', 'c'],
                     '3': ['d', 'e', 'f'],
                     '4': ['g', 'h', 'i'],
                     '5': ['j', 'k', 'l'],
                     '6': ['m', 'n', 'o'],
                     '7': ['p', 'q', 'r', 's'],
                     '8': ['t', 'u', 'v'],
                     '9': ['w', 'x', 'y', 'z']}

            def backtrack(path_str, nextdigit):
                if len(nextdigit) == 0:
                    res.append(path_str)
                else:
                    for letter in phone[nextdigit[0]]:
                        # ps: path_str + letter 会开辟一个新的 内存空间， 所以不用选择和撤销
                        # 坏处就是 浪费内存空间
                        backtrack(path_str + letter, nextdigit[1:])

            res = []
            backtrack('', digits)
            return res

``` 

## 2. 括号生成

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

22\. 括号生成（middle） [力扣](https://leetcode-cn.com/problems/generate-parentheses/description/)

示例 1:

```html
输入：n = 3
输出：[
       "((()))",
       "(()())",
       "(())()",
       "()(())",
       "()()()"
     ]

```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        '''
        Backtracking都是这样的思路：在当前局面下，你有若干种选择。那么尝试每一种选择。如果已经发现某种
        选择肯定不行（因为违反了某些限定条件），就返回；如果某种选择试到最后发现是正确解，就将其加入解集
        所以思考回溯题时，只要明确三点就行：选择 (Options)，限制 (Restraints)，结束条件 (Termination)
        '''

        # step4:定义递归函数(1:参数 2:终止条件 3:剪枝条件 4:调用递归逐步产生结果
        def backTrack(left, right, n, result):
            if left == n and right == n:
                res.append(result)
                return

            if left < n:
                backTrack(left+1, right, n, result+"(") # 会开辟新的内存，所以不用撤销选择

            if left > right and right < n:
                backTrack(left, right+1, n, result+")")

        res = [] # step1: 定义全局结果数组
        backTrack(0, 0, n, "") #step2: 调用递归函数

        return res #step3: 返回全局结果数组


    def generateParenthesis_v2(self, n: int) -> List[str]:
        # 推荐
        def backtrack(S, left, right, n):
            if left == n and right == n:
                ans.append(''.join(S))
                return

            if left < n:
                S.append('(') # 节省内存空间
                backtrack(S, left+1, right, n)
                S.pop()

            if right < left:
                S.append(')')
                backtrack(S, left, right+1, n)
                S.pop()

        ans = []
        backtrack([], 0, 0, n)

        return ans

``` 

## 3. 组合总和

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取,ps 所有数字（包括 target）都是正整数 解集不能包含重复的组合

39\. 组合总和（middle） [力扣](https://leetcode-cn.com/problems/combination-sum/description/)

示例 1:

```html
输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]


输入：candidates = [2,3,5], target = 8,
所求解集为：
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        def backtrack(candidates, total, path, level):
            # 1 recursion terminator
            if total == 0:
                res.append(path[:])
                return

            for i in range(level, len(candidates)):
                if total > 0:
                    # 2 process data
                    path.append(candidates[i])
    
                    # 3 drill down
                    backtrack(candidates, total-candidates[i], path, i) # 这里i 是关键
                    
                    # reverse the current level status
                    path.pop() # path是一个变量 内存地址一直都不会变，所以需要 撤销

                else: # 剪枝
                    break # 跳出循环

        res = []
        candidates.sort() #用来剪枝，配合break
        backtrack(candidates, target, [], 0)

        return res

``` 

## 4. 组合总和 II

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次

###说明
<ul>
	<li>所有数字（包括目标数）都是正整数。</li>
	<li>解集不能包含重复的组合。&nbsp;</li>
</ul>

candidates 中的数字可以无限制重复被选取,ps 所有数字（包括 target）都是正整数 解集不能包含重复的组合

39\. 组合总和 II（middle） [力扣](https://leetcode-cn.com/problems/combination-sum-ii/description/)

示例 1:

```html
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
```

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

        def backtrack(candidates, target, level, path, used):
            if target == 0:
                res.append(path[:])

            for i in range(level, len(candidates)):
                if target > 0: # 剪枝
                    # 剪枝条件去掉 [1, 2, 5], [1, 2, 5] 其中 第二个[1, 2, 5] 中的1为list中的第二1
                    if i > 0 and candidates[i] == candidates[i-1] and used[i-1]==False:
                        continue

                    path.append(candidates[i])
                    used[i] = True

                    # candidates 中的每个数字在每个组合中只能使用一次 故 用i加1
                    backtrack(candidates, target - candidates[i], i+1, path, used)

                    path.pop() # 这里3个都要撤销选择 是由于公用内存地址
                    used[i] = False

        candidates.sort() #用来剪枝
        used = [False] * len(candidates)
        res = []

        backtrack(candidates, target, 0, [], used)

        return res

```

## 5. 组合总和 III

找出所有相加之和为 n 的 k 个数的组合。**组合中只允许含有 1 - 9 的正整数**，并且每种组合中不存在重复的数字。

216\. 组合总和 III（middle） [力扣](https://leetcode-cn.com/problems/combination-sum-iii/description/)

示例 1:

```html
输入: k = 3, n = 7
输出: [[1,2,4]]

输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]

```

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:

        def backtrack(path, total, level):
            # recursion terminator condition
            if k == len(path) and total == 0 :
                res.append(path[:])
                return

            for i in range(level, 10):# 这里用10 替代 n+1 因为 组合中只允许含有 1 - 9 的正整数
                if total >= i: # 剪枝
                    path.append(i) # 选择
                    #backtrack(path+[i], total-i, i+1) # 这一行抵3行，但是浪费空间，不推荐
                    backtrack(path, total-i, i+1)
                    path.pop() # 撤销


        res = []
        backtrack([], n, 1)

        return res

``` 

## 6. 路径总和 II

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

说明: 叶子节点是指没有子节点的节点。

113\. 路径总和 II（middle） [力扣](https://leetcode-cn.com/problems/path-sum-ii/description/)

示例 1:

```html
给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1

返回:

[
   [5,4,11,2],
   [5,8,4,5]
]

```

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        def backtrack(root, total, path):
            if not root: return 
            # recursion terminator condition 根节点到叶子节点的路径
            if root.left is None and root.right is None:
                if total - root.val== 0:
                    res.append(path[:]+[root.val])
                    return 
                    
            path.append(root.val) # 选择
            
            backtrack(root.left, total-root.val, path)
            backtrack(root.right, total-root.val, path)
            
            path.pop() # 撤销 (因为path同一个变量 指向同一内存空间)
        
        if not root: return [] # bad case

        res = []
        backtrack(root, sum, [])
        return res

``` 

## 7. 路径总和 III

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数

437\. 路径总和 III（middle） [力扣](https://leetcode-cn.com/problems/path-sum-iii/description/)

示例 1:

```html
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11

```

```python
class Solution:
    def __init__(self): #初始化出一个 类属性用来计数
        self.path_num=0

    def pathSum(self, root: TreeNode, sum: int) -> int:
        if not root: return self.path_num

        self.dfs(root, sum)

        self.pathSum(root.left, sum) # 因为 路径不需要从根节点开始 
        self.pathSum(root.right, sum)

        return self.path_num
    
    def dfs(self, root, total):
        if not root: return 
        if total - root.val == 0: # 也不需要在叶子节点结束
            self.path_num +=1

        self.dfs(root.left, total - root.val) 
        self.dfs(root.right, total - root.val)

``` 

## 8. 全排列

给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

46\. 全排列（middle） [力扣](https://leetcode-cn.com/problems/permutations/description/)

示例 1:

```html
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        #回溯：树的深度遍历（选择，撤销选择，剪枝）
        def dfs(nums, path, depath, used):# path变量是一个栈
            if depath == len(nums):# 递归的终止条件是，数已经选够
                # 因为只用path一个变量，是指向一个内存空间，所以要用copy
                res.append(path[:]) #path[:] == path.copy()
                return  # 退出函数，不往下继续进行

            for i in range(len(nums)):
                if not used[i]: # 剪枝非常重要
                    path.append(nums[i]) #选择
                    used[i] = True # 代表该节点已经选择过

                    dfs(nums, path, depath+1, used)

                    path.pop()  # 撤销选择
                    used[i] = False

        res = []
        used = [False for _ in range(len(nums))] # 标记节点访问状态
        dfs(nums, [], 0, used)

        return res
``` 

## 9. 全排列 II

给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列

47\. 全排列 II（middle） [力扣](https://leetcode-cn.com/problems/permutations-ii/description/)

示例 1:

```html
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]


输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        #回溯：树的深度遍历（选择，撤销选择，剪枝）
        def dfs(nums, path, depath, used):# path变量是一个栈
            if depath == len(nums):# 递归的终止条件是，数已经选够
                # 因为只用path一个变量，是指向一个内存空间，所以要用copy
                res.append(path[:]) #path[:] == path.copy()
                return  # 退出函数，不往下继续进行

            for i in range(len(nums)):
                if not used[i]:
                    #增加剪枝条件,特别注意used[i-1]==False 条件，
                    #因为不能把 [1,1,2] 给去掉 选择第2个1时第一个1的状态为True
                    #相比 46. 全排列 只是增加了剪枝条件其余代码全不用变
                    if i>0 and nums[i]==nums[i-1] and used[i-1]==False:
                        continue # 进行下一次循环

                    path.append(nums[i]) #选择
                    used[i] = True # 代表该节点已经选择过

                    dfs(nums, path, depath+1, used)

                    path.pop()  # 撤销选择
                    used[i] = False

        res = []
        used = [False for _ in range(len(nums))]
        nums.sort() # 先对列表排序, 因为要 nums[i]==nums[i-1]
        dfs(nums, [], 0, used)

        return res
``` 

## 10. 子集

给定一组**不含重复元素**的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：**解集不能包含重复的子集**。

78\. 子集（middle） [力扣](https://leetcode-cn.com/problems/subsets/description/)

示例 1:

```html
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 回溯法
        def backtrack(nums, path, depath):
            # 递归结束条件 比较特殊， 都满足，只要进来就res apppend
            res.append(path[:]) # path[:] == path.copy()

            #注意这里用(depath, n)，不用(0, n) 过滤前面用过的数字
            for i in range(depath, n): # 通过depath 结束递归 ！！！
                path.append(nums[i]) # 选择 

                backtrack(nums, path, i+1) # 这里用 i+1 进行剪枝

                path.pop() # 撤销 (因为共用一个变量path 所以要会在现场)

        res = []
        n = len(nums)
        backtrack(nums, [], 0)
        
        return res
``` 

## 11. 子集 II

给定一个**可能包含重复元素**的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：**解集不能包含重复的子集**。

90\. 子集 II（middle） [力扣](https://leetcode-cn.com/problems/subsets-ii/description/)

示例 1:

```html
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]

```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        #相对与78.子集 增加了一个sort()和used节点标记，剪枝条件
        def backtrack(nums, path, depath, used):
            # 递归结束条件 比较特殊， 都满足，只要进来就res apppend
            res.append(path[:]) #path[:] == path.copy()

            #注意这里用(depath, n)，不用(0, n) 过滤前面用过的数字
            for i in range(depath, n):# 通过depath 结束递归
                # 剪枝 去重 
                if i > 0 and nums[i] == nums[i-1] and used[i-1]==False:
                    continue

                path.append(nums[i]) # path为栈
                used[i] = True

                backtrack(nums, path, i+1, used) # i+1 是不使用重复元素

                path.pop()
                used[i] = False 

        res = []
        n = len(nums)
        nums.sort() # 目的去重 nums[i] == nums[i-1]
        used = [False for _ in range(n)] # 增加一个节点是否访问过的标记
        backtrack(nums, [], 0, used)

        return res

``` 

## 12. 递增子序列

给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。

给定数组中可能包含重复数字，相等的数字应该被视为递增的一种情况

491\. 递增子序列（middle） [力扣](https://leetcode-cn.com/problems/increasing-subsequences/description/)

示例 1:

```html
输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]

```

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:

        def backtrace(nums, temp):
            if len(temp)>=2 and temp not in res:
                res.append(temp)

            if not nums:return

            for i in range(len(nums)):
                #temp是一个递增数组  进来的数必须大于等于temp 数组最后一个
                if not temp or nums[i] >= temp[-1]:
                    backtrace(nums[i+1:], temp+[nums[i]])

        res = []
        backtrace(nums, [])

        return res

``` 

## 13. 无重复字符串的排列组合

**无重复字符串**的排列组合。编写一种方法，计算某字符串的所有排列组合，**字符串每个字符均不相同**

493\. 无重复字符串的排列组合（middle） [力扣](https://leetcode-cn.com/problems/permutation-i-lcci/description/)

示例 1:

```html
 输入：S = "qwe"
 输出：["qwe", "qew", "wqe", "weq", "ewq", "eqw"]


 输入：S = "ab"
 输出：["ab", "ba"]

```

```python
class Solution:
    def permutation(self, S: str) -> List[str]:

        def backtrack(path_s, n):
            if len(path_s) == n:
                res.append(path_s)
                return  # 退出

            for i in range(n):
                if not used[i]:
                    used[i] = True # 标记使用了
                    backtrack(path_s+S[i], n)
                    used[i] = False

        res = []
        n =len(S)
        used =[False] * n #标记是否选择过了
        backtrack("", n)

        return res
``` 



## 14. 矩阵中的路径

```html
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
 ["s","f","c","s"],
 ["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
```

496\. 矩阵中的路径（middle） [力扣](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/description/)

示例 1:

```html
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false

```

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        def dfs(i, j, k):
            #边界溢出or不相等的情况
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: 
                return False
            
            # 如果word判断完了
            if k == len(word) - 1: return True

            board[i][j] = '' #置空 代表此元素已访问过，防止之后搜索时重复访问
            # 左、右、上、下 4个方向
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            board[i][j] = word[k] #board[i][j] 元素还原至初始值，即 word[k]

            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0): 
                    return True

        return False

``` 
