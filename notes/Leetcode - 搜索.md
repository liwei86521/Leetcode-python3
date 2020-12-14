# Leetcode 题解 - 搜索
<!-- GFM-TOC -->
* [Leetcode 题解 - 搜索](#leetcode-题解---搜索)
    * [BFS](#bfs)
        * [1. 二叉树的最大深度](#1-二叉树的最大深度)
        * [2. 二叉树的层次遍历 II](#2-二叉树的层次遍历-II)
        * [3. 员工的重要性](#3-员工的重要性)
        * [4. 二叉树的堂兄弟节点](#4-二叉树的堂兄弟节点)
        * [5. 二叉树的最小深度](#5-二叉树的最小深度)
    * [DFS](#dfs)
        * [1. 查找最大的连通面积](#1-查找最大的连通面积)
        * [2. 矩阵中的连通分量数目](#2-矩阵中的连通分量数目)
        * [3. 好友关系的连通分量数目](#3-好友关系的连通分量数目)
        * [4. 填充封闭区域](#4-填充封闭区域)
        * [5. 能到达的太平洋和大西洋的区域](#5-能到达的太平洋和大西洋的区域)
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

<img style="width: 432px; height: 302px;" src="https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg" alt="">

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
