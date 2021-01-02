# Leetcode - 数组与矩阵
<!-- GFM-TOC -->
* [Leetcode 数组与矩阵](#leetcode-数组与矩阵)
    * [1. 翻转图像](#1-翻转图像)
    * [2. 转置矩阵](#2-转置矩阵)
    * [3. 矩阵置零](#3-矩阵置零)
    * [4. 旋转图像](#4-旋转图像)
    * [5. 螺旋矩阵](#5-螺旋矩阵)
    * [6. 螺旋矩阵 II](#6-螺旋矩阵-II)
    * [7. 有效的数独](#7-有效的数独)
<!-- GFM-TOC -->

## 1. 翻转图像

```html
给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。

水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。

反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。
```

832\. 翻转图像（easy） [力扣](https://leetcode-cn.com/problems/flipping-an-image/description/)

示例 1:

```html
输入: [[1,1,0],[1,0,1],[0,0,0]]
输出: [[1,0,0],[0,1,0],[1,1,1]]
解释: 首先翻转每一行: [[0,1,1],[1,0,1],[0,0,0]]；
     然后反转图片: [[1,0,0],[0,1,0],[1,1,1]]

输入: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
输出: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
解释: 首先翻转每一行: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]]；
     然后反转图片: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]

```

```python

class Solution:
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        # j ^ 1  --> 0 全部被 1 替换， 1 全部被 0 替换

        return [[j ^ 1 for j in row[::-1]] for row in A]
``` 

## 2. 转置矩阵

给定一个矩阵 A， 返回 A 的转置矩阵。

矩阵的转置是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引

867\. 转置矩阵（easy） [力扣](https://leetcode-cn.com/problems/transpose-matrix/description/)

示例 1:

```html
1 <= A.length <= 1000
1 <= A[0].length <= 1000

输入：[[1,2,3],[4,5,6],[7,8,9]]
输出：[[1,4,7],[2,5,8],[3,6,9]]


输入：[[1,2,3],[4,5,6]]
输出：[[1,4],[2,5],[3,6]]

```

```python
class Solution:
    def transpose(self, A: List[List[int]]) -> List[List[int]]:
        # # 行变列   ps: 1 <= A[0].length <= 1000
        # res = []
        # for i in range(len(A[0])):
        #     res.append([row[i] for row in A])

        # return res

        # 上面的 浓缩成列表生成式
        return [[row[i] for row in A] for i in range(len(A[0]))]
``` 

## 3. 矩阵置零

给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用**原地**算法

73\. 矩阵置零（middle） [力扣](https://leetcode-cn.com/problems/set-matrix-zeroes/description/)

示例 1:

```html
输入: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]

输出: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]


输入: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]

输出: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

```

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # O(m+n) space complexity
        m = [None] * len(matrix) # 用2个1维数组替代一个二维数组
        n = [None] * len(matrix[0])
        
        for i in range(len(matrix)): # 行
            for j in range(len(matrix[0])): # 列
                if (matrix[i][j] == 0):
                    m[i] = 1
                    n[j] = 1
                    
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (m[i] == 1 or n[j] == 1):
                    matrix[i][j] = 0

    '''
    def setZeroes(self, matrix: List[List[int]]) -> None:
        # Do not return anything, modify matrix in-place instead.

        row_n, col_n = len(matrix), len(matrix[0])

        # 先找到 所以为 0 的坐标
        temp = [[i, j] for i in range(row_n) for j in range(col_n) if matrix[i][j]==0]

        row_ind_0 = set([row[0] for row in temp])
        col_ind_0 = set([row[1] for row in temp])

        for i in range(row_n):
            for j in range(col_n):
                if i in row_ind_0 or j in col_ind_0:
                    matrix[i][j] = 0
        
    '''
``` 

## 4. 旋转图像

给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

你必须在**原地**旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要**使用另一个矩阵来旋转图像。

48\. 旋转图像（middle） [力扣](https://leetcode-cn.com/problems/rotate-image/description/)

示例 1:

```html
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]


给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

```

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        #矩阵顺时针旋转90度，就相当于先做转置，再水平翻转一下
        n = len(matrix)        
        # transpose matrix
        for i in range(n):
            for j in range(i, n):# ps: j >= i
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j] 
        
        # reverse each row
        for i in range(n):
            #matrix[i].reverse() # 每行列表水平翻转一下
            matrix[i] = matrix[i][::-1]
``` 

## 5. 螺旋矩阵

给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素

54\. 螺旋矩阵（middle） [力扣](https://leetcode-cn.com/problems/spiral-matrix/description/)

示例 1:

```html
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]


输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

```

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        #与 59. 螺旋矩阵 II 是一样的意思,但是由于不是方形的，边界不能套用
        if len(matrix) == 0:
            return []

        rows = len(matrix)
        cols = len(matrix[0])
        left, right, low, high = 0, cols - 1, 0, rows - 1

        res = []

        while left <= right and low <= high:
            # left ---> right
            for i in range(left, right+1):
                res.append(matrix[low][i])
            low = low + 1
            if low > high:
                break  # 退出while 循环

            # low ---> high
            for i in range(low, high+1):
                res.append(matrix[i][right])
            right = right - 1
            if left > right:
                break  # 退出while 循环

            # right ---> left
            for j in range(right, left-1, -1):
                res.append(matrix[high][j])
            high = high - 1
            if low > high:
                break  # 退出while 循环

            # high ---> low
            for j in range(high, low-1, -1):
                res.append(matrix[j][left])
            left = left + 1
            if left > right:
                break  # 退出while 循环

        return res
``` 

## 6. 螺旋矩阵 II

给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

59\. 螺旋矩阵 II（middle） [力扣](https://leetcode-cn.com/problems/spiral-matrix-ii/description/)

示例 1:

```html
输入: 3

输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
```

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        #还是模拟过程,控制好边界,行的上下边界, 列的左右边界, 很经典
        #与 54. 螺旋矩阵 是一样的意思,但是由于是方形的，边界不能没有那么复杂
        
        res = [[0] * n for _ in range(n)] # 把结果矩阵全置0,然后下面按规则填充值
        
        above_row = 0 #上边界index
        below_row = n-1 #下边界index
        left_col = 0 #左边界index
        right_col = n - 1 #右边界index
        
        num = 1 #填充的初始值设置为1

        while(above_row <= below_row and left_col <= right_col):
            # 从左到右循环
            for i in range(left_col, right_col+1):
                res[above_row][i] = num #ps: 这里用above_row，是因为马上要更新它
                num = num + 1
            #更新above_row，上边界index加1
            above_row = above_row + 1

            # 从上到下循环
            for i in range(above_row, below_row + 1):
                res[i][right_col] = num  # ps: 这里用right_col，是因为马上要更新它
                num = num + 1
            # 更新right_col，右边界index减1
            right_col = right_col - 1

            # 从右到左循环
            for i in range(right_col, left_col-1, -1):
                res[below_row][i] = num  # ps: 这里用below_row，是因为马上要更新它
                num = num + 1
            # 更新below_row，下边界index减1
            below_row = below_row - 1

            # 从下到上循环
            for i in range(below_row, above_row-1, -1):
                res[i][left_col] = num  # ps: 这里用left_col，是因为马上要更新它
                num = num + 1
            # 更新left_col，右边界index加1
            left_col = left_col + 1

        return res
``` 

## 7. 有效的数独

```html
判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

	1.数字 1-9 在每一行只能出现一次。
	2.数字 1-9 在每一列只能出现一次。
	3.数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次

数独部分空格内已填入了数字，空白格用 '.' 表示

说明:
	1. 一个有效的数独（部分已被填充）不一定是可解的。
	2. 只需要根据以上规则，验证已经填入的数字是否有效即可。
	3. 给定数独序列只包含数字 1-9 和字符 '.' 。
	4. 给定数独永远是 9x9 形式的。

```

36\. 有效的数独（middle） [力扣](https://leetcode-cn.com/problems/valid-sudoku/description/)

示例 1:

```html
输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true

输入:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: false
解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
     但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
```

```python
class Solution:
    def isValidSudoku2(self, board: List[List[str]]) -> bool:
        #分3步： 分别判断行，列，3*3所组成的list里面是否有重复值
        for row in board: # 判断行是否有重复元素
            row_valide = [val for val in row if val != '.']
            if len(set(row_valide)) != len(row_valide):
                return False

        for col_ind in range(len(board[0])):#判断每列是否有重复元素
            col_valide = [row[col_ind] for row in board if row[col_ind] != '.']
            if len(set(col_valide)) != len(col_valide):
                return False


        for i in [2, 5, 8]:# 所有3*3所组成的list里面是否有重复值
            for j in [2, 5, 8]:
                #print('i, j ---->', i, j)
                tmp = []
                for m in range(3):
                    for n in range(3):
                        if board[i - m][j - n] != ".":
                            tmp.append(board[i - m][j - n])
                #print('temp----->', tmp)
                if len(set(tmp)) != len(tmp):
                    return False


        return True


    def isValidSudoku(self, board: List[List[str]]) -> bool:
        lows, columns, boxes = defaultdict(set), defaultdict(set), defaultdict(set)

        for low in range(9):
            for col in range(9):
                if board[low][col].isdigit():  # 或者用 board[low][col] != '.'也可以
                    # 以下三个if判断是不是在行、列和 3*3宫格内存在有重复数字
                    if board[low][col] in lows[low]:
                        return False

                    if board[low][col] in columns[col]:
                        return False

                    # 这里3*3宫格缩小1/3
                    if board[low][col] in boxes[low // 3, col // 3]:
                        return False

                    # 没存在加入行、列和 3*3宫格
                    lows[low].add(board[low][col])
                    columns[col].add(board[low][col])
                    boxes[low // 3, col // 3].add(board[low][col])

        """
        dic ={(0, 0): {'3', '8', '6'}, (0, 1): {'9', '7', '5', '1'}}
        print(dic[0, 1]) # {'8', '6', '3'}
        """

        return True

``` 
