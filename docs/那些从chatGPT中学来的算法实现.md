# 那些从chatGPT中学来的算法实现
chatGPT 中确实能有一些很实用的代码实现，让俺耳目一新，为之震撼。
## 1 大图切割小图，保留不足5000的小图
```python
width=10001
height=10000
step_size=5000
x_tiles = (width + step_size - 1) // step_size
y_tiles = (height + step_size - 1) // step_size
x_tiles, y_tiles
```
输出：
```python
(3, 2)
```
## 2 大图切割小图，有重叠，并保留不足5000的小图
```python
width=10001
height=10000
overlap = 500
title_size = 5000
step_size=title_size-overlap
x_tiles = (width + step_size - 1) // step_size
y_tiles = (height + step_size - 1) // step_size
x_tiles,y_tiles
```
输出：
```python
(3, 3)
```

## pdb调试的基本命令
在使用 `pdb.set_trace()` 进入调试模式后，除了之前提到的常用命令，Python 调试器 `pdb` 还提供了许多其他有用的命令。以下是一些常用的 `pdb` 命令及其功能：

### 常用调试命令：
1. **`n` (next)**：
   - 执行下一行代码，不会进入函数内部。

2. **`s` (step)**：
   - 逐步执行，进入函数内部。

3. **`c` (continue)**：
   - 继续执行程序，直到遇到下一个断点或程序结束。

4. **`p [变量名]`**：
   - 打印变量的值。

5. **`q` (quit)**：
   - 退出调试模式，程序终止。

6. **`l` (list)**：
   - 列出当前执行位置附近的代码，显示当前文件的代码上下文。

7. **`h` 或 `help`**：
   - 显示帮助信息，或输入 `help [命令]` 查看特定命令的帮助文档。

8. **`b [行号]` 或 `b [文件名]:[行号]` (break)**：
   - 设置断点。可以在当前文件的某一行，或在指定文件的某一行设置断点。
   - 例如：`b 10` 在当前文件的第 10 行设置断点，`b myfile.py:20` 在 `myfile.py` 文件的第 20 行设置断点。

9. **`cl [断点编号]` (clear)**：
   - 清除指定断点，也可以直接输入 `cl` 来清除所有断点。

10. **`tbreak [行号]` (temporary break)**：
    - 设置一个临时断点，程序执行到此断点时会暂停，但断点在触发一次后自动删除。

11. **`w` (where)** 或 **`bt` (backtrace)**：
    - 显示当前的调用堆栈，可以帮助查看代码执行到当前行之前的调用过程。

12. **`u` (up)** 和 **`d` (down)**：
    - 在调用栈中向上（`u`）或向下（`d`）移动，以查看不同层级的堆栈帧中的代码和变量。

13. **`j [行号]` (jump)**：
    - 改变当前执行的行号，让程序从指定的行继续执行。这不会执行跳跃之间的代码，慎用。

14. **`disable [断点编号]`**：
    - 禁用某个断点，输入 `disable` 可禁用所有断点。

15. **`enable [断点编号]`**：
    - 重新启用一个断点。

16. **`! [表达式]`**：
    - 执行 Python 表达式或语句。例如，`!x = 10` 会直接修改 `x` 变量的值。

17. **`a` (args)**：
    - 打印当前函数的参数列表及其值。

18. **`retval`**：
    - 打印最近一个返回值，显示最后执行的函数返回的值。

19. **`display [表达式]`**：
    - 自动显示表达式的值，每次进入调试模式时都会自动显示指定的表达式结果。
    - 例如：`display x`，在每次代码停顿时自动打印 `x` 的值。

20. **`undisplay [编号]`**：
    - 取消自动显示。编号是通过 `display` 命令设置的。

21. **`r` (return)**：
    - 执行到当前函数返回为止。

### 调试中的技巧：
- **调试过程中修改代码**：在 `pdb` 中，你可以修改变量的值或执行任意 Python 代码，这对于动态修复代码问题很有用。
- **条件断点**：使用 `b [行号], [条件]` 可以设置条件断点。只有条件满足时程序才会暂停。例如，`b 15, x == 5` 表示在第 15 行，只有当 `x == 5` 时才会暂停程序。

这些命令能帮助你更高效地调试代码，灵活排查程序中的问题。


取消自动显示。编号是通过 display 命令设置的。
r (return)：

执行到当前函数返回为止。

## 继续补充pdb
参考链接：https://blog.csdn.net/weixin_49131823/article/details/132392072
```bash
import pdb

直接在代码里需要调试的地方放一个pdb.set_trace()

n 执行下一条语句

w where 打印当前执行堆栈

d down 执行跳转到在当前堆栈的深一层

u up 执行跳转到当前堆栈的上一层

b break 添加断点

tbreak：（temporary break）临时断点

在第一次执行到这个断点之后，就自动删除这个断点，用法和b一样

cl clear 清楚断点

disable：停用断点，参数为bpnumber，和cl的区别是，断点依然存在，只是不启用

enable：激活断点，参数为bpnumber

s step 执行下一条命令 如果本句是函数调用，则s会执行到函数的第一句

r return 执行当前运行函数到结束

c continue 继续执行，直到遇到下一条断点

l list 列出源码 看下面代码

longlist 所有源吗

ll 查看当前函数的代码

a args 列出当前执行函数的函数

run 重新启动debug 相当于restart

q quit 退出debug

j jump 设置下条执行的语句函数 只能在堆栈的最底层跳转，向后重新执行，向前可直接执行到行号

unt：（until）执行到下一行（跳出循环），或者当前堆栈结束

conditon，给断点设置条件，当参数condition返回True的时候bpnumber断点有效，否则bpnumber断点无效

直接输入Enter，会执行上一条命令；

直接使用 p 变量名 查看值 print

pp 好看一点的 打印

bt 调用查看的堆栈

alias 查看所有命令别名和对应命令 相当于配置 ls 为 l

unalias 取消命名

whatis 查看类型

where 查看所在的位置

interact 启用交互式解释器

retval 打印函数的最后一次返回的返回值。

source 尝试获取给定对象的源代码并显示它。

display 每次在当前帧中停止执行时，显示表达式的值

undisplay 在当前帧中不再显示该表达式。如果没有表达式，请清除当前帧的所有显示表达式。

debug 输入一个递归调试器，它逐步遍历code参数（这是要在当前环境中执行的任意表达式或语句）。

ignore 设置给定断点号的忽略计数。如果忽略count，则忽略计数将设置为0。当忽略计数为零时，断点将变为活动状态。如果为非零值，则每次达到断点且不禁用断点时，计数都会递减，并且任何关联条件的评估结果为true。

commands 为断点设置一个新条件，该表达式必须在接受断点之前求值为true。如果条件不存在，任何现有的条件被移除; 即，将断点设为无条件
```
## 获取MM系列的log，实现打印log
```bash
import pdb;pdb.set_trace()
logger: MMLogger = MMLogger.get_current_instance()
```
