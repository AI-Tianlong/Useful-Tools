# 那些从chatGPT中学来的算法实现
chatGPT 中确实能有一些很实用的代码实现，让俺耳目一新，为之震撼。
## 大图切割小图，保留不足5000的小图
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
