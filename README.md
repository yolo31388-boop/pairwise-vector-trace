# pairwise-vector-trace

## 领域

游戏、图形与多媒体

## 主题

位图描摹矢量化

## 说明

光照估计阈值、误差收敛拟合与跨块合并。

初始状态下 `imaging/vector_trace.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
