# 任务 0164 解题过程记录

## Step 1：先做低复杂度规则探测

### think
- 题目说明只有一条全局规则，优先从可解释且可复现的模型入手。
- 先用浅层决策树观察是否存在单特征阈值或区间切分；如果很快达到高准确率，通常能直接还原规则形态。
- 同时跑逻辑回归和单特征阈值暴力搜索做交叉验证，避免被单一方法误导。

### 代码
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import itertools, numpy as np

path = '/Users/jinming.liu/Documents/base/arena-da/task_0164/data.csv'
df = pd.read_csv(path)
X = df.drop(columns=['target'])
y = df['target']

for d in [1, 2, 3, 4]:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    print('depth', d, 'train', accuracy_score(y, pred), 'cv', cross_val_score(clf, X, y, cv=5).mean())
    print(export_text(clf, feature_names=list(X.columns), max_depth=d))

lr = LogisticRegression(max_iter=10000)
lr.fit(X, y)
print('logistic train', lr.score(X, y))

best = None
for c in X.columns:
    vals = np.unique(X[c])
    ths = (vals[:-1] + vals[1:]) / 2
    for t in ths:
        for sign in [1, -1]:
            pred = (X[c] * sign > t * sign).astype(int)
            acc = (pred == y).mean()
            if best is None or acc > best[0]:
                best = (acc, c, sign, t)
print('best_1d', best)
```

### 观测
- 决策树 `max_depth=2` 即达到 `train=1.0`、`cv=1.0`。
- 树结构只使用 `feat_3`，并出现两段阈值：`<= -2.00`、`<= -0.60`。
- 逻辑回归只能到 `0.696`，提示规则更像“区间条件”而非线性超平面。
- 单特征单阈值的最佳准确率约 `0.70`，也支持“需要双阈值区间”这一判断。

---

## Step 2：精确确认阈值边界

### think
- 树里的阈值会有四舍五入显示，需回到原数据核对真实边界区间，保证规则可复现且无歧义。
- 按 `target` 分组查看 `feat_3` 的最小值、最大值，并检查正类区间上下两侧最近的负类点。

### 代码
```python
import pandas as pd

p = '/Users/jinming.liu/Documents/base/arena-da/task_0164/data.csv'
df = pd.read_csv(p)

for y in [0, 1]:
    s = df[df.target == y]['feat_3']
    print(y, s.min(), s.max(), s.nunique())

s1 = df[df.target == 1]['feat_3']
s0 = df[df.target == 0]['feat_3']
print('class1 min/max', s1.min(), s1.max())
print('max class0 below class1 min', s0[s0 < s1.min()].max())
print('min class0 above class1 max', s0[s0 > s1.max()].min())
```

### 观测
- `target=1` 的 `feat_3` 取值范围是 `[-1.992125, -0.606717]`。
- 负类在该区间外分布，最接近边界的点约为 `-2.007954` 与 `-0.589636`。
- 因此用可读阈值可写为：`feat_3 > -2.0` 且 `feat_3 <= -0.6`。


---

## 最终提交答案（按题目 XML 格式）

```xml
<answer_features>feat_3</answer_features>
<answer_rule>target = 1 if (feat_3 > -2.0) and (feat_3 <= -0.6) else 0</answer_rule>
```
