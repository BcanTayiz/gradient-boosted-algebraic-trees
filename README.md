# Gradient Boosted Algebraic Model Trees (GBAT)

A high-performance hybrid machine learning framework that combines axis-aligned decision tree partitioning with local continuous Ridge regression hyperplanes, optimized via gradient boosting. Designed for superior accuracy on complex tabular datasets.

---

## 🚀 Key Features

* **Hybrid Architecture:** Uses axis-aligned decision tree splits to capture sharp thresholds in tabular data, while fitting local algebraic hyperplanes (Ridge regression) inside each leaf instead of flat constant predictions.
* **Gradient Boosting Wrapper:** Iteratively fits weak Algebraic Model Trees to pseudo-residuals, yielding robust ensembles that outperform standard Random Forest and baseline Gradient Boosting Regressors on tabular benchmarks.
* **Continuous Precision:** Eliminates the "staircase" artifacts typical of traditional decision trees by introducing smooth local linear continuity across partitions.

---

## 📊 Benchmark Performance

Tested on standard tabular benchmarks (e.g., California Housing), the Gradient Boosted Algebraic Model Trees framework successfully outperforms standard global Ridge regression, Random Forest, and baseline Gradient Boosting in $R^2$ and RMSE metrics.

---

## ⚙️ Installation & Usage

### Requirements
* Python 3.8+
* `numpy`
* `scikit-learn`

### Quick Start
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import or include the GradientBoostedAlgebraicTrees class from the repository
# gbat = GradientBoostedAlgebraicTrees(n_estimators=50, learning_rate=0.1, max_depth=3, alpha=2.0)
# gbat.fit(X_train_scaled, y_train)
# preds = gbat.predict(X_test_scaled)

```
## Folder Structure

```

├── .gitignore
├── LICENSE.md
├── README.md
├── final_results/
│   ├── article.pdf
│   ├── code/
│   │   ├── final_code.ipynb

```



