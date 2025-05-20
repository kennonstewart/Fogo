# 🔥 Fogo: Online Gradient Boosted Decision Trees for Regression

**Fogo** is a powerful software package designed to bring **true online learning** to Gradient Boosted Decision Trees (GBDTs). Tailored for **regression** under non-stationary data streams, it offers an efficient and audit-ready solution for edge deployment, privacy compliance, and continuous learning.

---

## 🚀 Why Fogo?

### 1. 📦 Train on Batches
Train an ensemble model on any batch of data—initialize your predictive pipeline with standard offline learning before going fully online.

### 2. 🔁 Learn from Live Data
Update the model **incrementally** with each new data point. Perfect for **sensor streams**, **traffic models**, or any scenario where the data never stops flowing. No retraining. No buffering. No centralized servers required.

### 3. ❌ Unlearn on Demand (Decremental Learning)
Fogo supports **decremental updates**—remove the influence of a data point from the model entirely when you receive a deletion request. This makes Fogo **GDPR-compliant by design**, enabling logs and verification for privacy audits.

---

## 🧠 What Makes It Unique?

- **Built for Regression**: Native support for continuous targets—no need to hack classification models.
- **Truly Online**: Fit one point at a time using lightweight trees.
- **Edge Deployment Ready**: Can be launched to a device and learn in place—no streaming needed.
- **Pluggable Loss Functions**: MSE, Huber, Quantile Loss, and custom differentiable functions.
- **Privacy-first Architecture**: Track, delete, and unlearn data by design—not by exception.

---

## 📦 Installation

```bash
git clone https://github.com/kennonstewart/fogo.git
cd fogo
pip install -e .
```

---

## 🛠 Contributing

We welcome contributions, feedback, and discussions!  
Check out `contributing.md` for more on how to get involved.

---

## 🙏 Acknowledgements

Thanks to Huawei Lin, Jun Woo Chung, Yingjie Lao, and Weijjie Zhao for foundational work on Online GBDTs for classification, which inspired this extension to regression and online adaptation.