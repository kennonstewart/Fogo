# Online GBDT for Regression (OGBDT-R)

A lightweight, streaming-compatible implementation of **Online Gradient Boosted Decision Trees (GBDTs)** for regression tasks under **non-stationary data distributions**.  
This framework supports online learning, memory-efficient model updates, and plug-in loss functions like MSE, Huber, and Quantile Loss.

## Why use OGBDT-R?

- **Truly Online**: Process one sample at a time—no mini-batching or retraining required.
- **Regress, not classify**: Unlike many online GBDT frameworks, this one is purpose-built for real-valued targets.
- **Concept Drift Ready**: Detects and adapts to changing patterns using loss-driven or statistical drift signals.
- **Flexible Losses**: Plug in your own differentiable loss functions.
- **Edge Ready**: Optimized for low-latency environments—streaming IoT, traffic prediction, and time series forecasting.

---

## Installation

Clone this repo and install via pip:

```bash
git clone https://github.com/your-username/online-gbdt-regression.git
cd online-gbdt-regression
pip install -e .