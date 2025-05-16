:fire: # Fogo Means Fire :fire:
## A Software Package for Online Gradient Boosted Decision Trees -- Specifically Built for Regression

A lightweight, streaming-compatible implementation of **Online Gradient Boosted Decision Trees (GBDTs)** for regression tasks under **non-stationary data distributions**.  
This framework supports online learning, memory-efficient model updates, and plug-in loss functions like MSE, Huber, and Quantile Loss.

## Why use Fogo?

- **Truly Online**: Process one sample at a time—no mini-batching or retraining required.
- **Regress, not classify**: Unlike many online GBDT frameworks, this method is built for continuous response variables.
- **Concept Drift Ready**: Detects and adapts to changing patterns using loss-driven or statistical drift signals.
- **Flexible Losses**: Plug in your own differentiable loss functions.
- **Edge Ready**: Optimized for low-latency environments—streaming IoT, traffic prediction, and time series forecasting.

---

## Installation

Clone this repo and install via pip:

```bash
git clone https://github.com/kennonstewart/fogo.git
cd online-gbdt-regression
pip install -e .
```