# Time Series Forecasting with Recurrent Neural Networks

This project explores time series forecasting using recurrent neural network architectures.

Three models are implemented and compared:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM

The models are trained on the Airline Passengers dataset to learn temporal patterns such as trend and seasonality.

---

## Dataset

Airline Passengers Dataset

- Monthly international airline passenger totals
- Time period: 1949–1960
- Total observations: 144

This dataset is commonly used as a benchmark for time series forecasting.

---

## Methodology

A sliding window approach is used to construct input-output pairs.

- Window length: 24 past observations
- Forecast horizon: 1 step ahead

Each model predicts the next value of the time series.

---

## Model Architectures

Three recurrent neural network architectures were evaluated:

- LSTM
- GRU
- Bidirectional LSTM

All models were trained under the same configuration to ensure fair comparison.

Training setup:

- Hidden size: 64
- Recurrent layers: 1
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 16
- Maximum epochs: 300
- Early stopping with patience of 30 epochs

---

## Experiment

Two forecasting strategies were compared:

### 1. Direct Forecasting
The model predicts the next value of the time series.

### 2. Differencing
The model predicts the change between consecutive observations.

The predicted change is added to the last observed value to reconstruct the forecast.

---

## Results

### Without Differencing

| Model | MAE | RMSE |
|------|------|------|
| LSTM | 82.61 | 91.36 |
| GRU | 79.30 | 90.71 |
| BiLSTM | **66.37** | **77.76** |

### With Differencing

| Model | MAE | RMSE |
|------|------|------|
| LSTM | 37.72 | 47.35 |
| GRU | 31.42 | 41.04 |
| BiLSTM | **28.34** | **34.93** |

Differencing significantly improves forecasting accuracy for all models.

Among the evaluated architectures, Bidirectional LSTM achieves the best performance.

---


---

## Author

Keerthija Bontu  
M.Eng. Information Technology (Specialization: Artificial Intelligence)
