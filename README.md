# power-forecast  
**Energy Forecasting with Gradient Descent + Ridge Regression**

This repository implements an end-to-end backend pipeline for short-term household energy forecasting, formulated as a **convex optimization problem** and solved via **gradient-based learning**.

The project emphasizes **interpretability, optimization analysis, and reproducibility**, rather than black-box accuracy.

---

## Overview

The system predicts **interval-based energy consumption (kWh)** using a regularized linear regression model trained with Gradient Descent (GD).

### Key ideas
- Forecast **energy per interval** (kWh), not instantaneous power (kW)
- Preserve **time-series integrity** (no random shuffling)
- Use **Ridge (L2) regularization** for stability and conditioning
- Visualize **optimization geometry** (loss surface, contours, GD trajectory)

---

## Backend Pipeline

1. **Dataset acquisition**
   - UCI *Individual Household Electric Power Consumption*
   - Stored under `data/raw/`

2. **Missing value handling**
   - Short gaps (≤ 30 min): linear interpolation  
   - Long gaps (> 30 min): calendar-conditioned median  
     (same hour-of-day & day-of-week)

3. **Resampling**
   - User-defined granularity (default: **60 minutes**)

4. **Target construction**
   - Energy per interval:
     \[
     E_k = \sum_{t \in B_k} P_t \cdot \frac{1}{60}
     \]

5. **Feature engineering**
   - Lagged energy values (lookback window)
   - Calendar features (sin/cos encoding)
   - Bias term

6. **Time-series validation**
   - Blocked split or rolling (walk-forward)
   - No random shuffling (prevents leakage)

7. **Training**
   - Ridge regression
   - Mini-batch Gradient Descent

8. **Prediction**
   - User-defined date range
   - Interval-based forecasts

9. **Visualization**
   - Training curves
   - Loss surface (2D slice)
   - Contours + GD trajectory

---

## Project Structure

