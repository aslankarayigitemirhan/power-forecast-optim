# power-forecast (Energy Forecasting with GD + Ridge)

Backend pipeline:
- Download UCI dataset to `data/raw/`
- Missing handling (short gap -> linear interpolation, long gap -> calendar-conditioned fill)
- Resample to chosen granularity (default 60 minutes)
- Build target: Energy (kWh) per bucket
- Feature engineering: energy lags + calendar sin/cos
- Time-series split: blocked or rolling
- Train: Ridge regression via mini-batch GD
- Predict: date-range forecasts
- Visualizations: EDA, training curves, loss surface + GD trajectory

## Quickstart

### 1) Create venv (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
