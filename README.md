# Demand Forecasting & Operational Decision System

This project develops a data-driven system to support pricing and demand management decisions in an accommodation business.

It combines demand dynamics modeling, forecasting, and reliability calibration to transform operational data into actionable business signals.

The project originates from real operational work, where data was collected and structured to understand how demand behaves and whether it can be used for forward-looking decision-making, especially during low-demand periods.

---------------------------------------------------------------------

## Business Objective

The goal is not only to predict occupancy, but to answer:

- When should pricing be increased or reduced?
- How reliable are the model signals at any given time?
- When should decisions be avoided due to instability?

This project bridges the gap between **forecasting** and **operational decision-making**.

---------------------------------------------------------------------

## Approach Overview

The system is built in four layers:

### 1. Feature Engineering
Operational signals are constructed from raw data:

- Booking inflows and notice outflows  
- Demand pressure and timing-adjusted flows  
- Occupancy persistence (lags and trends)  
- Seasonal and calendar effects  

### 2. Multi-Horizon Forecasting

Linear models are trained to predict occupancy at:

- T+1 (short-term)
- T+7
- T+14
- T+21
- T+28 (strategic horizon)

These forecasts are benchmarked against naive baselines.

### 3. Directional Modeling

Classification models predict the **direction of change**:

- UP / DOWN / STABLE  
- Probabilities are used instead of hard predictions  
- Weak signals are filtered out to avoid overreaction  

### 4. Operational Calibration (Core Contribution)

A reliability-aware decision layer is built using:

- **Guardrail** → detects structural instability (high booking/notice pressure)  
- **Alignment Score** → rolling measure of recent model accuracy  
- **Strength Bands** → STRONG / MODERATE / UNCERTAIN signals  

These components are combined into calibrated actions such as:

- Increase pricing (strategic)  
- Prepare promotions  
- Maintain current settings  
- Monitor (unstable / low reliability)  

---------------------------------------------------------------------

## Understanding Demand Behavior

A central part of the project is modeling how demand evolves over time.

Rather than treating occupancy as an isolated time series, the analysis focuses on the mechanisms that drive it:

- booking inflows and notice outflows  
- timing of demand (early vs late bookings and cancellations)  
- pressure on capacity  
- persistence and delayed effects  
- recurring seasonal patterns  

This perspective leads to several key insights:

- Short-term occupancy is dominated by persistence  
- Medium-term occupancy reflects accumulated demand pressure  
- High-demand or high-cancellation periods introduce instability and reduce predictability  

These findings directly inform both the forecasting models and the operational decision layer.

---------------------------------------------------------------------

## Project Structure

├── notebooks/
  ├── 01_data_preparation.ipynb
  ├── 02_forecasting_and_modeling.ipynb
│ ├── 03_operational_calibration.ipynb
│ └── 04_operational_scoring_demo.ipynb
│
├── src/
│ ├── forecast_utils.py
│ ├── build_operational_visual_board.py
│ └── excel_dashboard.py
│
├── data/
│ └── sample_daily_cleaned_dataset.csv
│
├── outputs/
│ ├── scored_operational_output_sample.csv
│ ├── operational_calibration_output_sample.csv
│ └── operational_dashboard_example.xlsx
│
├── models/
│ └── (saved models not included)
│
├── requirements.txt
└── README.md

---------------------------------------------------------------------

## Example Output

The system produces a structured operational view combining:

- Forecasted occupancy (T+7 → T+28)
- Directional probabilities
- Signal strength
- Reliability assessment
- Final recommended action

Additionally, a visual decision board is generated:

- Separate T+21 and T+28 panels  
- Direction, reliability, stability, and action clearly displayed  
- Designed for operational readability  

