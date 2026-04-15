# Demand Forecasting & Operational Decision System

This project develops a data-driven system to support pricing and demand management decisions in an accommodation business, transforming operational data into actionable signals.

It combines demand dynamics modeling, forecasting, and reliability calibration to transform operational data into actionable business signals.

The project originates from real operational work, where data was collected and structured to understand how demand behaves and whether it can be used for forward-looking decision-making, especially during low-demand periods.

---------------------------------------------------------------------

## Live Demo

A simplified operational dashboard (no setup required):

[View the operational demo](https://adreanerpgh.github.io/demand-forecasting-operational-decision-system/notebooks/04_operational_demo.html)

---------------------------------------------------------------------

## Project Walkthrough (No Setup Required)

You can explore the full project directly through the HTML notebooks:

1. [View Data Preparation (documentation only)](https://adreanerpgh.github.io/demand-forecasting-operational-decision-system/notebooks/01_build_daily_dataset.html)
2. [View Modeling & Forecasting](https://adreanerpgh.github.io/demand-forecasting-operational-decision-system/notebooks/02_forecasting_and_spike_analysis_test=2019.html)
3. [Operational Calibration](https://adreanerpgh.github.io/demand-forecasting-operational-decision-system/notebooks/03_operational_calibration.html)  
4. [Operational Demo (final output)](https://adreanerpgh.github.io/demand-forecasting-operational-decision-system/notebooks/04_operational_demo.html)  

Note: Notebook 01 requires raw data and is included for documentation only.  
Notebooks 02–04 are fully reproducible using the provided dataset.

---------------------------------------------------------------------

## Business Objective

The goal is not only to predict occupancy levels, but to provide reliable signals for operational decision-making:

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

- **Guardrail**: detects structural instability (high booking/notice pressure)  
- **Alignment Score**: rolling measure of recent model accuracy  
- **Strength Bands**: STRONG / MODERATE / UNCERTAIN signals  

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
These observations explain why predicting exact occupancy levels is less reliable in unstable periods, and motivate the use of directional and reliability-aware decision signals instead.

---------------------------------------------------------------------

## Data Transformation

The original dataset was collected at customer and unit level and contained operational and behavioral information.

For this project, the data was transformed into a daily aggregated dataset to:

- remove all customer identifiers
- remove personal attributes
- remove unit-level tracking
- create consistent time-series inputs

The resulting dataset captures system-level demand dynamics while preserving confidentiality.

This transformation step is a key part of the project, enabling reliable modeling while maintaining data privacy.

---------------------------------------------------------------------

## Project Structure
```
├── notebooks/
│   ├── 01_data_preparation.ipynb 
│   ├── 01_data_preparation.html 
│   ├── 02_modeling.ipynb
│   ├── 02_modeling.html
│   ├── 03_operational_calibration.ipynb
│   ├── 03_operational_calibration.html
│   ├── 04_operational_demo.ipynb
│   ├── 04_operational_demo.html
├── src/
│   ├── forecast_utils.py
│   └── build_operational_visual_board.py
├── data/
│   └── daily_cleaned_dataset.csv
└── README.md
```
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

---------------------------------------------------------------------

## Key Insights

- Demand behavior is not uniform: short-term occupancy is persistence-driven, while medium-term occupancy reflects demand pressure and flow dynamics  
- Spikes are associated with unstable high-pressure periods and are harder to predict reliably  
- Medium-term forecasts (T+7 to T+28) outperform naive baselines  
- Directional models become more useful as the forecast horizon increases  
- The alignment score successfully identifies periods of higher and lower model reliability  
- Combining demand signals, directional predictions, and reliability layers improves decision quality  
