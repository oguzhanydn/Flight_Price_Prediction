# ✈️ End-to-End Flight Price Prediction: Web Scraping to ML Model

## Project Overview

Have you ever wondered what actually drives flight prices? This project attempts to answer that question by building a complete machine learning pipeline — from collecting raw data off the web to training a regression model that predicts flight prices across major European routes.

The project covers every step: **data collection, cleaning, feature engineering, model selection, and evaluation.**

This project was built as a learning exercise focused specifically on linear regression models and regularization techniques (Ridge, Lasso) — tree-based or ensemble methods were intentionally excluded from the scope.

---

## Dataset

- **Source:** Scraped from [ucuzabilet.com](https://www.ucuzabilet.com) using Python and BeautifulSoup library
- **Scraping date:** April 14, 2026
- **Flight dates:** June 1–30, 2026
- **Size:** 87,295 flights
- **Routes:** 156 routes across 13 major European airports (LHR, IST, CDG, AMS, MAD, FRA, BCN, FCO, MUC, DUB, LIS, ATH, ZRH)
- **Class types:** Economy & Business
- **Flight type:** Direct flights only (connecting flights excluded to keep the dataset manageable)

---

## Tools & Libraries

- **Scraping:** BeautifulSoup, Requests, ThreadPoolExecutor
- **Data processing:** Pandas, NumPy
- **Modeling:** Scikit-learn (Pipeline, ColumnTransformer, FunctionTransformer, TransformedTargetRegressor, Ridge, Lasso, GridSearchCV)
- **Visualization:** Matplotlib, Seaborn

---

## Step 1 — Web Scraping

The data did not exist as a ready-to-download dataset, so the first step was building a scraper from scratch.

**The challenge:** 13 airports × 12 destination airports × 30 days = **4,680 pages** to scrape.

**Key decisions made:**
- Used `ThreadPoolExecutor` with 5 workers for parallel scraping — significantly reducing total scraping time compared to sequential requests
- Added random delays (1.5–3.5 seconds) between requests to avoid getting blocked by the server
- Built retry logic (3 attempts per page) to handle timeouts and failed connections
- Extracted 10 fields per flight: date, airline, class type, departure/arrival time, airports, flight duration, baggage allowance, and price

---

## Step 2 — Data Cleaning & Type Transformation

Raw scraped data required several cleaning steps:

- Converted date columns from string to `datetime` format
- Detected overnight flights (arrival times marked with `*`) and created a binary `elapsed_day` feature
- Converted flight duration from "3sa 50dk" format to total minutes (numeric)
- Standardized baggage allowance categories

---

## Step 3 — Feature Engineering

This was the most critical and thought-intensive step. The goal was to extract meaningful signals from raw fields.

### 3.1 Time-based features

**Departure & arrival time → Cyclical encoding (sin/cos transform)**

Time is circular — 23:59 and 00:01 are only 2 minutes apart, but numerically they appear far apart. A standard scaler would not handle this correctly. To solve this, departure and arrival times were encoded using sine and cosine transformations, preserving the circular nature of time.

<img width="2400" height="1650" alt="circular" src="https://github.com/user-attachments/assets/e8538cc1-5a9c-4f59-9bd6-6204da81a16d" />

**Day of week → Cyclical encoding**

Same logic applied to the day of the week (Monday=0, Sunday=6).

**Is weekend:** Binary flag for Saturday/Sunday flights.

### 3.2 Days to flight

`days_to_flight` = flight date − scraping date

This captures how far in advance a ticket is being priced. Airlines typically increase prices as the departure date approaches — this feature was expected to be one of the strongest predictors.

### 3.3 Proximity to holidays

`days_to_holiday_dep` and `days_to_holiday_arr` measure how close the flight date is to a public holiday at the departure and arrival countries respectively.

The intuition: flights around holiday periods (school breaks, national holidays) tend to be significantly more expensive due to demand spikes.

---

## Step 4 — Preprocessing Pipeline

A `Pipeline` + `ColumnTransformer` architecture was used to keep all transformations clean, reproducible, and leak-free.

The pipeline applies transformations in two stages:

**Stage 1 — Deterministic transformations** (feature creation):
- Cyclical encoding for departure time, arrival time, day of week
- Log transformation on flight duration since it's distribution is right-skewed
- Since these transformations were custom-built functions rather than built-in sklearn transformers, `FunctionTransformer` was used to wrap them and make them compatible with the `ColumnTransformer` pipeline

**Stage 2 — Encoding & scaling:**
- `OneHotEncoder` for categorical features (airline, departure/arrival airport, baggage type)
- `OrdinalEncoder` for class type (Economy < Business)
- `StandardScaler` for numerical features

**Target variable:** Price was log-transformed using `TransformedTargetRegressor` to reduce the effect of extreme outliers (very expensive business class tickets skewing the distribution).

---

## Step 5 — Model Selection & Evaluation

### Cross-validation strategy

Since the data has a temporal structure (prices scraped on a single date for future flights), `TimeSeriesSplit` was used instead of standard k-fold cross-validation — to prevent future data from leaking into training folds.

### Models compared

| Model | Mean R² | Std R² | Mean RMSE |
|-------|---------|--------|-----------|
| Baseline (log-transformed LR) | 0.49 | 0.22 | 7,684 |
| Linear Regression | 0.44 | 0.34 | 7,923 |
| Ridge (α=1) | 0.49 | 0.21 | 7,667 |
| Ridge (α=100) | 0.57 | 0.06 | 7,151 |
| Lasso (α=1) | -0.06 | 0.02 | 11,182 |
| **Lasso (α=0.001)** | **0.58** | **0.03** | **7,027** |

**Why Lasso(α=0.001) was selected:**
- Best R² score (0.58)
- Lowest RMSE (7,027 TRY)
- Most importantly: lowest standard deviation (0.03) — meaning consistent performance across all time folds, not just lucky on some splits
- Lasso also performs implicit feature selection by driving irrelevant feature coefficients to zero

### Final model performance (on held-out test set)

- **R²:** 0.59
- **RMSE:** 6,806 TRY

---

## Key Takeaways

- Web scraping at scale requires careful rate limiting and retry logic
- Cyclical encoding is essential for time-based features — naive numeric encoding loses the circular structure
- Regularization (especially Lasso) significantly improves stability over plain linear regression on this dataset
- `TimeSeriesSplit` is the correct cross-validation strategy when data has temporal ordering

---

## Presentation

📊 **View the full project presentation:** [Canva Presentation](https://canva.link/c344an9gvm55pcb)

---

## Files

| File | Description |
|------|-------------|
| `Web_Scrapping.ipynb` | Data collection via web scraping |
| `Flight_Prices_Model.ipynb` | Data cleaning, feature engineering, modeling |
| `flight_data.csv` | Scraped dataset (87,295 rows) |

---

## How to Run

1. Clone the repository
2. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn beautifulsoup4 requests
```
3. Run `Web_Scrapping.ipynb` to collect fresh data (optional — `flight_data.csv` already included)
4. Run `Flight_Prices_Model.ipynb` for the full analysis and model  
