# demand-forecasting-retail
End-to-end retail demand forecasting on the M5 (Walmart) dataset — EDA, baseline vs. ML models (Prophet, XGBoost), error analysis, and safety-stock recommendations for replenishment decisions.

# Problem Statement
A regional retailer's demand planning team places weekly purchase orders for hundreds of SKUs across multiple stores. Forecast errors are costly in both directions: under-forecasting causes stockouts and lost sales, while over-forecasting ties up capital in excess inventory and increases holding costs.
This project builds an end-to-end demand forecasting pipeline on the M5 (Walmart) dataset to support weekly replenishment decisions. Beyond forecast accuracy, the goal is decision quality: forecasts are evaluated against naive and seasonal baselines (RMSE, MAE, WAPE), errors are analyzed by SKU demand segment, and residual uncertainty is translated into safety-stock recommendations under different service-level targets (95% vs. 99%).

Scope: [X] SKUs from the FOODS category across [N] stores, using [~3] years of daily sales, price, and calendar/event data.
