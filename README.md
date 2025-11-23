# Switzerland Weather Prediction Model

## Project Overview

This repository hosts a machine learning project focused on developing and evaluating models for **predicting ambient temperature in Bern, Switzerland**. The goal is to build robust regression models that can accurately forecast the temperature 12, 24, and 48 hours in advance, using diverse meteorological observations from 10 weather stations across the country.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Models and Evaluation](#models-and-evaluation)
5.  [Key Results](#key-results)
6.  [Repository Structure](#repository-structure)
7.  [Setup and Installation](#setup-and-installation)
8.  [Usage](#usage)
9.  [Future Work](#future-work)
10. [License](#license)

---

## Dataset

The project utilizes a **Switzerland Weather Dataset** (train.csv) containing hourly observations across the country. The target predictions are made for the Bern station.

* **Size**: Approximately **7,579 hourly records**.
* **Features**: Includes 92 raw features for 10 weather stations, covering meteorological variables such as:
    * Temperature (tre200h0_*)
    * Wind speed/gust (fkl010h0_*, fkl010h3_*)
    * Radiation (gre000h0_*, sre000h0_*)
    * Pressure (prestah0_*, pp0qffh0_*)
    * Relative Humidity (ure200h0_*)
    * Precipitation (rre150h0_*)
* **Temporal Features**: hour and season.
* **Target Variables**: target_tre200h0_plus12h, target_tre200h0_plus24h, and target_tre200h0_plus48h (future temperature predictions in °C).

---

## Methodology

The project follows a comprehensive machine learning pipeline designed to mitigate the effects of multicollinearity and data skewness while extracting meaningful meteorological signals.

1.  **Data Cleaning**: Rows with missing values were removed to ensure data integrity.
2.  **Feature Engineering**: Domain-specific features were created to capture weather dynamics:
    * **Pressure Gradients**: Calculated North-South and West-Center pressure differences to indicate wind flow patterns.
    * **Gust Factor**: Derived from wind speed and gust peak to model atmospheric stability.
    * **Net Solar Heating**: Interaction term between Global Radiation and the cosine of the hour to represent solar heating efficiency.
    * **Temperature Change**: Squared difference between current and 24-hour lagged temperature to capture extreme swings.
3.  **Transformation and Outlier Handling**:
    * **Box-Cox Transformation**: Applied to highly skewed features (Precipitation, Sunshine Duration, Global Radiation) to approximate normal distribution.
    * **Winsorization**: Wind speed features were capped at the 1st and 99th percentiles to limit the impact of extreme gusts.
4.  **Dimensionality Reduction**: To reduce multicollinearity, highly correlated station measurements (Temperature, Radiation, Sunshine) were aggregated into **mean and standard deviation** features.
5.  **Encoding**:
    * **Cyclical Encoding**: Sine and Cosine transformations applied to the hour feature.
    * **One-Hot Encoding**: Applied to the season variable.
6.  **Scaling**: All numerical features were standardized using **StandardScaler**.

---

## Models and Evaluation

The following regression models were implemented, optimized via **GridSearchCV** with 10-fold cross-validation:

* **Linear Regression** (Baseline)
* **Lasso Regression** (L1 regularization)
* **Ridge Regression** (L2 regularization)
* **K-Nearest Neighbors (KNN) Regressor**

**Evaluation Metric**: While Root Mean Squared Error (RMSE) is commonly used in regression tasks, this project utilizes **Mean Absolute Error (MAE)**. MAE was selected to provide a robust interpretation of error in degrees Celsius and to avoid excessive penalization of outliers which are natural in weather data.

---

## Key Results

The analysis revealed that current temperature is the strongest predictor for all horizons, but feature redundancy requires careful handling.

* **12-Hour Forecast**: **Linear Regression** achieved the best performance with a Test MAE of approximately **2.29**.
* **24-Hour Forecast**: **Lasso Regression** proved superior, achieving the lowest error with a Test MAE of approximately **1.44**.
* **48-Hour Forecast**: **Lasso Regression** again performed best with a Test MAE of approximately **1.63**, demonstrating that regularization helps generalize better over longer horizons.
* **KNN Performance**: The K-Nearest Neighbors model consistently underperformed compared to linear models, showing signs of overfitting despite hyperparameter tuning.

---

## Repository Structure

/Switzerland-Weather-Prediction    
|-- .gitignore    
|-- README.md    
|-- requirements.txt    
|-- data/    
|   |-- train.csv    
|-- notebooks/    
|   |-- 01_CH_Weather_Prediction_EDA.ipynb
|   |-- 02_CH_Weather_Prediction_Preprocessing_and_Models.ipynb
|-- models/    
|   |-- (Placeholder for future trained models)    
|-- src/    
|   |-- (Placeholder for future scripts/modules)    

---

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd Switzerland-Weather-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the weather datasets** (train.csv) and place them in the data/ directory.

---

## Usage

1.  **Explore the Notebooks**:
    * `01_CH_Weather_Prediction_EDA.ipynb`: Contains the exploratory data analysis, outlier detection, and target distribution analysis.
    * `02_CH_Weather_Prediction_Preprocessing_and_Models.ipynb`: Contains the complete preprocessing pipeline, feature engineering, model training, and evaluation.

2.  **Run the Notebooks**: Open the .ipynb files in Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

---

## Future Work

To further improve prediction accuracy and model robustness, the following areas are identified for future development:

1.  **Advanced Modeling Techniques**: Implement non-linear methods such as **Random Forest**, **Gradient Boosting** (XGBoost/LightGBM), or **LSTMs** (Long Short-Term Memory networks) to capture complex feature interactions.
2.  **Improvement Potential**:
    * **Data Preprocessing**: Further refinement of outlier handling strategies and exploration of alternative scaling methods for specific features.
    * **Fine tuning on the parameters**: Expanded hyperparameter tuning ranges and Bayesian optimization to maximize model performance.
3.  **Feature Strategy**: Experiment with PCA (Principal Component Analysis) to further reduce dimensionality and potential noise in the spatial data.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

> **Note:** Some portions of the code and this documentation were generated with the assistance of Generative AI to improve clarity and structure.