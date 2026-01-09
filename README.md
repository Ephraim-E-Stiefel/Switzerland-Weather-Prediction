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
9. [License](#license)

---

## Dataset

The project utilizes a **Switzerland Weather Dataset** (train.csv) containing hourly observations across the country. The target predictions are made for the Bern station.

* **Size**: 7,579 hourly records.
* **Features**: Includes 92 raw features for 10 weather stations, covering meteorological variables such as:
    * Temperature (tre200h0_)
    * Wind speed/gust (fkl010h0_, fkl010h3_)
    * Radiation (gre000h0_, sre000h0_)
    * Pressure (prestah0_, pp0qffh0_)
    * Relative Humidity (ure200h0_)
    * Precipitation (rre150h0_)
* **Temporal Features**: hour and season.
* **Target Variables**: target_tre200h0_plus12h, target_tre200h0_plus24h, and target_tre200h0_plus48h (future temperature predictions in °C).

---

## Methodology

The project follows a comprehensive machine learning pipeline designed to mitigate the effects of multicollinearity and data skewness while extracting meaningful meteorological signals.

1.  **Data Cleaning**: Rows with missing values were removed to ensure data integrity.
2.  **Exploratory Data Analysis**: Conducted correlation analysis to identify highly redundant station measurements (e.g., temperature and pressure correlations > 0.9) and visualized distributions using violin plots.
3.  **Feature Engineering**: To account for the fact that the dataset is not chronologically ordered (preventing the use of standard lagged variables), three key features were engineered for non-parametric models to provide temporal context:
    * **Mean Season-Hour Temperature ($\mu_{s,h}$)**: The average temperature for a specific hour within a specific season, representing the expected baseline.
    * **Standard Deviation Season-Hour Temperature ($\sigma_{s,h}$)**: The variability expected for that specific hour and season.
    * **Temperature Anomaly**: Interaction term between Global Radiation and the cosine of the hour to represent solar heating efficiency.
    * **Temperature Change**: Calculated as the current temperature minus the mean season-hour temperature ($tre200h0 - \mu_{s,h}$), capturing whether the current conditions are unusually warm or cold relative to the baseline.
3.  **Preprocessing for Parametric Models**:
    * **Box-Cox Transformation**: Applied to highly skewed features (Wind speed/gust, Precipitation, Sunshine Duration, Global Radiation) to approximate normal distribution.
    * **Winsorization**: Wind speed and Precipitation features were capped at the 98th percentiles to limit the impact of rare measures which can disturb the accuracy of the models.
    * **Dimensionality Reduction**: To reduce multicollinearity, highly correlated station measurements (Temperature, Radiation, Sunshine) were aggregated into **mean and standard deviation** features.
    * **Cyclical Encoding**: Sine and Cosine transformations applied to the hour feature.
    * **One-Hot Encoding**: Applied to the season variable.
    * **Scaling**: All numerical features were standardized using **RobustScaler**.
4.  **Preprocessing for Non-Parametric Models**:
    * For tree-based models (Random Forest, Gradient Boosting), scaling and outlier handling were kept minimal, as these models are naturally robust to feature scales and extreme values.
    * **Feature Engineering**: To account for the fact that the dataset is not chronologically ordered (preventing the use of standard lagged variables), three key features were engineered for non-parametric models to provide temporal context:
      * **Mean Season-Hour Temperature ($\mu_{s,h}$)**: The average temperature for a specific hour within a specific season, representing the expected baseline.
      * **Standard Deviation Season-Hour Temperature ($\sigma_{s,h}$)**: The variability expected for that specific hour and season.
      * **Temperature Anomaly**: Interaction term between Global Radiation and the cosine of the hour to represent solar heating efficiency.
      * **Temperature Change**: Calculated as the current temperature minus the mean season-hour temperature ($tre200h0 - \mu_{s,h}$), capturing whether the current conditions are unusually warm or cold relative to the baseline.
      * The hour feature was dropped after creating the engineered variables to avoid redundancy.

---

## Models and Evaluation

The following regression models were implemented, optimized via **GridSearchCV** with 10-fold cross-validation or manual grid search for hyperperameter tuning:

* **Linear Regression** (Baseline)
* **Lasso Regression** (L1 regularization)
* **Ridge Regression** (L2 regularization)
* **K-Nearest Neighbors (KNN) Regressor**
* **Random Forest Regressor**
* **Gradient Boosting Regressor**

**Evaluation Metric**: While Root Mean Squared Error (RMSE) is commonly used in regression tasks, this project utilizes **Mean Absolute Error (MAE)**. MAE was selected to provide a robust interpretation of error in degrees Celsius and to avoid excessive penalization of outliers which are natural in weather data.

---

## Key Results

The analysis demonstrated that non-linear, tree-based models significantly outperformed traditional linear models by better capturing the complex interactions between meteorological variables.

* **12-Hour Forecast**: **Gradient Boosting** achieved the best performance with a Test MAE of approximately **1.56**.
* **24-Hour Forecast**: **Gradient Boosting** proved superior, achieving the lowest error with a Test MAE of approximately **1.84**.
* **48-Hour Forecast**: **Gradient Boosting** again performed best with a Test MAE of approximately **2.42**.
* **Model Comparison**: While Linear and Lasso regression provided a solid baseline (~2.10 MAE for 12h), they lacked the flexibility of **Random Forest** and **Gradient Boosting**, which were able to leverage the engineered "Temperature Anomaly" features more effectively.

---

## Repository Structure

/Switzerland-Weather-Prediction    
|-- .gitignore    
|-- README.md    
|-- requirements.txt    
|-- data/    
|   |-- train.csv    
|-- notebooks/    
|   |-- 01_Weather_Prediction.ipynb
|-- report/    
|   |-- Weather_Prediction_Report.pdf

---

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd Switzerland-Weather-Prediction
    ```

2.  **Create a virtual environment:**
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
    * `01_Weather_Prediction.ipynb`: Contains the complete pipeline including the specific feature engineering for season-hour means, model training, and performance comparison.

2.  **Run the Notebooks**: Open the .ipynb files in Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3. **Read the Report:** A detailed breakdown of the model selection process, evaluation criteria, and the final recommendation can be found in report/Weather_Prediction_Report.pdf.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

> **Note:** Some portions of the code and this documentation were generated with the assistance of Generative AI to improve clarity and structure.