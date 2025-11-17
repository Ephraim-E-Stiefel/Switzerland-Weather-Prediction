# Switzerland Weather Prediction Model

## Project Overview

This repository hosts a machine learning project focused on developing and evaluating models for ***predicting ambient temperature in Bern, Switzerland***. The goal is to build robust regression models that can accurately forecast the temperature 12, 24, and 48 hours in advance, using diverse meteorological observations from 10 weather stations across the country.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Models Used](#models-used)
5.  [Key Results](#key-results)
6.  [Repository Structure](#repository-structure)
7.  [Setup and Installation](#setup-and-installation)
8.  [Usage](#usage)
9.  [Future Work](#future-work)
10. [License](#license)

---

## Dataset

The project utilizes a **Switzerland Weather Dataset** (`train.csv`) containing hourly observations across the country. The target predictions are made for the Bern station.

* ***Size***: Approximately **7,579 hourly records**.
* ***Features***: Includes 92 raw features for 10 weather stations, covering meteorological variables such as:
    * Temperature (`tre200h0_*`)
    * Wind speed/gust (`fkl010h0_*`, `fkl010h3_*`)
    * Radiation (`gre000h0_*`, `sre000h0_*`)
    * Pressure (`prestah0_*`)
    * Relative Humidity (`ure200h0_*`)
    * Precipitation (`rre150h0_*`)
* ***Temporal Features***: `hour` and `season`.
* ***Target Variables***: `target_tre200h0_plus12h`, `target_tre200h0_plus24h`, and `target_tre200h0_plus48h` (future temperature predictions in °C).

---

## Methodology

The project follows a comprehensive machine learning pipeline designed to mitigate the effects of multicollinearity and data skewness, which were identified during the initial analysis:

1.  ***Data Cleaning***: Rows with missing values were dropped, accounting for a low percentage (1.98%) of the dataset.
2.  ***Outlier and Skewness Handling***:
    * **Transformation**: Log and Box-Cox transformations were applied to highly skewed radiation and precipitation features (`sre000h0_*`, `gre000h0_*`, `rre150h0_*`) to approximate a normal distribution.
    * **Outlier Capping**: **Winsorization** (capping) was applied to extreme wind speed/gust outliers at the 99th percentile to mitigate their influence on linear models.
3.  ***Feature Engineering***:
    * **Spatial Aggregation**: New features were engineered by calculating the **mean, median, and standard deviation** across all 10 stations for each variable (e.g., `tre200h0_mean`, `fkl010h3_std`) to reduce multicollinearity and capture overall spatial trends.
    * **Time Encoding**: The cyclical nature of the `hour` feature was captured using **sine and cosine transformation**. The `season` variable was processed using **One-Hot Encoding**.
4.  ***Feature Selection***: Redundant and highly correlated individual station measurements were dropped, retaining only the new engineered features and the final Box-Cox transformed spatial aggregates.
5.  ***Data Scaling***: All remaining numerical features were standardized using **StandardScaler** for model readiness.
6.  ***Model Training***: Data was split into a **Training** and **Testing** set (`test_size=0.2`), and initial regression models were trained against the three target horizons.

---

## Models Used

Two simple, interpretable machine learning models were implemented and compared as baselines:

* **K-Nearest Neighbors (KNN) Regressor**.
* **Linear Regression**.

---

## Key Results

The initial baseline models demonstrated that predicting temperature across the various horizons is a **highly complex task**:

* The simple **K-Nearest Neighbors** and **Linear Regression** models resulted in a **high Mean Absolute Error (MAE)**, indicating they were insufficient for accurate prediction.
* A key observation during the analysis was that while current temperature showed a strong correlation (up to $\approx 93\%$ variance explained) for the 24-hour forecast, its predictive power dropped significantly for the 12-hour forecast (only $\approx 72\%$ variance explained), highlighting the need for more complex models to capture the **advective and diabatic processes** that drive short-term temperature changes.

---

## Repository Structure

/Switzerland-Weather-Prediction/   
|-- .gitignore   
|-- README.md   
|-- requirements.txt   
|-- data/   
|   |-- train.csv   
|-- notebooks/   
|   |-- 01_Switzerland_Weather_Prediction.ipynb   
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
    (Note: You will need to generate `requirements.txt` from the libraries used in the notebook.)

4.  **Download the weather datasets** (`train.csv`) and place them in the `data/` directory.

---

## Usage

1.  **Explore the Notebook**:
    * `01_Switzerland_Weather_Prediction.ipynb`: Contains the complete data cleaning, feature engineering, model training, and baseline evaluation pipeline.

2.  **Run the Notebook**: Open the `.ipynb` file in Jupyter Notebook or JupyterLab to execute the pipeline.
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

---

## Future Work

To significantly improve prediction accuracy beyond the baseline models, the project requires further development in the following areas:

1.  **Advanced Modeling Techniques**: Switch to non-linear methods such as **Random Forest**, **Gradient Boosting** (XGBoost/LightGBM), or **LSTMs** (Long Short-Term Memory networks) to better capture feature interactions.
2.  **Feature and Data Strategy**:
    * Experiment with **interaction terms** between features (e.g., wind speed and seasonality) for greater predictive power.
    * Utilize **Principal Component Analysis (PCA)** to simplify the high dimensionality and multicollinearity of the feature set.
3.  **Model Rigor**: Implement systematic optimization using **Hyperparameter Tuning** (via Grid Search or Random Search) to find the best configuration for the chosen advanced models.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

> **Note:** Some portions of the code and this documentation were generated with the assistance of **Generative AI** to improve clarity and structure.
