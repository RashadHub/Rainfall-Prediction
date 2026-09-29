# Rainfall Prediction -- Predicting Whether It Will Rain Tomorrow

## Problem
Predict whether it will rain tomorrow in Saudi cities using historical weather data from two sources.

## Dataset Sources
- **saudi_weather_ksa.csv**: Hourly weather observations from weather.com for 13 Saudi cities (2017-2019), 249,023 rows, 15 columns
- **saudi_weather_openmeteo.csv**: Daily reanalysis weather data from Open-Meteo API for 10 locations (2015-2024), 36,530 rows, 13 columns

## Data Integration
- Dataset 1 cities are mapped to Dataset 2 location IDs using geographic coordinates (latitude/longitude)
- Dataset 1 is aggregated to daily level (daily means for numerical, mode for weather)
- Both datasets are merged on `(location_id, date)` using inner join
- Final dataset: 7,199 rows with 19 features + 1 target

## Methodology
1. **Data Integration**: Two sources merged by location and date
2. **Feature Engineering**: Created `Rainfall_Index` from humidity, cloud cover, wind speed, and pressure
3. **Preprocessing**: Median imputation, StandardScaler, OneHotEncoder via sklearn Pipeline and ColumnTransformer
4. **Models**: Logistic Regression, Random Forest, K-Nearest Neighbors
5. **Evaluation**: Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC
6. **Hyperparameter Tuning**: GridSearchCV on Random Forest (n_estimators, max_depth) with 5-fold CV

## Final Model

- **Algorithm**: Tuned Logistic Regression with optimized `C` and `solver` hyperparameters using GridSearchCV and 5-fold Cross-Validation.
- **Pipeline**: Complete preprocessing + tuned Logistic Regression saved as `rainfall_model.pkl`.
- **Primary Selection Criterion**: Recall, because False Negatives represent missed rainfall events and are considered more costly in this application.

## Main Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| DummyClassifier | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.340 | 0.746 | 0.467 | 0.827 |
| Random Forest | 0.588 | 0.244 | 0.344 | 0.846 |
| KNN | 0.509 | 0.301 | 0.378 | 0.798 |
| **Tuned Logistic Regression** | **0.327** | **0.762** | **0.457** | **0.826** |

### Final Model Selection

**Tuned Logistic Regression** was selected as the final model because it achieved the highest **Recall (0.762)** among the evaluated models. This means it detected approximately **76.2% of the actual rainfall events** in the test set.

Although Random Forest achieved a higher ROC-AUC (**0.846**), Recall is the primary metric in this project because **False Negatives are considered more costly than False Positives**.

The tuned model improved Recall from **0.746 to 0.762** compared with the original Logistic Regression, while its F1-Score changed from **0.467 to 0.457**. Therefore, the final model prioritizes the detection of rainfall events while Precision and F1-Score are retained as supporting evaluation metrics.

## How to Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Project

### Run the Jupyter Notebook
```bash
jupyter notebook Rainfall_Prediction.ipynb
```

### Run the Web Application
```bash
python app_flask.py
```
**then**
```bash
http://127.0.0.1:5000 
```

## Files
- `Rainfall_Prediction.ipynb` - Complete notebook with documentation and executable code
- `rainfall_model.pkl` - Saved final pipeline (preprocessing + tuned model)
- `app.py` - Streamlit web application for interactive predictions
- `requirements.txt` - Project dependencies
- `data_dictionary.csv` - Complete data dictionary for all features
- `merged_rainfall_dataset.csv` - Final merged dataset
- `model_comparison.csv` - Model comparison results
