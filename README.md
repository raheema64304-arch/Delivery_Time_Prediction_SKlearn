# Delivery_Time_Prediction_SKlearn
Machine learning project that predicts delivery time using distance, preparation time, traffic, weather, and vehicle type. Built with Python, Pandas, Scikit-learn, Random Forest, Joblib, Matplotlib, and Streamlit.
# Delivery Time Prediction using Scikit-learn

## Project Overview

**Delivery Time Prediction** is a machine learning project that predicts the estimated delivery time of an order based on important delivery-related factors such as distance, food preparation time, traffic conditions, weather, and vehicle type.

The project uses a **Random Forest Regressor** with a Scikit-learn preprocessing pipeline to handle both numerical and categorical features. It also includes an interactive **Streamlit web application** where users can enter delivery details and receive an estimated delivery time.

This project is designed as an educational machine learning application and demonstrates the complete workflow from dataset preparation and model training to prediction and deployment through a web interface.

## Features

* Predict delivery time in minutes
* Random Forest regression model
* Numerical and categorical feature preprocessing
* One-hot encoding for categorical variables
* Train/test dataset splitting
* Model evaluation using:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * R² Score
* Feature importance visualization
* Command-line prediction interface
* Interactive Streamlit web application
* Training dataset preview through the web application
* Saved trained model using Joblib

## Machine Learning Workflow

The project follows these steps:

1. Load the delivery dataset.
2. Select relevant input features.
3. Separate features and target variable.
4. Preprocess numerical and categorical features.
5. Split the data into training and testing sets.
6. Train a Random Forest Regressor.
7. Evaluate the model on test data.
8. Save the trained model.
9. Generate feature importance visualization.
10. Use the trained model for new delivery-time predictions.

## Dataset

The project includes a synthetic dataset containing **500 delivery records**.

### Dataset Features

| Feature                | Description                     | Type        |
| ---------------------- | ------------------------------- | ----------- |
| `order_id`             | Unique order identifier         | Integer     |
| `distance_km`          | Delivery distance in kilometers | Numerical   |
| `preparation_time_min` | Food/order preparation time     | Numerical   |
| `traffic_level`        | Traffic condition               | Categorical |
| `weather`              | Weather condition               | Categorical |
| `vehicle_type`         | Delivery vehicle used           | Categorical |
| `delivery_time_min`    | Actual delivery time            | Target      |

### Input Features Used by the Model

The model uses the following five features:

* Distance
* Preparation time
* Traffic level
* Weather
* Vehicle type

The target variable is:

```text
delivery_time_min
```

## Machine Learning Model

The project uses:

**RandomForestRegressor**

Model configuration:

```text
n_estimators = 150
max_depth = 10
random_state = 42
```

A Scikit-learn `Pipeline` is used to combine preprocessing and model training.

### Preprocessing

Numerical features:

* `distance_km`
* `preparation_time_min`

Categorical features:

* `traffic_level`
* `weather`
* `vehicle_type`

Categorical features are transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This allows the model to work with categorical delivery information while keeping preprocessing and prediction consistent.

## Model Evaluation

The model is evaluated using three regression metrics:

### Mean Absolute Error

Measures the average absolute difference between the predicted and actual delivery times.

### Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

### R² Score

Measures how well the model explains the variation in delivery time.

The exact metric values are calculated automatically when `train_model.py` is executed.

## Streamlit Web Application

The project includes an interactive Streamlit application.

The application provides three main sections:

### 1. Live ETA Predictor

Users can enter:

* Delivery distance
* Kitchen/preparation time
* Traffic condition
* Weather condition
* Vehicle type

The application then displays the predicted delivery time.

### 2. Feature Importance & Insights

Displays the feature importance chart generated during model training.

This helps understand which processed features contribute most to the Random Forest predictions.

### 3. Delivery Dataset

Displays information about the training dataset and provides a preview of the delivery records.

## Project Structure

```text
Delivery_Time_Prediction_Sklearn/
│
├── Delivery_Time_Prediction_Sklearn/
│   ├── data/
│   │   └── delivery_data.csv
│   │
│   ├── app.py
│   ├── predict.py
│   ├── train_model.py
│   ├── delivery_time_model.pkl
│   ├── feature_importance.png
│   ├── requirements.txt
│   ├── README.md
│   └── HOW_TO_RUN.md
│
├── app.py
├── HOW_TO_RUN.md
└── run_frontend.bat
```

## Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and preprocessing
* **Random Forest Regressor** – Regression algorithm
* **Matplotlib** – Feature importance visualization
* **Joblib** – Model serialization
* **Streamlit** – Interactive web application

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Delivery_Time_Prediction_Sklearn.git
```

Navigate to the project directory:

```bash
cd Delivery_Time_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

The training script will:

* Load `delivery_data.csv`
* Train the Random Forest model
* Calculate MAE, RMSE, and R²
* Save the trained model as:

```text
delivery_time_model.pkl
```

It will also generate:

```text
feature_importance.png
```

## Make Predictions from the Terminal

Run:

```bash
python predict.py
```

The program will ask for delivery information such as:

```text
Distance in km:
Food/order preparation time in minutes:
Traffic level (Low/Medium/High):
Weather (Clear/Rainy/Cloudy):
Vehicle type (Bike/Scooter/Car):
```

It will then display the predicted delivery time.

## Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

You can then enter delivery details through the web interface and receive a predicted ETA.

## Example Prediction

Example input:

```text
Distance: 5.5 km
Preparation Time: 15 minutes
Traffic: Medium
Weather: Clear
Vehicle: Bike
```

The trained model will process these values and return an estimated delivery time in minutes.

The exact prediction depends on the trained model stored in `delivery_time_model.pkl`.

## Applications

This type of machine learning system can be used for:

* Food delivery platforms
* E-commerce delivery systems
* Logistics applications
* Courier services
* Restaurant order management
* Delivery ETA estimation
* Route and operational analysis

## Limitations

This project uses a **synthetic dataset**, so the model should not be treated as a production-ready delivery forecasting system.

Real-world delivery prediction can also depend on factors such as:

* Road closures
* GPS location
* Driver availability
* Route selection
* Peak-hour demand
* Accidents
* Special events
* Exact weather conditions
* Restaurant workload
* Real-time traffic data

For production use, the model should be trained and validated using reliable historical delivery data.

## Future Improvements

Possible improvements include:

* Use real-world delivery datasets
* Add GPS and location information
* Integrate real-time traffic APIs
* Add time-of-day and day-of-week features
* Include driver availability
* Perform hyperparameter tuning
* Compare Random Forest with XGBoost, Gradient Boosting, and other algorithms
* Add cross-validation
* Deploy the Streamlit application online
* Add prediction confidence or an ETA range
* Build a dashboard for delivery analytics

## Author

Developed as a machine learning project demonstrating regression, preprocessing, model evaluation, and interactive deployment using Python and Scikit-learn.
