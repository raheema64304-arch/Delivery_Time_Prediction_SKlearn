# How to Run: Delivery Time Prediction using Python + Scikit-learn

This machine learning project predicts delivery time using distance, preparation time, traffic, weather, and vehicle type.

---

## 1. Prerequisites

- **Python**: Python 3.9+ (Python 3.14 tested)
- **Terminal**: Windows PowerShell, Command Prompt, or VS Code integrated terminal
- **Dependencies**: `pandas`, `scikit-learn`, `numpy`, `joblib`, `matplotlib`

---

## 2. Open Terminal & Navigate to Project Directory

Open PowerShell or Command Prompt, then change directory to the project folder:

```powershell
# Note: Project files are inside the nested "Delivery_Time_Prediction_Sklearn" folder:
cd "c:\Users\user\Desktop\jj college\Delivery_Time_Prediction_Sklearn\Delivery_Time_Prediction_Sklearn"
```

---

## 3. Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

> **Windows Note**: If `python` or `pip` opens the Microsoft Store, specify the direct Python path:
> ```powershell
> & "C:\Users\user\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
> ```

---

## 4. Step 1: Train the Machine Learning Model

Execute the training script to train and save the model:

```powershell
python train_model.py
```
*(or `py train_model.py` / `& "C:\Users\user\AppData\Local\Python\bin\python.exe" train_model.py`)*

### What this does:
1. Loads the dataset from `data/delivery_data.csv`.
2. Trains a **RandomForestRegressor** model predicting `delivery_time_min`.
3. Evaluates model performance (Mean Absolute Error, Mean Squared Error, R2 Score).
4. Saves the trained model (`.pkl` file).
5. Generates and saves visualization plots/charts (e.g., feature importance or segment visualizations).

---

## 5. Step 2: Run Predictions

Run the prediction script to test predictions:

```powershell
python predict.py
```

### Interactive Inputs:
This script prompts for inputs in the terminal. Here is an example of what to enter when prompted:

| Prompt / Field | Example Value |
| :--- | :--- |
| `Distance in km:` | `5` |
| `Food/order preparation time in minutes:` | `5` |
| `Traffic level (Low/Medium/High` | `5` |
| `Weather (Clear/Rainy/Cloudy` | `0.0` |
| `Vehicle type (Bike/Scooter/Car` | `5` |

---

## 6. Directory Structure

```text
Delivery_Time_Prediction_Sklearn/
├── data/
│   └── delivery_data.csv
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```
