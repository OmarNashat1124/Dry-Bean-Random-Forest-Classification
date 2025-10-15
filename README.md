# 🫘 Dry Bean Classification Project

This project trains and evaluates a machine learning model to classify dry beans using numerical features. It handles data preprocessing (skewness correction and scaling) directly inside the model pipeline.

---

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

##  How to Run

1. Ensure `Dry_Bean_Dataset.csv` is in the same directory as the scripts.
2. Run the main script:

```bash
python main.py
```

---

##  What Each File Does

### `main.py`
- Loads the dataset from the CSV file.
- Calls the training function to preprocess data and train a classifier.
- Prints the model’s accuracy and classification report.

### `utils.py`
- **`load_dataset(path)`**: Loads the dataset using pandas.
- **`train_model(df, target_col="Class")`**:
  - Cleans and converts all numeric features.
  - Applies skewness correction with `PowerTransformer`.
  - Scales features with `StandardScaler`.
  - Trains a `RandomForestClassifier`.
  - Outputs accuracy and a detailed classification report.

---

## 🧩 Preprocessing Pipeline

The numeric preprocessing inside the model pipeline includes:
1. **PowerTransformer (Yeo–Johnson)** → Fixes skewed distributions.  
2. **StandardScaler** → Normalizes feature scales.  

This ensures the model works optimally even when the raw features are not normally distributed.

---

## 📊 Output

After running `main.py`, you’ll see:

- Model **Accuracy**  
- **Classification Report** (Precision, Recall, F1-score)




