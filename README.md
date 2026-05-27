# Customer Churn Prediction Platform 🔮

so basically i built this thing that predicts whether a customer is gonna leave a company or not
before they actually do. companies lose mad money when customers dip so like... it matters

built the whole thing with python, machine learning, fastapi and docker. its not just a notebook
its an actual system with an api and everything

---

## what it actually does

you send in some customer info → it tells you if theyre gonna churn, how likely, and why

thats it. simple concept but the engineering behind it is kinda clean ngl

---

## the stack

| thing | why i used it |
|---|---|
| Python | its python bro |
| Pandas + NumPy | cleaning and crunching the data |
| Scikit-learn | training the ML models |
| FastAPI | building the prediction api |
| Docker | so it runs the same everywhere |
| Joblib | saving and loading the trained model |
| Pytest | testing so stuff doesnt randomly break |

---

## how the whole system flows

```
raw customer data
      ↓
clean it up (fix nulls, bad types)
      ↓
feature engineering (tenure groups, risk flags, avg spend)
      ↓
train ml model (logistic regression vs random forest, picks the best)
      ↓
save the model
      ↓
fastapi picks it up
      ↓
docker wraps everything
      ↓
hit /predict and get churn probability back
```

---

## project structure

```
customer-churn-platform/
│
├── app/
│   ├── main.py               ← fastapi app lives here
│   ├── model/
│   │   ├── train.py          ← trains the model
│   │   ├── predict.py        ← runs predictions
│   │   └── churn_model.pkl   ← saved model (generated)
│   │
│   ├── preprocessing/
│   │   └── preprocess.py     ← cleans + engineers features
│   │
│   ├── api/
│   │   └── routes.py         ← all the api endpoints
│   │
│   └── utils/
│       └── logger.py         ← logging stuff
│
├── data/
│   ├── raw/                  ← drop the kaggle csv here
│   └── processed/            ← gets generated after training
│
├── notebooks/                ← for experimenting
├── tests/                    ← pytest tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## running it locally (with docker)

**step 1** — grab the dataset from kaggle
[telco customer churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
drop the csv into `data/raw/`

**step 2** — train the model first
```bash
pip install -r requirements.txt
python -m app.model.train
```

**step 3** — spin up docker
```bash
docker-compose up --build
```

**step 4** — go to swagger ui
```
http://localhost:8000/docs
```
you'll see all the endpoints right there, you can test it live

---

## the api endpoints

### `GET /api/v1/health`
just checks if the api is alive and the model is loaded
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /api/v1/model-info`
shows which model won + the metrics
```json
{
  "model_name": "Random Forest",
  "metrics": { "f1": 0.62, "precision": 0.68, "recall": 0.57, "roc_auc": 0.84 },
  "top_churn_factors": [["Contract_Two_year", 0.12], ["tenure", 0.11], ...]
}
```

### `POST /api/v1/predict`
send customer data, get churn prediction back

example request:
```json
{
  "tenure": 3,
  "MonthlyCharges": 95.50,
  "TotalCharges": 286.50,
  "gender": 0,
  "SeniorCitizen": 0,
  "Partner": 0,
  "Dependents": 0,
  "PhoneService": 1,
  "PaperlessBilling": 1,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Credit_card": 0,
  "PaymentMethod_Mailed_check": 0
}
```

example response:
```json
{
  "will_churn": true,
  "churn_probability": 0.81,
  "risk_level": "HIGH",
  "top_churn_factors": [["Contract_Two_year", 0.12], ["tenure", 0.11]],
  "model_used": "Random Forest"
}
```

---

## the ml part

### why not just accuracy?

accuracy is kinda useless for churn prediction bc the dataset is imbalanced
(most customers dont churn). so i used:

- **F1 score** — balance between catching churners and not crying wolf
- **Recall** — catching as many real churners as possible matters more than false alarms
- **ROC-AUC** — how well the model separates churners from stayers overall

### feature engineering (the actually important bit)

raw data isnt enough. i made extra features:

- **tenure_group** — 0-1yr / 1-2yr / 2-4yr / 4+yr (new customers churn way more)
- **avg_spend_per_month** — total charges divided by tenure, shows value vs risk
- **high_risk flag** — customer under 12 months AND paying over $65/month = 🚩

### models trained

trained both logistic regression and random forest, auto picks whichever gets better F1

---

## running the tests

```bash
pytest tests/ -v
```

tests cover:
- api endpoints (valid inputs, bad inputs, missing fields)
- preprocessing pipeline (null handling, encoding, feature creation)
- model loading and prediction output shape

---

## what i'd add if i had more time

- streamlit frontend so non-technical people can use it
- mlflow for tracking experiments
- postgresql to log every prediction
- retrain pipeline when model drift is detected
- batch predictions endpoint for processing whole customer lists at once

---

## dataset

[IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7043 customers, 21 features, binary churn target

---

built this to learn how real ML systems actually work, not just notebooks
