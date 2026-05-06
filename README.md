🚗
Used Car Price & Quality Predictor

A machine learning project that predicts the price of used cars and classifies their quality (Bad / Average / Good) using regression and classification models. Includes a fully interactive Streamlit web application for real-time predictions.

• Project Overview

This project was built as part of an MSc Data Science course. It walks through a complete ML pipeline — from raw data cleaning to model deployment — using a real-world used car dataset.

Two prediction tasks:
- 🔢 Regression — Predict the exact price of a used car
- 🏷️ Classification — Label car quality as `Bad`, `Average`, or `Good`

---

• 🗂️ Project Structure

```
📦 car-price-predictor/
├── car_price_prediction.ipynb     # Main Jupyter Notebook
├── app.py                         # Streamlit Web Application
├── best_car_price_model.pkl       # Saved Random Forest Regressor
├── car_quality_model.pkl          # Saved KNN Classifier
├── scaler.pkl                     # Saved StandardScaler
├── feature_columns.pkl            # Saved feature column list
└── README.md
```

---

• 🔧 Tech Stack

 Category          Libraries / Tools                              

 Data Processing   pandas, numpy                             
 Visualization     matplotlib, seaborn                        
 Machine Learning  scikit-learn                                 
 Deployment        streamlit, joblib                          
 Environment       Python 3.x, Google Colab / Jupyter Notebook    



• 📊 ML Pipeline

 1. Data Cleaning
- Stripped `$` and `,` from the `price` column
- Cleaned mileage string formatting
- Filled missing values (`fuel_type`, `accident`, `clean_title`)
- Dropped low-value columns (`brand`, `model`, `engine`, `ext_col`, `int_col`)

 2. Outlier Removal
- Applied **IQR method** on the `price` column to remove extreme outliers

3. Feature Engineering
- Created `car_age` from `model_year`
- Applied **Target Encoding** on high-cardinality columns (`brand`, `model`)
- Applied **One-Hot Encoding** on low-cardinality columns (`fuel_type`, `transmission`, `accident`, `clean_title`)
- Removed highly correlated features (threshold > 0.8)

 4. Log Transformation
- Applied `np.log1p()` to both `price` and `milage` to reduce skewness

 5. EDA
- Price distribution histogram
- Mileage vs. Price scatter plot
- Car age vs. Price box plot
- Top-10 feature correlation heatmap

6. Model Training

Regression Models (Price Prediction):

| Model                      | Metric |
|----------------------------|--------|
| Multiple Linear Regression | RMSE, R² |
| KNN Regressor (k=7)        | RMSE, R² |
| Random Forest Regressor    | RMSE, R² ✅ Best |

Classification Models (Car Quality):

 Model                  Metric   
Logistic Regression    Accuracy 
 KNN Classifier (k=7)   Accuracy 

 7. Model Saving
 Best models saved using `joblib` for deployment in the Streamlit app



 🖥️ Streamlit Web App

The app allows users to input car details via a sidebar and get instant predictions.

Input fields:
- Car Age (years)
- Mileage (miles)
- Brand
- Fuel Type
- Accident History
- Clean Title status

Outputs:
- 💰 Predicted Price (USD)
- 🏷️ Car Quality Label (Good  / Average  / Bad )

Run the app locally

pip install streamlit scikit-learn pandas numpy joblib
streamlit run app.py




 🚀 Getting Started

1. Clone the repository
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor

 2. Install dependencies
pip install -r requirements.txt

 3. Run the notebook
jupyter notebook car_price_prediction.ipynb

 4. Or launch the web app
streamlit run app.py
```

---

 📁 Dataset

The project uses a used car dataset (`used_cars.csv`) with features including:

`model_year`, `milage`, `fuel_type`, `transmission`, `accident`, `clean_title`, `price`, `brand`, `model`, `engine`, `ext_col`, `int_col`

> ⚠️ Dataset not included in this repo. You can source a similar dataset from [Kaggle](https://www.kaggle.com/datasets?search=used+cars).



 📈 Results

Fill in your actual model scores after running the notebook.

 Model                       RMSE    R² Score 
 Linear Regression           .0425     .5751      
 KNN Regressor               .0422     .5759       
 Random Forest Regressor     .0409     .6057     

 Model                   Accuracy 

 Logistic Regression     0.6773      
 KNN Classifier          0.6348      
---

🙋 Author

NISHANT KUMAR
MSc Data Science  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)





