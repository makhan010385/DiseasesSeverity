import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page settings
st.set_page_config(page_title="🌾 Soybean Disease Prediction", layout="wide")
st.title("🌾 Soybean Disease Prediction App")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Updated_CSV6.csv")
    df.columns = df.columns.str.strip()
    df['Variety'] = df['Variety'].astype(str)
    return df

df = load_data()

# Remove rows with missing critical values
features = ['PDI', 'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity',
            'No_of_Rainy_Days', 'Rainfall', 'Wind_Velocity', 'Variety']
df.dropna(subset=features + ['Disease'], inplace=True)

# Label encoding
le_variety = LabelEncoder()
df['Variety_encoded'] = le_variety.fit_transform(df['Variety'])

le_disease = LabelEncoder()
df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])

# Define features and target
X = df[['PDI', 'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity',
        'No_of_Rainy_Days', 'Rainfall', 'Wind_Velocity', 'Variety_encoded']]
y = df['Disease_encoded']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Disease severity classifier
def classify_severity(pdi):
    if pdi == 0:
        return "No Disease"
    elif 1 <= pdi <= 10:
        return "Low Disease"
    elif 11 <= pdi <= 30:
        return "Medium Disease"
    elif 31 <= pdi <= 50:
        return "High Disease"
    elif 51 <= pdi <= 75:
        return "Very High Disease"
    elif 76 <= pdi <= 100:
        return "Severe Disease"
    else:
        return "Unknown"

# Predict for all rows in the dataset
X_all = df[['PDI', 'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity',
            'No_of_Rainy_Days', 'Rainfall', 'Wind_Velocity', 'Variety_encoded']]

df['Predicted Disease'] = le_disease.inverse_transform(model.predict(X_all))
df['Disease Severity'] = df['PDI'].apply(classify_severity)

# Save updated dataset
df.to_csv("Updated_CSV4_with_predictions.csv", index=False)

# Sidebar input
st.sidebar.header("🧑‍🌾 Input for Prediction")
location_input = st.sidebar.selectbox("Select Location", sorted(df['Location'].unique()))
variety_input = st.sidebar.selectbox("Select Variety", sorted(df['Variety'].unique()))
sowing_date = st.sidebar.date_input("Select Sowing Date")

# Filtered prediction preview
if st.sidebar.button("🔍 Show Prediction"):
    filtered = df[(df['Location'] == location_input) & (df['Variety'] == variety_input)]

    if filtered.empty:
        st.error("No records found for selected Location and Variety.")
    else:
        filtered['Parsed_Date'] = pd.to_datetime(
            filtered['Date'].str.extract(r'(\d{2}-\d{2}-\d{4})')[0], dayfirst=True, errors='coerce'
        )
        filtered['Crop_Age'] = (filtered['Parsed_Date'] - pd.to_datetime(sowing_date)).dt.days
        valid = filtered[filtered['Crop_Age'] >= 0]
        closest = valid.loc[valid['Crop_Age'].idxmin()] if not valid.empty else filtered.iloc[0]

        st.success(f"🌿 Predicted Disease: **{closest['Predicted Disease']}**")
        st.info(f"🩺 Disease Severity Level: **{closest['Disease Severity']}**")

        st.dataframe(closest[['Date', 'Crop_Week', 'PDI', 'Predicted Disease', 'Disease Severity']].to_frame().T)

# Show distribution
st.subheader("📊 Disease Distribution")
st.bar_chart(df['Disease'].value_counts())

# Model Evaluation
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)

st.write(f"✅ Accuracy: **{accuracy_score(y_test, y_pred):.2f}**")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred, target_names=le_disease.classes_))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=le_disease.classes_, yticklabels=le_disease.classes_,
            cmap="YlGnBu")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Sample output
st.subheader("📂 Sample Predictions")
st.dataframe(df[['Location', 'Variety', 'Date', 'PDI', 'Predicted Disease', 'Disease Severity']].head(10))
