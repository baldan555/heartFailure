import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

logistic_params = {'C': 0.1, 'solver': 'liblinear'}
random_forest_params = {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 200}
gradient_boosting_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}
svc_params = {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}

models = {
    "Logistic Regression": LogisticRegression(**logistic_params),
    "Random Forest": RandomForestClassifier(**random_forest_params),
    "Gradient Boosting": GradientBoostingClassifier(**gradient_boosting_params),
    "Support Vector Classifier": SVC(**svc_params, probability=True)
}

categorical_features = ['ChestPainType', 'RestingECG', 'ST_Slope']
numerical_features = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

def load_data():
    df = pd.read_csv('heart.csv')

    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'M' else 0)
    df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda x: 1 if x == 'Y' else 0)

    return df

if 'entries' not in st.session_state:
    st.session_state.entries = pd.DataFrame(columns=['Name', 'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

def get_user_input():
    st.sidebar.header("User Input")

    name = st.sidebar.text_input("Name", "")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=40)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=500, value=200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["N", "Y"])
    oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    user_data = {
        'Name': name,
        'Age': age,
        'Sex': 1 if sex == "M" else 0,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': 1 if exercise_angina == "Y" else 0,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    return user_data

def upload_file():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for prediction", type="csv")
    if uploaded_file is not None:
        file_df = pd.read_csv(uploaded_file)
        file_df['Sex'] = file_df['Sex'].apply(lambda x: 1 if x == 'M' else 0)
        file_df['ExerciseAngina'] = file_df['ExerciseAngina'].apply(lambda x: 1 if x == 'Y' else 0)
        missing_cols = set(numerical_features + categorical_features) - set(file_df.columns)
        for col in missing_cols:
            file_df[col] = 0 if col in numerical_features else 'Normal'
        return file_df
    return None

st.title("Heart Disease Prediction")

df = load_data()

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[model_name]

user_input = get_user_input()

if st.sidebar.button("Insert"):
    st.session_state.entries = pd.concat([st.session_state.entries, pd.DataFrame([user_input])], ignore_index=True)

st.subheader("User Input Data")
st.dataframe(st.session_state.entries)

st.subheader("File Upload Data")
file_df = upload_file()
if file_df is not None:
    st.dataframe(file_df)

def train_model_and_predict(input_df):
    try:
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']

        results = {}
        probabilities = {}
        
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X, y)

            predictions = pipeline.predict(input_df)
            probas = pipeline.predict_proba(input_df)

            results[model_name] = input_df.copy()
            results[model_name]['Prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in predictions]
            results[model_name]['Probability'] = [f"{prob[1]*100:.2f}%" for prob in probas]
            
            probabilities[model_name] = probas[:, 1]  

        return results, probabilities
        
    except NotFittedError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")

# Predict based on user input or file
if st.button("Train and Predict"):
    if file_df is not None:
        predictions, probabilities = train_model_and_predict(file_df)
        input_names = file_df['Name'].tolist()  # Names from file upload
    else:
        predictions, probabilities = train_model_and_predict(st.session_state.entries)
        input_names = st.session_state.entries['Name'].tolist()  # Names from user input
    
    for model_name, result_df in predictions.items():
        st.subheader(f"Prediction Results for {model_name}")
        st.dataframe(result_df)

    st.subheader("Model Probability Visualizations")
    

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Model Probability Comparisons')
    
    for i, (model_name, prob) in enumerate(probabilities.items()):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        
        ax.plot(input_names, prob, marker='o', linestyle='-', label=model_name)
        ax.set_title(model_name)
        ax.set_xlabel('Names')
        ax.set_ylabel('Probability of Heart Disease')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
