import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load

# Load the trained model and vectorizer
nb_classifier = load('naive_bayes_model.joblib')
vectorizer = load('count_vectorizer.joblib')

# Load the test data
test_data = pd.read_csv('processed_comments.csv')

# Create a function to make predictions
def make_prediction(text):
    text_count = vectorizer.transform([text])
    prediction = nb_classifier.predict(text_count)
    return prediction[0]

# Create a Streamlit app
st.title("Sentiment Analysis Dashboard")

# Add a text input for user input
user_input = st.text_input("Enter a comment to analyze:", value="")

# Add a button to make predictions
if st.button("Analyze"):
    prediction = make_prediction(user_input)
    st.write("Predicted Sentiment:", prediction)

# Add a section to visualize the results
st.header("Results Visualization")

# Load the test data
test_data = pd.read_csv('processed_comments.csv')

# Create a confusion matrix
y_pred = nb_classifier.predict(vectorizer.transform(test_data['comment_text']))
conf_mat = confusion_matrix(test_data['Sentiment'], y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, cmap="Oranges", fmt="d")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix (Naive Bayes)")
st.pyplot(plt)

# Add a section to display the classification report
st.header("Classification Report")
report = classification_report(test_data['Sentiment'], y_pred)
st.write(report)

# Add a section to display the accuracy score
st.header("Accuracy Score")
accuracy = accuracy_score(test_data['Sentiment'], y_pred)
st.write("Accuracy:", accuracy)