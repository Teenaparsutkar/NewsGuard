
import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Spam Detection Dashboard")
st.caption("Machine Learning powered message classification")

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", "95.27%")
col2.metric("Unique Messages", "5,171")
col3.metric("Algorithm", "Naive Bayes")

st.divider()

st.subheader("📩 Check Your Message")

message = st.text_area(
    "Enter a message below:",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

if st.button("🔍 Check Message", type="primary"):
    if message.strip():
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
            st.write("This message has been classified as spam.")
        else:
            st.success("✅ HAM — LIKELY A LEGITIMATE MESSAGE")
            st.write("This message has been classified as ham.")
    else:
        st.warning("Please enter a message first.")

st.divider()

st.subheader("📊 Model Evaluation")

st.write("The model was evaluated on 1,035 test messages.")

col4, col5 = st.columns(2)
col4.metric("Spam Precision", "100%")
col5.metric("Spam Recall", "62.60%")

st.info(
    "Limitation: The model missed 49 spam messages in the test set. "
    "A 'ham' prediction does not guarantee that a message is safe."
)

st.caption(
    "Dataset: UCI SMS Spam Collection | "
    "Model: TF-IDF + Multinomial Naive Bayes"
)
