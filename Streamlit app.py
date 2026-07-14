import streamlit as st
import requests


st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="centered",
)

API_URL = "https://fake-news-classisier.onrender.com"  
PREDICT_ENDPOINT = f"{API_URL}/predict"

st.title("📰 Fake News Classifier")
st.markdown(
    "TF-IDF + Logistic Regression model trained on the **WELFake** dataset "
    "(95.3% accuracy). Paste a news headline or article body below."
)

with st.expander("ℹ️ About this project"):
    st.write(
        """
        - **Model:** Logistic Regression on TF-IDF features
        - **Dataset:** WELFake (~72,000 labeled news articles)
        - **Accuracy:** 95.3%
        - **Backend:** FastAPI (deployed on Render)
        - **Frontend:** Streamlit (deployed on Streamlit Cloud)
        """
    )

news_title = st.text_input(
    "Headline (optional, but improves accuracy)",
    placeholder="e.g. Scientists discover new exoplanet in habitable zone",
)

news_text = st.text_area(
    "Article text",
    height=200,
    placeholder="Paste the article body here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Classify", type="primary", use_container_width=True)

if predict_clicked:
    if not news_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    PREDICT_ENDPOINT,
                    json={"title": news_title, "text": news_text},
                    timeout=30,  # Render free tier can cold-start slowly
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]

                    if prediction == "Fake":
                        st.error(f"🚨 **Prediction: {prediction}**")
                    else:
                        st.success(f"✅ **Prediction: {prediction}**")

                    st.metric("Confidence", f"{confidence * 100:.1f}%")

                    st.subheader("Probability breakdown")
                    st.bar_chart(probabilities)

                else:
                    st.error(f"API error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the API. It may be waking up from Render's "
                    "free-tier sleep — wait ~30-50 seconds and try again."
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again in a moment.")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.divider()
st.caption("Built by Amey Dande · [GitHub](https://github.com/AmeyDande/Fake_News_Classisier)")
