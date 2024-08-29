import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

# Streamlit App
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Sentiment Analysis Settings")
st.sidebar.write("Upload a CSV file with Amazon reviews.")

# Main Title
st.title("📊 Amazon Reviews Sentiment Analysis")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    h1 {
        color: #336699;
        font-family: 'Trebuchet MS', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.write("""
    This interactive web app allows you to perform sentiment analysis on Amazon reviews using the VADER model. 
    Upload your CSV file, and the app will analyze the sentiments and provide insightful visualizations.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head(), height=200)
    
    # Check if the necessary columns exist
    if 'Text' in df.columns:
        # Apply sentiment analysis
        df['sentiment_score'] = df['Text'].apply(get_sentiment_score)
        
        # Display results
        st.write("### Sentiment Analysis Results")
        st.dataframe(df[['Text', 'sentiment_score']].head(), height=200)
        
        # Plot sentiment score distribution
        st.write("### Sentiment Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['sentiment_score'], bins=30, kde=True, ax=ax, color="#ff6347")
        ax.set_title('Distribution of Sentiment Scores', fontsize=15, color="#336699")
        ax.set_xlabel('Sentiment Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)
        
        # Plot sentiment scores by review length
        df['review_length'] = df['Text'].apply(len)
        st.write("### Sentiment Score vs. Review Length")
        fig, ax = plt.subplots()
        sns.scatterplot(x='review_length', y='sentiment_score', data=df, ax=ax, color="#2e8b57")
        ax.set_title('Sentiment Score vs. Review Length', fontsize=15, color="#336699")
        ax.set_xlabel('Review Length', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        st.pyplot(fig)
        
    else:
        st.error("The uploaded file must contain a 'Text' column with review text.")
else:
    st.info("Please upload a CSV file to begin sentiment analysis.")
