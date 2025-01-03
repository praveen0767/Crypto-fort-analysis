import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import datetime

# Streamlit UI Configuration
st.set_page_config(
    page_title="Crypto Fort Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("CRYPTO FORT ANALYSIS")
st.subheader("Analyze Password Similarities to Detect Threats in Real-Time")

# Initialize Session State for Tracking Attempts
if "attempts_log" not in st.session_state:
    st.session_state["attempts_log"] = []

# Define Constants
ATTEMPT_LIMIT = 8  # Max attempts allowed
TIME_WINDOW_HOURS = 4  # Time window in hours

# Helper Function to Check Attempt Status
def can_attempt():
    now = datetime.datetime.now()
    # Remove attempts older than the time window
    st.session_state["attempts_log"] = [
        timestamp for timestamp in st.session_state["attempts_log"]
        if now - timestamp <= datetime.timedelta(hours=TIME_WINDOW_HOURS)
    ]
    return len(st.session_state["attempts_log"]) < ATTEMPT_LIMIT

# Display Input Block Message if Limit Exceeded
if not can_attempt():
    next_attempt_time = min(
        st.session_state["attempts_log"]
    ) + datetime.timedelta(hours=TIME_WINDOW_HOURS)
    remaining_time = next_attempt_time - datetime.datetime.now()
    st.error(
        f"Attempt limit reached! Please wait {remaining_time.seconds // 3600}h "
        f"{(remaining_time.seconds % 3600) // 60}m to try again."
    )
else:
    # Sidebar Inputs
    st.sidebar.header("Input Passwords")
    correct_password = st.sidebar.text_input("Correct Password", type="password")
    wrong_passwords = st.sidebar.text_area(
        "8 Incorrect Passwords",
        placeholder="Enter each incorrect password on a new line",
    )
    custom_threshold = st.sidebar.slider(
        "Set Threat Level Threshold (%)", min_value=0, max_value=100, value=50
    )

    # Process Inputs
    if st.sidebar.button("Analyze"):
        # Validate Inputs
        if not correct_password:
            st.error("Correct password is required!")
        elif len(wrong_passwords.strip().split("\n")) != 8:
            st.error("Please enter exactly 8 incorrect passwords!")
        else:
            # Log Current Attempt
            st.session_state["attempts_log"].append(datetime.datetime.now())

            # Prepare Data
            wrong_passwords_list = wrong_passwords.strip().split("\n")
            all_passwords = [correct_password] + wrong_passwords_list

            # Vectorize Passwords
            vectorizer = CountVectorizer(analyzer="char")
            password_vectors = vectorizer.fit_transform(all_passwords)

            # Compute Similarity (Cosine Similarity)
            similarity_scores = cosine_similarity(password_vectors[0:1], password_vectors[1:]).flatten()
            similarity_scores_percentage = similarity_scores * 100

            # Calculate Overall Statistics
            average_similarity = np.mean(similarity_scores_percentage)
            min_similarity = np.min(similarity_scores_percentage)
            max_similarity = np.max(similarity_scores_percentage)
            median_similarity = np.median(similarity_scores_percentage)
            threat_status = "Safe" if average_similarity < custom_threshold else "Threat"

            # Create DataFrame for Dashboard
            df = pd.DataFrame({
                "Password": wrong_passwords_list,
                "Similarity (%)": similarity_scores_percentage,
            }).sort_values(by="Similarity (%)", ascending=False)

            df["Threat"] = df["Similarity (%)"].apply(
                lambda x: "Exceeds Threshold" if x > custom_threshold else "Below Threshold"
            )

            # Password Strength Indicator
            st.sidebar.subheader("Password Strength")
            password_strength = len(set(correct_password)) / len(correct_password) * 100 if correct_password else 0
            st.sidebar.progress(int(password_strength))
            st.sidebar.text(f"Strength: {round(password_strength, 2)}%")

            # Dashboard Layout
            st.markdown("### Password Analysis Dashboard")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.metric("Total Attempts", len(wrong_passwords_list))
                st.metric("Average Similarity (%)", round(average_similarity, 2))
                st.metric("Min Similarity (%)", round(min_similarity, 2))
                st.metric("Max Similarity (%)", round(max_similarity, 2))
                st.metric("Median Similarity (%)", round(median_similarity, 2))
                st.metric("Threat Status", threat_status)

            with col2:
                st.markdown("#### Bar Chart of Similarity Scores")
                bar_chart = px.bar(
                    df,
                    x="Password",
                    y="Similarity (%)",
                    title="Similarity Scores",
                    color="Threat",
                    color_discrete_map={"Exceeds Threshold": "red", "Below Threshold": "blue"},
                )
                bar_chart.update_layout(yaxis_title="Similarity (%)", xaxis_title="Passwords")
                st.plotly_chart(bar_chart, use_container_width=True)

            with col3:
                st.markdown("#### Heatmap of Similarity Matrix")
                similarity_matrix = cosine_similarity(password_vectors)
                heatmap_fig = px.imshow(
                    similarity_matrix,
                    labels=dict(x="Password Index", y="Password Index", color="Similarity"),
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)

            # Additional Insights
            col4, col5 = st.columns([2, 1])

            with col4:
                st.markdown("#### Line Chart of Similarity Trends")
                line_chart = px.line(
                    df,
                    x="Password",
                    y="Similarity (%)",
                    title="Similarity Trends",
                    markers=True,
                )
                st.plotly_chart(line_chart, use_container_width=True)

            with col5:
                st.markdown("#### Boxplot of Similarity Distribution")
                boxplot_fig = px.box(
                    df,
                    y="Similarity (%)",
                    points="all",
                    title="Similarity Distribution",
                )
                st.plotly_chart(boxplot_fig, use_container_width=True)

            # Downloadable Report
            st.markdown("### Download Report")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Report as CSV",
                data=csv,
                file_name="password_similarity_report.csv",
                mime="text/csv",
            )
