import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt



# Load your dataset (ensure it's available in the server)
df = pd.read_csv('premierLeague.csv')

# Page title
st.title("Premier League Match Prediction")

# Sidebar for team selection
st.sidebar.header("Select Teams")
team1 = st.sidebar.selectbox("Select Team 1", df['Team'].unique())
team2 = st.sidebar.selectbox("Select Team 2", df['Team'].unique())

# Helper function to predict the next score using ARIMA
def predict_next_score(score_series):
    try:
        model = ARIMA(score_series, order=(2, 0, 2))
        model_fit = model.fit()
        
        # Forecast the next period (e.g., next match or day)
        forecast = model_fit.forecast(steps=1)  # Adjust steps if more predictions are needed
        forecast_results = forecast.round().astype(int).tolist()  # Convert to integer list
        
        return forecast_results[0]
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {e}")
        return None

# Process data for the selected teams
if team1 and team2:
    # Display historical data
    st.header("Historical Match Data")
    
    df_team1 = df[df['Team'] == team1][['Match_Date', 'Score']].copy()
    df_team2 = df[df['Team'] == team2][['Match_Date', 'Score']].copy()
    
    # Convert Match_Date to datetime
    df_team1['Match_Date'] = pd.to_datetime(df_team1['Match_Date'], errors='coerce')
    df_team2['Match_Date'] = pd.to_datetime(df_team2['Match_Date'], errors='coerce')
    
    df_team1.dropna(subset=['Match_Date'], inplace=True)
    df_team2.dropna(subset=['Match_Date'], inplace=True)
    
    # Display data tables
    st.subheader(f"{team1} Scores")
    st.dataframe(df_team1)
    st.subheader(f"{team2} Scores")
    st.dataframe(df_team2)

    # Resample and interpolate
    df_team1.set_index('Match_Date', inplace=True)
    df_team2.set_index('Match_Date', inplace=True)
    df_team1 = df_team1.resample('D').asfreq().interpolate(method='linear')
    df_team2 = df_team2.resample('D').asfreq().interpolate(method='linear')
    
    # Predict next scores
    next_score_team1 = predict_next_score(df_team1['Score'])
    next_score_team2 = predict_next_score(df_team2['Score'])
    
    # Display predictions
    st.header("Predicted Next Match Scores")
    st.write(f"Predicted score for {team1}: **{next_score_team1}**")
    st.write(f"Predicted score for {team2}: **{next_score_team2}**")
    
    # Plot scores
    st.header("Score Trends")
    fig, ax = plt.subplots()
    ax.plot(df_team1.index, df_team1['Score'], label=f"{team1} Scores", color='blue')
    ax.plot(df_team2.index, df_team2['Score'], label=f"{team2} Scores", color='red')
    ax.legend()
    ax.set_title("Score Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    st.pyplot(fig)
