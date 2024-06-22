
import streamlit as st
import pickle
import pandas as pd

#load the ml model
model_data = pickle.load(open('model/ml_model.pkl','rb'))

model = model_data['model']
teams = model_data['teams']
cities = model_data['cities']

st.title('IPL Winner Team Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select a batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select a bowling team',sorted(teams))

selected_city = st.selectbox('Select the host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Probability Prediction'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    data = {'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[selected_city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets],
            'total_runs_x':[target],
            'crr':[crr],'rrr':[rrr]}
    input_df = pd.DataFrame(data)

    result = model.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.write(batting_team + "- " + str(round(win*100)) + "%")
    st.write(bowling_team + "- " + str(round(loss*100)) + "%")

