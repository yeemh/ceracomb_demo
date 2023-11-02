import os
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


### 분류 예측 ###
def get_data() -> pd.DataFrame:
    df = pd.read_csv('./data/slided_without_error.csv',
                    dtype={'date_time': str, 'id': int, 'lat': float,
                           'lng': float, 'speed': int, 'temp': int,
                           'pres': int, 'noxIn': int, 'tempIn': int, 
                           'tempOut': int, 'dRate': float, 'ureaLv': int,
                           'maf': int, 'rideStart': bool, 'ride_id': int,
                           'speed-1': int, 'temp-1': int, 'pres-1': int,
                           'noxIn-1': int, 'tempIn-1': int, 'tempOut-1': int,
                           'dRate-1': float, 'ureaLv-1': int, 'maf-1': int, 
                           'elapsed_time': str,
                          },
                    parse_dates=['date_time']
                   )
    df.elapsed_time = pd.to_timedelta(df.elapsed_time).dt.seconds
    return df

def resample_data(df: pd.DataFrame) -> (pd.DataFrame, np.array) :
    tmp = df.sample(n = 100000,random_state= 42).drop(columns = ['date_time', 'id', 'lat','lng', 
                                           'dRate', 'rideStart', 'ride_id'])
    
    tmp.replace([np.inf,-np.inf],np.nan, inplace = True)
    tmp.dropna(inplace=True)

    labels = pd.read_csv("./data/weird trio.csv").loc[tmp.index,'weird3'].values
    return (tmp, labels)

def confusion_matrix_test(x,y):
    if y:
        if x: 
            res = 'TP'
        else:
            res = 'FP'
    else:
        if x:
            res = 'FN'
        else:
            res = 'TN'
    return res

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def classify_test(exp_num: int):
    with st.status("Testing...", expanded=True) as status:
        st.write("Test is starting...")
        st.write("At first, Program read the raw data ... ")
        df = get_data()

        st.write(f"The shape of the raw data is {df.shape}\n")
        st.write(f"Resample Data and train_test_split for experiment #{exp_num}")

        data_feature, data_labels = resample_data(df)
        
        st.write(f"The shape of the data_feature is {data_feature.shape}")


        train_features, test_features, train_labels, test_labels = train_test_split(data_feature, data_labels, test_size = 30000, random_state = 121)
        st.write(f'Training Features Shape: {train_features.shape}')
        st.write(f'Training Labels Shape: {train_labels.shape}')
        st.write(f'Testing Features Shape: {test_features.shape}')
        st.write(f'Testing Labels Shape: {test_labels.shape}')
        st.write(f'Get trained model and save the result')
        model = pickle.load(open('./model/rfc.pkl','rb'))
        st.write(f'(Train) Out-of-bag score estimate: {model.oob_score_:.3}')

        # take random 10000 test_data
        random_ = test_features.reset_index(drop=True).sample(n = 10000)
        predicted = model.predict(random_)
        accuracy = accuracy_score(test_labels[random_.index], predicted)
        cm = confusion_matrix(test_labels[random_.index],predicted)
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]

        st.write(f'(Testing)Experiment\n#{exp_num} Mean accuracy score: {accuracy:.3}')
        st.write(f'TP - {TP}, FP - {FP}, FN - {FN}, TN - {TN}')

        res = pd.DataFrame({'actual': test_labels[random_.index], 'predicted': predicted})
        res['status'] = res.apply(lambda x : confusion_matrix_test(x[0],x[1]),axis = 1)

        status.update(label="Test complete!", state="complete", expanded=True)
    
    csv = convert_df(res)
    st.download_button(
        label="Download result as CSV",
        data=csv,
        file_name=f'result_{exp_num}.csv',
        mime='text/csv',
    )


st.set_page_config(page_title="고장 진단")

### Title ###
st.title("순환자원 - 특수차량 재제조 기술")

### 시스템 정보 ###
st.header("미세먼지 저감을 위한 노후 특수∙화물차량(3.5~8톤급) 엔진 및 전자화 연동 후처리장치 재제조 기술 개발")

st.subheader("Raw 데이터 예시")
df = pd.read_excel('./data/230207.xlsx')
st.dataframe(df.head(30))


st.subheader("예지 보전 알고리즘 고장 분류 예측")

number = st.number_input("Write experiment number for the result check", min_value=1, value=1)
if st.button('고장 분류 예측 실행'):
    classify_test(number)