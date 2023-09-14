import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.set_option('display.max_columns', None) # coluna
pd.set_option('display.max_rows', None) # linha
import missingno as msno
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import datetime as dt
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

#Code Starts Here
def Regressao( path : str, filter : str, coluna_filter : str):

    df = pd.read_excel(path, sheet_name='Export')
    df_tratado = df[df['Fat. c/ ADF'] > 0]
    df_roic = df_tratado[['UNB_PDV','Cluster CP','GEO','Segmento','Comercial','PMP c/ ADF','Taxa média','% Fat C/ ADF']]
    df_roic = df_roic.fillna(0)
    df_roic = df_roic[df_roic['Taxa média'] < 0.3]


    #filter
    #filter = 'SUB'
    #coluna_filter = 'Segmento'
    df_roic = df_roic[df_roic[coluna_filter] == filter]



    print('Base lida e tratada')

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['PMP c/ ADF']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )
    df_roic['PMP c/ ADF_norm'] = df_scaled
    #df_roic = df_roic.drop(columns = 'PMP c/ ADF')

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Taxa média']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Taxa média_norm'] = df_scaled




    print('Variaveis normalizadas')


    identificadores = df_roic['GEO']
    X = df_roic[['Taxa média_norm', 'PMP c/ ADF_norm']]
    y = df_roic['% Fat C/ ADF']


    from sklearn.linear_model import LinearRegression
    modelo = LogisticRegression()
    test_model = LogisticRegression()

    X_train, X_test , y_train ,y_test = train_test_split(X,y,random_state =0)
    test_model.fit(X_train,y_train)



    yhat = test_model.predict(X_test)

    mae = mean_absolute_error(y_test, yhat)
    mse = mean_squared_error(y_test, yhat)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f} - Erro médio das previsões')
    print(f'Mean squared error: {mse:.2f} - Erro médio quadratico (da mais peso para erros maiores, e menos pra erros pequenos e numerosos)')
    print(f'Root mean squared error: {rmse:.2f} - Indica o maior erro, o valor mais distante previsto')

    feature_names = X.columns
    model_coefficients = modelo.coef_

    coefficients_df = pd.DataFrame(data = model_coefficients, 
                                index = feature_names, 
                                columns = ['Coefficient value'])
    print(coefficients_df)

    print('Modelo de testes pronto')


    modelo.fit(X, y)

    print('Modelo Oficial pronto')

    limit1 = min(df_roic['PMP c/ ADF_norm'])
    limit2 = max(df_roic['PMP c/ ADF_norm'])
    limit3 = min(df_roic['Taxa média_norm'])
    limit4 = max(df_roic['Taxa média_norm'])

    # Criar uma lista de dicionários com combinações aleatórias
    data = []
    for _ in range(100000):
        data.append({
            'Taxa média_norm': random.uniform(limit3, limit4),
            'PMP c/ ADF_norm': random.uniform(limit1, limit2),
            
        })

    # Criar o DataFrame a partir da lista de dicionários
    df_teste = pd.DataFrame(data)
    yhat= modelo.predict(df_teste)
    df_plot = df_teste
    df_plot['Previsões'] = yhat


    print('Combinações aleatorias geradas')


    # Plotar o gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_plot['PMP c/ ADF_norm'], df_plot['Taxa média_norm'], df_plot['Previsões'], c='b', marker='o')
    ax.set_xlabel('PMP c/ ADF_norm')
    ax.set_ylabel('Taxa média')
    ax.set_zlabel('Previsões')
    ax.set_title('Gráfico 3D com PMP c/ ADF_norm, Taxa media e Previsões')
    plt.show()
    df_top50 = df_plot.nlargest(50, 'Previsões')

    df_top50.to_excel(r'50 maiores fat c adf- log - '+ filter,'.xlsx', index = False)

    print('Base e graficos gerados')



Regressao(r'C:\Users\99829465\Documents\Squad Roic.xlsx','Geo CO','GEO')