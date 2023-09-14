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

#Code Starts Here

def Regressao( df: pd.DataFrame, filter : str, coluna_filter : str, GEO : str, tipo : str):

    #df = pd.read_excel(path, sheet_name='Export')
    df_tratado = df[df['Fat. c/ ADF'] > 0]
    df_roic = df_tratado[['UNB_PDV','Cluster CP','GEO','Segmento','Comercial','PMP c/ ADF','PMP s/ ADF','Taxa média','ADF/TTV','Prazo Isento','Chance de Pagamento','Classe Risco Atual','Perc de atraso','Lista de Restrição','% Fat C/ ADF']]
    df_roic = df_roic.fillna(0)
    df_roic = df_roic[df_roic['Taxa média'] < 0.3]


    #filter
    #filter = 'SUB'
    #coluna_filter = 'Segmento'
    df_roic = df_roic[df_roic[coluna_filter] == filter]
    df_roic['Lista de Restrição'] = df_roic['Lista de Restrição'].map({'S': 1, 'N': 0})


    
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

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['ADF/TTV']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['ADF/TTV_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['PMP s/ ADF']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['PMP s/ ADF_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Prazo Isento']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Prazo Isento_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Chance de Pagamento']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Chance de Pagamento_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Classe Risco Atual']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Classe Risco Atual_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Perc de atraso']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Perc de atraso_norm'] = df_scaled

    nrm_scaler = MinMaxScaler()
    nume = df_roic[['Lista de Restrição']]
    colunas = nume.columns
    df_scaled = nrm_scaler.fit_transform(nume.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns= colunas )

    df_roic['Lista de Restrição_norm'] = df_scaled






    print('Variaveis normalizadas')


    identificadores = df_roic['GEO']
    X = df_roic[['Taxa média_norm', 'PMP c/ ADF_norm','ADF/TTV_norm','PMP s/ ADF_norm','Prazo Isento_norm','Chance de Pagamento_norm','Classe Risco Atual_norm','Perc de atraso_norm','Lista de Restrição_norm']]
    y = df_roic['% Fat C/ ADF']


    from sklearn.linear_model import LinearRegression
    modelo = LinearRegression()
    test_model = LinearRegression()

    X_train, X_test , y_train ,y_test = train_test_split(X,y,random_state =0)
    test_model.fit(X_train,y_train)



    yhat = test_model.predict(X_test)

    mae = mean_absolute_error(y_test, yhat)
    mse = mean_squared_error(y_test, yhat)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f} - Erro médio das previsões')
    print(f'Mean squared error: {mse:.2f} - Erro médio quadratico (da mais peso para erros maiores, e menos pra erros pequenos e numerosos)')
    print(f'Root mean squared error: {rmse:.2f} - Indica o maior erro, o valor mais distante previsto')



    print('Modelo de testes pronto')


    modelo.fit(X, y)
    feature_names = X.columns
    model_coefficients = modelo.coef_

    coefficients_df = pd.DataFrame(data = model_coefficients, 
                                index = feature_names, 
                                columns = ['Coefficient value'])
    print(coefficients_df)
    print('Modelo Oficial pronto')

    limit1 = min(df_roic['PMP c/ ADF_norm'])
    limit2 = max(df_roic['PMP c/ ADF_norm'])
    limit3 = min(df_roic['Taxa média_norm'])
    limit4 = max(df_roic['Taxa média_norm'])
    limit6 = min(df_roic['ADF/TTV_norm'])
    limit7 = max(df_roic['ADF/TTV_norm'])

    # Criar uma lista de dicionários com combinações aleatórias
    data = []
    for _ in range(100000):
        data.append( {

            'Taxa média_norm': random.uniform(limit3, limit4),
            'PMP c/ ADF_norm': random.uniform(limit1, limit2),
            'ADF/TTV_norm': random.uniform(limit6, limit7),
            'PMP s/ ADF_norm' : random.uniform(0, 1),
            'Prazo Isento_norm' :  random.uniform(0, 1),
            'Chance de Pagamento_norm' : random.uniform(0, 1),
            'Classe Risco Atual_norm' : random.uniform(0, 1),
            'Perc de atraso_norm' : random.uniform(0, 1),
            'Lista de Restrição_norm' : random.uniform(0, 1),
            
        })

    # Criar o DataFrame a partir da lista de dicionários
    df_teste = pd.DataFrame(data)
    yhat= modelo.predict(df_teste)
    df_plot = df_teste
    df_plot['Previsões'] = yhat
    df_plot = df_plot[df_plot['Previsões']<1]

    print('Combinações aleatorias geradas')


    # Plotar o gráfico 3D
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_plot['PMP c/ ADF_norm'], df_plot['Taxa média_norm'], df_plot['Previsões'], c='b', marker='o')
    ax.set_xlabel('PMP c/ ADF_norm')
    ax.set_ylabel('Taxa média')
    ax.set_zlabel('Previsões')
    ax.set_title('Gráfico 3D com PMP c/ ADF_norm, Taxa media e Previsões')
    plt.show()'''

    df_top50 = df_plot.nlargest(50, 'Previsões')

    a = min(df_roic['Taxa média'])
    b = max(df_roic['Taxa média'])

    c = min(df_roic['PMP c/ ADF'])
    d = max(df_roic['PMP c/ ADF'])
    e = min(df_roic['ADF/TTV'])
    f = max(df_roic['ADF/TTV'])
    g = min(df_roic['PMP s/ ADF'])
    h = max(df_roic['PMP s/ ADF'])
    m = min(df_roic['Prazo Isento'])
    n = max(df_roic['Prazo Isento'])
    o = min(df_roic['Chance de Pagamento'])
    p = max(df_roic['Chance de Pagamento'])
    q = min(df_roic['Classe Risco Atual'])
    r = max(df_roic['Classe Risco Atual'])
    s = min(df_roic['Perc de atraso'])
    t = max(df_roic['Perc de atraso'])
    u = min(df_roic['Lista de Restrição'])
    v = max(df_roic['Lista de Restrição'])

    def min_max_inverse(normalized_list, original_min, original_max):
        denormalized_list = [(x * (original_max - original_min)) + original_min for x in normalized_list]
        return denormalized_list
    df_top50['Taxa média'] = min_max_inverse(df_top50['Taxa média_norm'], a, b)
    df_top50['PMP c/ ADF'] = min_max_inverse(df_top50['PMP c/ ADF_norm'], c, d)
    df_top50['ADF/TTV'] = min_max_inverse(df_top50['ADF/TTV_norm'], e, f)
    df_top50['PMP s/ ADF'] = min_max_inverse(df_top50['PMP s/ ADF_norm'], g, h)
    df_top50['Prazo Isento'] = min_max_inverse(df_top50['Prazo Isento_norm'], m, n)
    df_top50['Chance de Pagamento'] = min_max_inverse(df_top50['Chance de Pagamento_norm'], o, p)
    df_top50['Classe Risco Atual'] = min_max_inverse(df_top50['Classe Risco Atual_norm'], q, r)
    df_top50['Lista de Restrição'] = min_max_inverse(df_top50['Lista de Restrição_norm'], u, v)
    df_top50['Perc de atraso'] = min_max_inverse(df_top50['Perc de atraso_norm'], s, t)
    




    #df_top50.to_excel(r'50 maiores fat c adf - '+GEO+"-"+filter+'.xlsx', index = False)

    print('Base e graficos gerados')
    saida = []
    #print("TESTE:",df_top50['Taxa média'].mean())
    saida.append(GEO)
    saida.append(filter)
    saida.append(tipo)
    saida.append(df_top50['Taxa média'].mean())
    saida.append(df_top50['PMP c/ ADF'].mean())
    saida.append(df_top50['ADF/TTV'].mean())
    saida.append(df_top50['PMP s/ ADF'].mean())
    saida.append(df_top50['Prazo Isento'].mean())
    saida.append(df_top50['Chance de Pagamento'].mean())
    saida.append(df_top50['Classe Risco Atual'].mean())
    saida.append(df_top50['Perc de atraso'].mean())
    saida.append(df_top50['Lista de Restrição'].mean())
    saida.append(df_top50['Previsões'].mean())
    #print(saida)
    return saida


df = pd.read_excel(r'C:\Users\99829465\Documents\Merge Squad Roic.xlsx', sheet_name='Sheet1')

df = df[df['Fat. c/ ADF'] > 0]

lista_geo = df['GEO'].unique()

ind = 0
results = []
for i in lista_geo:
    df_f = df[df['GEO'] == i]
    lista_seg = df_f['Segmento'].unique()
    for j in lista_seg:
        segseg = df_f[df_f['Segmento']==j]
        lista_tipo = segseg['Tipo de Pessoa'].unique()
        if ind==0:
            print(lista_tipo)
        for k in lista_tipo:

            teste = segseg[segseg['Tipo de Pessoa']==k]
            print("Na GEO:",i,"Seg:",j,"2 tipos tem shape ",segseg.shape)
            print("Seg:",j,"TIPO:",k," ",teste.shape)
            if(teste.shape[0] > 20):
                    
                    x = Regressao(teste,j,'Segmento',i,k)
                    #print("GEO: ", i ,"\n Segmento:",j, "\n Tipo:",k," ok.")
                    #results["Taxa Média Otimizada"]=x[0]
                    #results["PMP c/ ADF Otimizado"]=x[1]
                    #results["% de Faturamento Médio esperado"]=x[2]
                    results.append(x)
                    #results.append("PMP c/ ADF Otimizado":x[1])
                    #results.append("% de Faturamento Médio esperado":x[2])
                    #print(results)
                    if(ind == 0):
                        saida = pd.DataFrame(results, columns=["GEO","Segmento","Tipo de Pessoa","Taxa Média Otimizada", "PMP c/ ADF Otimizado", 'ADF/TTV otimizado','PMP s/ ADF','Prazo Isento','Chance de Pagamento','Classe Risco Atual','Perc de atraso','Lista de Restrição',"% de Faturamento Médio esperado"])

                    else:
                        y = pd.DataFrame(results, columns=["GEO","Segmento","Tipo de Pessoa","Taxa Média Otimizada", "PMP c/ ADF Otimizado", 'ADF/TTV otimizado','PMP s/ ADF','Prazo Isento','Chance de Pagamento','Classe Risco Atual','Perc de atraso','Lista de Restrição',"% de Faturamento Médio esperado"])
                        saida = pd.concat([saida,y], ignore_index=True)

                    ind = ind + 1
                
            else:
                print("GEO: ", i ,"\n Segmento:",j, "\n Tipo:",k," NOK.")



saida.to_excel(r'Base resumo.xlsx', index = False)