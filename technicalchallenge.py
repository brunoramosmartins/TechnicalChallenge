#!/usr/bin/env python
# coding: utf-8

# # Desafio FieldPRO
# 
# Neste notebook, abordaremos o desafio técnico proposto pela **FieldPro**, onde exploraremos e aplicaremos técnicas de análise de dados e _machine learning_ para resolver um problema específico. Ao longo deste notebook, iremos importar as bibliotecas necessárias, explorar os dados, criar e avaliar modelos preditivos. Vamos começar!

# ## Desafio
# 
# O objetivo deste desafio é construir um modelo de calibração de um sensor de chuva baseado em impactos mecânicos.
# 
# O Sistema de medição de chuva funciona por meio de uma placa eletrônica com um piezoelétrico, um acumulador de carga e um sensor de temperatura. Os dados são transmitidos de hora em hora.
# 
# O impacto das gotas de chuva gera vibrações no piezoelétrico, que induzem uma corrente elétrica. A corrente elétrica não é medida diretamente, mas é acumulada ao longo do tempo e gera uma queda na carga do acumulador.
# 
# A carga do acumulador é medida de hora em hora e transmitida com o nome de `piezo_charge`. A temperatura da placa é transmitida sob o nome `piezo_temperature` e pode ser importante na calibração.
# 
# Um evento de reset na placa pode afetar o comportamento do acumulador de carga, e o número total de resets da placa desde que foi ligada pela primeira vez é transmitido com nome `num_of_resets`.
# 
# As medidas realizadas pelo sensor estão no arquivo **Sensor_FieldPRO.csv**, para comparação, foram utilizadas medidas de uma estação metereológica próxima, que estão no arquivo **Estacao_Convencional.csv**.
# 
# Outras medidas do sensor incluem a carga medida no acumulador, a temperatura da placa, o número de resets da placa e as condições atmosféricas do ambiente.
# 
# **Bônus**: Realizar o deploy do modelo em uma plataforma de cloud.

# ## Entendendo o problema
# 
# Com o objetivo de compreender melhor o problema e obter o máximo proveito do conjunto de dados disponível, iniciei uma pesquisa para entender o funcionamento de um sensor de chuva baseado em impactos mecânicos. Além disso, busquei explorar as possíveis relações entre a temperatura do ar, a umidade do ar e a pressão atmosférica, a fim de incorporar mais informações no treinamento do modelo e torná-lo mais robusto.
# 
# A seguir, apresentam-se os títulos associados aos links consultados para esse propósito:
# 
# - [Dew point](https://en.wikipedia.org/wiki/Dew_point)
# - [Estudo e desenvolvimento de um sensor de chuva piezoelétrico para automóveis](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://repositorio.ipl.pt/bitstream/10400.21/2544/1/Disserta%C3%A7%C3%A3o.pdf)
# - [Relações entre temperatura, umidade relativa do ar e pressão atmosférica em área urbana](https://periodicos.ufmg.br/index.php/geografias/article/view/13313)
# - [How do Rain Sensors Work](https://wiki.dfrobot.com/How_do_Rain_Sensors_Work)

# ## Bibliotecas

# In[34]:


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# ## Conjunto de dados
# 
# Os dados do sensor estão armazenados no arquivo `Sensor_FieldPRO.csv`, enquanto para fins de comparação, foram utilizadas também medições de uma estação meteorológica próxima, que estão contidas no arquivo `Estacao_Convencional.csv`.

# In[2]:


df_sensor = pd.read_csv("dados/Sensor_FieldPRO.csv")
df_estacao = pd.read_csv("dados/Estacao_Convencional.csv")


# ## Verificação dos Tipos de Dados nos DataFrames
# 
# Antes de iniciar qualquer análise ou modelagem, é importante conhecer os tipos de dados presentes nos DataFrames. Isso nos permitirá entender a natureza das informações que temos disponíveis e, se necessário, realizar conversões ou tratamentos específicos para preparar os dados para o modelo.
# 
# Além de verificar os tipos de dados, também será útil visualizar as primeiras linhas dos DataFrames. Essa visualização inicial nos dará uma ideia geral do formato e conteúdo dos dados, permitindo identificar padrões ou possíveis problemas nas informações.

# In[3]:


df_sensor.head()


# In[4]:


df_sensor.dtypes


# In[5]:


df_estacao.head()


# In[6]:


df_estacao.dtypes


# Como a coluna `Datetime – utc` está no formato universal de hora (por exemplo: '2020-09-30T23:00:00Z'), faremos a conversão para o horário de Brasília. Em seguida, criaremos duas novas colunas: uma contendo apenas a data e outra com o horário no fuso horário de Brasília.

# In[7]:


df_sensor["data-hora(brasilia)"] = pd.to_datetime(df_sensor['Datetime – utc'], format='mixed').dt.tz_convert("America/Sao_Paulo")
df_sensor["data"] = df_sensor["data-hora(brasilia)"].dt.strftime("%Y-%m-%d")
df_sensor["hora"] = df_sensor["data-hora(brasilia)"].dt.strftime("%H:%M:%S")


# Observando o conjunto de dados, podemos perceber que a variável `piezo_charge` diminui ao longo do tempo. Portanto, criaremos uma nova coluna chamada `timeOn` que representa o tempo ligado a partir do reset.

# In[8]:


df_sensor['timeOn'] = 0

# primeira data-hora do reset
valores_num_of_resets = df_sensor['num_of_resets'].unique()
primeira_ocorrencia = {}
for valor in valores_num_of_resets:
    mask = df_sensor['num_of_resets'] == valor
    primeira_data = df_sensor.loc[mask, 'data-hora(brasilia)'].min()
    primeira_ocorrencia[valor] = primeira_data

for valor in valores_num_of_resets:
    mask = df_sensor['num_of_resets'] == valor
    primeira_data = primeira_ocorrencia[valor]
    df_sensor.loc[mask, 'timeOn'] = (df_sensor.loc[mask, 'data-hora(brasilia)'] - primeira_data).dt.total_seconds() / 3600


# ## Merge
# 
# Para o treinamento do modelo, faremos a união dos DataFrames utilizando a data e hora como chave para a operação de merge. Essa abordagem garantirá que os dados sejam combinados de maneira coesa e organizada, preparando-os adequadamente para o processo de treinamento. Além disso, nesta etapa, realizaremos a remoção de dados nulos e a criação de novas features, com base em estudos das referências iniciais, visando enriquecer e aprimorar a qualidade dos dados para o desenvolvimento do modelo.

# In[9]:


merge_df = df_sensor.merge(df_estacao, how='left', left_on=['data', 'hora'], right_on=['data', 'Hora (Brasília)'])	


# In[10]:


df = merge_df[['data', 'hora', 'air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature','timeOn', 'chuva']].copy()


# In[11]:


df.dropna(subset=['air_humidity_100', 'air_temperature_100','chuva'], inplace=True)


# ### Ponto de orvalho
# 
# O ponto de orvalho é uma temperatura crucial que pode auxiliar no modelo de análise. Ele representa a temperatura na qual o ar deve esfriar para que o vapor de água presente nele se condense e forme orvalho. Essa feature pode ser utilizada como uma variável relevante para enriquecer a análise do modelo, permitindo compreender melhor as condições ambientais e seus efeitos sobre a umidade do ar. Além disso, ao incorporar o ponto de orvalho como uma feature, o modelo pode obter insights mais precisos sobre a saturação de vapor de água no ar e sua relação com outros parâmetros meteorológicos, tornando-o mais robusto e confiável.

# In[12]:


# Nessa parte, estamos calculando a pressão parcial do vapor d'água no ar usando a fórmula empírica de August-Roche-Magnus. 
# Ela requer a umidade relativa do ar (como uma fração entre 0 e 1) e a temperatura do ar (em Celsius).

df['pParcial'] = 243.04 * (np.log(df['air_humidity_100'] / 100) + (17.625 * df['air_temperature_100']) / (243.04 + df['air_temperature_100']))

# Nessa parte, estamos calculando a pressão de vapor saturado do ar usando a mesma fórmula empírica de August-Roche-Magnus. 
# Novamente, ela requer a umidade relativa do ar (como uma fração entre 0 e 1) e a temperatura do ar (em Celsius).

df['pVapor'] = 17.625 - (np.log(df['air_humidity_100']) + (17.625 * df['air_temperature_100']) / (243.04 + df['air_temperature_100']))

# Aqui, dividimos a pressão parcial do vapor d'água pela pressão de vapor saturado do ar para calcular o ponto de orvalho

df['ponto_de_orvalho'] = df['pParcial'] / df['pVapor'] - 273.15


# ## Modelagem e Avaliação de Modelos de Machine Learning
# Nesta etapa, realizaremos a modelagem dos dados após o tratamento e preparação dos mesmos. Vamos explorar diferentes modelos de machine learning para encontrar aquele que melhor se ajusta ao nosso conjunto de dados.
# 
# É importante ressaltar que nosso conjunto de dados é relativamente pequeno, o que requer atenção especial na escolha das métricas de avaliação. Dessa forma, daremos prioridade a métricas específicas de regressão, adequadas para avaliar o desempenho dos modelos que visam prever um valor numérico contínuo, que é a quantidade de chuva em milímetros.

# As principais métricas que iremos considerar para avaliar os modelos de regressão são: 
# 
# - Mean Absolute Error (MAE): Mede o erro médio absoluto entre as previsões do modelo e os valores reais. Essa métrica é menos sensível a outliers e pode ser mais estável em conjuntos de dados pequenos.
# 
# - Mean Squared Error (MSE): Mede a média dos quadrados das diferenças entre as previsões do modelo e os valores reais. É mais sensível a erros maiores devido à sua natureza quadrática.
# 
# - R² Score (Coefficient of Determination): Mede a proporção da variabilidade dos dados que é explicada pelo modelo. Um valor mais próximo de 1 indica um modelo que se ajusta bem aos dados.

# In[15]:


X = df.drop(columns=['data', 'hora', 'chuva'])
y = df['chuva']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Regressão linear

# In[24]:


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculando as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibindo os resultados
print("Métricas de Regressão:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score (Coefficient of Determination):", r2)


# ### Decision tree

# In[25]:


tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)

print("Métricas de Regressão:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score (Coefficient of Determination):", r2_score)


# ### Polynomial Features

# In[32]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

grau = 2
poly = PolynomialFeatures(grau)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

y_pred = linear_model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)  # Renamed the variable to r_squared

print("Métricas de Regressão:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score (Coefficient of Determination):", r_squared)  # Renamed the variable here as well


# ### modelo SVR

# In[22]:


svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Métricas de Regressão:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score (Coefficient of Determination):", r2)


# ### Random Forest Regressor

# In[35]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score (Coefficient of Determination):", r2)


# ### Conclusão.

# Após uma análise detalhada das métricas de desempenho dos modelos de regressão utilizados, chegou-se à conclusão de que o modelo SVR se destacou, demonstrando um desempenho superior em relação aos outros modelos testados. Isso indica uma maior capacidade de predição e uma melhor adaptação aos dados de treinamento.

# No entanto, ao observar o R² Score (Coefficient of Determination), nota-se que o valor foi próximo de zero e negativo. Esse resultado sugere que o modelo ainda não consegue explicar a variação total dos dados e pode indicar problemas na predição, como subajustamento aos dados de treinamento. Contudo, é fundamental ressaltar que esse comportamento pode ser atribuído ao tamanho relativamente pequeno do conjunto de dados utilizado no projeto.

# À medida que o conjunto de dados é enriquecido com mais informações, o modelo SVR tem o potencial de alcançar um desempenho mais satisfatório, permitindo que ele aprenda padrões mais complexos e melhore sua capacidade de generalização. Portanto, futuros esforços de aprimoramento do modelo devem considerar a expansão do conjunto de dados, a fim de obter resultados mais representativos e precisos.

# Em resumo, os resultados obtidos até o momento fornecem uma base sólida para prosseguir com o aperfeiçoamento do modelo SVR. O próximo passo consiste em buscar a otimização dos hiperparâmetros C e epsilon, além de continuar a coletar mais dados relevantes para melhorar a capacidade de predição do modelo e, assim, torná-lo mais aplicável a cenários complexos. Com essas melhorias, espera-se que o modelo SVR alcance seu máximo potencial e possa ser empregado de forma mais robusta em aplicações práticas.

# ### Otimizando o modelo

# In[41]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100, 1000, 10000],
    'epsilon': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]   
    }

svr_model = SVR(kernel='rbf')

grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores parâmetros:", best_params)

best_svr_model = grid_search.best_estimator_

y_pred_best = best_svr_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("Métricas de Regressão (Melhor Modelo):")
print("Mean Absolute Error (MAE):", mae_best)
print("Mean Squared Error (MSE):", mse_best)
print("R² Score (Coefficient of Determination):", r2_best)


# ## **Bônus**: Realizar o deploy do modelo em uma plataforma de cloud.

# In[42]:


joblib.dump(svr_model, 'svr_model.pkl')


# In[ ]:




