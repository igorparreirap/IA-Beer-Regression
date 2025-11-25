# -*- coding: utf-8 -*-
"""
Análise de Regressão Múltipla para o Consumo de Cerveja.

Este script realiza uma análise completa do consumo de cerveja com base em 
variáveis climáticas e sazonais, seguindo as etapas do trabalho da disciplina
de Inteligência Artificial.

Produtos Finais Gerados:
- describe.csv: Estatísticas descritivas das variáveis.
- vif.csv: Fatores de Inflação da Variância (VIF).
- ols_summary.txt: Sumário estatístico completo do modelo OLS.
- coeficientes.csv: Coeficientes da regressão.
- metrics.csv: Métricas de avaliação do modelo (RMSE e R²).
- Gráficos em formato .png:
  - scatter_temp_consumo.png
  - scatter_precip_consumo.png
  - boxplot_fds_consumo.png
  - residuals_vs_fitted.png
  - qq_plot.png
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Iniciando a análise de regressão do consumo de cerveja...")

# --- 1. Carregamento e Preparação dos Dados ---
try:
    # Carrega o dataset
    df = pd.read_csv('beer_consuption.csv', decimal=',')
    
    # Renomeia as colunas
    df.rename(columns={
        'Data': 'data',
        'Temperatura Media (C)': 'temperatura_media',
        'Temperatura Minima (C)': 'temperatura_minima',
        'Temperatura Maxima (C)': 'temperatura_maxima',
        'Precipitacao (mm)': 'precipitacao',
        'Final de Semana': 'final_de_semana',
        'Consumo de cerveja (litros)': 'consumo_cerveja'
    }, inplace=True)
    
    # Converte colunas para tipo numérico, tratando erros com 'coerce'
    numeric_cols = ['temperatura_media', 'temperatura_minima', 'temperatura_maxima', 
                    'precipitacao', 'consumo_cerveja']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove linhas com valores ausentes (NaN)
    df.dropna(inplace=True)
    
    # Converte a coluna de data para o formato datetime
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

    print("Dados carregados e preparados com sucesso.")

except FileNotFoundError:
    print("Erro: O arquivo 'beer_consuption.csv' não foi encontrado.")
    exit()


# --- 2. Análise Exploratória e Geração de Gráficos ---

# Estatísticas descritivas
desc_stats = df[['consumo_cerveja', 'temperatura_media', 'precipitacao', 'final_de_semana']].describe()
desc_stats.to_csv('describe.csv')
print("Arquivo 'describe.csv' com estatísticas descritivas foi salvo.")

# Configurações de estilo para os gráficos
sns.set_style("whitegrid")

# Gráfico de Dispersão: Temperatura Média vs. Consumo
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='temperatura_media', y='consumo_cerveja', line_kws={"color": "red"})
plt.title('Consumo de Cerveja vs. Temperatura Média')
plt.xlabel('Temperatura Média (°C)')
plt.ylabel('Consumo de Cerveja (litros)')
plt.tight_layout()
plt.savefig('scatter_temp_consumo.png')
plt.close()
print("Gráfico 'scatter_temp_consumo.png' foi salvo.")

# Gráfico de Dispersão: Precipitação vs. Consumo
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='precipitacao', y='consumo_cerveja')
plt.title('Consumo de Cerveja vs. Precipitação')
plt.xlabel('Precipitação (mm)')
plt.ylabel('Consumo de Cerveja (litros)')
plt.tight_layout()
plt.savefig('scatter_precip_consumo.png')
plt.close()
print("Gráfico 'scatter_precip_consumo.png' foi salvo.")


# Boxplot: Final de Semana vs. Consumo
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='final_de_semana', y='consumo_cerveja')
plt.title('Consumo de Cerveja por Tipo de Dia')
plt.xlabel('Tipo de Dia')
plt.ylabel('Consumo de Cerveja (litros)')
plt.xticks(ticks=[0, 1], labels=['Dia Útil', 'Final de Semana'])
plt.tight_layout()
plt.savefig('boxplot_fds_consumo.png')
plt.close()
print("Gráfico 'boxplot_fds_consumo.png' foi salvo.")


# --- 3. Seleção de Variáveis ---
X = df[['temperatura_media', 'precipitacao', 'final_de_semana']]
y = df['consumo_cerveja']

# Adiciona a constante (intercepto) para o modelo statsmodels
X_sm = sm.add_constant(X)


# --- 4. Diagnóstico de Multicolinearidade (VIF) ---
vif_data = pd.DataFrame()
vif_data["Variável"] = X.columns
# Cálculo do Fator de Inflação da Variância (VIF)
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data.to_csv('vif.csv', index=False)
print("Arquivo 'vif.csv' com o diagnóstico VIF foi salvo.")


# --- 5. Divisão em Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")


# --- 6. Treinamento do Modelo (Scikit-learn) ---
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
print("Modelo de regressão linear treinado com scikit-learn.")


# --- 7. Avaliação do Modelo ---
y_pred = model_sklearn.predict(X_test)

# Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Salvando as métricas em um arquivo
metrics = pd.DataFrame({'Métrica': ['RMSE', 'R²'], 'Valor': [rmse, r2]})
metrics.to_csv('metrics.csv', index=False)
print("Arquivo 'metrics.csv' com as métricas de avaliação foi salvo.")


# --- 8 & 9. Relatório Estatístico Detalhado (Statsmodels) e Diagnóstico de Resíduos ---
# Ajuste do modelo OLS para relatório final
model_ols = sm.OLS(y, X_sm).fit()

# Salva o sumário completo em um arquivo de texto
with open('ols_summary.txt', 'w') as f:
    f.write(str(model_ols.summary()))
print("Arquivo 'ols_summary.txt' com o sumário OLS completo foi salvo.")

# Extrai e salva os coeficientes
coeficientes = model_ols.params.reset_index()
coeficientes.columns = ['Variável', 'Coeficiente']
coeficientes.to_csv('coeficientes.csv', index=False)
print("Arquivo 'coeficientes.csv' com os coeficientes do modelo foi salvo.")

# Diagnóstico dos resíduos
fitted_values = model_ols.fittedvalues
residuals = model_ols.resid

# Gráfico de Resíduos vs. Valores Ajustados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Resíduos vs. Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.tight_layout()
plt.savefig('residuals_vs_fitted.png')
plt.close()
print("Gráfico 'residuals_vs_fitted.png' foi salvo.")

# QQ-plot para normalidade dos resíduos
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='s')
plt.title('QQ-Plot dos Resíduos')
plt.tight_layout()
plt.savefig('qq_plot.png')
plt.close()
print("Gráfico 'qq_plot.png' foi salvo.")

print("\nAnálise concluída com sucesso! Todos os arquivos e gráficos foram gerados.")
