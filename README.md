<<<<<<< HEAD
# 🍺 IA-Beer-Regression: Previsão do Consumo de Cerveja


### 🎯 Objetivo do Projeto
Desenvolver um modelo de **Regressão Linear Múltipla** para prever o consumo diário de cerveja (em litros) com base em fatores climáticos e sazonais:
* Temperatura Média
* Precipitação
* Final de Semana (variável binária)

---

## ⚙️ Metodologia e Pipeline

O projeto seguiu uma metodologia de modelagem estatística clássica, utilizando as bibliotecas `statsmodels` e `scikit-learn` em Python:

1.  **Análise Exploratória de Dados (EDA):** Verificação da relação entre as variáveis (gráficos de dispersão e boxplots).
2.  **Diagnóstico de Multicolinearidade:** Análise do VIF (Fator de Inflação da Variância), que confirmou a ausência de multicolinearidade significativa entre as preditoras, pois os valores VIF ficaram abaixo de 1,55.
    * **VIFs (Variáveis Selecionadas):** `temperatura_media` (1,544), `precipitacao` (1,175) e `final_de_semana` (1,374).
3.  **Treinamento do Modelo:** Regressão OLS (Mínimos Quadrados Ordinários).
4.  **Diagnóstico de Resíduos:** Verificação das premissas de normalidade (QQ-Plot) e homocedasticidade (Resíduos vs. Valores Ajustados).

---

## 📈 Resultados da Regressão (Conjunto de Teste)

O modelo final foi treinado em 75% dos dados e avaliado nos 25% restantes, demonstrando boa capacidade preditiva.

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **$R^{2}$ (R-squared)** | $0,6743$ | O modelo explica **67,43%** da variância no consumo de cerveja. |
| **RMSE** (Root Mean Squared Error) | $2,708$ | O erro médio das previsões é de $\approx 2,7$ litros. |

### Coeficientes do Modelo (Modelo OLS Completo)

Estes coeficientes determinam a contribuição de cada variável para o consumo:

| Variável | Coeficiente | Significado |
| :--- | :--- | :--- |
| **Temperatura Média** | $0,8397$ | Para cada aumento de $1^\circ C$ na temperatura média, o consumo aumenta $\approx 0,84$ litros. |
| **Final de Semana** | $5,2279$ | O consumo é $\approx 5,23$ litros maior nos finais de semana (mantendo outras variáveis constantes). |
| **Precipitação** | $-0,0742$ | A precipitação tem uma pequena relação inversa com o consumo. |

---

## 🛠️ Como Clonar e Rodar o Projeto

### Pré-requisitos
* Python 3.x
* Dataset `beer_consuption.csv` (incluso neste repositório)

### Instalação das Dependências
Instale todas as bibliotecas necessárias usando o arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
=======
# IA-Beer-Regression
Regressão Linear Múltipla para prever o consumo de cerveja com base em temperatura e fatores sazonais.
>>>>>>> 1f3380e72f8494932a8274f04cc0f60eebde97c5
