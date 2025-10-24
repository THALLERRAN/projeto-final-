import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Importações OBRIGATÓRIAS para o Pickle ---
# O pickle do sklearn precisa que as definições das classes
# usadas no pipeline estejam importadas no script.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# --- Fim das importações obrigatórias ---


# Lista de colunas que o modelo foi treinado para esperar.
# (Baseado no pipeline que construímos na Célula 2 anterior)
FEATURES_ESPERADAS = [
    'idade', 
    'tempo_emprego', 
    'qt_pessoas_residencia', # Numéricas
    'renda',                                  # Outlier (log)
    'sexo', 
    'posse_de_veiculo', 
    'posse_de_imovel', # Categóricas
    'tipo_renda', 
    'educacao', 
    'estado_civil', 
    'tipo_residencia'
]

# --- 1. Função para Carregar o Modelo ---
# @st.cache_resource garante que o modelo seja carregado 
# apenas uma vez, otimizando a performance.
@st.cache_resource
def carregar_modelo(model_path='model_final.pkl'):
    """
    Carrega o pipeline completo (pré-processamento + PCA + modelo) 
    salvo em um arquivo .pkl.
    """
    # Verifica se o arquivo existe
    if not os.path.exists(model_path):
        st.error(f"Erro: Arquivo do modelo '{model_path}' não foi encontrado.")
        st.error("Por favor, certifique-se que o 'model_final.pkl' "
                 "está no mesmo diretório que o app.py.")
        return None
        
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# --- 2. Interface Principal do Streamlit ---
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("Aplicação de Escoragem de Risco de Crédito")

# Carregar o modelo
model = carregar_modelo()

if model:
    st.success("Modelo (pipeline completo) carregado com sucesso!")
    
    # --- 3. Carregador de CSV ---
    uploaded_file = st.file_uploader(
        "Selecione o arquivo CSV para escoragem", 
        type="csv"
    )
    
    if uploaded_file is not None:
        try:
            # --- 4. Ler o CSV ---
            df_para_escorar = pd.read_csv(uploaded_file)
            st.write("--- 1. Amostra dos Dados Carregados ---")
            st.dataframe(df_para_escorar.head())

            # --- 5. Verificar Colunas ---
            missing_cols = [col for col in FEATURES_ESPERADAS 
                            if col not in df_para_escorar.columns]
            
            if missing_cols:
                st.error(f"Erro: O arquivo CSV não contém as colunas necessárias "
                         f"para o modelo: {missing_cols}")
            else:
                # Selecionar apenas as colunas que o modelo usa
                X_score = df_para_escorar[FEATURES_ESPERADAS]
                
                st.write("--- 2. Processando e Escorando... ---")
                
                # --- 6. Utilizar o Pipeline para Escorar ---
                # O pipeline aplica TODO o pré-processamento (nulos, dummies, pca)
                # e depois gera as predições
                
                # .predict_proba() retorna [prob_classe_0, prob_classe_1]
                # Queremos a probabilidade da classe 1 (mau)
                scores = model.predict_proba(X_score)[:, 1]
                
                # Gerar a predição (label 0 ou 1)
                labels = model.predict(X_score)
                
                # --- 7. Exibir Resultados ---
                st.write("--- 3. Resultados da Escoragem ---")
                df_resultados = df_para_escorar.copy()
                df_resultados['score_mau'] = scores # Score (probabilidade)
                df_resultados['predicao_mau'] = labels # Classe (0=Bom, 1=Mau)
                
                st.dataframe(df_resultados)
                
                # --- 8. Botão de Download ---
                @st.cache_data # Cache para conversão do CSV
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_results = convert_df_to_csv(df_resultados)
                
                st.download_button(
                    label="Baixar resultados em CSV",
                    data=csv_results,
                    file_name="resultados_scoring.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")