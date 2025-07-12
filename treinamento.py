import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import re

print("Iniciando o processo de treinamento...")

# Baixar recursos do NLTK (necessário apenas na primeira vez)
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando stopwords do NLTK...")
    nltk.download('stopwords')
    nltk.download('punkt')

# --- Funções de Pré-processamento (do seu Colab) ---
def limpar_e_tokenizar_texto(texto):
    # Converte para minúsculas
    texto = texto.lower()
    # Remove pontuações e caracteres especiais
    texto = re.sub(r'[^\w\s]', '', texto)
    # Tokeniza o texto
    tokens = word_tokenize(texto)
    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stop_words]
    return set(tokens_filtrados) # Retorna um conjunto para a Similaridade de Jaccard

def calcular_similaridade_jaccard(set1, set2):
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    return intersecao / uniao if uniao != 0 else 0

# --- Carregamento e Preparação dos Dados ---
df = pd.read_csv('base_decision.csv')

# Limpa e tokeniza as habilidades
df['habilidades_candidato_set'] = df['habilidades_candidato'].apply(limpar_e_tokenizar_texto)
df['habilidades_vaga_set'] = df['habilidades_vaga'].apply(limpar_e_tokenizar_texto)

# Calcula a similaridade de Jaccard para cada linha
df['similaridade_jaccard'] = df.apply(
    lambda row: calcular_similaridade_jaccard(row['habilidades_candidato_set'], row['habilidades_vaga_set']),
    axis=1
)

# --- Treinamento do Modelo ---
X = df[['similaridade_jaccard']]
y = df['contratado']

# Dividindo os dados (não é estritamente necessário para o deploy, mas mantém a consistência)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criando e treinando o modelo de Regressão Logística
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

# --- Salvando o Modelo ---
joblib.dump(modelo, 'modelo_decision_v1.pkl')

print("Modelo salvo como 'modelo_decision_v1.pkl'. Processo concluído.")