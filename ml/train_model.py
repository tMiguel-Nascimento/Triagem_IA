"""
train_model.py
-----------------------------------------------------------------------------
Treina o modelo de produção do site: Naive Bayes Gaussiano (GaussianNB),
em Classificação Multirrótulo, usando o StressLevelDataset.csv.

Este script é a versão "de produção" do que está explorado com mais calma
(gráficos, comparação com Random Forest, curva de aprendizado, validação
cruzada, testes de sanidade...) em `notebooks/triagem_ml_naive_bayes.ipynb`.
Para a análise completa, abra o notebook no Google Colab ou no Jupyter —
aqui fica só o essencial para treinar e salvar o modelo que o site usa.

Se quiser reproduzir a comparação Naive Bayes x Random Forest com os
gráficos salvos em arquivo (sem precisar do Jupyter), rode
`comparar_modelos.py`, nesta mesma pasta.

POR QUE NAIVE BAYES: testamos e o desempenho ficou equivalente ao Random
Forest usado nas versões anteriores do projeto (ansiedade ~80%, depressão
~74%, estresse ~86% de F1-score) — e Naive Bayes é o algoritmo pedido para
esta etapa do trabalho. A desvantagem conhecida dele (tratar cada fator como
independente, "diluindo" o efeito de poucos fatores ruins num mar de fatores
médios) está documentada no notebook, seção 7.

O QUE ESTE SCRIPT FAZ:
  1. Carrega StressLevelDataset.csv (+ feedback_profissionais.csv, se existir).
  2. Separa ENTRADAS (17 fatores do formulário) de SAÍDAS (risco_ansiedade,
     risco_depressao, risco_estresse — multirrótulo: mais de um pode sair
     "Alto" ao mesmo tempo).
  3. ColumnTransformer normaliza os números (StandardScaler) — importante
     para Naive Bayes Gaussiano, que assume cada fator ~Normal por classe.
  4. GaussianNB dentro de MultiOutputClassifier (um classificador por rótulo).
  5. Mede Recall/F1 num conjunto de teste (20% dos dados, nunca visto no treino).
  6. Calcula a IMPORTÂNCIA de cada fator (permutation importance) para cada
     rótulo, e guarda os 3 mais importantes de cada — é o que vira "Fatores
     mais determinantes" no dashboard do site.
  7. Salva tudo em modelo_triagem.joblib.
-----------------------------------------------------------------------------
"""

import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

CSV_PATH = "StressLevelDataset.csv"
FEEDBACK_CSV_PATH = "feedback_profissionais.csv"  # gerado por retrain_from_feedback.py (opcional)
MODEL_OUT = "modelo_triagem.joblib"
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)

FEATURES = [
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying", "mental_health_history",
]
LABEL_COLS = ["risco_ansiedade", "risco_depressao", "risco_estresse"]


def faixa_ansiedade(score):  # anxiety_level: 0-21 (estilo GAD-7)
    if score < 8:
        return "Baixo"
    elif score < 15:
        return "Medio"
    return "Alto"


def faixa_depressao(score):  # depression: 0-27 (estilo PHQ-9)
    if score < 10:
        return "Baixo"
    elif score < 20:
        return "Medio"
    return "Alto"


MAPA_ESTRESSE = {0: "Baixo", 1: "Medio", 2: "Alto"}  # stress_level já vem 0/1/2

df["risco_ansiedade"] = df["anxiety_level"].apply(faixa_ansiedade)
df["risco_depressao"] = df["depression"].apply(faixa_depressao)
df["risco_estresse"] = df["stress_level"].map(MAPA_ESTRESSE)

df = df[FEATURES + LABEL_COLS]

# Incorpora feedback real de profissionais, se já existir (ver
# retrain_from_feedback.py) — assim o sistema aprende com o uso real, sempre
# com um humano revisando antes (nunca com as próprias previsões ainda não
# confirmadas, pra não criar um ciclo vicioso de erro).
if os.path.exists(FEEDBACK_CSV_PATH):
    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)[FEATURES + LABEL_COLS]
    df = pd.concat([df, feedback_df], ignore_index=True)
    print(f"[info] {len(feedback_df)} triagens reais validadas por profissionais foram incorporadas ao treino.")

X = df[FEATURES].copy()

encoders = {}
y = pd.DataFrame()
for col in LABEL_COLS:
    le = LabelEncoder()
    y[col] = le.fit_transform(df[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y["risco_estresse"]
)

# Todas as entradas já são numéricas (escalas 0-5, 0-3, 1-3 ou 0/1): só
# precisa normalizar. GaussianNB assume que cada fator segue uma
# distribuição normal dentro de cada classe — por isso o StandardScaler
# aqui não é só "boa prática", é parte do que faz o algoritmo funcionar bem.
preprocessador = ColumnTransformer(transformers=[("num", StandardScaler(), FEATURES)])

pipeline = Pipeline(steps=[
    ("preprocessador", preprocessador),
    ("modelo", MultiOutputClassifier(GaussianNB())),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=" * 70)
print("RESULTADOS DA AVALIAÇÃO — Naive Bayes Gaussiano (20% dos dados em teste)")
print("=" * 70)
for i, col in enumerate(LABEL_COLS):
    print(f"\n--- {col} ---")
    print(classification_report(
        y_test[col], y_pred[:, i], target_names=encoders[col].classes_, zero_division=0
    ))

# -----------------------------------------------------------------------
# Importância dos fatores por rótulo (permutation importance), calculada
# UMA VEZ aqui no treino — não a cada requisição da API, que seria lento.
# Treinamos um pipeline simples (sem multirrótulo) por label só para medir
# a importância isolada de cada fator naquele rótulo específico.
# -----------------------------------------------------------------------
importancias_por_label = {}
for col in LABEL_COLS:
    pipeline_simples = Pipeline([("prep", StandardScaler()), ("nb", GaussianNB())])
    pipeline_simples.fit(X_train, y_train[col])
    resultado_perm = permutation_importance(
        pipeline_simples, X_test, y_test[col],
        n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro"
    )
    ordenado = pd.Series(resultado_perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    importancias_por_label[col] = ordenado.index.tolist()

joblib.dump({
    "pipeline": pipeline,
    "encoders": encoders,
    "features": FEATURES,
    "label_cols": LABEL_COLS,
    "importancias_por_label": importancias_por_label,
    "algoritmo": "GaussianNB (Naive Bayes)",
}, MODEL_OUT)
print(f"\nModelo salvo em: {MODEL_OUT}")
print("Fatores mais determinantes para risco_ansiedade:", importancias_por_label["risco_ansiedade"][:3])
