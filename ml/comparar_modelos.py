"""
comparar_modelos.py
-----------------------------------------------------------------------------
Reproduz, como script (sem precisar abrir o Jupyter/Colab), a análise
comparativa entre Naive Bayes (o modelo usado no site) e Random Forest
(usado nas versões anteriores do projeto) que está em
notebooks/triagem_ml_naive_bayes.ipynb.

Gera e SALVA em ml/graficos/ (não precisa de tela gráfica, roda liso num
servidor): matriz de confusão dos dois modelos, comparação de F1-score,
curva de aprendizado e importância das variáveis — tudo pronto para colar
no relatório/slides do grupo.

Não é usado pelo site; é só uma ferramenta de apoio ao relatório. Para a
explicação linha a linha de cada gráfico (o porquê, não só o como), veja o
notebook.

COMO USAR:
    cd ml
    python3 comparar_modelos.py
-----------------------------------------------------------------------------
"""

import os
import matplotlib
matplotlib.use("Agg")  # não precisa de tela gráfica — só salva os arquivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
CSV_PATH = "StressLevelDataset.csv"
OUT_DIR = "graficos"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying", "mental_health_history",
]
LABEL_COLS = ["risco_ansiedade", "risco_depressao", "risco_estresse"]

print("Carregando dados e preparando rótulos...")
df = pd.read_csv(CSV_PATH)


def faixa_ansiedade(s):
    return "Baixo" if s < 8 else ("Medio" if s < 15 else "Alto")


def faixa_depressao(s):
    return "Baixo" if s < 10 else ("Medio" if s < 20 else "Alto")


df["risco_ansiedade"] = df["anxiety_level"].apply(faixa_ansiedade)
df["risco_depressao"] = df["depression"].apply(faixa_depressao)
df["risco_estresse"] = df["stress_level"].map({0: "Baixo", 1: "Medio", 2: "Alto"})

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

print("Treinando Naive Bayes e Random Forest...")
pipeline_nb = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
    ("modelo", MultiOutputClassifier(GaussianNB())),
])
pipeline_rf = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
    ("modelo", MultiOutputClassifier(RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE))),
])
pipeline_nb.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)

print("\n" + "=" * 70)
for nome, y_pred in [("NAIVE BAYES (usado no site)", y_pred_nb), ("RANDOM FOREST (comparação)", y_pred_rf)]:
    print(nome)
    print("=" * 70)
    for i, col in enumerate(LABEL_COLS):
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred[:, i], target_names=encoders[col].classes_, zero_division=0))

# ---- Gráfico 1: Matrizes de confusão ----
print("Gerando gráfico: matrizes de confusão...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for linha, (nome, y_pred) in enumerate([("Naive Bayes", y_pred_nb), ("Random Forest", y_pred_rf)]):
    for i, col in enumerate(LABEL_COLS):
        cm = confusion_matrix(y_test[col], y_pred[:, i])
        ax = axes[linha, i]
        ax.imshow(cm, cmap="Blues")
        labels = encoders[col].classes_
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for a in range(len(labels)):
            for b in range(len(labels)):
                ax.text(b, a, cm[a, b], ha="center", va="center")
        ax.set_title(f"{nome} — {col}")
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/matrizes_confusao.png", dpi=120)
plt.close(fig)

# ---- Gráfico 2: Comparação F1 ----
print("Gerando gráfico: comparação de F1-score...")
f1_nb = [f1_score(y_test[c], y_pred_nb[:, i], average="macro") for i, c in enumerate(LABEL_COLS)]
f1_rf = [f1_score(y_test[c], y_pred_rf[:, i], average="macro") for i, c in enumerate(LABEL_COLS)]
x = np.arange(len(LABEL_COLS))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width / 2, f1_nb, width, label="Naive Bayes")
ax.bar(x + width / 2, f1_rf, width, label="Random Forest")
ax.set_xticks(x); ax.set_xticklabels(LABEL_COLS)
ax.set_ylabel("F1-score (macro)"); ax.set_ylim(0, 1); ax.legend()
ax.set_title("Naive Bayes x Random Forest")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/comparacao_f1.png", dpi=120)
plt.close(fig)

# ---- Gráfico 3: Curva de aprendizado (Naive Bayes) ----
print("Gerando gráfico: curva de aprendizado...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, LABEL_COLS):
    modelo_simples = Pipeline([("prep", StandardScaler()), ("nb", GaussianNB())])
    train_sizes, train_scores, val_scores = learning_curve(
        modelo_simples, X, y[col], cv=5, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE,
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validação")
    ax.set_title(col); ax.set_xlabel("Nº de exemplos de treino"); ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1); ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/curva_aprendizado.png", dpi=120)
plt.close(fig)

# ---- Validação cruzada ----
print("\nVALIDAÇÃO CRUZADA (5-fold) — Naive Bayes")
for col in LABEL_COLS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    modelo_simples = Pipeline([("prep", StandardScaler()), ("nb", GaussianNB())])
    scores = cross_val_score(modelo_simples, X, y[col], cv=skf, scoring="f1_macro")
    print(f"{col}: média={scores.mean():.3f}  desvio={scores.std():.3f}  | folds: {np.round(scores, 3)}")

# ---- Gráfico 4: Importância das variáveis ----
print("\nGerando gráfico: importância das variáveis (risco_estresse)...")
pipeline_estresse = Pipeline([("prep", StandardScaler()), ("nb", GaussianNB())])
pipeline_estresse.fit(X_train, y_train["risco_estresse"])
resultado_perm = permutation_importance(
    pipeline_estresse, X_test, y_test["risco_estresse"],
    n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro",
)
importancias = pd.Series(resultado_perm.importances_mean, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importancias.plot(kind="barh", ax=ax)
ax.set_title("Importância das variáveis — risco_estresse (Naive Bayes)")
ax.set_xlabel("Queda no F1-score ao embaralhar a variável")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/importancia_variaveis.png", dpi=120)
plt.close(fig)

print(f"\nPronto! 4 gráficos salvos em {OUT_DIR}/:")
for arq in ["matrizes_confusao.png", "comparacao_f1.png", "curva_aprendizado.png", "importancia_variaveis.png"]:
    print(f"  - {OUT_DIR}/{arq}")
