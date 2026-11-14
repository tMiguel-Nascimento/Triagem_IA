"""
retrain_from_feedback.py
-----------------------------------------------------------------------------
"A IA aprende sozinha" — da forma cientificamente correta: só com triagens
que um profissional humano já revisou (`laudo_validado = 1`), usando a
correção dele quando existir, ou confirmando a previsão da IA quando ele
concordou. Isso evita o modelo entrar num ciclo vicioso de aprender com os
próprios erros não revisados (ver README para a explicação completa).

O QUE ESTE SCRIPT FAZ:
  1. Abre o banco SQL (backend/triagem.db).
  2. Pega só as triagens com `laudo_validado = 1`.
  3. Para cada rótulo (ansiedade/depressão/estresse), usa a CORREÇÃO do
     profissional se ela existir; senão, usa a própria previsão da IA
     (porque foi validada como certa).
  4. Salva isso em ml/feedback_profissionais.csv.
  5. Roda train_model.py de novo, que junta StressLevelDataset.csv + esse
     feedback real e treina um modelo novo.

COMO USAR:
    cd ml
    python3 retrain_from_feedback.py

Ou pelo site, chamando (autenticado) o endpoint POST /api/retreinar.
-----------------------------------------------------------------------------
"""

import os
import json
import sqlite3
import subprocess
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "backend", "triagem.db")
FEEDBACK_OUT = os.path.join(BASE_DIR, "feedback_profissionais.csv")
MAPEAMENTO_PATH = os.path.join(BASE_DIR, "mapeamento_campos.json")

LABEL_COLS = ["risco_ansiedade", "risco_depressao", "risco_estresse"]

# Nomes das colunas no banco SQL (prefixos em português) -> nomes que o
# StressLevelDataset usa (os mesmos esperados por train_model.py). Lido de
# um único arquivo JSON, também usado por backend/app.py — assim os dois
# nunca ficam dessincronizados (foi exatamente essa dessincronia que causou
# o bug corrigido nesta versão: o backend buscava os campos pelo nome
# errado e mandava quase tudo em branco para o modelo).
with open(MAPEAMENTO_PATH, "r", encoding="utf-8") as f:
    MAPA_COLUNAS = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
FEATURES = list(MAPA_COLUNAS.values())


def extrair_feedback():
    if not os.path.exists(DB_PATH):
        print("[aviso] Banco triagem.db ainda não existe — nenhuma triagem feita ainda.")
        return 0

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        linhas = [dict(l) for l in db.execute(
            "SELECT * FROM triagens WHERE laudo_validado = 1"
        ).fetchall()]

    if not linhas:
        print("[aviso] Nenhuma triagem validada por profissional ainda. Nada para aprender.")
        return 0

    registros = []
    for linha in linhas:
        registro = {}
        for col_banco, col_dataset in MAPA_COLUNAS.items():
            valor = linha[col_banco]
            if col_dataset == "mental_health_history":
                valor = 1 if valor == "Yes" else 0
            registro[col_dataset] = valor
        registro["risco_ansiedade"] = linha["risco_ansiedade_corrigido"] or linha["risco_ansiedade"]
        registro["risco_depressao"] = linha["risco_depressao_corrigido"] or linha["risco_depressao"]
        registro["risco_estresse"] = linha["risco_estresse_corrigido"] or linha["risco_estresse"]
        registros.append(registro)

    df = pd.DataFrame(registros)
    df.to_csv(FEEDBACK_OUT, index=False)
    print(f"[ok] {len(df)} triagens validadas exportadas para {FEEDBACK_OUT}")
    return len(df)


if __name__ == "__main__":
    qtd = extrair_feedback()
    print("Retreinando o modelo (StressLevelDataset + feedback real, quando houver)...")
    subprocess.run([sys.executable, "train_model.py"], cwd=BASE_DIR, check=True)
    import shutil
    shutil.copy(
        os.path.join(BASE_DIR, "modelo_triagem.joblib"),
        os.path.join(BASE_DIR, "..", "backend", "modelo_triagem.joblib"),
    )
    print("Concluído. backend/modelo_triagem.joblib atualizado.")
    if qtd == 0:
        print("[lembrete] Retreinado só com o StressLevelDataset (nenhum feedback humano novo ainda).")
