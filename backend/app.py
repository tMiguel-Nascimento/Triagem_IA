"""
app.py
-----------------------------------------------------------------------------
Servidor backend (API RESTful) que conecta o site ao modelo de IA treinado
em StressLevelDataset.csv (ver ml/train_model.py).

O QUE ESTE ARQUIVO FAZ:
  1. Ao iniciar, carrega o modelo treinado (modelo_triagem.joblib).
  2. Cria/abre o banco SQL (triagem.db).
  3. Serve os arquivos do frontend (index.html, style.css, script.js).
  4. Expõe os endpoints da API:
       POST /api/triagem          -> roda a IA, calcula os indicadores de
                                      regra, salva no banco e devolve o
                                      resultado (o e-mail de feedback é
                                      manual, feito pelo próprio participante
                                      — ver frontend/script.js, sem nenhuma
                                      credencial de servidor envolvida).
       GET  /api/triagem/<id>     -> busca uma triagem específica.
       PUT  /api/triagem/<id>     -> salva anotações/validação/correção do
                                      profissional (alimenta o reaprendizado).
       GET  /api/triagens         -> lista as últimas triagens.
       POST /api/retreinar        -> dispara o reaprendizado com feedback.

COMO RODAR: ver README.md.
-----------------------------------------------------------------------------
"""

import os
import json
import sqlite3
import subprocess
import sys

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g, send_from_directory

from indicadores import calcular_indicadores_complementares

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
ML_DIR = os.path.join(BASE_DIR, "..", "ml")
DB_PATH = os.path.join(BASE_DIR, "triagem.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_triagem.joblib")

# Token simples para proteger o endpoint de retreino (/api/retreinar), que
# ninguém além de você deveria poder chamar. Pode trocar por uma variável de
# ambiente (TRIAGEM_RETREINO_TOKEN) se quiser, mas não é obrigatório — o
# projeto não depende de nenhum arquivo de credenciais para rodar.
RETREINO_TOKEN = os.environ.get("TRIAGEM_RETREINO_TOKEN", "troque-este-token-antes-de-publicar")

app = Flask(__name__, static_folder=None)


# -----------------------------------------------------------------------
# Carregar o modelo (uma vez, ao subir o servidor)
# -----------------------------------------------------------------------
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' não encontrado. Rode 'python3 train_model.py' dentro de ml/ "
            f"e copie o modelo_triagem.joblib gerado para backend/."
        )
    return joblib.load(MODEL_PATH)


MODELO = carregar_modelo()

# Nomes dos campos em português (formulário/banco) -> nomes das colunas em
# inglês que o modelo espera (StressLevelDataset.csv). Um único arquivo
# (ml/mapeamento_campos.json) é a fonte da verdade, lido também por
# ml/retrain_from_feedback.py — assim os dois nunca ficam dessincronizados.
MAPEAMENTO_PATH = os.path.join(ML_DIR, "mapeamento_campos.json")
with open(MAPEAMENTO_PATH, "r", encoding="utf-8") as f:
    MAPA_PT_PARA_EN = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
MAPA_EN_PARA_PT = {en: pt for pt, en in MAPA_PT_PARA_EN.items()}

CAMPOS_FORMULARIO = [c for c in MAPA_PT_PARA_EN if c != "mental_health_history"]


# -----------------------------------------------------------------------
# Banco de dados
# -----------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            db.executescript(f.read())


# -----------------------------------------------------------------------
# Servir o frontend
# -----------------------------------------------------------------------
def _sem_cache(resposta):
    # Evita que o navegador guarde uma versão antiga de index.html/style.css/
    # script.js em cache — foi exatamente isso que causou o "o resultado não
    # muda"/"a porcentagem ficou travada" em testes: o servidor já estava
    # com o código novo, mas o navegador seguia mostrando o JS antigo.
    resposta.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resposta


@app.route("/")
def home():
    return _sem_cache(send_from_directory(FRONTEND_DIR, "index.html"))


@app.route("/<path:filename>")
def frontend_files(filename):
    return _sem_cache(send_from_directory(FRONTEND_DIR, filename))



# -----------------------------------------------------------------------
# Rodar o modelo de IA
# -----------------------------------------------------------------------
def prever(dados):
    pipeline = MODELO["pipeline"]
    encoders = MODELO["encoders"]
    features = MODELO["features"]
    label_cols = MODELO["label_cols"]

    linha = {}
    for f in features:  # f = nome da coluna em inglês, que o modelo espera
        campo_pt = MAPA_EN_PARA_PT[f]  # nome correspondente no formulário/banco (português)
        if f == "mental_health_history":
            linha[f] = 1 if dados.get(campo_pt) == "Yes" else 0
        else:
            linha[f] = dados.get(campo_pt)
    X = pd.DataFrame([linha])

    faltando = [MAPA_EN_PARA_PT[f] for f in features if linha[f] is None]
    if faltando:
        raise ValueError(f"Campos obrigatórios faltando ou inválidos: {faltando}")
    probas_por_rotulo = pipeline.predict_proba(X)

    resultado = {}
    for i, col in enumerate(label_cols):
        le = encoders[col]
        probas = probas_por_rotulo[i][0]
        idx = probas.argmax()
        distribuicao = {
            le.inverse_transform([j])[0]: round(float(p) * 100, 1)
            for j, p in enumerate(probas)
        }

        # ÍNDICE DE RISCO (0 a 100): combina as três probabilidades em vez de
        # olhar só a chance de "Alto" sozinha. Isso existe porque o Naive
        # Bayes, nesse problema, quase sempre dá 0% de chance de "Alto" para
        # qualquer entrada que não seja bem extrema — mesmo quando a
        # classificação já oscila entre Baixo e Médio. Mostrar só a chance de
        # "Alto" fazia o número parecer "travado em 0%" o tempo todo, mesmo
        # com o resultado mudando de verdade por trás. O índice pondera:
        # Baixo conta 0, Médio conta 50, Alto conta 100 — então ele se move
        # de forma suave mesmo quando a chance de "Alto" sozinha é zero.
        indice_risco = round(
            distribuicao.get("Baixo", 0) * 0
            + distribuicao.get("Medio", 0) * 0.5
            + distribuicao.get("Alto", 0) * 1.0,
            1,
        )

        resultado[col] = {
            "classe": le.inverse_transform([idx])[0],
            "probabilidade": round(float(probas[idx]) * 100, 1),
            "distribuicao": distribuicao,
            "indice_risco": indice_risco,
        }

    # "Fatores mais determinantes": o Naive Bayes não tem um atributo pronto
    # tipo o `feature_importances_` do Random Forest, então a importância de
    # cada fator foi calculada por PERMUTATION IMPORTANCE uma única vez, no
    # treino (ml/train_model.py) — aqui só lemos os 3 primeiros da lista já
    # pronta que está salva dentro do modelo_triagem.joblib. (Explicação
    # simplificada e global, não uma explicação individual por paciente, do
    # tipo que SHAP daria — isso fugiria do escopo deste projeto.)
    fatores_top = MODELO.get("importancias_por_label", {}).get("risco_ansiedade", [])[:3]

    indicadores = calcular_indicadores_complementares(dados)

    return resultado, fatores_top, indicadores


# -----------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------
@app.route("/api/triagem", methods=["POST"])
def criar_triagem():
    dados = request.get_json(force=True)

    faltando = [c for c in ["matricula", "nome"] if not dados.get(c)]
    if faltando:
        return jsonify({"erro": f"Campos obrigatórios faltando: {faltando}"}), 400

    try:
        resultado, fatores_top, indicadores = prever(dados)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400

    db = get_db()
    cur = db.execute(
        "INSERT INTO pacientes (matricula, nome, semestre, idade, genero) VALUES (?, ?, ?, ?, ?)",
        (dados["matricula"], dados["nome"], dados.get("semestre"), dados.get("idade"), dados.get("genero")),
    )
    paciente_id = cur.lastrowid

    valores_campos = [dados.get(c) for c in CAMPOS_FORMULARIO]

    db.execute(
        f"""
        INSERT INTO triagens (
            paciente_id, {", ".join(CAMPOS_FORMULARIO)}, mental_health_history,
            risco_ansiedade, risco_ansiedade_prob,
            risco_depressao, risco_depressao_prob,
            risco_estresse, risco_estresse_prob,
            fatores_determinantes, indicadores_complementares
        ) VALUES ({", ".join(["?"] * (2 + len(CAMPOS_FORMULARIO) + 8))})
        """,
        (
            paciente_id, *valores_campos, dados.get("mental_health_history"),
            resultado["risco_ansiedade"]["classe"], resultado["risco_ansiedade"]["probabilidade"],
            resultado["risco_depressao"]["classe"], resultado["risco_depressao"]["probabilidade"],
            resultado["risco_estresse"]["classe"], resultado["risco_estresse"]["probabilidade"],
            json.dumps(fatores_top, ensure_ascii=False),
            json.dumps(indicadores, ensure_ascii=False),
        ),
    )
    db.commit()
    triagem_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    return jsonify({
        "triagem_id": triagem_id,
        "paciente_id": paciente_id,
        "resultado": resultado,
        "fatores_determinantes": fatores_top,
        "indicadores_complementares": indicadores,
    }), 201


@app.route("/api/triagem/<int:triagem_id>", methods=["GET"])
def obter_triagem(triagem_id):
    db = get_db()
    row = db.execute(
        """
        SELECT t.*, p.nome, p.matricula, p.semestre, p.idade, p.genero
        FROM triagens t JOIN pacientes p ON p.id = t.paciente_id
        WHERE t.id = ?
        """,
        (triagem_id,),
    ).fetchone()
    if row is None:
        return jsonify({"erro": "Triagem não encontrada"}), 404
    return jsonify(dict(row))


@app.route("/api/triagem/<int:triagem_id>", methods=["PUT"])
def atualizar_triagem(triagem_id):
    """Painel de edição do profissional: anotações, validação e correções."""
    dados = request.get_json(force=True)
    db = get_db()
    db.execute(
        """
        UPDATE triagens SET
            observacoes_profissional = ?,
            laudo_validado = ?,
            risco_ansiedade_corrigido = ?,
            risco_depressao_corrigido = ?,
            risco_estresse_corrigido = ?
        WHERE id = ?
        """,
        (
            dados.get("observacoes_profissional", ""),
            int(dados.get("laudo_validado", 0)),
            dados.get("risco_ansiedade_corrigido") or None,
            dados.get("risco_depressao_corrigido") or None,
            dados.get("risco_estresse_corrigido") or None,
            triagem_id,
        ),
    )
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/triagens", methods=["GET"])
def listar_triagens():
    db = get_db()
    rows = db.execute(
        """
        SELECT t.id, p.nome, p.matricula, p.semestre, t.risco_ansiedade,
               t.risco_depressao, t.risco_estresse, t.criado_em
        FROM triagens t JOIN pacientes p ON p.id = t.paciente_id
        ORDER BY t.id DESC LIMIT 50
        """
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/retreinar", methods=["POST"])
def retreinar():
    """Dispara o reaprendizado do modelo com o feedback acumulado."""
    token = request.headers.get("X-Retreino-Token")
    if token != RETREINO_TOKEN:
        return jsonify({"erro": "Token inválido."}), 403

    resultado = subprocess.run(
        [sys.executable, "retrain_from_feedback.py"],
        cwd=ML_DIR, capture_output=True, text=True,
    )
    if resultado.returncode != 0:
        return jsonify({"ok": False, "log": resultado.stdout + resultado.stderr}), 500

    global MODELO
    MODELO = carregar_modelo()

    return jsonify({"ok": True, "log": resultado.stdout})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
