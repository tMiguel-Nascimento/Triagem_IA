-- schema.sql
-- Banco de dados relacional (SQLite) do sistema de triagem.
--
-- pacientes -> identificação de quem está sendo atendido
-- triagens  -> cada rodada de triagem: os 17 fatores coletados, o
--              resultado do modelo de IA (ansiedade/depressão/estresse),
--              os 4 indicadores de regra (não-IA), e o espaço de
--              correção do profissional (usado no reaprendizado — ver
--              ml/retrain_from_feedback.py).

CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    matricula TEXT NOT NULL,
    nome TEXT NOT NULL,
    semestre INTEGER,
    idade INTEGER,          -- registro clínico; NÃO é usado como entrada do modelo de IA
    genero TEXT,             -- idem (o StressLevelDataset não tem essas colunas)
    criado_em TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS triagens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL,

    -- ===== Entradas (os 17 fatores que o modelo usa) =====
    dor_cabeca INTEGER,
    pressao_arterial INTEGER,
    qualidade_sono_percebida INTEGER,
    dificuldade_respiratoria INTEGER,
    nivel_ruido_ambiente INTEGER,
    condicoes_moradia INTEGER,
    sensacao_seguranca INTEGER,
    necessidades_basicas INTEGER,
    desempenho_academico INTEGER,
    carga_estudos INTEGER,
    relacao_professores INTEGER,
    preocupacao_futuro INTEGER,
    apoio_social_academico INTEGER,
    pressao_colegas INTEGER,
    atividades_extracurriculares INTEGER,
    bullying INTEGER,
    mental_health_history TEXT,   -- 'Yes' / 'No'

    -- ===== Resultados do modelo de IA =====
    risco_ansiedade TEXT,
    risco_ansiedade_prob REAL,
    risco_depressao TEXT,
    risco_depressao_prob REAL,
    risco_estresse TEXT,
    risco_estresse_prob REAL,
    fatores_determinantes TEXT,        -- JSON

    -- ===== Indicadores complementares (regra clínica simplificada, não-IA) =====
    indicadores_complementares TEXT,   -- JSON

    -- ===== Painel de edição do profissional =====
    observacoes_profissional TEXT DEFAULT '',
    laudo_validado INTEGER DEFAULT 0,

    -- Correções (viram "verdade" para o reaprendizado — ver retrain_from_feedback.py)
    risco_ansiedade_corrigido TEXT,
    risco_depressao_corrigido TEXT,
    risco_estresse_corrigido TEXT,

    criado_em TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);
