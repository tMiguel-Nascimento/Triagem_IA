"""
indicadores.py
-----------------------------------------------------------------------------
Indicadores COMPLEMENTARES, calculados por REGRA (não por Machine Learning).

O modelo de IA (train_model.py) já prevê ansiedade, depressão e estresse.
Esses 4 indicadores abaixo são contas diretas e transparentes a partir dos
mesmos campos do formulário — do tipo que qualquer triagem já faz ("dorme
mal? sinaliza"). Separamos isso do que é Machine Learning de propósito: é
mais honesto para a banca do que fingir que o modelo "aprendeu" algo que o
dataset não tem sinal suficiente para ensinar com confiança.

Cada função devolve ("Baixo" | "Medio" | "Alto", texto curto explicando).
-----------------------------------------------------------------------------
"""


def risco_burnout(carga_estudos, desempenho_academico, qualidade_sono_percebida):
    # Carga de estudos alta + baixo desempenho + sono ruim = combinação
    # clássica associada a esgotamento (burnout) na literatura acadêmica.
    pontos = 0
    if carga_estudos >= 4:
        pontos += 1
    if desempenho_academico <= 2:
        pontos += 1
    if qualidade_sono_percebida <= 2:
        pontos += 1
    if pontos >= 2:
        return "Alto", "Carga de estudos alta combinada com baixo desempenho e/ou sono ruim."
    elif pontos == 1:
        return "Medio", "Um fator de alerta presente; vale observar a evolução."
    return "Baixo", "Sem combinação relevante de fatores de esgotamento."


def qualidade_sono(qualidade_sono_percebida):
    # Escala 0-5 do próprio formulário (0 = muito ruim, 5 = excelente).
    if qualidade_sono_percebida <= 1:
        return "Alto", "Sono autoavaliado como muito ruim."
    elif qualidade_sono_percebida <= 2:
        return "Medio", "Sono autoavaliado abaixo do ideal."
    return "Baixo", "Sono autoavaliado como bom."


def risco_isolamento_social(apoio_social_academico):
    # Escala 0-3 do formulário.
    if apoio_social_academico == 0:
        return "Alto", "Apoio social percebido muito baixo."
    elif apoio_social_academico == 1:
        return "Medio", "Apoio social percebido moderado."
    return "Baixo", "Boa percepção de rede de apoio."


def clima_bullying(bullying):
    # Escala 0-5 do formulário.
    if bullying >= 3:
        return "Alto", "Relato de bullying frequente/intenso."
    elif bullying >= 1:
        return "Medio", "Relato de episódios ocasionais de bullying."
    return "Baixo", "Sem relato relevante de bullying."


def calcular_indicadores_complementares(dados: dict) -> dict:
    """Recebe os mesmos dados do formulário e devolve os 4 indicadores."""
    classe_burnout, exp_burnout = risco_burnout(
        dados["carga_estudos"], dados["desempenho_academico"], dados["qualidade_sono_percebida"]
    )
    classe_sono, exp_sono = qualidade_sono(dados["qualidade_sono_percebida"])
    classe_isolamento, exp_isolamento = risco_isolamento_social(dados["apoio_social_academico"])
    classe_bullying, exp_bullying = clima_bullying(dados["bullying"])

    return {
        "risco_burnout": {"classe": classe_burnout, "explicacao": exp_burnout},
        "qualidade_sono": {"classe": classe_sono, "explicacao": exp_sono},
        "risco_isolamento_social": {"classe": classe_isolamento, "explicacao": exp_isolamento},
        "clima_bullying": {"classe": classe_bullying, "explicacao": exp_bullying},
    }
