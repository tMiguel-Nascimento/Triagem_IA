/*
 * script.js
 * -----------------------------------------------------------------------
 * 1. Mostra o valor atual de cada slider.
 * 2. Envia o formulário via fetch() (POST /api/triagem).
 * 3. Renderiza o dashboard com o resultado da IA + os indicadores de regra.
 * 4. Permite ao profissional corrigir/validar e salvar (PUT).
 * 5. Monta o link de feedback por e-mail do participante.
 * -----------------------------------------------------------------------
 */

const API_BASE = "";

const form = document.getElementById("form-triagem");
const telaFormulario = document.getElementById("tela-formulario");
const telaResultado = document.getElementById("tela-resultado");
const erroEl = document.getElementById("form-erro");

let triagemAtualId = null;

// ---- sliders mostrando o valor atual ----
function religarSliders() {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const label = document.querySelector(`.slider-value[data-for="${slider.name}"]`);
    const atualizar = () => { if (label) label.textContent = slider.value; };
    slider.removeEventListener("input", slider._atualizarHandler || (() => {}));
    slider._atualizarHandler = atualizar;
    slider.addEventListener("input", atualizar);
    atualizar();
  });
}
religarSliders();

const CAMPOS_FORMULARIO = [
  "dor_cabeca", "pressao_arterial", "qualidade_sono_percebida", "dificuldade_respiratoria",
  "nivel_ruido_ambiente", "condicoes_moradia", "sensacao_seguranca", "necessidades_basicas",
  "desempenho_academico", "carga_estudos", "relacao_professores", "preocupacao_futuro",
  "apoio_social_academico", "pressao_colegas", "atividades_extracurriculares", "bullying",
];

// ---- envio do formulário ----
form.addEventListener("submit", async (evento) => {
  evento.preventDefault();
  erroEl.textContent = "";

  const formData = new FormData(form);
  const dados = {
    matricula: formData.get("matricula"),
    nome: formData.get("nome"),
    semestre: formData.get("semestre") ? Number(formData.get("semestre")) : null,
    idade: formData.get("idade") ? Number(formData.get("idade")) : null,
    genero: formData.get("genero") || null,
    mental_health_history: form.mental_health_history.checked ? "Yes" : "No",
  };
  CAMPOS_FORMULARIO.forEach((campo) => {
    dados[campo] = Number(formData.get(campo));
  });

  try {
    const resposta = await fetch(`${API_BASE}/api/triagem`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dados),
    });

    if (!resposta.ok) {
      const erro = await resposta.json();
      throw new Error(erro.erro || "Erro ao processar a triagem.");
    }

    const corpo = await resposta.json();
    triagemAtualId = corpo.triagem_id;
    mostrarResultado(dados, corpo);

  } catch (e) {
    erroEl.textContent = "Não foi possível conectar à API. " +
      "Verifique se o servidor (python3 app.py) está rodando. Detalhe: " + e.message;
  }
});

const NOMES_RISCO = {
  risco_ansiedade: "Indícios de ansiedade",
  risco_depressao: "Indícios de depressão",
  risco_estresse: "Nível de estresse",
};
const NOMES_INDICADOR = {
  risco_burnout: "Risco de burnout",
  qualidade_sono: "Qualidade do sono",
  risco_isolamento_social: "Isolamento social",
  clima_bullying: "Clima de bullying",
};

function mostrarResultado(dados, corpo) {
  const { resultado, fatores_determinantes: fatores, indicadores_complementares: indicadores } = corpo;

  document.getElementById("resultado-paciente").textContent =
    `${dados.nome} — matrícula ${dados.matricula}`;

  const container = document.getElementById("blocos-risco");
  container.innerHTML = "";
  ["risco_ansiedade", "risco_depressao", "risco_estresse"].forEach((chave) => {
    const info = resultado[chave];
    // O número em destaque é o ÍNDICE DE RISCO (0-100), calculado no backend
    // combinando as três probabilidades (Baixo conta 0, Médio conta 50, Alto
    // conta 100) — não só a chance de "Alto" sozinha. Isso evita o número
    // ficar "travado em 0%" sempre que a chance de Alto é zero (o que é
    // muito comum nesse modelo), mesmo com a classificação mudando de
    // verdade por trás. Quanto menor o índice, melhor — sempre.
    const indice = info.indice_risco;
    const div = document.createElement("div");
    div.className = `bloco-risco risco-${info.classe}`;
    div.innerHTML = `
      <h4>${NOMES_RISCO[chave] || chave}</h4>
      <span class="selo selo-${info.classe}">${info.classe}</span>
      <div class="valor">${indice}%</div>
      <div class="legenda-valor">índice de risco (quanto menor, melhor)</div>
      <div class="barra"><div class="barra__preenchida" style="width:${indice}%"></div></div>
    `;
    container.appendChild(div);
  });

  const containerIndicadores = document.getElementById("blocos-indicadores");
  containerIndicadores.innerHTML = "";
  Object.entries(indicadores || {}).forEach(([chave, info]) => {
    const div = document.createElement("div");
    div.className = "bloco-indicador";
    div.innerHTML = `
      <span>
        <span class="nome">${NOMES_INDICADOR[chave] || chave}</span>
        <span class="explicacao">${info.explicacao}</span>
      </span>
      <span class="selo selo-${info.classe}">${info.classe}</span>
    `;
    containerIndicadores.appendChild(div);
  });

  const listaFatores = document.getElementById("lista-fatores");
  listaFatores.innerHTML = "";
  fatores.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = traduzirFator(f);
    listaFatores.appendChild(li);
  });

  // Link de feedback por e-mail (mailto:), pré-preenchido já com os dados
  // da triagem — assim quem recebe o feedback já vê o que a IA respondeu,
  // sem precisar abrir o sistema. Isso é só um link que abre o APP DE E-MAIL
  // DO PRÓPRIO PARTICIPANTE (Gmail, Outlook...): quem efetivamente envia é a
  // pessoa, pela conta dela — o site não manda nada sozinho nem precisa de
  // nenhuma senha configurada no servidor para isso.
  const resumoResultado =
    `Risco de ansiedade: ${resultado.risco_ansiedade.classe} ` +
    `(índice de risco: ${resultado.risco_ansiedade.indice_risco}%)\n` +
    `Risco de depressão: ${resultado.risco_depressao.classe} ` +
    `(índice de risco: ${resultado.risco_depressao.indice_risco}%)\n` +
    `Risco de estresse: ${resultado.risco_estresse.classe} ` +
    `(índice de risco: ${resultado.risco_estresse.indice_risco}%)`;

  const assunto = encodeURIComponent(`Feedback - Triagem Inteligente (triagem #${triagemAtualId})`);
  const mensagem = encodeURIComponent(
    `Olá! Participei da triagem (#${triagemAtualId}, matrícula ${dados.matricula}).\n\n` +
    `--- Resultado que o sistema me deu ---\n${resumoResultado}\n\n` +
    `--- Meu feedback ---\n` +
    `O que achei do site:\n(escreva aqui)\n\n` +
    `A IA acertou ou errou o resultado acima em relação ao que eu realmente sinto?\n(escreva aqui)\n\n` +
    `Sugestões/críticas:\n(escreva aqui)`
  );
  document.getElementById("link-feedback").href = `mailto:tmiguel2909@gmail.com?subject=${assunto}&body=${mensagem}`;

  document.getElementById("observacoes").value = "";
  document.getElementById("laudo-validado").checked = false;
  document.getElementById("corrigir-ansiedade").value = "";
  document.getElementById("corrigir-depressao").value = "";
  document.getElementById("corrigir-estresse").value = "";
  document.getElementById("salvo-ok").classList.add("oculto");

  telaFormulario.classList.add("oculto");
  telaResultado.classList.remove("oculto");
  telaResultado.scrollIntoView({ behavior: "smooth" });
}

// O backend devolve o nome técnico da coluna como está no dataset de
// treino (StressLevelDataset.csv), que é em INGLÊS — por isso o mapa de
// tradução usa esses nomes como chave, e não os nomes em português do
// formulário (esse era justamente o bug: o mapa antigo tinha as chaves
// trocadas e nunca batia com nada, então sempre caía no "devolve sem traduzir").
function traduzirFator(nomeTecnico) {
  const mapa = {
    headache: "Dor de cabeça",
    blood_pressure: "Pressão arterial",
    sleep_quality: "Qualidade do sono",
    breathing_problem: "Dificuldade respiratória",
    noise_level: "Ruído no ambiente",
    living_conditions: "Condições de moradia",
    safety: "Sensação de segurança",
    basic_needs: "Necessidades básicas atendidas",
    academic_performance: "Desempenho acadêmico",
    study_load: "Carga de estudos",
    teacher_student_relationship: "Relação com professores",
    future_career_concerns: "Preocupação com o futuro profissional",
    social_support: "Apoio social",
    peer_pressure: "Pressão dos colegas",
    extracurricular_activities: "Atividades extracurriculares",
    bullying: "Bullying",
    mental_health_history: "Histórico de saúde mental",
  };
  for (const chave in mapa) {
    if (nomeTecnico.startsWith(chave)) return mapa[chave];
  }
  return nomeTecnico;
}

// ---- salvar anotações do profissional ----
document.getElementById("btn-salvar").addEventListener("click", async () => {
  if (!triagemAtualId) return;
  const observacoes = document.getElementById("observacoes").value;
  const validado = document.getElementById("laudo-validado").checked;

  const resposta = await fetch(`${API_BASE}/api/triagem/${triagemAtualId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observacoes_profissional: observacoes,
      laudo_validado: validado ? 1 : 0,
      risco_ansiedade_corrigido: document.getElementById("corrigir-ansiedade").value,
      risco_depressao_corrigido: document.getElementById("corrigir-depressao").value,
      risco_estresse_corrigido: document.getElementById("corrigir-estresse").value,
    }),
  });

  const aviso = document.getElementById("salvo-ok");
  if (resposta.ok) {
    aviso.classList.remove("oculto");
    setTimeout(() => aviso.classList.add("oculto"), 2500);
  }
});

// ---- nova triagem ----
document.getElementById("btn-nova").addEventListener("click", () => {
  form.reset();
  religarSliders();
  triagemAtualId = null;
  telaResultado.classList.add("oculto");
  telaFormulario.classList.remove("oculto");
});
