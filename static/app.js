const form = document.querySelector('#chat-form');
const promptInput = document.querySelector('#prompt');
const answer = document.querySelector('#answer');
const status = document.querySelector('#status');
const submit = document.querySelector('#submit');
const requestLog = document.querySelector('#request-log');
const temperature = document.querySelector('#temperature');
const topP = document.querySelector('#top-p');
const maxTokens = document.querySelector('#max-tokens');
const tabList = document.querySelector('[role="tablist"]');
const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const tabPanels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
const fillTestCase = document.querySelector('#fill-test-case');
const runAllReasoning = document.querySelector('#run-all-reasoning');
const reasoningModeHint = document.querySelector('#reasoning-mode-hint');
const reasoningResultsCard = document.querySelector('#reasoning-results-card');
const reasoningResults = document.querySelector('#reasoning-results');
const reasoningSummary = document.querySelector('#reasoning-summary');
const reasoningComparison = document.querySelector('#reasoning-comparison');
const showPreparedPrompt = document.querySelector('#show-prepared-prompt');

const sampleTask = 'Четыре задачи A, B, C и D нужно выполнить в четыре последовательных слота: 1, 2, 3, 4. Условия: A выполняется раньше B; C выполняется сразу после A; B выполняется сразу перед D. В каком порядке выполняются задачи? Кратко проверьте, что все условия соблюдены.';
const sampleReference = 'A → C → B → D';
const reasoningApproaches = {
  direct: {
    title: '1. Прямой ответ',
    hint: 'Прямой вариант создаёт базовую точку сравнения: оценивайте итог без дополнительного управления ходом решения.',
  },
  step_by_step: {
    title: '2. «Решай пошагово»',
    hint: 'Здесь добавляется только инструкция «решай пошагово»: она должна сделать ход решения прозрачнее, но не меняет саму задачу.',
  },
  prompt_designer: {
    title: '3. Конструктор промпта',
    hint: 'Сначала отдельный API-вызов проектирует рабочую инструкцию, затем второй вызов решает исходную задачу по этой инструкции.',
  },
  expert_panel: {
    title: '4. Группа экспертов',
    hint: 'В одном промпте создаётся группа: аналитик, инженер и критик дают самостоятельные блоки, затем формируется общий вывод.',
  },
};
const reasoningOrder = Object.keys(reasoningApproaches);
const reasoningRuns = new Map();
let reasoningTask = '';
let savedMaxTokens = maxTokens.value;

function activeTabName() {
  return tabList.dataset.activeTab;
}

function updateSubmitLabel() {
  submit.textContent = activeTabName() === 'reasoning' ? 'Запустить выбранный способ' : 'Спросить DeepSeek';
}

function activateTab(tab, focus = false) {
  const panelID = tab.getAttribute('aria-controls');
  tabList.dataset.activeTab = tab.dataset.tab;

  tabs.forEach((item) => {
    const isActive = item === tab;
    item.setAttribute('aria-selected', String(isActive));
    item.tabIndex = isActive ? 0 : -1;
  });
  tabPanels.forEach((panel) => {
    panel.hidden = panel.id !== panelID;
  });
  updateSubmitLabel();

  if (focus) tab.focus();
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let nextIndex;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = tabs.length - 1;
    if (nextIndex === undefined) return;

    event.preventDefault();
    activateTab(tabs[nextIndex], true);
  });
});

function showValue(inputId, outputId) {
  const input = document.querySelector(inputId);
  const output = document.querySelector(outputId);
  input.addEventListener('input', () => { output.value = input.value; });
}

showValue('#temperature', '#temperature-value');
showValue('#top-p', '#top-p-value');
showValue('#max-tokens', '#max-tokens-value');

function syncSamplingMode() {
  const mode = document.querySelector('input[name="sampling-mode"]:checked').value;
  temperature.disabled = mode !== 'temperature';
  topP.disabled = mode !== 'top-p';
}

document.querySelectorAll('input[name="sampling-mode"]').forEach((input) => {
  input.addEventListener('change', syncSamplingMode);
});

function syncResponseMode() {
  const mode = document.querySelector('input[name="response-mode"]:checked').value;
  const fixedLimits = { length: 120, all: 180 };
  if (fixedLimits[mode]) {
    if (!maxTokens.disabled) savedMaxTokens = maxTokens.value;
    maxTokens.value = fixedLimits[mode];
    document.querySelector('#max-tokens-value').value = maxTokens.value;
    maxTokens.disabled = true;
    return;
  }
  if (maxTokens.disabled) {
    maxTokens.value = savedMaxTokens;
    document.querySelector('#max-tokens-value').value = maxTokens.value;
  }
  maxTokens.disabled = false;
}

document.querySelectorAll('input[name="response-mode"]').forEach((input) => {
  input.addEventListener('change', syncResponseMode);
});

maxTokens.addEventListener('input', () => { savedMaxTokens = maxTokens.value; });
syncSamplingMode();
syncResponseMode();

function selectedReasoningApproach() {
  return document.querySelector('input[name="reasoning-approach"]:checked').value;
}

function syncReasoningModeHint() {
  reasoningModeHint.textContent = reasoningApproaches[selectedReasoningApproach()].hint;
}

document.querySelectorAll('input[name="reasoning-approach"]').forEach((input) => {
  input.addEventListener('change', syncReasoningModeHint);
});
syncReasoningModeHint();

function prettyJSON(value) {
  return JSON.stringify(value, null, 2);
}

function readableFetchError(error, fallback) {
  if (error instanceof TypeError && /fetch/i.test(error.message)) {
    return 'Не удалось подключиться к backend. Запустите `go run .` в папке проекта и откройте http://localhost:8080.';
  }
  return error.message || fallback;
}

function currentSettings() {
  const mode = document.querySelector('input[name="sampling-mode"]:checked').value;
  const settings = { maxTokens: Number(maxTokens.value) };
  if (mode === 'temperature') settings.temperature = Number(temperature.value);
  if (mode === 'top-p') settings.topP = Number(topP.value);
  return settings;
}

function setReasoningBusy(isBusy) {
  submit.disabled = isBusy;
  runAllReasoning.disabled = isBusy;
}

function resetReasoningResults() {
  reasoningRuns.clear();
  reasoningTask = '';
  reasoningResultsCard.hidden = true;
  reasoningResults.replaceChildren();
}

function referenceDetected(text) {
  return text.toUpperCase().replace(/[^ABCD]/g, '').includes('ACBD');
}

function normalizedAnswer(text) {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

function createResultCard(result) {
  const card = document.createElement('article');
  card.className = `reasoning-result${result.error ? ' reasoning-result-error' : ''}`;

  const heading = document.createElement('div');
  heading.className = 'reasoning-result-heading';
  const title = document.createElement('h3');
  title.textContent = reasoningApproaches[result.approach].title;
  const badge = document.createElement('span');
  badge.className = 'result-badge';
  badge.textContent = result.error ? 'Ошибка запуска' : referenceDetected(result.answer) ? `Эталон ${sampleReference} найден` : 'Сверьте с эталоном';
  heading.append(title, badge);
  card.append(heading);

  if (result.error) {
    const message = document.createElement('p');
    message.className = 'error';
    message.textContent = result.error;
    card.append(message);
    return card;
  }

  if (result.preparedPrompt && showPreparedPrompt.checked) {
    const promptDetails = document.createElement('details');
    promptDetails.className = 'prepared-prompt';
    const promptSummary = document.createElement('summary');
    promptSummary.textContent = 'Шаг 1. Промпт, составленный моделью';
    const promptText = document.createElement('pre');
    promptText.textContent = result.preparedPrompt;
    promptDetails.append(promptSummary, promptText);
    card.append(promptDetails);
  }

  const answerLabel = document.createElement('strong');
  answerLabel.textContent = result.preparedPrompt ? 'Шаг 2. Решение по созданному промпту' : 'Решение';
  const answerText = document.createElement('p');
  answerText.className = 'reasoning-answer';
  answerText.textContent = result.answer;
  const metadata = document.createElement('p');
  metadata.className = 'result-metadata';
  const finishReasons = result.debug.finishReasons?.filter(Boolean).join(', ') || 'не указана';
  metadata.textContent = `${result.debug.requests} API-запрос(а) · ${result.debug.durationMs} мс · причина остановки: ${finishReasons}`;
  card.append(answerLabel, answerText, metadata);
  return card;
}

function updateReasoningComparison() {
  const results = reasoningOrder.map((approach) => reasoningRuns.get(approach)).filter(Boolean);
  if (results.length === 0) {
    reasoningResultsCard.hidden = true;
    return;
  }

  reasoningResultsCard.hidden = false;
  reasoningResults.replaceChildren(...reasoningOrder
    .map((approach) => reasoningRuns.get(approach))
    .filter(Boolean)
    .map(createResultCard));

  const successful = results.filter((result) => !result.error && result.answer);
  const uniqueAnswers = new Set(successful.map((result) => normalizedAnswer(result.answer)));
  const matchingReference = successful.filter((result) => referenceDetected(result.answer));
  const notRun = 4 - results.length;

  if (notRun > 0) {
    reasoningSummary.textContent = `${results.length} из 4 способов запущено`;
    reasoningComparison.textContent = `Для одной задачи получено ${successful.length} решение(й) и ${uniqueAnswers.size} текстово различающихся ответа(ов). Запустите ещё ${notRun}, чтобы сравнение было полным.`;
    return;
  }

  reasoningSummary.textContent = 'Все 4 способа готовы';
  const names = matchingReference.map((result) => reasoningApproaches[result.approach].title).join(' · ');
  if (matchingReference.length === 0) {
    reasoningComparison.textContent = `Формулировки различаются: ${uniqueAnswers.size} уникальных текста из ${successful.length} успешных запусков. В явном виде эталон ${sampleReference} не найден — проверьте порядок и условия вручную.`;
  } else if (matchingReference.length === 1) {
    reasoningComparison.textContent = `Формулировки различаются: ${uniqueAnswers.size} уникальных текста из ${successful.length} успешных запусков. По явной сверке с эталоном ${sampleReference} наиболее точным оказался вариант «${names}».`;
  } else {
    reasoningComparison.textContent = `Формулировки различаются: ${uniqueAnswers.size} уникальных текста из ${successful.length} успешных запусков. С эталоном ${sampleReference} совпали: ${names}. Среди них выберите наиболее точный по полноте проверяемой проверки условий.`;
  }
}

async function requestReasoning(approach, progressText = '') {
  const task = promptInput.value.trim();
  const requestBody = { task, approach, settings: currentSettings() };
  status.textContent = progressText || 'Модель решает задачу…';
  answer.textContent = 'Пожалуйста, подождите.';
  answer.classList.remove('error');
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPOST /api/reasoning\n${prettyJSON(requestBody)}\n\nОжидание ответа…`;

  try {
    const response = await fetch('/api/reasoning', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(requestBody),
    });
    const payload = await response.json();
    const debug = payload.debug || { httpStatus: response.status };
    requestLog.textContent += `\n\nBACKEND → БРАУЗЕР\n${prettyJSON(debug)}`;
    if (!response.ok) throw new Error(payload.error || 'Не удалось получить решение.');
    if (payload.preparedPrompt) requestLog.textContent += `\n\nСОЗДАННЫЙ ПРОМПТ\n${payload.preparedPrompt}`;
    requestLog.textContent += `\n\nОТВЕТ МОДЕЛИ\n${payload.answer}`;
    reasoningRuns.set(approach, payload);
    reasoningTask = task;
    updateReasoningComparison();
    answer.textContent = payload.answer;
    status.textContent = `Готово: ${reasoningApproaches[approach].title}`;
    return payload;
  } catch (error) {
    const message = readableFetchError(error, 'Не удалось получить решение.');
    reasoningRuns.set(approach, { approach, error: message });
    reasoningTask = task;
    updateReasoningComparison();
    answer.textContent = message;
    answer.classList.add('error');
    status.textContent = 'Ошибка';
    throw error;
  }
}

async function runSelectedReasoning() {
  const task = promptInput.value.trim();
  if (!task) {
    answer.textContent = 'Введите задачу, чтобы продолжить.';
    answer.classList.add('error');
    status.textContent = 'Ошибка';
    promptInput.focus();
    return;
  }
  setReasoningBusy(true);
  try {
    await requestReasoning(selectedReasoningApproach());
  } catch (_) {
    // The error is already shown both in the main result and the comparison card.
  } finally {
    setReasoningBusy(false);
  }
}

async function runAllReasoningApproaches() {
  const task = promptInput.value.trim();
  if (!task) {
    answer.textContent = 'Введите задачу, чтобы продолжить.';
    answer.classList.add('error');
    status.textContent = 'Ошибка';
    promptInput.focus();
    return;
  }

  resetReasoningResults();
  setReasoningBusy(true);
  for (const [index, approach] of reasoningOrder.entries()) {
    try {
      await requestReasoning(approach, `Запуск ${index + 1} из 4: ${reasoningApproaches[approach].title}`);
    } catch (_) {
      // Keep running the remaining approaches, so one failed request does not hide comparison data.
    }
  }
  setReasoningBusy(false);
  if (reasoningRuns.size === 4) status.textContent = 'Все 4 способа завершены';
}

fillTestCase.addEventListener('click', () => {
  promptInput.value = sampleTask;
  resetReasoningResults();
  answer.textContent = 'Тестовая задача подставлена. Выберите способ в третьем табе и запустите его.';
  answer.classList.remove('error');
  status.textContent = 'Готово к запуску';
  promptInput.focus();
});

runAllReasoning.addEventListener('click', runAllReasoningApproaches);
showPreparedPrompt.addEventListener('change', updateReasoningComparison);
promptInput.addEventListener('input', () => {
  if (reasoningTask && promptInput.value.trim() !== reasoningTask) resetReasoningResults();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (activeTabName() === 'reasoning') {
    await runSelectedReasoning();
    return;
  }

  const prompt = promptInput.value.trim();
  if (!prompt) {
    activateTab(document.querySelector('#generation-tab'));
    answer.textContent = 'Введите вопрос, чтобы продолжить.';
    answer.classList.add('error');
    status.textContent = 'Ошибка';
    promptInput.focus();
    return;
  }

  const responseMode = document.querySelector('input[name="response-mode"]:checked').value;
  const requestBody = { prompt, mode: responseMode, settings: currentSettings() };

  submit.disabled = true;
  status.textContent = 'Модель отвечает…';
  answer.textContent = 'Пожалуйста, подождите.';
  answer.classList.remove('error');
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPOST /api/chat\n${prettyJSON(requestBody)}\n\nОжидание ответа…`;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(requestBody),
    });
    const payload = await response.json();
    const debug = payload.debug || { httpStatus: response.status };
    requestLog.textContent += `\n\nBACKEND → БРАУЗЕР\n${prettyJSON(debug)}`;
    if (!response.ok) throw new Error(payload.error || 'Не удалось получить ответ.');
    requestLog.textContent += `\n\nОТВЕТ МОДЕЛИ\n${payload.answer}`;
    answer.textContent = payload.answer;
    status.textContent = 'Готово';
  } catch (error) {
    answer.textContent = readableFetchError(error, 'Не удалось получить ответ.');
    answer.classList.add('error');
    status.textContent = 'Ошибка';
  } finally {
    submit.disabled = false;
  }
});

updateSubmitLabel();
