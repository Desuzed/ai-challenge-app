const form = document.querySelector('#chat-form');
const promptInput = document.querySelector('#prompt');
const answer = document.querySelector('#answer');
const status = document.querySelector('#status');
const submit = document.querySelector('#submit');
const requestLog = document.querySelector('#request-log');
const temperature = document.querySelector('#temperature');
const topP = document.querySelector('#top-p');
const maxTokens = document.querySelector('#max-tokens');
let savedMaxTokens = maxTokens.value;

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
syncResponseMode();

function prettyJSON(value) {
  return JSON.stringify(value, null, 2);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();

  if (!prompt) {
    answer.textContent = 'Введите вопрос, чтобы продолжить.';
    answer.classList.add('error');
    status.textContent = 'Ошибка';
    promptInput.focus();
    return;
  }

  const mode = document.querySelector('input[name="sampling-mode"]:checked').value;
  const settings = { maxTokens: Number(maxTokens.value) };
  if (mode === 'temperature') settings.temperature = Number(temperature.value);
  if (mode === 'top-p') settings.topP = Number(topP.value);
  const responseMode = document.querySelector('input[name="response-mode"]:checked').value;
  const requestBody = { prompt, mode: responseMode, settings };

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
    answer.textContent = error.message || 'Не удалось получить ответ.';
    answer.classList.add('error');
    status.textContent = 'Ошибка';
  } finally {
    submit.disabled = false;
  }
});
