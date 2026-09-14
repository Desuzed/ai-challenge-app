const agentForm = document.querySelector('#agent-form');
const agentMessage = document.querySelector('#agent-message');
const agentSubmit = document.querySelector('#agent-submit');
const agentHistory = document.querySelector('#agent-history');
const clearAgentHistory = document.querySelector('#clear-agent-history');
const openContextSettings = document.querySelector('#open-context-settings');
const contextSettingsDialog = document.querySelector('#context-settings-dialog');
const contextStrategy = document.querySelector('#context-strategy');
const recentMessages = document.querySelector('#recent-messages');
const strategyDescription = document.querySelector('#strategy-description');
const status = document.querySelector('#status');
const chatStatus = document.querySelector('#chat-status');
const requestLog = document.querySelector('#request-log');

const strategyDescriptions = {
  sliding_window: 'В модель отправляются только последние N сообщений. Ранние реплики удаляются.',
  facts: 'В модель отправляются sticky facts и последние N сообщений. Summary не используется.',
};

openContextSettings.addEventListener('click', () => contextSettingsDialog.showModal());
contextSettingsDialog.addEventListener('close', () => openContextSettings.focus());

function prettyJSON(value) {
  return JSON.stringify(value, null, 2);
}

function readableFetchError(error, fallback) {
  if (error instanceof TypeError && /fetch/i.test(error.message)) {
    return 'Не удалось подключиться к серверу. Запустите `go run .` в папке проекта и откройте http://localhost:8080.';
  }
  return error.message || fallback;
}

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle('error', isError);
}

function renderAgentHistory(messages) {
  agentHistory.replaceChildren();
  if (!messages || messages.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'hint';
    empty.textContent = 'История диалога появится здесь после первого сообщения.';
    agentHistory.append(empty);
    return;
  }
  messages.forEach((message) => {
    const item = document.createElement('article');
    item.className = `agent-message agent-message-${message.role}`;
    const role = document.createElement('strong');
    role.textContent = message.role === 'user' ? 'Вы' : 'Агент';
    const content = document.createElement('p');
    content.textContent = message.content;
    item.append(role, content);
    agentHistory.append(item);
  });
  agentHistory.scrollTop = agentHistory.scrollHeight;
}

function renderContextState(payload) {
  if (payload.strategy && strategyDescriptions[payload.strategy]) {
    contextStrategy.value = payload.strategy;
  }
  if (payload.recentMessages) recentMessages.value = payload.recentMessages;
  strategyDescription.textContent = strategyDescriptions[contextStrategy.value];
}

async function readAgentResponse(response) {
  const body = await response.text();
  try {
    return JSON.parse(body);
  } catch (_) {
    if (response.status === 404) {
      throw new Error('Сервер запущен в старой версии. Остановите его и снова выполните go run .');
    }
    throw new Error('Сервер вернул ответ в неожиданном формате.');
  }
}

async function saveStrategy() {
  const response = await fetch('/api/agent/chat', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store',
    body: JSON.stringify({ action: 'set_strategy', strategy: contextStrategy.value }),
  });
  const payload = await readAgentResponse(response);
  if (!response.ok) throw new Error(payload.error || 'Не удалось изменить стратегию.');
  renderContextState(payload);
  renderAgentHistory(payload.messages);
}

contextStrategy.addEventListener('change', async () => {
  strategyDescription.textContent = strategyDescriptions[contextStrategy.value];
  try {
    await saveStrategy();
    setStatus('Стратегия контекста обновлена.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось изменить стратегию.'), true);
  }
});

recentMessages.addEventListener('change', () => {
  const n = Number(recentMessages.value);
  if (!Number.isInteger(n) || n < 2 || n > 40) {
    setStatus('N должен быть целым числом от 2 до 40.', true);
    recentMessages.focus();
  }
});

async function loadAgentHistory() {
  try {
    const response = await fetch('/api/agent/chat', { cache: 'no-store' });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось загрузить чат.');
    renderAgentHistory(payload.messages);
    renderContextState(payload);
  } catch (error) {
    chatStatus.textContent = readableFetchError(error, 'Не удалось загрузить чат.');
  }
}

agentForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = agentMessage.value.trim();
  const n = Number(recentMessages.value);
  if (!message) {
    setStatus('Введите сообщение для агента.', true);
    agentMessage.focus();
    return;
  }
  if (!Number.isInteger(n) || n < 2 || n > 40) {
    setStatus('N должен быть целым числом от 2 до 40.', true);
    recentMessages.focus();
    return;
  }

  agentSubmit.disabled = true;
  setStatus('Агент отвечает…');
  const requestBody = { message, recentMessages: n, strategy: contextStrategy.value };
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPOST /api/agent/chat\n${prettyJSON(requestBody)}\n\nОжидание ответа…`;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(requestBody),
    });
    const payload = await readAgentResponse(response);
    requestLog.textContent += `\n\nBACKEND → БРАУЗЕР\n${prettyJSON({ httpStatus: response.status, messagesInSession: payload.messages?.length })}`;
    if (!response.ok) throw new Error(payload.error || 'Не удалось получить ответ агента.');
    requestLog.textContent += `\n\nОТВЕТ АГЕНТА\n${payload.answer}`;
    renderAgentHistory(payload.messages);
    renderContextState(payload);
    agentMessage.value = '';
    agentMessage.focus();
    setStatus('Готово.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось получить ответ агента.'), true);
  } finally {
    agentSubmit.disabled = false;
  }
});

clearAgentHistory.addEventListener('click', async () => {
  if (!window.confirm('Удалить всю историю этого чата? Это действие нельзя отменить.')) return;
  clearAgentHistory.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', { method: 'DELETE', cache: 'no-store' });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось удалить историю.');
    renderAgentHistory(payload.messages);
    renderContextState(payload);
    requestLog.textContent = 'БРАУЗЕР → BACKEND\nDELETE /api/agent/chat\n\nИстория текущей сессии удалена.';
    setStatus('Вся история удалена.');
    agentMessage.focus();
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось удалить историю.'), true);
  } finally {
    clearAgentHistory.disabled = false;
  }
});

loadAgentHistory();
