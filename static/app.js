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
const tokenReport = document.querySelector('#token-report');
const agentModel = document.querySelector('#agent-model');
const modelDescription = document.querySelector('#model-description');
const memoryLayers = document.querySelector('#memory-layers');
const clearLongTermMemory = document.querySelector('#clear-long-term-memory');
const profileForm = document.querySelector('#profile-form');
const profileSelect = document.querySelector('#profile-select');
const newProfile = document.querySelector('#new-profile');
const deleteProfile = document.querySelector('#delete-profile');
const profileName = document.querySelector('#profile-name');
const profilePerspective = document.querySelector('#profile-perspective');
const profileStyle = document.querySelector('#profile-style');
const profileFormat = document.querySelector('#profile-format');
const profileConstraints = document.querySelector('#profile-constraints');
const saveProfile = document.querySelector('#save-profile');
let selectedAgentModel = 'deepseek-flash';
let activeProfileID = '';

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
    const memoryActions = document.createElement('div');
    memoryActions.className = 'message-memory-actions';
    [
      ['В рабочую', 'working', ''],
      ['В решения', 'long_term', 'decisions'],
      ['В знания', 'long_term', 'knowledge'],
    ].forEach(([label, layer, category]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'message-memory-button';
      button.textContent = label;
      button.addEventListener('click', () => saveMessageToMemory(message.content, layer, category));
      memoryActions.append(button);
    });
    item.append(role, content, memoryActions);
    agentHistory.append(item);
  });
  agentHistory.scrollTop = agentHistory.scrollHeight;
}

function renderContextState(payload) {
  if (payload.strategy && strategyDescriptions[payload.strategy]) {
    contextStrategy.value = payload.strategy;
  }
  if (payload.recentMessages) recentMessages.value = payload.recentMessages;
  if (payload.model) {
    selectedAgentModel = payload.model;
    agentModel.value = selectedAgentModel;
  }
  strategyDescription.textContent = strategyDescriptions[contextStrategy.value];
	if (payload.memory) renderMemoryLayers(payload.memory);
	if (payload.profiles) renderProfiles(payload);
}

function renderProfile(profile = {}) {
  profileName.value = profile.name || '';
  profilePerspective.value = profile.perspective || '';
  profileStyle.value = profile.style || '';
  profileFormat.value = profile.format || '';
  profileConstraints.value = profile.constraints || '';
}

function renderProfiles(payload) {
  activeProfileID = payload.activeProfileId || '';
  profileSelect.replaceChildren();
  const none = document.createElement('option');
  none.value = '';
  none.textContent = 'Без профиля';
  profileSelect.append(none);
  (payload.profiles || []).forEach((profile) => {
    const option = document.createElement('option');
    option.value = profile.id;
    option.textContent = profile.name;
    profileSelect.append(option);
  });
  profileSelect.value = activeProfileID;
  deleteProfile.disabled = !activeProfileID;
  saveProfile.textContent = activeProfileID ? 'Сохранить изменения' : 'Создать профиль';
  renderProfile(activeProfileID ? payload.profile : {});
}

function renderTokenReport(payload) {
  const tokens = payload.tokens;
  if (!tokens || (tokens.requestTokens === 0 && tokens.responseTokens === 0)) return;
  const total = tokens.requestTokens + tokens.responseTokens;
  const longTermSent = (payload.requestMessages || []).some((message) => message.role === 'system' && message.content.includes('Долговременная память'));
  const workingSent = (payload.requestMessages || []).some((message) => message.role === 'system' && message.content.includes('Рабочая память'));
  const profileSent = (payload.requestMessages || []).some((message) => message.role === 'system' && message.content.includes('Активный профиль пользователя'));
  tokenReport.replaceChildren();
  const entries = [
    `Модель: ${modelLabel(payload.model || selectedAgentModel)}`,
    `Вход: ${tokens.requestTokens} токенов`,
    `Выход: ${tokens.responseTokens} токенов`,
    `Всего: ${total} токенов`,
    `Долгосрочная память: ${longTermSent ? 'передана в контекст' : 'не передавалась'}`,
    `Рабочая память: ${workingSent ? 'передана в контекст' : 'не передавалась'}`,
    `Профиль: ${profileSent ? 'применён автоматически' : 'не задан'}`,
  ];
  entries.forEach((entry, index) => {
    const item = document.createElement('span');
    item.textContent = entry;
    tokenReport.append(item);
    if (index < entries.length - 1) tokenReport.append(document.createTextNode(' · '));
  });
}

profileForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const action = activeProfileID ? 'update_profile' : 'create_profile';
  const profile = {
    id: activeProfileID,
    name: profileName.value.trim(),
    perspective: profilePerspective.value.trim(),
    style: profileStyle.value.trim(),
    format: profileFormat.value.trim(),
    constraints: profileConstraints.value.trim(),
  };
  saveProfile.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ action, profile }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось сохранить профиль.');
    renderContextState(payload);
    requestLog.textContent = `БРАУЗЕР → BACKEND\nPATCH /api/agent/chat\n${prettyJSON({ action, profile })}\n\nАктивный профиль будет автоматически добавляться к каждому запросу этой сессии.`;
    setStatus('Профиль сохранён. Следующий ответ будет написан через выбранную перспективу.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось сохранить профиль.'), true);
  } finally {
    saveProfile.disabled = false;
  }
});

newProfile.addEventListener('click', async () => {
  newProfile.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, cache: 'no-store',
      body: JSON.stringify({ action: 'set_active_profile', profileId: '' }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось начать создание профиля.');
    renderContextState(payload);
    profileName.focus();
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось начать создание профиля.'), true);
  } finally {
    newProfile.disabled = false;
  }
});

profileSelect.addEventListener('change', async () => {
  const profileId = profileSelect.value;
  profileSelect.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, cache: 'no-store',
      body: JSON.stringify({ action: 'set_active_profile', profileId }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось выбрать профиль.');
    renderContextState(payload);
    setStatus(profileId ? 'Профиль выбран для этой сессии.' : 'Профиль отключён для этой сессии.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось выбрать профиль.'), true);
  } finally {
    profileSelect.disabled = false;
  }
});

deleteProfile.addEventListener('click', async () => {
  if (!activeProfileID || !window.confirm('Удалить выбранный профиль? Он перестанет быть доступен во всех ваших сессиях.')) return;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, cache: 'no-store',
      body: JSON.stringify({ action: 'delete_profile', profileId: activeProfileID }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось удалить профиль.');
    renderContextState(payload);
    setStatus('Профиль удалён.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось удалить профиль.'), true);
  }
});

function modelLabel(model) {
  if (model === 'deepseek-flash') return 'DeepSeek Flash';
  if (model === 'deepseek-v4-pro') return 'DeepSeek V4 Pro';
  return model;
}

async function loadAgentModels() {
  try {
    const response = await fetch('/api/agent/models', { cache: 'no-store' });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось загрузить список моделей.');
    agentModel.replaceChildren();
    (payload.models || []).forEach((model) => {
      const option = document.createElement('option');
      option.value = model;
      option.textContent = modelLabel(model);
      agentModel.append(option);
    });
    if (!agentModel.options.length) throw new Error('DeepSeek не вернул доступных моделей для чата.');
    if (![...agentModel.options].some((option) => option.value === selectedAgentModel)) selectedAgentModel = agentModel.options[0].value;
    agentModel.value = selectedAgentModel;
    agentModel.disabled = false;
    modelDescription.textContent = 'Выбор сохраняется для текущего браузерного чата и применяется к следующему сообщению.';
  } catch (error) {
    modelDescription.textContent = readableFetchError(error, 'Не удалось загрузить список моделей.');
  }
}

agentModel.addEventListener('change', async () => {
  const previous = selectedAgentModel;
  selectedAgentModel = agentModel.value;
  agentModel.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ action: 'set_model', model: selectedAgentModel }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось сменить модель.');
    renderContextState(payload);
    setStatus(`Выбрана модель: ${modelLabel(selectedAgentModel)}.`);
  } catch (error) {
    selectedAgentModel = previous;
    agentModel.value = previous;
    setStatus(readableFetchError(error, 'Не удалось сменить модель.'), true);
  } finally {
    agentModel.disabled = false;
  }
});

function memorySection(title, description, items, emptyText) {
  const section = document.createElement('section');
  section.className = 'memory-layer';
  const heading = document.createElement('h3');
  heading.textContent = title;
  const note = document.createElement('p');
  note.className = 'hint';
  note.textContent = description;
  const list = document.createElement('ul');
  if (!items || items.length === 0) {
    const item = document.createElement('li');
    item.className = 'hint';
    item.textContent = emptyText;
    list.append(item);
  } else {
    items.forEach((entry) => {
      const item = document.createElement('li');
      const text = document.createElement('span');
      const prefix = entry.category ? `${entry.category}.` : '';
      text.textContent = entry.key.startsWith('message-') ? entry.value : `${prefix}${entry.key}: ${entry.value}`;
      item.append(text);
      if (entry.layer !== 'short_term') {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'memory-remove';
        remove.textContent = 'Удалить';
        remove.addEventListener('click', () => deleteMemory(entry));
        item.append(remove);
      }
      list.append(item);
    });
  }
  section.append(heading, note, list);
  return section;
}

function renderMemoryLayers(memory) {
  memoryLayers.replaceChildren(
    memorySection('Краткосрочная', 'Последние реплики текущего диалога. Добавляется автоматически.', (memory.shortTerm || []).map((message, index) => ({ key: `${index + 1}. ${message.role}`, value: message.content, layer: 'short_term' })), 'Пока нет реплик.'),
    memorySection('Рабочая', 'Данные текущей задачи. Добавляется только этой формой.', memory.working, 'Нет явно сохранённых данных задачи.'),
    memorySection('Долговременная', 'Глобальные факты пользователя: решения и знания. Не зависит от профиля и сессии.', memory.longTerm, 'Нет глобальных фактов.')
  );
  clearLongTermMemory.disabled = !memory.longTerm || memory.longTerm.length === 0;
}

async function deleteMemory(entry) {
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ action: 'delete_memory', layer: entry.layer, category: entry.category, key: entry.key }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось удалить запись памяти.');
    renderContextState(payload);
    setStatus('Запись памяти удалена.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось удалить запись памяти.'), true);
  }
}

clearLongTermMemory.addEventListener('click', async () => {
  if (!window.confirm('Удалить все глобальные решения и знания пользователя?')) return;
  clearLongTermMemory.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ action: 'clear_memory_layer', layer: 'long_term' }),
    });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось очистить долговременную память.');
    renderContextState(payload);
    setStatus('Долговременная память очищена.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось очистить долговременную память.'), true);
  }
});

async function saveMessageToMemory(value, layer, category) {
  try {
    const body = { action: 'save_message', layer, category, value };
    const response = await fetch('/api/agent/chat', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, cache: 'no-store', body: JSON.stringify(body) });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось сохранить реплику в памяти.');
    renderContextState(payload);
    requestLog.textContent = `БРАУЗЕР → BACKEND\nPATCH /api/agent/chat\n${prettyJSON(body)}\n\nПолная реплика сохранена в ${layer === 'working' ? 'рабочей' : 'долговременной'} памяти.`;
    setStatus('Полная реплика сохранена и будет добавлена к следующему запросу агента.');
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось сохранить реплику в памяти.'), true);
  }
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
    renderTokenReport(payload);
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
  const requestBody = { message, recentMessages: n, strategy: contextStrategy.value, model: selectedAgentModel };
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPOST /api/agent/chat\n${prettyJSON(requestBody)}\n\nОжидание ответа…`;
  try {
    const response = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(requestBody),
    });
    const payload = await readAgentResponse(response);
    requestLog.textContent += `\n\nBACKEND → БРАУЗЕР\n${prettyJSON({ httpStatus: response.status, messagesInSession: payload.messages?.length, memory: payload.memory, contextSentToModel: payload.requestMessages })}`;
    if (!response.ok) throw new Error(payload.error || 'Не удалось получить ответ агента.');
    requestLog.textContent += `\n\nОТВЕТ АГЕНТА\n${payload.answer}`;
    renderAgentHistory(payload.messages);
    renderContextState(payload);
    renderTokenReport(payload);
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
    tokenReport.textContent = 'История очищена. Следующий запрос покажет точный расход токенов.';
    setStatus('Вся история удалена.');
    agentMessage.focus();
  } catch (error) {
    setStatus(readableFetchError(error, 'Не удалось удалить историю.'), true);
  } finally {
    clearAgentHistory.disabled = false;
  }
});

loadAgentHistory();
loadAgentModels();
