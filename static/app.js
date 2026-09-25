const agentForm = document.querySelector('#agent-form');
const agentMessage = document.querySelector('#agent-message');
const agentSubmit = document.querySelector('#agent-submit');
const agentHistory = document.querySelector('#agent-history');
const clearAgentHistory = document.querySelector('#clear-agent-history');
const openContextSettings = document.querySelector('#open-context-settings');
const contextSettingsDialog = document.querySelector('#context-settings-dialog');
const closeContextSettings = document.querySelector('#close-context-settings');
const doneContextSettings = document.querySelector('#done-context-settings');
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
const taskState = document.querySelector('#task-state');
const pauseTask = document.querySelector('#pause-task');
const resumeTask = document.querySelector('#resume-task');
const approvePlan = document.querySelector('#approve-plan');
const completeImplementation = document.querySelector('#complete-implementation');
const passValidation = document.querySelector('#pass-validation');
const returnForRework = document.querySelector('#return-for-rework');
const resetTask = document.querySelector('#reset-task');
const togglePlanner = document.querySelector('#toggle-planner');
const invariantForm = document.querySelector('#invariant-form');
const invariantScope = document.querySelector('#invariant-scope');
const invariantRule = document.querySelector('#invariant-rule');
const invariantLayers = document.querySelector('#invariant-layers');
const mcpStatus = document.querySelector('#mcp-status');
const mcpRefresh = document.querySelector('#mcp-refresh');
const mcpServers = document.querySelector('#mcp-servers');
const mcpTools = document.querySelector('#mcp-tools');
const mcpTokenNote = document.querySelector('#mcp-token-note');
const mcpResult = document.querySelector('#mcp-result');

let selectedAgentModel = 'deepseek-flash';
let activeTask = {};
let activeProfileID = '';
const strategyDescriptions = {
  sliding_window: 'В модель отправляются только последние N сообщений. Ранние реплики удаляются.',
  facts: 'В модель отправляются sticky facts и последние N сообщений. Summary не используется.',
};

openContextSettings.addEventListener('click', () => contextSettingsDialog.showModal());
contextSettingsDialog.addEventListener('close', () => openContextSettings.focus());
closeContextSettings.addEventListener('click', () => contextSettingsDialog.close());
doneContextSettings.addEventListener('click', () => contextSettingsDialog.close());

function prettyJSON(value) { return JSON.stringify(value, null, 2); }
function readableFetchError(error, fallback) {
  if (error instanceof TypeError && /fetch/i.test(error.message)) return 'Не удалось подключиться к серверу. Запустите `go run .` в папке проекта и откройте http://localhost:8080.';
  return error.message || fallback;
}
function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle('error', isError);
}
function modelLabel(model) {
  if (model === 'deepseek-flash') return 'DeepSeek Flash';
  if (model === 'deepseek-v4-pro') return 'DeepSeek V4 Pro';
  return model;
}

async function readJSONResponse(response, operation) {
  const body = await response.text();
  try { return JSON.parse(body); } catch (_) {
    if (response.status === 404 || /^404\s+page not found/i.test(body.trim())) {
      throw new Error(`${operation}: backend запущен со старым кодом. Остановите его, снова выполните \`go run .\` и обновите страницу.`);
    }
    throw new Error(`${operation}: сервер вернул не JSON (HTTP ${response.status}): ${body.trim().slice(0, 180) || 'пустой ответ'}`);
  }
}

async function loadMCP(event) {
  mcpRefresh.disabled = true;
  const manualRefresh = event?.type === 'click';
  if (manualRefresh) {
    mcpStatus.textContent = 'Выполняю tools/list…';
    mcpStatus.className = 'mcp-status';
    mcpResult.textContent = 'GET /api/mcp/tools → ожидание ответа…';
  }
  try {
    const response = await fetch('/api/mcp/tools', { cache: 'no-store' });
    const payload = await readJSONResponse(response, 'tools/list');
    if (!response.ok) throw new Error(payload.error || 'MCP-соединение не установлено.');
    mcpStatus.textContent = payload.connected ? `Подключено · ${payload.servers.length} сервера` : 'MCP недоступен';
    mcpStatus.className = `mcp-status${payload.connected ? ' connected' : ''}`;
    mcpServers.replaceChildren();
    payload.servers.forEach((server) => {
      const item = document.createElement('article'); item.className = 'mcp-server';
      const name = document.createElement('strong'); name.textContent = server.name;
      const detail = document.createElement('span');
      detail.textContent = `${server.connected ? 'подключён' : 'недоступен'} · ${server.toolCount} инструментов · ${server.detail}${server.configured ? '' : ' · запись требует токен'}`;
      item.append(name, detail); mcpServers.append(item);
    });
    mcpTools.replaceChildren();
    payload.tools.forEach((tool) => {
      const item = document.createElement('article'); item.className = 'mcp-tool';
      const name = document.createElement('strong'); name.textContent = tool.name;
      const description = document.createElement('span'); description.textContent = tool.description;
      item.append(name, description); mcpTools.append(item);
    });
    const refreshedAt = new Date().toLocaleTimeString('ru-RU');
    mcpTokenNote.textContent = `tools/list выполнен в ${refreshedAt}. Получено инструментов: ${payload.tools.length}. Схемы ≈ ${payload.estimatedDefinitionTokens} токенов, модель использовала ${payload.modelTokensUsedByThisRequest}.`;
    mcpResult.textContent = prettyJSON({
      request: 'GET /api/mcp/tools',
      protocolMethod: 'tools/list',
      connected: payload.connected,
      servers: payload.servers,
      toolNames: payload.tools.map((tool) => tool.name),
      receivedAt: refreshedAt,
    });
  } catch (error) {
    mcpStatus.textContent = 'Ошибка MCP'; mcpStatus.className = 'mcp-status error';
    mcpResult.textContent = readableFetchError(error, 'Не удалось подключить MCP.');
  } finally { mcpRefresh.disabled = false; }
}

mcpRefresh.addEventListener('click', loadMCP);

function renderAgentHistory(messages) {
  agentHistory.replaceChildren();
  if (!messages?.length) {
    const empty = document.createElement('p');
    empty.className = 'hint';
    empty.textContent = 'История диалога появится здесь после первого сообщения.';
    agentHistory.append(empty);
    return;
  }
  messages.forEach((message) => {
    if (message.content?.startsWith('[[phase-transition]]')) return;
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

function statusLabel(task) {
  if (task.status === 'paused') return 'на паузе';
  if (task.status === 'done') return 'завершена';
  return 'выполняется';
}
function dashboardSection(title, value, items) {
  const section = document.createElement('section');
  section.className = 'plan-detail';
  const heading = document.createElement('h3');
  heading.textContent = title;
  section.append(heading);
  if (value) {
    const text = document.createElement('p');
    text.textContent = value;
    section.append(text);
  }
  if (items?.length) {
    const list = document.createElement('ul');
    items.forEach((item) => { const row = document.createElement('li'); row.textContent = item; list.append(row); });
    section.append(list);
  }
  return section;
}

function artifactSection(task) {
  const section = dashboardSection('Итоговый артефакт', task.artifactContent || 'Финальное ТЗ будет сформировано автоматически на последнем этапе, когда все пункты плана закрыты.');
  if (!task.artifactContent) return section;
  const title = document.createElement('p');
  title.className = 'artifact-title';
  title.textContent = task.artifactTitle || 'Итоговое текстовое ТЗ';
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'secondary-button compact-button';
  copy.textContent = 'Скопировать ТЗ';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(task.artifactContent);
      setStatus('Итоговое ТЗ скопировано в буфер обмена.');
    } catch (_) {
      setStatus('Не удалось скопировать ТЗ автоматически. Выделите текст в разделе артефакта.', true);
    }
  });
  section.insertBefore(title, section.querySelector('h3').nextSibling);
  section.append(copy);
  return section;
}

function renderTask(task = {}, tokens = {}) {
  activeTask = task || {};
  const configured = Boolean(task.goal && task.phases?.length);
  taskState.replaceChildren();
  if (!configured) {
    const note = document.createElement('p');
    note.className = 'hint';
    note.textContent = 'Пока нет плана. Отправьте в чат сообщение с «Цель:», «Этапы:» и «Шаг:» — агент-проектировщик создаст дашборд.';
    const example = document.createElement('pre');
    example.className = 'plan-example';
    example.textContent = 'Разработка MVP приложения\n\n- Цель: «Сделать MVP приложения доставки еды».\n- Этапы: planning → execution → validation → done.\n- Шаг: «Собрать список ключевых экранов».\n\nplanning — согласовать границы MVP и экраны.\nexecution — подготовить подробное текстовое ТЗ: сценарии, экраны, данные, API и критерии приёмки.\nvalidation — проверить ТЗ на пробелы, риски и тестовые сценарии.\ndone — выдать финальное согласованное ТЗ.';
    taskState.append(note, example);
  } else {
    const goal = document.createElement('p');
    goal.className = 'plan-goal';
    goal.textContent = task.goal;
    taskState.append(goal);
    const phases = document.createElement('div');
    phases.className = 'task-phases';
    task.phases.forEach((phase, index) => {
      const chip = document.createElement('span');
      chip.className = `task-chip${index === task.phaseIndex ? ' task-chip-current' : ''}`;
      chip.textContent = phase;
      phases.append(chip);
    });
    taskState.append(phases);
    const meta = document.createElement('p');
    meta.className = 'task-meta';
    meta.textContent = `Этап: ${task.phase} · Статус: ${statusLabel(task)}\nТекущий шаг: ${task.currentStep || 'не задан'}\nОжидаемое действие: ${task.expectedAction || 'не задано'}`;
    taskState.append(meta);
    const lifecycle = dashboardSection('Контроль переходов', null, [
      `План: ${task.planApproved ? 'утверждён' : 'ожидает утверждения'}.`,
      `Реализация: ${task.implementationCompleted ? 'готова' : 'не подтверждена'}.`,
      `Валидация: ${task.validationPassed ? 'успешна' : 'не подтверждена'}.`,
    ]);
    taskState.append(
      lifecycle,
      dashboardSection('Текстовое ТЗ', task.specification || 'Агент сформирует ТЗ после первого ответа.'),
      dashboardSection('Открытые вопросы', null, task.openQuestions?.length ? task.openQuestions : ['Нет открытых вопросов.']),
      dashboardSection('Принятые решения', null, task.decisions?.length ? task.decisions : ['Пока нет зафиксированных решений.']),
      dashboardSection('Следующие шаги', null, task.nextSteps?.length ? task.nextSteps : ['Продолжайте диалог с агентом-проектировщиком.']),
      artifactSection(task),
      dashboardSection('Использование токенов', `Последний запрос: ${tokens.requestTokens || 0} входных + ${tokens.responseTokens || 0} выходных. Всего в задаче: ${tokens.cumulativeInputTokens || 0} входных + ${tokens.cumulativeOutputTokens || 0} выходных.`)
    );
  }
  pauseTask.disabled = !configured || task.status !== 'active';
  resumeTask.disabled = !configured || task.status !== 'paused';
  approvePlan.disabled = !configured || task.status !== 'active' || task.phaseIndex !== 0 || task.planApproved;
  completeImplementation.disabled = !configured || task.status !== 'active' || task.phaseIndex !== 1 || task.implementationCompleted;
  passValidation.disabled = !configured || task.status !== 'active' || task.phaseIndex !== 2 || task.validationPassed;
  returnForRework.disabled = !configured || task.phaseIndex === 0 || task.status === 'paused';
  resetTask.disabled = !configured;
  togglePlanner.disabled = false;
  togglePlanner.textContent = window.currentPlannerMode === 'disabled' ? 'Включить планировщик' : 'Отключить планировщик';
}

function renderInvariants(payload) {
  invariantLayers.replaceChildren();
  const groups = [
    ['Текущая задача', payload.task?.taskInvariants || []],
    ['Переходы состояния', payload.task?.stateInvariants || []],
    ['Глобальные', payload.globalInvariants || []],
  ];
  groups.forEach(([title, items]) => {
    const heading = document.createElement('p'); heading.className = 'hint'; heading.textContent = `${title}: ${items.length || 'нет'}`; invariantLayers.append(heading);
    items.forEach((item) => {
      const row = document.createElement('div'); row.className = 'invariant-item';
      const text = document.createElement('span'); const scope = document.createElement('strong'); scope.textContent = `${title}.`; text.append(scope, document.createTextNode(` ${item.rule}`));
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'memory-remove'; remove.textContent = 'Удалить';
      remove.addEventListener('click', async () => {
        try { await patchAgent({ action: 'delete_invariant', invariant: { id: item.id, scope: item.scope } }, 'Не удалось удалить инвариант.'); setStatus('Инвариант удалён.'); }
        catch (error) { setStatus(readableFetchError(error, 'Не удалось удалить инвариант.'), true); }
      });
      row.append(text, remove); invariantLayers.append(row);
    });
  });
}

function renderContextState(payload) {
  if (payload.strategy && strategyDescriptions[payload.strategy]) contextStrategy.value = payload.strategy;
  if (payload.recentMessages) recentMessages.value = payload.recentMessages;
  if (payload.model) {
    selectedAgentModel = payload.model;
    agentModel.value = selectedAgentModel;
  }
  strategyDescription.textContent = strategyDescriptions[contextStrategy.value];
  window.currentPlannerMode = payload.plannerMode || 'enabled';
  renderTask(payload.task, payload.tokens);
  renderInvariants(payload);
  if (payload.memory) renderMemoryLayers(payload.memory);
  renderProfiles(payload);
}

function renderTokenReport(payload) {
  const tokens = payload.tokens;
  if (!tokens || (tokens.requestTokens === 0 && tokens.responseTokens === 0)) return;
  tokenReport.textContent = [
    `Модель: ${modelLabel(payload.model || selectedAgentModel)}`,
    `Вход: ${tokens.requestTokens} токенов`,
    `Выход: ${tokens.responseTokens} токенов`,
    `Всего: ${tokens.requestTokens + tokens.responseTokens} токенов`,
  ].join(' · ');
}

async function readAgentResponse(response) {
  const body = await response.text();
  try { return JSON.parse(body); } catch (_) {
    if (response.status === 404) throw new Error('Сервер запущен в старой версии. Остановите его и снова выполните go run .');
    throw new Error('Сервер вернул ответ в неожиданном формате.');
  }
}
async function patchAgent(command, fallback) {
  const response = await fetch('/api/agent/chat', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, cache: 'no-store', body: JSON.stringify(command) });
  const payload = await readAgentResponse(response);
  if (!response.ok) throw new Error(payload.error || fallback);
  if (command.action === 'pause_task' && !payload.messages) {
    renderTask(payload.task);
    requestLog.textContent = `БРАУЗЕР → BACKEND\nPATCH /api/agent/chat\n${prettyJSON(command)}\n\nПАУЗА → сервер получил сигнал остановки активной работы.`;
    return payload;
  }
  renderContextState(payload);
  renderAgentHistory(payload.messages);
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPATCH /api/agent/chat\n${prettyJSON(command)}`;
  return payload;
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

function memorySection(title, description, items, emptyText) {
  const section = document.createElement('section');
  section.className = 'memory-layer';
  const heading = document.createElement('h3');
  heading.textContent = title;
  const note = document.createElement('p');
  note.className = 'hint';
  note.textContent = description;
  const list = document.createElement('ul');
  if (!items?.length) {
    const item = document.createElement('li');
    item.className = 'hint';
    item.textContent = emptyText;
    list.append(item);
  } else {
    items.forEach((entry) => {
      const item = document.createElement('li');
      const text = document.createElement('span');
      const prefix = entry.category ? `${entry.category}.` : '';
      text.textContent = entry.key?.startsWith('message-') ? entry.value : `${prefix}${entry.key}: ${entry.value}`;
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
    memorySection('Рабочая', 'Данные текущей задачи. Добавляется выбором реплики.', memory.working, 'Нет явно сохранённых данных задачи.'),
    memorySection('Долговременная', 'Глобальные решения и знания пользователя.', memory.longTerm, 'Нет глобальных фактов.')
  );
  clearLongTermMemory.disabled = !memory.longTerm?.length;
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
    await patchAgent({ action, profile }, 'Не удалось сохранить профиль.');
    setStatus('Профиль сохранён. Он будет добавлен к следующему запросу агента.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось сохранить профиль.'), true); }
  finally { saveProfile.disabled = false; }
});

newProfile.addEventListener('click', async () => {
  newProfile.disabled = true;
  try {
    await patchAgent({ action: 'set_active_profile', profileId: '' }, 'Не удалось начать создание профиля.');
    profileName.focus();
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось начать создание профиля.'), true); }
  finally { newProfile.disabled = false; }
});

profileSelect.addEventListener('change', async () => {
  profileSelect.disabled = true;
  try {
    await patchAgent({ action: 'set_active_profile', profileId: profileSelect.value }, 'Не удалось выбрать профиль.');
    setStatus(profileSelect.value ? 'Профиль выбран для этой сессии.' : 'Профиль отключён для этой сессии.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось выбрать профиль.'), true); }
  finally { profileSelect.disabled = false; }
});

deleteProfile.addEventListener('click', async () => {
  if (!activeProfileID || !window.confirm('Удалить выбранный профиль? Он перестанет быть доступен во всех ваших сессиях.')) return;
  try {
    await patchAgent({ action: 'delete_profile', profileId: activeProfileID }, 'Не удалось удалить профиль.');
    setStatus('Профиль удалён.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось удалить профиль.'), true); }
});

async function deleteMemory(entry) {
  try {
    await patchAgent({ action: 'delete_memory', layer: entry.layer, category: entry.category, key: entry.key }, 'Не удалось удалить запись памяти.');
    setStatus('Запись памяти удалена.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось удалить запись памяти.'), true); }
}

clearLongTermMemory.addEventListener('click', async () => {
  if (!window.confirm('Удалить все глобальные решения и знания пользователя?')) return;
  clearLongTermMemory.disabled = true;
  try {
    await patchAgent({ action: 'clear_memory_layer', layer: 'long_term' }, 'Не удалось очистить долговременную память.');
    setStatus('Долговременная память очищена.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось очистить долговременную память.'), true); }
});

invariantForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const rule = invariantRule.value.trim();
  if (!rule) { setStatus('Введите текст инварианта.', true); invariantRule.focus(); return; }
  try {
    await patchAgent({ action: 'save_invariant', invariant: { scope: invariantScope.value, rule } }, 'Не удалось сохранить инвариант.');
    invariantRule.value = '';
    setStatus('Инвариант зафиксирован и будет проверяться в каждом следующем ответе.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось сохранить инвариант.'), true); }
});

async function saveMessageToMemory(value, layer, category) {
  try {
    await patchAgent({ action: 'save_message', layer, category, value }, 'Не удалось сохранить реплику в памяти.');
    setStatus('Реплика сохранена и будет добавлена к следующему запросу агента.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось сохранить реплику в памяти.'), true); }
}

pauseTask.addEventListener('click', async () => {
  try { await patchAgent({ action: 'pause_task' }, 'Не удалось поставить задачу на паузу.'); setStatus('Задача поставлена на паузу.'); }
  catch (error) { setStatus(readableFetchError(error, 'Не удалось изменить состояние задачи.'), true); }
});

async function applyLifecycleAction(action, success) {
  try {
    await patchAgent({ action }, 'Не удалось изменить этап задачи.');
    setStatus(success);
  } catch (error) { setStatus(readableFetchError(error, 'Недопустимый переход состояния.'), true); }
}

approvePlan.addEventListener('click', () => applyLifecycleAction('approve_plan', 'План утверждён: задача перешла к реализации.'));
completeImplementation.addEventListener('click', () => applyLifecycleAction('complete_implementation', 'Реализация подтверждена: задача перешла к валидации.'));
passValidation.addEventListener('click', () => applyLifecycleAction('pass_validation', 'Валидация подтверждена: задача завершена.'));
returnForRework.addEventListener('click', () => applyLifecycleAction('previous_task', 'Задача возвращена на предыдущий этап. Подтверждения следующих этапов сброшены.'));

resumeTask.addEventListener('click', async () => {
  try {
    const payload = await patchAgent({ action: 'resume_task' }, 'Не удалось продолжить задачу.');
    if (payload.pendingMessage) {
      agentMessage.value = payload.pendingMessage;
      setStatus('Задача продолжена: проектировщик возобновляет остановленный запрос…');
      agentForm.requestSubmit();
      return;
    }
    setStatus('Задача продолжена с сохранённого этапа и шага.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось продолжить задачу.'), true); }
});

resetTask.addEventListener('click', async () => {
  if (!window.confirm('Сбросить состояние задачи? История диалога останется.')) return;
  try { await patchAgent({ action: 'reset_task' }, 'Не удалось сбросить задачу.'); setStatus('Состояние задачи сброшено.'); }
  catch (error) { setStatus(readableFetchError(error, 'Не удалось сбросить задачу.'), true); }
});

togglePlanner.addEventListener('click', async () => {
  const disabled = window.currentPlannerMode === 'disabled';
  try {
    await patchAgent({ action: 'set_planner_mode', plannerMode: disabled ? 'enabled' : 'disabled' }, 'Не удалось изменить режим планировщика.');
    setStatus(disabled ? 'Планировщик включён: следующие ответы снова будут обновлять ТЗ и этапы.' : 'Планировщик отключён: агент отвечает как обычный чат, план сохранён.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось изменить режим планировщика.'), true); }
});

async function loadAgentModels() {
  try {
    const response = await fetch('/api/agent/models', { cache: 'no-store' });
    const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось загрузить список моделей.');
    agentModel.replaceChildren();
    (payload.models || []).forEach((model) => {
      const option = document.createElement('option'); option.value = model; option.textContent = modelLabel(model); agentModel.append(option);
    });
    if (!agentModel.options.length) throw new Error('DeepSeek не вернул доступных моделей для чата.');
    if (![...agentModel.options].some((option) => option.value === selectedAgentModel)) selectedAgentModel = agentModel.options[0].value;
    agentModel.value = selectedAgentModel; agentModel.disabled = false;
    modelDescription.textContent = 'Выбор сохраняется для текущего браузерного чата и применяется к следующему сообщению.';
  } catch (error) { modelDescription.textContent = readableFetchError(error, 'Не удалось загрузить список моделей.'); }
}
agentModel.addEventListener('change', async () => {
  const previous = selectedAgentModel; selectedAgentModel = agentModel.value; agentModel.disabled = true;
  try { await patchAgent({ action: 'set_model', model: selectedAgentModel }, 'Не удалось сменить модель.'); setStatus(`Выбрана модель: ${modelLabel(selectedAgentModel)}.`); }
  catch (error) { selectedAgentModel = previous; agentModel.value = previous; setStatus(readableFetchError(error, 'Не удалось сменить модель.'), true); }
  finally { agentModel.disabled = false; }
});
async function saveStrategy() { return patchAgent({ action: 'set_strategy', strategy: contextStrategy.value }, 'Не удалось изменить стратегию.'); }
contextStrategy.addEventListener('change', async () => {
  strategyDescription.textContent = strategyDescriptions[contextStrategy.value];
  try { await saveStrategy(); setStatus('Стратегия контекста обновлена.'); }
  catch (error) { setStatus(readableFetchError(error, 'Не удалось изменить стратегию.'), true); }
});
recentMessages.addEventListener('change', () => {
  const n = Number(recentMessages.value);
  if (!Number.isInteger(n) || n < 2 || n > 40) { setStatus('N должен быть целым числом от 2 до 40.', true); recentMessages.focus(); }
});

async function loadAgentHistory() {
  try {
    const response = await fetch('/api/agent/chat', { cache: 'no-store' }); const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось загрузить чат.');
    renderAgentHistory(payload.messages); renderContextState(payload); renderTokenReport(payload);
  } catch (error) { chatStatus.textContent = readableFetchError(error, 'Не удалось загрузить чат.'); }
}
agentForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = agentMessage.value.trim(); const n = Number(recentMessages.value);
  if (!message) { setStatus('Введите сообщение для агента.', true); agentMessage.focus(); return; }
  if (!Number.isInteger(n) || n < 2 || n > 40) { setStatus('N должен быть целым числом от 2 до 40.', true); recentMessages.focus(); return; }
  agentSubmit.disabled = true;
  setStatus(activeTask.goal ? 'Агент проектирует… Можно нажать «Пауза», чтобы остановить работу.' : 'Агент отвечает…');
  const requestBody = { message, recentMessages: n, strategy: contextStrategy.value, model: selectedAgentModel };
  requestLog.textContent = `БРАУЗЕР → BACKEND\nPOST /api/agent/chat\n${prettyJSON(requestBody)}\n\nОжидание ответа…`;
  try {
    const response = await fetch('/api/agent/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, cache: 'no-store', body: JSON.stringify(requestBody) });
    const payload = await readAgentResponse(response);
    requestLog.textContent += `\n\nBACKEND → БРАУЗЕР\n${prettyJSON({
      httpStatus: response.status,
      taskState: payload.task,
      modelCallSkipped: payload.task?.status === 'paused',
      contextSentToModel: payload.requestMessages,
      mcpToolExecutions: payload.toolExecutions || [],
    })}`;
    if (!response.ok) throw new Error(payload.error || 'Не удалось получить ответ агента.');
    requestLog.textContent += `\n\nОТВЕТ АГЕНТА\n${payload.answer}`;
    renderAgentHistory(payload.messages); renderContextState(payload); renderTokenReport(payload);
    agentMessage.value = ''; agentMessage.focus();
    setStatus(payload.task?.status === 'paused' ? 'Задача на паузе: запрос к модели не выполнялся.' : 'Готово.');
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось получить ответ агента.'), true); }
  finally { agentSubmit.disabled = false; }
});
clearAgentHistory.addEventListener('click', async () => {
  if (!window.confirm('Удалить всю историю этого чата? Это действие нельзя отменить.')) return;
  clearAgentHistory.disabled = true;
  try {
    const response = await fetch('/api/agent/chat', { method: 'DELETE', cache: 'no-store' }); const payload = await readAgentResponse(response);
    if (!response.ok) throw new Error(payload.error || 'Не удалось удалить историю.');
    renderAgentHistory(payload.messages); renderContextState(payload); tokenReport.textContent = 'История очищена. Следующий запрос покажет точный расход токенов.'; setStatus('Вся история удалена.'); agentMessage.focus();
  } catch (error) { setStatus(readableFetchError(error, 'Не удалось удалить историю.'), true); }
  finally { clearAgentHistory.disabled = false; }
});

renderTask({});
loadAgentHistory();
loadAgentModels();
loadMCP();
