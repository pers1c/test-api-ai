/* ============================================================
   AI API Tester — client-side SPA logic
   ============================================================ */

(() => {
  'use strict';

  // Состояние
  const state = {
    currentView: 'new',
    activeRunId: null,
    eventSource: null,
    stages: { parsing: false, generating: false, executing: false },
  };

  // Утилиты DOM
  const $  = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls)  e.className   = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  // Инициализация
  document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupFileDrop();
    setupAuthToggle();
    setupForm();
    setupRunsList();
    setupReportView();
    setupEstimate();
    checkHealth();
  });

  // Навигация между вкладками
  function setupTabs() {
    $$('.tab').forEach(tab => {
      tab.addEventListener('click', () => switchView(tab.dataset.view));
    });
  }

  function switchView(view) {
    state.currentView = view;
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
    $('#view-new').hidden    = view !== 'new';
    $('#view-runs').hidden   = view !== 'runs';
    $('#view-report').hidden = view !== 'report';

    if (view === 'runs') loadRunsList();
  }

  // Health-check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      const dot  = $('#health-indicator .health-dot');
      const text = $('#health-indicator .health-text');
      if (data.gptunnel_key_present) {
        dot.classList.add('ok');
        text.textContent = 'готов';
      } else {
        dot.classList.add('err');
        text.textContent = 'нет ключа';
        $('#health-indicator').title = 'GPTUNNEL_API_KEY не задан в переменных окружения сервера';
      }
    } catch {
      $('#health-indicator .health-text').textContent = 'offline';
    }
  }

  // File drop
  function setupFileDrop() {
    const drop = $('#file-drop');
    const input = $('#spec-file');
    const text = $('#file-drop-text');

    const updateDisplay = () => {
      if (input.files && input.files.length) {
        text.textContent = input.files[0].name;
        text.classList.add('selected');
      } else {
        text.textContent = 'Перетащите файл или нажмите, чтобы выбрать';
        text.classList.remove('selected');
      }
    };

    input.addEventListener('change', updateDisplay);

    ['dragenter', 'dragover'].forEach(ev => {
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.add('dragging');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.remove('dragging');
      });
    });
    drop.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        updateDisplay();
      }
    });
  }

  // Auth toggle (показывать header_name только для api_key)
  function setupAuthToggle() {
    const sel = $('#auth-type');
    const field = $('#auth-header-field');
    const toggle = () => {
      field.style.display = sel.value === 'api_key' ? '' : 'none';
    };
    sel.addEventListener('change', toggle);
    toggle();
  }

  // Submit формы
  function setupForm() {
    $('#run-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const btn  = $('#submit-btn');
      btn.disabled = true;
      btn.querySelector('.btn-label').textContent = 'Запуск…';

      const fd = new FormData(form);
      // Checkbox без галочки в FormData не попадает — указываем явно
      if (!fd.has('include_negative'))   fd.set('include_negative',   'false');
      else                                fd.set('include_negative',   'true');
      if (!fd.has('include_edge_cases')) fd.set('include_edge_cases', 'false');
      else                                fd.set('include_edge_cases', 'true');

      try {
        const res = await fetch('/api/runs', { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Ошибка создания прогона');
        }
        const { run_id } = await res.json();
        startLiveView(run_id);
      } catch (err) {
        alert('Ошибка: ' + err.message);
        btn.disabled = false;
        btn.querySelector('.btn-label').textContent = 'Запустить';
      }
    });
  }

  // Live-панель и SSE
  function startLiveView(runId) {
    state.activeRunId = runId;
    state.stages = { parsing: false, generating: false, executing: false };
    state.costEstimate = null;
    state.costActual = null;

    $('#live-panel').hidden = false;
    $('#live-summary').hidden = true;
    $('#live-log').innerHTML = '';
    $('#live-run-id').textContent = runId;
    updateStatus('running', 'в процессе');

    // Сбрасываем стадии
    $$('.stage').forEach(s => s.classList.remove('active', 'done'));

    // Подписываемся на SSE
    if (state.eventSource) state.eventSource.close();
    state.eventSource = new EventSource(`/api/runs/${runId}/events`);

    state.eventSource.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        handleEvent(event);
      } catch (err) {
        console.error('Bad SSE payload', err, e.data);
      }
    };

    state.eventSource.onerror = () => {
      // Браузер сам попробует переподключиться — не закрываем здесь
      console.warn('SSE connection error (auto-reconnect)');
    };

    // Разрешаем форму снова после завершения (обрабатывается в complete/error)
  }

  function handleEvent(event) {
    const log = $('#live-log');
    const entry = el('div', `log-entry ${event.type}`);

    const time = new Date((event.timestamp || Date.now()/1000) * 1000);
    const hh = String(time.getHours()).padStart(2, '0');
    const mm = String(time.getMinutes()).padStart(2, '0');
    const ss = String(time.getSeconds()).padStart(2, '0');
    entry.appendChild(el('span', 'log-time', `${hh}:${mm}:${ss}`));
    entry.appendChild(el('span', 'log-icon'));

    const msg = el('span', 'log-message');

    // Специально рендерим progress с цветным статусом
    if (event.type === 'progress' && event.data && event.data.status) {
      msg.textContent = event.message.replace(/:\s*\w+$/, ': ') || event.message;
      const statusSpan = el('span', `status-${event.data.status}`, event.data.status);
      msg.appendChild(statusSpan);
    } else {
      msg.textContent = event.message || '';
    }
    entry.appendChild(msg);
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;

    // Обновляем стадии
    if (event.type === 'stage' && event.data && event.data.stage) {
      advanceStage(event.data.stage);
    }

    // Накапливаем cost-данные из log-событий, эмитящих их в поле data
    if (event.type === 'log' && event.data) {
      if (event.data.estimate) state.costEstimate = event.data.estimate;
      if (event.data.actual)   state.costActual   = event.data.actual;
    }

    // Завершение
    if (event.type === 'complete') {
      updateStatus('complete', 'завершено');
      Object.keys(state.stages).forEach(s => {
        $(`.stage[data-stage="${s}"]`).classList.add('done');
      });
      renderSummary(event.data && event.data.summary);
      $('#submit-btn').disabled = false;
      $('#submit-btn .btn-label').textContent = 'Запустить';
      if (state.eventSource) state.eventSource.close();
    }

    if (event.type === 'error') {
      updateStatus('error', 'ошибка');
      $('#submit-btn').disabled = false;
      $('#submit-btn .btn-label').textContent = 'Запустить';
      if (state.eventSource) state.eventSource.close();
    }
  }

  function advanceStage(stage) {
    // Помечаем предыдущие стадии завершёнными
    const order = ['parsing', 'generating', 'executing'];
    const idx = order.indexOf(stage);
    order.forEach((s, i) => {
      const node = $(`.stage[data-stage="${s}"]`);
      if (!node) return;
      if (i < idx) {
        node.classList.add('done');
        node.classList.remove('active');
      } else if (i === idx) {
        node.classList.add('active');
        state.stages[s] = true;
      }
    });
  }

  function updateStatus(cls, text) {
    const dot = $('#live-dot');
    dot.className = 'live-dot ' + cls;
    $('#live-status-text').textContent = text;
  }

  function renderSummary(summary) {
    if (!summary) return;
    const container = $('#summary-stats');
    container.innerHTML = '';

    const stats = [
      { label: 'Всего',    value: summary.total,   cls: '' },
      { label: 'Успех',    value: summary.passed,  cls: 'ok' },
      { label: 'Провал',   value: summary.failed,  cls: summary.failed ? 'err' : '' },
      { label: 'Ошибки',   value: summary.errors,  cls: summary.errors ? 'err' : '' },
      { label: 'Время',    value: (summary.duration_ms / 1000).toFixed(1) + 'с', cls: '' },
    ];
    stats.forEach(s => {
      const wrap = el('div', 'stat');
      wrap.appendChild(el('div', 'stat-value ' + s.cls, s.value));
      wrap.appendChild(el('div', 'stat-label', s.label));
      container.appendChild(wrap);
    });

    $('#live-summary').hidden = false;

    // Сравнение факт vs оценка
    renderCostComparison();

    $('#view-report-btn').onclick = () => showReport(state.activeRunId);
    $('#download-report-btn').href = `/api/runs/${state.activeRunId}/report/download`;
  }

  function renderCostComparison() {
    // Удаляем прошлую версию, если была
    const existing = $('#cost-comparison');
    if (existing) existing.remove();

    const est = state.costEstimate;
    const act = state.costActual;
    // Если нет фактических данных — не показываем блок сравнения
    if (!act) return;

    const container = el('div', 'cost-comparison');
    container.id = 'cost-comparison';

    // Ячейка "Ожидалось"
    const estCell = el('div', 'cost-comparison-cell');
    estCell.appendChild(el('div', 'cost-comparison-label', 'Ожидалось'));
    if (est && est.price_known) {
      estCell.appendChild(el('div', 'cost-comparison-value', window._formatRub(est.cost_rub)));
      estCell.appendChild(el('div', 'cost-comparison-delta',
        `~${est.total_tokens.toLocaleString('ru-RU')} токенов`));
    } else {
      estCell.appendChild(el('div', 'cost-comparison-value', '—'));
    }
    container.appendChild(estCell);

    // Ячейка "Факт"
    const actCell = el('div', 'cost-comparison-cell');
    actCell.appendChild(el('div', 'cost-comparison-label', 'Факт'));
    if (act.price_known) {
      actCell.appendChild(el('div', 'cost-comparison-value actual', window._formatRub(act.cost_rub)));
    } else {
      actCell.appendChild(el('div', 'cost-comparison-value actual',
        `${act.total_tokens.toLocaleString('ru-RU')} токенов`));
    }
    // Если есть и оценка и факт — показываем отклонение
    if (est && est.price_known && act.price_known && est.cost_rub > 0) {
      const delta = ((act.cost_rub - est.cost_rub) / est.cost_rub) * 100;
      const sign = delta > 0 ? '+' : '';
      actCell.appendChild(el('div', 'cost-comparison-delta',
        `${sign}${delta.toFixed(0)}% от оценки`));
    } else if (act.total_tokens) {
      actCell.appendChild(el('div', 'cost-comparison-delta',
        `${act.total_tokens.toLocaleString('ru-RU')} токенов`));
    }
    container.appendChild(actCell);

    $('#live-summary').appendChild(container);
  }

  // Список прогонов
  function setupRunsList() {
    $('#refresh-runs').addEventListener('click', loadRunsList);
  }

  async function loadRunsList() {
    const container = $('#runs-list');
    container.innerHTML = '<div class="empty">Загрузка…</div>';

    try {
      const res = await fetch('/api/runs');
      const data = await res.json();
      renderRunsList(data.runs || []);
    } catch (err) {
      container.innerHTML = `<div class="empty">Ошибка загрузки: ${err.message}</div>`;
    }
  }

  function renderRunsList(runs) {
    const container = $('#runs-list');
    if (!runs.length) {
      container.innerHTML = '<div class="empty">Прогонов пока нет. Запустите первый.</div>';
      return;
    }
    container.innerHTML = '';

    runs.forEach(run => {
      const card = el('div', 'run-card');
      card.addEventListener('click', () => showReport(run.run_id));

      card.appendChild(el('span', `run-status-badge ${run.status}`, run.status));

      const info = el('div', 'run-info');
      info.appendChild(el('div', 'run-url', run.base_url));
      const created = run.created_at ? new Date(run.created_at).toLocaleString('ru-RU') : '';
      info.appendChild(el('div', 'run-meta', `${run.spec_filename} · ${run.model} · ${created}`));
      card.appendChild(info);

      const summary = el('div', 'run-summary');
      if (run.summary) {
        summary.appendChild(el('span', 'passed', `✓ ${run.summary.passed}`));
        summary.appendChild(el('span', 'failed', `✕ ${run.summary.failed}`));
        if (run.summary.errors) summary.appendChild(el('span', 'errors', `! ${run.summary.errors}`));
      } else if (run.error) {
        summary.appendChild(el('span', 'errors', run.error.slice(0, 50)));
      }
      card.appendChild(summary);

      // Стоимость: показываем факт если есть, иначе оценку
      if (run.cost) {
        const costWrap = el('div', 'run-cost');
        const actual = run.cost.actual;
        const estimated = run.cost.estimated;
        if (actual && actual.price_known) {
          costWrap.appendChild(el('div', '', window._formatRub(actual.cost_rub)));
          if (estimated) {
            costWrap.appendChild(el('div', 'run-cost-estimate',
              `оц. ${window._formatRub(estimated.cost_rub)}`));
          }
        } else if (estimated && estimated.price_known) {
          costWrap.appendChild(el('div', 'run-cost-estimate',
            `~${window._formatRub(estimated.cost_rub)}`));
        }
        card.appendChild(costWrap);
      } else {
        card.appendChild(el('span', ''));  // плейсхолдер для grid
      }

      card.appendChild(el('span', 'run-arrow', '→'));
      container.appendChild(card);
    });
  }

  // Отчёт — детальный просмотр
  function setupReportView() {
    $('#back-to-runs').addEventListener('click', () => switchView('runs'));
  }

  async function showReport(runId) {
    switchView('report');
    const container = $('#report-tests');
    const summaryBox = $('#report-summary');
    container.innerHTML = '<div class="empty">Загрузка отчёта…</div>';
    summaryBox.innerHTML = '';
    $('#report-title').textContent = runId;
    $('#report-download').href = `/api/runs/${runId}/report/download`;

    try {
      const res = await fetch(`/api/runs/${runId}/report`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Отчёт не найден');
      }
      const report = await res.json();
      renderReport(report);
    } catch (err) {
      container.innerHTML = `<div class="empty">${err.message}</div>`;
    }
  }

  function renderReport(report) {
    // Заголовок
    $('#report-title').textContent = report.base_url;

    // Сводка
    const summaryBox = $('#report-summary');
    summaryBox.innerHTML = '';
    const s = report.summary;
    const stats = [
      { label: 'Всего',   value: s.total,  cls: '' },
      { label: 'Успех',   value: s.passed, cls: 'ok' },
      { label: 'Провал',  value: s.failed, cls: s.failed ? 'err' : '' },
      { label: 'Ошибки',  value: s.errors, cls: s.errors ? 'err' : '' },
      { label: 'Пропуск', value: s.skipped, cls: '' },
      { label: 'Время',   value: (s.duration_ms / 1000).toFixed(1) + 'с', cls: '' },
    ];
    stats.forEach(st => {
      const wrap = el('div', 'stat');
      wrap.appendChild(el('div', 'stat-value ' + st.cls, st.value));
      wrap.appendChild(el('div', 'stat-label', st.label));
      summaryBox.appendChild(wrap);
    });

    // Стоимость (если есть в отчёте)
    if (report.cost && report.cost.actual && report.cost.actual.price_known) {
      const wrap = el('div', 'stat');
      wrap.appendChild(el('div', 'stat-value', window._formatRub(report.cost.actual.cost_rub)));
      wrap.appendChild(el('div', 'stat-label', 'Стоимость'));
      summaryBox.appendChild(wrap);
    }

    // Тесты
    const container = $('#report-tests');
    container.innerHTML = '';
    (report.test_cases || []).forEach(tc => {
      const card = el('div', 'test-card');

      const header = el('div', 'test-card-header');
      header.appendChild(el('span', 'test-card-id', tc.test_case_id));
      header.appendChild(el('span', 'test-card-type', tc.type));
      header.appendChild(el('span', 'test-card-name', tc.test_case_name));
      header.appendChild(el('span', 'test-card-duration', `${Math.round(tc.duration_ms)}мс`));
      header.appendChild(el('span', `test-card-status ${tc.status}`, tc.status));
      header.addEventListener('click', () => card.classList.toggle('open'));
      card.appendChild(header);

      const body = el('div', 'test-card-body');
      (tc.steps_results || []).forEach(step => {
        body.appendChild(renderStep(step));
      });
      card.appendChild(body);

      container.appendChild(card);
    });

    if (!(report.test_cases || []).length) {
      container.innerHTML = '<div class="empty">Тест-кейсов в отчёте нет.</div>';
    }
  }

  function renderStep(step) {
    const wrap = el('div', 'test-step');

    const header = el('div', 'step-header');
    header.appendChild(el('span', `step-method ${step.method}`, step.method));
    header.appendChild(el('span', 'step-endpoint', step.endpoint));

    if (step.skipped) {
      header.appendChild(el('span', 'step-status', 'пропущен'));
    } else {
      const statusText = `${step.expected_status} / ${step.actual_status ?? '—'}`;
      header.appendChild(el('span', `step-status ${step.passed ? 'ok' : 'err'}`, statusText));
    }
    wrap.appendChild(header);

    wrap.appendChild(el('div', 'step-description', step.step_description));

    // Тело запроса — если есть
    if (step.request_body) {
      const det = el('div', 'step-details');
      det.appendChild(el('span', 'step-details-label', 'Запрос'));
      det.appendChild(document.createTextNode(JSON.stringify(step.request_body, null, 2)));
      wrap.appendChild(det);
    }

    // Ответ — если есть
    if (step.response_body !== null && step.response_body !== undefined) {
      const det = el('div', 'step-details');
      det.appendChild(el('span', 'step-details-label', 'Ответ'));
      const body = typeof step.response_body === 'string'
        ? step.response_body
        : JSON.stringify(step.response_body, null, 2);
      det.appendChild(document.createTextNode(body));
      wrap.appendChild(det);
    }

    if (step.error_message) {
      const det = el('div', 'step-details');
      det.appendChild(el('span', 'step-details-label', 'Ошибка'));
      det.appendChild(document.createTextNode(step.error_message));
      wrap.appendChild(det);
    }

    return wrap;
  }

  // ============================================================
  // Калькулятор стоимости
  // ============================================================
  let estimateDebounce = null;

  function setupEstimate() {
    // Запускаем оценку при изменении файла, модели или max_tests
    ['#spec-file', '#model', '#max-tests'].forEach(sel => {
      const node = $(sel);
      if (!node) return;
      node.addEventListener('change', scheduleEstimate);
    });
    // Ручная кнопка обновления
    $('#estimate-refresh').addEventListener('click', (e) => {
      e.preventDefault();
      triggerEstimate();
    });
  }

  function scheduleEstimate() {
    clearTimeout(estimateDebounce);
    estimateDebounce = setTimeout(triggerEstimate, 300);
  }

  async function triggerEstimate() {
    const file = $('#spec-file').files[0];
    const panel = $('#estimate-panel');
    if (!file) {
      panel.hidden = true;
      return;
    }

    panel.hidden = false;
    $('#estimate-price').textContent = 'считаем…';
    $('#estimate-price').className = 'estimate-price';
    $('#estimate-meta').textContent = '';

    const fd = new FormData();
    fd.append('spec', file);
    const model = $('#model').value;
    if (model) fd.append('model', model);
    const maxTests = $('#max-tests').value;
    if (maxTests) fd.append('max_tests', maxTests);

    try {
      const res = await fetch('/api/estimate', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'ошибка оценки' }));
        $('#estimate-price').textContent = 'ошибка';
        $('#estimate-price').className = 'estimate-price unknown';
        $('#estimate-meta').textContent = err.detail || '';
        return;
      }
      const data = await res.json();
      renderEstimate(data);
    } catch (err) {
      $('#estimate-price').textContent = 'ошибка';
      $('#estimate-price').className = 'estimate-price unknown';
      $('#estimate-meta').textContent = err.message;
    }
  }

  function renderEstimate(data) {
    const est = data.estimate;
    const priceEl = $('#estimate-price');
    const metaEl  = $('#estimate-meta');

    if (est.price_known) {
      priceEl.textContent = formatRub(est.cost_rub);
      priceEl.className = 'estimate-price';
    } else {
      priceEl.textContent = 'цена модели не задана';
      priceEl.className = 'estimate-price unknown';
    }

    const tokens = est.total_tokens.toLocaleString('ru-RU');
    const inputT = est.input_tokens.toLocaleString('ru-RU');
    const outputT = est.output_tokens.toLocaleString('ru-RU');
    metaEl.textContent =
      `${tokens} токенов · ${inputT} вход + ${outputT} выход · ` +
      `модель: ${est.model} · ${data.endpoints} эндпоинтов`;
  }

  function formatRub(value) {
    // 12.345 → "12,35 ₽", 0.045 → "0,05 ₽"
    const digits = value < 1 ? 3 : 2;
    return value.toLocaleString('ru-RU', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }) + ' ₽';
  }

  // Экспортируем formatRub для renderRunsList и renderSummary
  window._formatRub = formatRub;
})();

