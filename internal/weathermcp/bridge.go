// Package weathermcp provides a small scheduled MCP server for Moscow weather.
package weathermcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"ai-challenge-app/internal/models"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const sourceURL = "https://wttr.in/Moscow?format=j1"

type Config struct {
	DataFile         string
	Interval         time.Duration
	HTTPClient       *http.Client
	Endpoint         string
	DisableScheduler bool
}
type Observation struct {
	CollectedAt          time.Time `json:"collectedAt"`
	TemperatureC         float64   `json:"temperatureC"`
	ApparentTemperatureC float64   `json:"apparentTemperatureC"`
	HumidityPercent      int       `json:"humidityPercent"`
	PrecipitationMM      float64   `json:"precipitationMM"`
	WindSpeedKMH         float64   `json:"windSpeedKMH"`
	WeatherCode          int       `json:"weatherCode"`
	Condition            string    `json:"condition"`
}
type schedulerState struct {
	Enabled          bool       `json:"enabled"`
	IntervalSeconds  int        `json:"intervalSeconds"`
	MeasurementCount int        `json:"measurementCount"`
	LastRunAt        *time.Time `json:"lastRunAt,omitempty"`
	NextRunAt        *time.Time `json:"nextRunAt,omitempty"`
	LastStatus       string     `json:"lastStatus"`
	LastError        string     `json:"lastError,omitempty"`
}
type persisted struct {
	Observations []Observation  `json:"observations"`
	Scheduler    schedulerState `json:"scheduler"`
}
type Bridge struct {
	client *mcp.ClientSession
	server *mcp.ServerSession
	cfg    Config
	mu     sync.Mutex
	data   persisted
	cancel context.CancelFunc
	wake   chan struct{}
}
type ToolView struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
	ReadOnly    bool   `json:"readOnly"`
}
type Status struct {
	Connected       bool           `json:"connected"`
	Server          string         `json:"server"`
	IntervalSeconds int            `json:"intervalSeconds"`
	Measurements    int            `json:"measurements"`
	Scheduler       schedulerState `json:"scheduler"`
	Tools           []ToolView     `json:"tools"`
}
type summaryInput struct {
	PeriodHours int `json:"period_hours,omitempty" jsonschema:"period in hours, 1-720; default 24"`
}
type clearHistoryInput struct {
	Confirm bool `json:"confirm" jsonschema:"must be true after explicit user confirmation"`
}
type scheduleInput struct {
	IntervalSeconds int `json:"interval_seconds" jsonschema:"collection interval in seconds, from 60 to 86400"`
}
type historyInput struct {
	Limit int `json:"limit,omitempty" jsonschema:"1-720; default 100"`
}

func New(ctx context.Context, cfg Config) (*Bridge, error) {
	if cfg.DataFile == "" {
		cfg.DataFile = filepath.Join(".local", "weather-observations.json")
	}
	if cfg.Interval <= 0 {
		cfg.Interval = time.Minute
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 20 * time.Second}
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = sourceURL
	}
	b := &Bridge{cfg: cfg}
	_ = b.load()
	if b.data.Scheduler.IntervalSeconds >= 60 && b.data.Scheduler.IntervalSeconds <= 86400 {
		b.cfg.Interval = time.Duration(b.data.Scheduler.IntervalSeconds) * time.Second
	}
	b.data.Scheduler.IntervalSeconds = int(b.cfg.Interval.Seconds())
	server := mcp.NewServer(&mcp.Implementation{Name: "moscow-weather-monitor-mcp", Version: "1.0.0"}, &mcp.ServerOptions{Instructions: "For a current Moscow weather request call collect_weather, then use weather_latest. Use weather_summary for historical aggregates. The server saves observations and collects them periodically."})
	readOnly, falseValue, openWorld := true, false, true
	mcp.AddTool(server, &mcp.Tool{Name: "collect_weather", Title: "Собрать погоду Москвы", Description: "Получает текущую погоду Москвы и сохраняет снимок в JSON-хранилище.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, Observation, error) {
		out, err := b.collect(ctx)
		return nil, out, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "weather_latest", Title: "Последняя погода Москвы", Description: "Возвращает последнее сохранённое измерение погоды Москвы.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		b.mu.Lock()
		defer b.mu.Unlock()
		if len(b.data.Observations) == 0 {
			return nil, map[string]string{"message": "Измерений пока нет."}, nil
		}
		return nil, b.data.Observations[len(b.data.Observations)-1], nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "weather_summary", Title: "Сводка погоды Москвы", Description: "Агрегирует сохранённые измерения температуры, осадков и ветра за выбранный период.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, in summaryInput) (*mcp.CallToolResult, any, error) {
		out, err := b.summary(in.PeriodHours)
		return nil, out, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "scheduler_status", Title: "Статус погодного планировщика", Description: "Показывает период, время последнего и следующего фонового сбора, а также количество сохранённых измерений.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, schedulerState, error) {
		b.mu.Lock()
		defer b.mu.Unlock()
		return nil, b.data.Scheduler, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "set_weather_schedule", Title: "Настроить период сбора погоды", Description: "Меняет период фонового сбора Москвы. Принимает interval_seconds от 60 до 86400 и запускает новый сбор сразу.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, in scheduleInput) (*mcp.CallToolResult, schedulerState, error) {
		if in.IntervalSeconds < 60 || in.IntervalSeconds > 86400 {
			return nil, schedulerState{}, errors.New("interval_seconds должен быть от 60 до 86400")
		}
		b.mu.Lock()
		b.cfg.Interval = time.Duration(in.IntervalSeconds) * time.Second
		b.data.Scheduler.Enabled = true
		b.data.Scheduler.IntervalSeconds = in.IntervalSeconds
		next := time.Now().UTC().Add(b.cfg.Interval)
		b.data.Scheduler.NextRunAt = &next
		err := b.saveLocked()
		state := b.data.Scheduler
		b.mu.Unlock()
		if err != nil {
			return nil, schedulerState{}, err
		}
		b.signalWake()
		return nil, state, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "stop_weather_scheduler", Title: "Остановить сбор погоды", Description: "Останавливает только фоновый сбор погоды Москвы. Уже сохранённая история остаётся доступной.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, schedulerState, error) {
		b.mu.Lock()
		b.data.Scheduler.Enabled = false
		b.data.Scheduler.NextRunAt = nil
		err := b.saveLocked()
		state := b.data.Scheduler
		b.mu.Unlock()
		if err != nil {
			return nil, schedulerState{}, err
		}
		b.signalWake()
		return nil, state, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "weather_history", Title: "История измерений погоды", Description: "Возвращает сохранённые измерения Москвы в хронологическом порядке, включая одинаковые значения.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &falseValue, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, in historyInput) (*mcp.CallToolResult, any, error) {
		if in.Limit == 0 {
			in.Limit = 100
		}
		if in.Limit < 1 || in.Limit > 720 {
			return nil, nil, errors.New("limit должен быть от 1 до 720")
		}
		b.mu.Lock()
		defer b.mu.Unlock()
		start := len(b.data.Observations) - in.Limit
		if start < 0 {
			start = 0
		}
		out := append([]Observation(nil), b.data.Observations[start:]...)
		return nil, map[string]any{"location": "Москва", "measurementCount": len(out), "observations": out}, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "clear_weather_history", Title: "Очистить историю погоды", Description: "Удаляет все сохранённые измерения погоды Москвы после confirm=true. Планировщик продолжает работать.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &openWorld, OpenWorldHint: &falseValue}}, func(_ context.Context, _ *mcp.CallToolRequest, in clearHistoryInput) (*mcp.CallToolResult, any, error) {
		if !in.Confirm {
			return nil, nil, errors.New("очистка истории погоды требует явного confirm=true")
		}
		b.mu.Lock()
		defer b.mu.Unlock()
		cleared := len(b.data.Observations)
		b.data.Observations = nil
		b.data.Scheduler.MeasurementCount = 0
		message := "История погоды очищена."
		if b.data.Scheduler.Enabled {
			message += " Планировщик продолжает сбор."
		} else {
			message += " Планировщик остановлен."
		}
		if err := b.saveLocked(); err != nil {
			return nil, nil, err
		}
		return nil, map[string]any{"clearedMeasurements": cleared, "message": message}, nil
	})
	ct, st := mcp.NewInMemoryTransports()
	ss, err := server.Connect(ctx, st, nil)
	if err != nil {
		return nil, err
	}
	b.server = ss
	client := mcp.NewClient(&mcp.Implementation{Name: "ai-challenge-agent", Version: "1.0.0"}, nil)
	cs, err := client.Connect(ctx, ct, nil)
	if err != nil {
		_ = ss.Close()
		return nil, err
	}
	b.client = cs
	if !cfg.DisableScheduler {
		runCtx, cancel := context.WithCancel(context.Background())
		b.cancel = cancel
		b.wake = make(chan struct{}, 1)
		go b.run(runCtx)
	}
	return b, nil
}
func (b *Bridge) run(ctx context.Context) {
	for {
		b.mu.Lock()
		enabled := b.data.Scheduler.Enabled
		interval := b.cfg.Interval
		b.mu.Unlock()
		if !enabled {
			select {
			case <-ctx.Done():
				return
			case <-b.wake:
				continue
			}
		}
		b.runOnce(ctx)
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-b.wake:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		case <-timer.C:
		}
	}
}
func (b *Bridge) signalWake() {
	if b.wake == nil {
		return
	}
	select {
	case b.wake <- struct{}{}:
	default:
	}
}
func (b *Bridge) runOnce(ctx context.Context) {
	_, err := b.collect(ctx)
	completedAt := time.Now().UTC()
	b.mu.Lock()
	next := completedAt.Add(b.cfg.Interval)
	b.data.Scheduler.LastRunAt = &completedAt
	b.data.Scheduler.IntervalSeconds = int(b.cfg.Interval.Seconds())
	b.data.Scheduler.MeasurementCount = len(b.data.Observations)
	b.data.Scheduler.NextRunAt = &next
	if err != nil {
		b.data.Scheduler.LastStatus = "error"
		b.data.Scheduler.LastError = err.Error()
	} else {
		b.data.Scheduler.LastStatus = "ok"
		b.data.Scheduler.LastError = ""
	}
	_ = b.saveLocked()
	b.mu.Unlock()
}

func (b *Bridge) collect(ctx context.Context) (Observation, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.cfg.Endpoint, nil)
	if err != nil {
		return Observation{}, err
	}
	resp, err := b.cfg.HTTPClient.Do(req)
	if err != nil {
		return Observation{}, fmt.Errorf("получить погоду: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Observation{}, fmt.Errorf("погодный API вернул HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Current []struct {
			TempC       string `json:"temp_C"`
			Feels       string `json:"FeelsLikeC"`
			Humidity    string `json:"humidity"`
			Precip      string `json:"precipMM"`
			Wind        string `json:"windspeedKmph"`
			Code        string `json:"weatherCode"`
			Description []struct {
				Value string `json:"value"`
			} `json:"weatherDesc"`
		} `json:"current_condition"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return Observation{}, err
	}
	if len(payload.Current) == 0 {
		return Observation{}, errors.New("погодный API не вернул текущие данные")
	}
	c := payload.Current[0]
	var out Observation
	if _, err := fmt.Sscan(c.TempC, &out.TemperatureC); err != nil {
		return Observation{}, err
	}
	fmt.Sscan(c.Feels, &out.ApparentTemperatureC)
	fmt.Sscan(c.Humidity, &out.HumidityPercent)
	fmt.Sscan(c.Precip, &out.PrecipitationMM)
	fmt.Sscan(c.Wind, &out.WindSpeedKMH)
	fmt.Sscan(c.Code, &out.WeatherCode)
	out.CollectedAt = time.Now().UTC()
	if len(c.Description) > 0 {
		out.Condition = c.Description[0].Value
	}
	b.mu.Lock()
	b.data.Observations = append(b.data.Observations, out)
	if len(b.data.Observations) > 720 {
		b.data.Observations = b.data.Observations[len(b.data.Observations)-720:]
	}
	b.data.Scheduler.MeasurementCount = len(b.data.Observations)
	err = b.saveLocked()
	b.mu.Unlock()
	return out, err
}
func (b *Bridge) summary(hours int) (any, error) {
	if hours == 0 {
		hours = 24
	}
	if hours < 1 || hours > 720 {
		return nil, errors.New("period_hours должен быть от 1 до 720")
	}
	since := time.Now().UTC().Add(-time.Duration(hours) * time.Hour)
	b.mu.Lock()
	defer b.mu.Unlock()
	var rows []Observation
	for _, o := range b.data.Observations {
		if !o.CollectedAt.Before(since) {
			rows = append(rows, o)
		}
	}
	if len(rows) == 0 {
		return map[string]any{"location": "Москва", "periodHours": hours, "measurementCount": 0, "message": "Нет измерений за выбранный период."}, nil
	}
	min, max, sum := rows[0].TemperatureC, rows[0].TemperatureC, 0.0
	precip, maxWind := 0.0, 0.0
	for _, o := range rows {
		if o.TemperatureC < min {
			min = o.TemperatureC
		}
		if o.TemperatureC > max {
			max = o.TemperatureC
		}
		sum += o.TemperatureC
		precip += o.PrecipitationMM
		if o.WindSpeedKMH > maxWind {
			maxWind = o.WindSpeedKMH
		}
	}
	return map[string]any{"location": "Москва", "periodHours": hours, "measurementCount": len(rows), "temperatureC": map[string]float64{"min": min, "max": max, "average": sum / float64(len(rows)), "trend": rows[len(rows)-1].TemperatureC - rows[0].TemperatureC}, "precipitationMM": precip, "maxWindSpeedKMH": maxWind, "latest": rows[len(rows)-1]}, nil
}
func (b *Bridge) load() error {
	data, err := os.ReadFile(b.cfg.DataFile)
	if errors.Is(err, os.ErrNotExist) {
		b.data.Scheduler.Enabled = true
		return nil
	}
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &b.data); err != nil {
		return err
	}
	// JSON files created before the pause feature did not contain Enabled.
	// Treat them as active to preserve their previous behaviour on upgrade.
	var raw struct {
		Scheduler map[string]json.RawMessage `json:"scheduler"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if _, exists := raw.Scheduler["enabled"]; !exists {
		b.data.Scheduler.Enabled = true
	}
	return nil
}
func (b *Bridge) saveLocked() error {
	if err := os.MkdirAll(filepath.Dir(b.cfg.DataFile), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(b.data, "", "  ")
	if err != nil {
		return err
	}
	tmp := b.cfg.DataFile + ".tmp"
	if err = os.WriteFile(tmp, data, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, b.cfg.DataFile)
}
func (b *Bridge) Close() error {
	if b.cancel != nil {
		b.cancel()
	}
	return errors.Join(b.client.Close(), b.server.Close())
}
func (b *Bridge) Status(ctx context.Context) (Status, error) {
	r, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return Status{}, err
	}
	tools := make([]ToolView, 0, len(r.Tools))
	for _, t := range r.Tools {
		v := ToolView{Name: t.Name, Title: t.Title, Description: t.Description, InputSchema: t.InputSchema}
		if t.Annotations != nil {
			v.ReadOnly = t.Annotations.ReadOnlyHint
		}
		tools = append(tools, v)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return Status{Connected: true, Server: "moscow-weather-monitor-mcp 1.0.0", IntervalSeconds: int(b.cfg.Interval.Seconds()), Measurements: len(b.data.Observations), Scheduler: b.data.Scheduler, Tools: tools}, nil
}
func (b *Bridge) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	r, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	out := make([]models.ToolDefinition, 0, len(r.Tools))
	for _, t := range r.Tools {
		out = append(out, models.ToolDefinition{Type: "function", Function: models.ToolFunction{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}})
	}
	return out, nil
}
func (b *Bridge) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	r, err := b.client.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return "", true, err
	}
	if r.StructuredContent != nil {
		raw, e := json.Marshal(r.StructuredContent)
		return string(raw), r.IsError, e
	}
	return "", r.IsError, nil
}
func sortObservations(rows []Observation) {
	sort.Slice(rows, func(i, j int) bool { return rows[i].CollectedAt.Before(rows[j].CollectedAt) })
}
