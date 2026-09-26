package weathermcp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWeatherToolsPersistAndAggregateObservations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"current_condition":[{"temp_C":"12","FeelsLikeC":"10","humidity":"71","precipMM":"0.4","windspeedKmph":"18","weatherCode":"296","weatherDesc":[{"value":"Moderate rain"}]}]}`))
	}))
	defer server.Close()
	bridge, err := New(context.Background(), Config{DataFile: filepath.Join(t.TempDir(), "weather.json"), Interval: 24 * time.Hour, Endpoint: server.URL, DisableScheduler: true})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()

	result, isError, err := bridge.CallForModel(context.Background(), "collect_weather", map[string]any{})
	if err != nil || isError {
		t.Fatalf("collect: result=%s isError=%v err=%v", result, isError, err)
	}
	if !strings.Contains(result, `"temperatureC":12`) {
		t.Fatalf("unexpected collect result: %s", result)
	}
	schedule, isError, err := bridge.CallForModel(context.Background(), "set_weather_schedule", map[string]any{"interval_seconds": 120})
	if err != nil || isError || !strings.Contains(schedule, `"intervalSeconds":120`) {
		t.Fatalf("schedule: result=%s isError=%v err=%v", schedule, isError, err)
	}
	history, isError, err := bridge.CallForModel(context.Background(), "weather_history", map[string]any{"limit": 10})
	if err != nil || isError || !strings.Contains(history, `"measurementCount":1`) {
		t.Fatalf("history: result=%s isError=%v err=%v", history, isError, err)
	}
	summary, isError, err := bridge.CallForModel(context.Background(), "weather_summary", map[string]any{"period_hours": 1})
	if err != nil || isError {
		t.Fatalf("summary: result=%s isError=%v err=%v", summary, isError, err)
	}
	if !strings.Contains(summary, `"measurementCount":1`) || !strings.Contains(summary, `"precipitationMM":0.4`) {
		t.Fatalf("unexpected summary: %s", summary)
	}
	cleared, isError, err := bridge.CallForModel(context.Background(), "clear_weather_history", map[string]any{"confirm": true})
	if err != nil || isError || !strings.Contains(cleared, `"clearedMeasurements":1`) {
		t.Fatalf("clear: result=%s isError=%v err=%v", cleared, isError, err)
	}
	empty, isError, err := bridge.CallForModel(context.Background(), "weather_summary", map[string]any{"period_hours": 1})
	if err != nil || isError || !strings.Contains(empty, `"measurementCount":0`) {
		t.Fatalf("empty summary: result=%s isError=%v err=%v", empty, isError, err)
	}
}
