package mcpbridge

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBridgeConnectsListsAndCallsTools(t *testing.T) {
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "lesson-16.mp4"), make([]byte, 2*1024*1024), 0o600); err != nil {
		t.Fatal(err)
	}
	bridge, err := New(context.Background(), Config{VideoDir: directory})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()

	status, err := bridge.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !status.Connected || len(status.Tools) != 4 || status.ModelTokensUsedByThisRequest != 0 {
		t.Fatalf("status = %#v", status)
	}
	gotNames := []string{status.Tools[0].Name, status.Tools[1].Name, status.Tools[2].Name, status.Tools[3].Name}
	if strings.Join(gotNames, ",") != "analyze_desktop_video,estimate_video_upload,list_desktop_videos,upload_video_to_yandex" {
		t.Fatalf("tools = %v", gotNames)
	}

	listed, err := bridge.Call(context.Background(), "list_desktop_videos", map[string]any{})
	if err != nil || listed.IsError {
		t.Fatalf("list result = %#v, err = %v", listed, err)
	}
	var listOutput ListVideosOutput
	decodeStructured(t, listed.StructuredContent, &listOutput)
	if len(listOutput.Videos) != 1 || listOutput.Videos[0].Name != "lesson-16.mp4" {
		t.Fatalf("videos = %#v", listOutput.Videos)
	}

	estimated, err := bridge.Call(context.Background(), "estimate_video_upload", map[string]any{
		"videoName": "lesson-16.mp4",
	})
	if err != nil || estimated.IsError {
		t.Fatalf("estimate result = %#v, err = %v", estimated, err)
	}
	var estimateOutput EstimateOutput
	decodeStructured(t, estimated.StructuredContent, &estimateOutput)
	if estimateOutput.VideoBytesSentToModel != 0 || estimateOutput.SizeBytes != 2*1024*1024 || estimateOutput.APIRequestsForUpload != 2 {
		t.Fatalf("estimate = %#v", estimateOutput)
	}
}

func TestUploadToolSendsVideoDirectlyToYandexDisk(t *testing.T) {
	directory := t.TempDir()
	videoBody := []byte("not-a-real-video")
	if err := os.WriteFile(filepath.Join(directory, "demo.mov"), videoBody, 0o600); err != nil {
		t.Fatal(err)
	}
	var gotDiskPath, gotAuthorization string
	var disk *httptest.Server
	disk = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/resources/upload":
			gotDiskPath = r.URL.Query().Get("path")
			gotAuthorization = r.Header.Get("Authorization")
			_ = json.NewEncoder(w).Encode(map[string]string{"href": disk.URL + "/upload-target"})
		case r.Method == http.MethodPut && r.URL.Path == "/upload-target":
			body, _ := io.ReadAll(r.Body)
			if string(body) != string(videoBody) {
				t.Errorf("body = %q", body)
			}
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	}))
	defer disk.Close()

	bridge, err := New(context.Background(), Config{
		OAuthToken: "test-token", RootFolder: "AI Challenge", VideoDir: directory,
		Endpoint: disk.URL, HTTPClient: disk.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()

	result, err := bridge.Call(context.Background(), "upload_video_to_yandex", map[string]any{
		"videoName": "demo.mov", "lessonFolder": "lession 16", "confirm": true,
	})
	if err != nil || result.IsError {
		t.Fatalf("upload result = %#v, err = %v", result, err)
	}
	var output UploadOutput
	decodeStructured(t, result.StructuredContent, &output)
	if gotAuthorization != "OAuth test-token" || gotDiskPath != "AI Challenge/lession 16/demo.mov" {
		t.Fatalf("disk path=%q authorization=%q", gotDiskPath, gotAuthorization)
	}
	if output.DiskPath != "disk:/AI Challenge/lession 16/demo.mov" || output.SizeBytes != int64(len(videoBody)) {
		t.Fatalf("output = %#v", output)
	}
}

func TestEstimateReadsYandexDiskQuota(t *testing.T) {
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "demo.mp4"), make([]byte, 1024), 0o600); err != nil {
		t.Fatal(err)
	}
	disk := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "OAuth test-token" {
			t.Errorf("authorization = %q", r.Header.Get("Authorization"))
		}
		_ = json.NewEncoder(w).Encode(map[string]int64{"total_space": 10_000, "used_space": 2_000})
	}))
	defer disk.Close()
	bridge, err := New(context.Background(), Config{OAuthToken: "test-token", VideoDir: directory, Endpoint: disk.URL, HTTPClient: disk.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()
	result, err := bridge.Call(context.Background(), "estimate_video_upload", map[string]any{"videoName": "demo.mp4"})
	if err != nil || result.IsError {
		t.Fatalf("estimate result = %#v, err = %v", result, err)
	}
	var output EstimateOutput
	decodeStructured(t, result.StructuredContent, &output)
	if output.DiskFreeBytes != 8_000 || output.FitsAvailableSpace == nil || !*output.FitsAvailableSpace {
		t.Fatalf("output = %#v", output)
	}
}

func TestUploadTreatsSameExistingDiskFileAsSuccess(t *testing.T) {
	directory := t.TempDir()
	body := []byte("same-video")
	if err := os.WriteFile(filepath.Join(directory, "demo.mov"), body, 0o600); err != nil {
		t.Fatal(err)
	}
	disk := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/resources/upload":
			w.WriteHeader(http.StatusConflict)
			_, _ = io.WriteString(w, `{"error":"DiskResourceAlreadyExistsError"}`)
		case "/resources":
			_ = json.NewEncoder(w).Encode(map[string]any{"type": "file", "size": len(body)})
		default:
			http.NotFound(w, r)
		}
	}))
	defer disk.Close()
	bridge, err := New(context.Background(), Config{OAuthToken: "test-token", RootFolder: "AI Challenge", VideoDir: directory, Endpoint: disk.URL, HTTPClient: disk.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()
	result, err := bridge.Call(context.Background(), "upload_video_to_yandex", map[string]any{
		"videoName": "demo.mov", "lessonFolder": "lession 16", "confirm": true,
	})
	if err != nil || result.IsError {
		t.Fatalf("upload result = %#v, err = %v", result, err)
	}
	var output UploadOutput
	decodeStructured(t, result.StructuredContent, &output)
	if !output.AlreadyExisted || output.DiskPath != "disk:/AI Challenge/lession 16/demo.mov" {
		t.Fatalf("output = %#v", output)
	}
}

func decodeStructured(t *testing.T, value any, target any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatal(err)
	}
}
