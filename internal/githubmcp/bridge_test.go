package githubmcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBridgeListsReadsAndWritesTextFiles(t *testing.T) {
	var wrote map[string]string
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/repos/acme/demo":
			_ = json.NewEncoder(w).Encode(map[string]any{"full_name": "acme/demo", "default_branch": "main", "html_url": "https://github.com/acme/demo"})
		case r.Method == http.MethodGet && r.URL.Path == "/repos/acme/demo/contents/README.md":
			_ = json.NewEncoder(w).Encode(map[string]string{"path": "README.md", "sha": "old-sha", "encoding": "base64", "content": base64.StdEncoding.EncodeToString([]byte("hello"))})
		case r.Method == http.MethodGet && r.URL.Path == "/repos/acme/demo/contents/obsolete.txt":
			_ = json.NewEncoder(w).Encode(map[string]string{"sha": "obsolete-sha"})
		case r.Method == http.MethodGet && r.URL.Path == "/repos/acme/demo/git/trees/main":
			_ = json.NewEncoder(w).Encode(map[string]any{"tree": []map[string]any{{"path": "README.md", "type": "blob", "size": 5}, {"path": "internal", "type": "tree"}}})
		case r.Method == http.MethodPut && r.URL.Path == "/repos/acme/demo/contents/notes.txt":
			if err := json.NewDecoder(r.Body).Decode(&wrote); err != nil {
				t.Fatal(err)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"content": map[string]string{"html_url": "https://github.com/acme/demo/blob/main/notes.txt"}, "commit": map[string]string{"sha": "new-sha"}})
		case r.Method == http.MethodDelete && r.URL.Path == "/repos/acme/demo/contents/obsolete.txt":
			if err := json.NewDecoder(r.Body).Decode(&wrote); err != nil {
				t.Fatal(err)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"commit": map[string]string{"sha": "delete-sha"}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer api.Close()
	bridge, err := New(context.Background(), Config{Token: "test-token", Repository: "acme/demo", Endpoint: api.URL, HTTPClient: api.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()
	status, err := bridge.Status(context.Background())
	if err != nil || !status.Connected || len(status.Tools) != 6 {
		t.Fatalf("status=%#v err=%v", status, err)
	}
	file, err := bridge.Call(context.Background(), "github_get_file", map[string]any{"path": "README.md"})
	if err != nil || file.IsError {
		t.Fatalf("file=%#v err=%v", file, err)
	}
	var read getFileOutput
	data, _ := json.Marshal(file.StructuredContent)
	if err := json.Unmarshal(data, &read); err != nil || read.Content != "hello" || read.SHA != "old-sha" {
		t.Fatalf("read=%#v err=%v", read, err)
	}
	listing, err := bridge.Call(context.Background(), "github_list_files", map[string]any{})
	if err != nil || listing.IsError {
		t.Fatalf("listing=%#v err=%v", listing, err)
	}
	data, _ = json.Marshal(listing.StructuredContent)
	var files listFilesOutput
	if err := json.Unmarshal(data, &files); err != nil || files.Ref != "main" || len(files.Files) != 2 {
		t.Fatalf("files=%#v err=%v", files, err)
	}
	written, err := bridge.Call(context.Background(), "github_put_file", map[string]any{"path": "notes.txt", "content": "saved text", "message": "docs: add notes", "confirm": true})
	if err != nil || written.IsError {
		t.Fatalf("written=%#v err=%v", written, err)
	}
	decoded, _ := base64.StdEncoding.DecodeString(wrote["content"])
	if string(decoded) != "saved text" || wrote["branch"] != "main" {
		t.Fatalf("write body=%#v", wrote)
	}
	deleted, err := bridge.Call(context.Background(), "github_delete_file", map[string]any{"path": "obsolete.txt", "message": "docs: remove obsolete file", "confirm": true})
	if err != nil || deleted.IsError || wrote["sha"] != "obsolete-sha" {
		t.Fatalf("deleted=%#v write body=%#v err=%v", deleted, wrote, err)
	}
}
