package main

import (
	"errors"
	"io/fs"
	"testing"
)

func TestLoadDeepSeekAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		fileData string
		readErr  error
		want     string
		wantRead bool
		wantErr  bool
	}{
		{
			name:     "environment takes priority",
			envValue: "from-environment",
			fileData: "DEEPSEEK_API_KEY=from-file",
			want:     "from-environment",
			wantRead: false,
		},
		{
			name:     "file fallback supports comments CRLF and equals",
			fileData: "# local settings\r\nOTHER_KEY=value\r\nDEEPSEEK_API_KEY=from-file=with-equals\r\n",
			want:     "from-file=with-equals",
			wantRead: true,
		},
		{
			name:     "missing file keeps no key behavior",
			readErr:  fs.ErrNotExist,
			wantRead: true,
		},
		{
			name:     "unreadable file returns an error",
			readErr:  errors.New("permission denied"),
			wantRead: true,
			wantErr:  true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wasRead := false
			got, err := loadDeepSeekAPIKey(
				func(name string) string {
					if name != apiKeyEnvVar {
						t.Fatalf("lookup variable = %q, want %q", name, apiKeyEnvVar)
					}
					return test.envValue
				},
				func(path string) ([]byte, error) {
					wasRead = true
					if path != localEnvFile {
						t.Fatalf("read path = %q, want %q", path, localEnvFile)
					}
					return []byte(test.fileData), test.readErr
				},
			)

			if (err != nil) != test.wantErr {
				t.Fatalf("error = %v, want error = %t", err, test.wantErr)
			}
			if got != test.want {
				t.Fatalf("API key = %q, want %q", got, test.want)
			}
			if wasRead != test.wantRead {
				t.Fatalf("file read = %t, want %t", wasRead, test.wantRead)
			}
		})
	}
}
