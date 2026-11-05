// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// captureBody spins up a fake Anthropic endpoint, points a Model at it via
// BaseURL, fires one non-streaming request and returns the raw JSON body
// the SDK actually put on the wire. This exercises reasoningRequestOptions
// end to end (the option mutates the serialized body via sjson, so the only
// faithful way to assert on it is to inspect the real outgoing request).
func captureBody(t *testing.T, cfg Config) map[string]any {
	t.Helper()
	return captureRequestBody(t, cfg, &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	})
}

// captureRequestBody 同 captureBody,发送调用方给的请求。
func captureRequestBody(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		// Minimal valid Messages response so convertResponse succeeds.
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	if cfg.ModelName == "" {
		cfg.ModelName = "claude-test"
	}
	m := New(cfg)

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if len(captured) == 0 {
		t.Fatalf("server captured no request body")
	}
	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	return body
}

// The adaptive path is the one that fixed the Opus 4.5+ 400. The wire body
// must carry thinking.type=adaptive AND output_config.effort=<level>.
func TestWireBody_Adaptive(t *testing.T) {
	body := captureBody(t, Config{ThinkingEffort: "high"})

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("expected a thinking object, got %v", body["thinking"])
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want \"adaptive\"", thinking["type"])
	}
	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected an output_config object, got %v", body["output_config"])
	}
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want \"high\"", oc["effort"])
	}
}

// The classic path must send thinking.type=enabled with the budget, and must
// NOT carry output_config.effort.
func TestWireBody_Enabled(t *testing.T) {
	body := captureBody(t, Config{ThinkingBudgetTokens: 4096, MaxOutputTokens: 8192})

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("expected a thinking object, got %v", body["thinking"])
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want \"enabled\"", thinking["type"])
	}
	if got, want := thinking["budget_tokens"], float64(4096); got != want {
		t.Errorf("thinking.budget_tokens = %v, want %v", got, want)
	}
	if _, present := body["output_config"]; present {
		t.Errorf("classic mode must not send output_config, got %v", body["output_config"])
	}
}

// 工具 schema 的根级关键字要原样上线:properties 里的 $ref 指向根级 $defs,
// $defs 必须随 input_schema 一起发出,引用才不悬空。
func TestWireBody_ToolSchemaKeepsRootKeywords(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "compose",
				ParametersJsonSchema: map[string]any{
					"type":                 "object",
					"description":          "Compose sections.",
					"additionalProperties": false,
					"required":             []any{"sections"},
					"properties": map[string]any{
						"sections": map[string]any{"type": "array", "items": map[string]any{"$ref": "#/$defs/section"}},
					},
					"$defs": map[string]any{
						"section": map[string]any{"type": "object", "properties": map[string]any{"title": map[string]any{"type": "string"}}},
					},
				},
			}},
		}}},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	body := captureRequestBody(t, Config{}, req)

	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want one tool", body["tools"])
	}
	schema := tools[0].(map[string]any)["input_schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Errorf("input_schema.type = %v, want \"object\"", schema["type"])
	}
	defs, ok := schema["$defs"].(map[string]any)
	if !ok || defs["section"] == nil {
		t.Fatalf("input_schema.$defs = %v, want the section definition", schema["$defs"])
	}
	if schema["additionalProperties"] != false {
		t.Errorf("input_schema.additionalProperties = %v, want false", schema["additionalProperties"])
	}
	if schema["description"] != "Compose sections." {
		t.Errorf("input_schema.description = %v, want the root description", schema["description"])
	}
	items := schema["properties"].(map[string]any)["sections"].(map[string]any)["items"].(map[string]any)
	if items["$ref"] != "#/$defs/section" {
		t.Errorf("sections.items.$ref = %v, want \"#/$defs/section\"", items["$ref"])
	}
}

// With no reasoning configured the body must carry no thinking and no
// output_config at all.
func TestWireBody_NoReasoning(t *testing.T) {
	body := captureBody(t, Config{})

	if v, present := body["thinking"]; present {
		t.Errorf("no reasoning must not send thinking, got %v", v)
	}
	if v, present := body["output_config"]; present {
		t.Errorf("no reasoning must not send output_config, got %v", v)
	}
}
