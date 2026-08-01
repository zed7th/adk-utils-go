// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// minimalResponseJSON is a complete-enough /v1/responses payload for
// convertResponse to succeed.
const minimalResponseJSON = `{
	"id": "resp_1",
	"status": "completed",
	"output": [{
		"type": "message",
		"id": "msg_1",
		"role": "assistant",
		"status": "completed",
		"content": [{"type": "output_text", "text": "ok"}]
	}],
	"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
		"input_tokens_details": {"cached_tokens": 0},
		"output_tokens_details": {"reasoning_tokens": 0}}
}`

// captureBody points a Model at a local fake endpoint via BaseURL, fires one
// non-streaming request, and returns the JSON body the openai-go SDK actually
// put on the wire. Asserting on this (not on the pre-SDK params) is what
// proves the bytes a real Responses API server receives are correct.
func captureBody(t *testing.T, req *model.LLMRequest) map[string]any {
	return captureBodyFor(t, Config{}, req)
}

// captureBodyFor mirrors captureBody but lets the test override Config
// fields (the integration tier needs a real model name for its paid step);
// BaseURL always points at the local fake server.
func captureBodyFor(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, minimalResponseJSON)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "gpt-test"
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

// A replayed tool round-trip must reach the wire with store=false, the
// encrypted-reasoning include, and "{}" (never "null") for nil tool
// payloads: strict OpenAI-compatible parsers reject "null" there.
func TestWireBody_StatelessToolReplay(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "ping the tool"}}},
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "ping"}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "ping"}},
			}},
		},
	}

	body := captureBody(t, req)

	if body["model"] != "gpt-test" {
		t.Errorf("model = %v, want gpt-test", body["model"])
	}
	if body["store"] != false {
		t.Errorf("store = %v, want false", body["store"])
	}
	includes, _ := body["include"].([]any)
	found := false
	for _, inc := range includes {
		if inc == "reasoning.encrypted_content" {
			found = true
		}
	}
	if !found {
		t.Errorf("include = %v, want reasoning.encrypted_content", includes)
	}

	input, _ := body["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input items = %d, want 3", len(input))
	}
	call, _ := input[1].(map[string]any)
	if call["type"] != "function_call" || call["arguments"] != "{}" {
		t.Errorf("function_call item = %v, want arguments {}", call)
	}
	output, _ := input[2].(map[string]any)
	if output["type"] != "function_call_output" || output["output"] != "{}" {
		t.Errorf("function_call_output item = %v, want output {}", output)
	}
}

// Streaming requests must carry stream=true on the wire, and the terminal
// response.completed event must come back as the final aggregated turn.
func TestWireBody_StreamFlag(t *testing.T) {
	// SSE data fields are single-line, so the response JSON is compacted
	// before being embedded in the terminal event.
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(minimalResponseJSON)); err != nil {
		t.Fatalf("compact: %v", err)
	}

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"ok","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":`+compact.String()+"}\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}

	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	if body["stream"] != true {
		t.Errorf("stream = %v, want true", body["stream"])
	}
	if final == nil {
		t.Fatalf("no final (non-partial) response was yielded")
	}
	if got := extractText(final.Content); got != "ok" {
		t.Errorf("final text = %q, want ok", got)
	}
}

// countingTransport records how many requests pass through so tests can
// prove an injected client actually carries the traffic.
type countingTransport struct {
	base http.RoundTripper
	hits int
}

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	c.hits++
	return c.base.RoundTrip(r)
}

// A custom HTTP client injected through HTTPOptions must carry the request:
// consumers rely on it for OAuth transports, proxies, and test servers.
func TestCustomHTTPClientIsUsed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, minimalResponseJSON)
	}))
	defer srv.Close()

	transport := &countingTransport{base: http.DefaultTransport}
	m := New(Config{
		BaseURL:     srv.URL,
		APIKey:      "test-key",
		ModelName:   "gpt-test",
		HTTPOptions: HTTPOptions{Client: &http.Client{Transport: transport}},
	})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if transport.hits != 1 {
		t.Errorf("injected transport hits = %d, want 1", transport.hits)
	}
}

// A terminal event whose payload carries malformed function arguments must
// surface as an error, not degrade into the delta fallback: gateways that
// omit output_item.done would otherwise silently drop the failure.
func TestWireBody_StreamMalformedTerminalArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp_1","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"delete_things","arguments":"{"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatalf("expected an error for malformed terminal arguments, got none")
	}
}

// When a gateway sends complete output_item.done events but a terminal event
// without aggregated output, the fallback must rebuild the turn from those
// items: message IDs, phase, and encrypted reasoning survive, instead of
// degrading to bare delta text.
func TestWireBody_StreamFallbackPrefersDoneItems(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hello","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc-blob"}}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":3,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"hello"}]}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (reasoning + message), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	thought := final.Content.Parts[0]
	if !thought.Thought || thought.PartMetadata["encrypted_content"] != "enc-blob" {
		t.Errorf("first part = %+v, want a thought carrying encrypted content", thought)
	}
	msg := final.Content.Parts[1]
	if msg.Text != "hello" || msg.PartMetadata["message_id"] != "msg_1" || msg.PartMetadata["phase"] != "final_answer" {
		t.Errorf("second part = %+v, want text with message_id and phase", msg)
	}
}

// closeTrackingBody flags when the response body is closed so tests can
// prove the stream does not leak the connection.
type closeTrackingBody struct {
	io.Reader
	closed *bool
}

func (b closeTrackingBody) Close() error {
	*b.closed = true
	return nil
}

type sseTransport struct {
	payload string
	closed  *bool
}

func (s *sseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       closeTrackingBody{Reader: bytes.NewReader([]byte(s.payload)), closed: s.closed},
		Request:    r,
	}, nil
}

// The stream returns early on terminal events, so the response body must be
// closed explicitly; leaving it open leaks the connection.
func TestStreamBodyClosed(t *testing.T) {
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(minimalResponseJSON)); err != nil {
		t.Fatalf("compact: %v", err)
	}
	closed := false
	payload := "event: response.completed\n" +
		`data: {"type":"response.completed","sequence_number":1,"response":` + compact.String() + "}\n\n"
	m := New(Config{
		APIKey:      "test-key",
		ModelName:   "gpt-test",
		BaseURL:     "http://fake.local",
		HTTPOptions: HTTPOptions{Client: &http.Client{Transport: &sseTransport{payload: payload, closed: &closed}}},
	})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if !closed {
		t.Fatalf("stream body was not closed")
	}
}

// Done items that cover only part of the turn (e.g. reasoning) must not
// discard text that streamed as bare deltas: the consumer already saw it,
// so the final event and the next turn's history must keep it.
func TestWireBody_StreamFallbackMergesUncoveredDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hello","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc-blob"}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (reasoning + merged text), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	if !final.Content.Parts[0].Thought || final.Content.Parts[0].PartMetadata["encrypted_content"] != "enc-blob" {
		t.Errorf("first part = %+v, want the reasoning item", final.Content.Parts[0])
	}
	if final.Content.Parts[1].Text != "hello" {
		t.Errorf("second part = %+v, want the merged delta text", final.Content.Parts[1])
	}
}

// Refusal deltas must be accumulated: without a terminal output or done
// item, the fallback would otherwise persist an empty response.
func TestWireBody_StreamRefusalDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.refusal.delta\n")
		io.WriteString(w, `data: {"type":"response.refusal.delta","delta":"I cannot help.","item_id":"msg_1","content_index":0,"output_index":0,"sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	part := final.Content.Parts[0]
	if part.Text != "I cannot help." || part.PartMetadata["refusal"] != true {
		t.Errorf("part = %+v, want refusal text with refusal metadata", part)
	}
}
