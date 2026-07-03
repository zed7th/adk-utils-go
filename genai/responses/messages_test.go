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
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// convertContentToInputItems bridges ADK Content into Responses API typed
// input items. A single Content may produce multiple items: text/media
// coalesce into a message, while FunctionCall and FunctionResponse become
// separate typed items.
func TestConvertContentToInputItems(t *testing.T) {
	cases := []struct {
		name      string
		content   *genai.Content
		wantCount int
		wantTypes []string // expected item type discriminators
		assert    func(t *testing.T, items []responses.ResponseInputItemUnionParam)
	}{
		{
			name: "user text becomes a single message item",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello"}},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected EasyInputMessage, got %+v", items[0])
				}
			},
		},
		{
			name: "model text becomes an assistant message",
			content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hi back"}},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected EasyInputMessage, got %+v", items[0])
				}
				if items[0].OfMessage.Role != responses.EasyInputMessageRoleAssistant {
					t.Errorf("role = %q, want assistant", items[0].OfMessage.Role)
				}
			},
		},
		{
			name: "function response becomes a function_call_output item",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_42",
						Response: map[string]any{"ok": true},
					}},
				},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfFunctionCallOutput == nil {
					t.Errorf("expected FunctionCallOutput, got %+v", items[0])
				}
			},
		},
		{
			name: "model text plus function call produce two items",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "calling tool"},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_xyz",
						Name: "do_thing",
						Args: map[string]any{"foo": "bar"},
					}},
				},
			},
			wantCount: 2,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("first item should be a message")
				}
				if items[1].OfFunctionCall == nil {
					t.Errorf("second item should be a function call")
				}
				if items[1].OfFunctionCall.Name != "do_thing" {
					t.Errorf("function name = %q, want do_thing", items[1].OfFunctionCall.Name)
				}
			},
		},
		{
			name: "user text plus inline image produces a single multi-part message",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "describe this"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("fakepng")}},
				},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected message item")
				}
			},
		},
		{
			name: "function response then text produce two items",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "tool_1", Response: map[string]any{"result": "ok"}}},
					{Text: "all done"},
				},
			},
			wantCount: 2,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfFunctionCallOutput == nil {
					t.Errorf("first item should be function_call_output")
				}
				if items[1].OfMessage == nil {
					t.Errorf("second item should be a message")
				}
			},
		},
		{
			name: "empty parts produce no items",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{}},
			},
			wantCount: 0,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertContentToInputItems(c.content)
			if err != nil {
				t.Fatalf("convertContentToInputItems error: %v", err)
			}
			if len(got) != c.wantCount {
				t.Fatalf("item count = %d, want %d", len(got), c.wantCount)
			}
			if c.assert != nil {
				c.assert(t, got)
			}
		})
	}
}

// Model output with phase metadata must use ResponseOutputMessageParam
// to preserve the phase field across turns. Without this, GPT-5.3-Codex+
// models suffer performance degradation.
func TestConvertContentToInputItems_PhasePreserved(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "thinking...",
				PartMetadata: map[string]any{"phase": "commentary"},
			},
		},
	}

	items, err := convertContentToInputItems(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfOutputMessage
	if msg == nil {
		t.Fatalf("expected OutputMessage for phase-carrying content, got %+v", items[0])
	}
	if msg.Phase != "commentary" {
		t.Errorf("Phase = %q, want commentary", msg.Phase)
	}
}

// Thought parts without encrypted content reference server-side IDs that only
// resolve in the originating response, so they must be silently dropped to
// avoid "Item not found" errors in stateless flows.
func TestConvertContentToInputItems_ThoughtPartsWithoutEncryptedContentSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "Let me think.",
				Thought:      true,
				PartMetadata: map[string]any{"reasoning_id": "rs-1"},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item (thought skipped), got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("item should be message, got %+v", items[0])
	}
}

// Thought parts carrying encrypted content must be replayed as reasoning
// items, preceding the rest of the assistant turn: reasoning models require
// the reasoning item that led to a function_call to be present in the input.
func TestConvertContentToInputItems_ThoughtPartsWithEncryptedContentReplayed(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "Let me think.",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
				},
			},
			{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "get_weather"}},
		},
	}

	items, err := convertContentToInputItems(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items (reasoning + function call), got %d", len(items))
	}
	reasoning := items[0].OfReasoning
	if reasoning == nil {
		t.Fatalf("first item should be a reasoning item, got %+v", items[0])
	}
	if reasoning.ID != "rs-1" {
		t.Errorf("reasoning ID = %q, want rs-1", reasoning.ID)
	}
	if !reasoning.EncryptedContent.Valid() || reasoning.EncryptedContent.Value != "enc-blob" {
		t.Errorf("EncryptedContent = %+v, want enc-blob", reasoning.EncryptedContent)
	}
	if len(reasoning.Summary) != 1 || reasoning.Summary[0].Text != "Let me think." {
		t.Errorf("Summary = %+v, want single summary with original text", reasoning.Summary)
	}
	if items[1].OfFunctionCall == nil {
		t.Fatalf("second item should be the function call, got %+v", items[1])
	}
}

// A reasoning item with encrypted content but no summary text (common for
// reasoning models) must still be replayed, with an empty summary list.
func TestConvertContentToInputItems_EncryptedThoughtWithoutSummary(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-2",
					"encrypted_content": "enc-blob-2",
				},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items (reasoning + message), got %d", len(items))
	}
	reasoning := items[0].OfReasoning
	if reasoning == nil {
		t.Fatalf("first item should be a reasoning item, got %+v", items[0])
	}
	if len(reasoning.Summary) != 0 {
		t.Errorf("Summary = %+v, want empty", reasoning.Summary)
	}
}

// Model output without phase should use the simpler EasyInputMessage path.
func TestConvertContentToInputItems_NoPhaseFallback(t *testing.T) {
	content := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "plain response"}},
	}

	items, err := convertContentToInputItems(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("expected EasyInputMessage for non-phase content, got %+v", items[0])
	}
}
