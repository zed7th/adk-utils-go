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
	"encoding/json"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

func TestConvertSchema(t *testing.T) {
	cases := []struct {
		name   string
		schema *genai.Schema
		want   map[string]any
	}{
		{
			name:   "nil schema yields a default empty object",
			schema: nil,
			want: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		{
			name: "primitive with description",
			schema: &genai.Schema{
				Type:        genai.TypeString,
				Description: "a string",
			},
			want: map[string]any{
				"type":        "string",
				"description": "a string",
			},
		},
		{
			name: "object with required and nested integer property",
			schema: &genai.Schema{
				Type:     genai.TypeObject,
				Required: []string{"a"},
				Properties: map[string]*genai.Schema{
					"a": {Type: genai.TypeInteger},
				},
			},
			want: map[string]any{
				"type":     "object",
				"required": []string{"a"},
				"properties": map[string]any{
					"a": map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "array with item schema",
			schema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			want: map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertSchema(c.schema)
			if err != nil {
				t.Fatalf("convertSchema() error = %v", err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertSchema() = %#v\nwant %#v", got, c.want)
			}
		})
	}
}

func TestConvertToFunctionParams(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want map[string]any
	}{
		{
			name: "required auto-filled and optional fields made nullable",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt":      map[string]any{"type": "string"},
					"heightRatio": map[string]any{"type": "integer"},
				},
				"required": []any{"prompt"},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"heightRatio", "prompt"},
				"properties": map[string]any{
					"prompt":      map[string]any{"type": "string"},
					"heightRatio": map[string]any{"type": []any{"integer", "null"}},
				},
			},
		},
		{
			name: "nil input returns valid empty object schema",
			in:   nil,
			want: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"required":             []any{},
				"additionalProperties": false,
			},
		},
		{
			name: "optional nested object gets strict treatment too",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"opts": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"color": map[string]any{"type": "string"},
						},
						"required": []any{"color"},
					},
				},
				"required": []any{"name"},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"name", "opts"},
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"opts": map[string]any{
						"type":                 []any{"object", "null"},
						"additionalProperties": false,
						"required":             []any{"color"},
						"properties": map[string]any{
							"color": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
		{
			name: "schema with properties and null required is treated as object",
			in: map[string]any{
				"properties": map[string]any{
					"env": map[string]any{"type": "string"},
				},
				"required": nil,
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"env"},
				"properties": map[string]any{
					"env": map[string]any{"type": []any{"string", "null"}},
				},
			},
		},
		{
			name: "$defs branches normalised and oneOf renamed to anyOf",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"item": map[string]any{"$ref": "#/$defs/item"},
				},
				"required": []any{"item"},
				"$defs": map[string]any{
					"item": map[string]any{
						"oneOf": []any{
							map[string]any{
								"type": "object",
								"properties": map[string]any{
									"text":    map[string]any{"type": "string"},
									"caption": map[string]any{"type": "string"},
									"source":  map[string]any{"$ref": "#/$defs/source"},
								},
								"required": []any{"text"},
							},
						},
					},
				},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"item"},
				"properties": map[string]any{
					"item": map[string]any{"$ref": "#/$defs/item"},
				},
				"$defs": map[string]any{
					"item": map[string]any{
						"anyOf": []any{
							map[string]any{
								"type":                 "object",
								"additionalProperties": false,
								"required":             []any{"caption", "source", "text"},
								"properties": map[string]any{
									"text":    map[string]any{"type": "string"},
									"caption": map[string]any{"type": []any{"string", "null"}},
									"source": map[string]any{
										"anyOf": []any{
											map[string]any{"$ref": "#/$defs/source"},
											map[string]any{"type": "null"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Results are compared verbatim: the "required" order must be
			// deterministic (sorted), because an unstable order changes the
			// serialized tool definition on every request and breaks OpenAI
			// prompt-cache prefix matching.
			got := convertToStrictFunctionParams(c.in)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertToStrictFunctionParams() = %#v\nwant %#v", got, c.want)
			}
		})
	}
}

// convertToStrictFunctionParams must deep-copy the input so callers who
// reuse schemas across multiple tool registrations don't see mutations.
func TestConvertToStrictFunctionParams_DeepCopy(t *testing.T) {
	original := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
	}

	_ = convertToStrictFunctionParams(original)

	// The original must still have a plain string type, not ["string","null"]
	prop := original["properties"].(map[string]any)["a"].(map[string]any)
	if _, ok := prop["type"].(string); !ok {
		t.Errorf("original schema was mutated: a.type = %#v, want string", prop["type"])
	}
	if _, has := original["additionalProperties"]; has {
		t.Errorf("original schema was mutated: has additionalProperties")
	}
}

// The normalised schema must serialize to identical bytes on every call:
// tool definitions are part of the prompt-cache prefix, so any instability
// (e.g. map-iteration order leaking into the "required" array) would cause
// a cache miss on every request.
func TestConvertToStrictFunctionParams_Deterministic(t *testing.T) {
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"alpha": map[string]any{"type": "string"},
			"beta":  map[string]any{"type": "integer"},
			"gamma": map[string]any{"type": "boolean"},
			"delta": map[string]any{"type": "number"},
			"omega": map[string]any{"type": "string"},
		},
	}

	first, err := json.Marshal(convertToStrictFunctionParams(in))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for range 50 {
		next, err := json.Marshal(convertToStrictFunctionParams(in))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if string(next) != string(first) {
			t.Fatalf("serialized schema is unstable:\nfirst = %s\n next = %s", first, next)
		}
	}
}

func TestLowercaseTypes(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]any
		want map[string]any
	}{
		{
			name: "root only",
			in:   map[string]any{"type": "OBJECT"},
			want: map[string]any{"type": "object"},
		},
		{
			name: "deeply nested",
			in: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"name": map[string]any{"type": "STRING"},
					"tags": map[string]any{
						"type":  "ARRAY",
						"items": map[string]any{"type": "STRING"},
					},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"tags": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
			},
		},
		{
			name: "non-type fields untouched",
			in: map[string]any{
				"type":        "STRING",
				"description": "DO NOT TOUCH",
				"enum":        []any{"FOO", "BAR"},
			},
			want: map[string]any{
				"type":        "string",
				"description": "DO NOT TOUCH",
				"enum":        []any{"FOO", "BAR"},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			lowercaseTypes(c.in)
			if !reflect.DeepEqual(c.in, c.want) {
				t.Errorf("after lowercaseTypes:\n got = %#v\nwant = %#v", c.in, c.want)
			}
		})
	}
}
