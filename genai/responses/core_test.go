// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
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
			name: "uppercase types get normalised and properties injected",
			in: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"name": map[string]any{"type": "STRING"},
					"items": map[string]any{
						"type":  "ARRAY",
						"items": map[string]any{"type": "OBJECT"},
					},
				},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"items": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type":                 "object",
							"properties":          map[string]any{},
							"additionalProperties": false,
						},
					},
				},
			},
		},
		{
			name: "nil input returns valid empty object schema",
			in:   nil,
			want: map[string]any{
				"type":                 "object",
				"properties":          map[string]any{},
				"additionalProperties": false,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := convertToStrictFunctionParams(c.in)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertToStrictFunctionParams() = %#v\nwant %#v", got, c.want)
			}
		})
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
