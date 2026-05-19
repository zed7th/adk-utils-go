// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package responses provides an OpenAI Responses API implementation for the ADK.
// It targets the /v1/responses endpoint, which is required for GPT-5.4+ models
// and natively supports features like reasoning items, phase metadata, and
// structured output that are unavailable through the Chat Completions API.
//
// For OpenAI-compatible providers (Ollama, vLLM, DeepSeek, Kimi, etc.) that
// expose the Chat Completions endpoint, use the genai/openai package instead.
package responses

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var _ model.LLM = &Model{}

var (
	ErrNoOutputInResponse = errors.New("no output items in OpenAI Responses API response")
)

// Model implements model.LLM using the OpenAI Responses API.
type Model struct {
	client    *openai.Client
	modelName string
}

// HTTPOptions holds optional HTTP-level configuration for the OpenAI client.
type HTTPOptions struct {
	Headers http.Header
}

// Config holds the configuration for creating a Responses API Model.
type Config struct {
	// APIKey for authentication. Falls back to OPENAI_API_KEY env var if empty.
	APIKey string
	// BaseURL for the API endpoint.
	BaseURL string
	// ModelName specifies which model to use (e.g., "gpt-4o", "gpt-5.4").
	ModelName string
	// HTTPOptions holds optional HTTP-level overrides (e.g. extra headers).
	HTTPOptions HTTPOptions
}

// New creates a new Responses API Model with the given configuration.
func New(cfg Config) *Model {
	var opts []option.RequestOption

	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	for k, vals := range cfg.HTTPOptions.Headers {
		for _, v := range vals {
			opts = append(opts, option.WithHeaderAdd(k, v))
		}
	}

	client := openai.NewClient(opts...)

	return &Model{
		client:    &client,
		modelName: cfg.ModelName,
	}
}

// Name returns the model name.
func (m *Model) Name() string {
	return m.modelName
}

// GenerateContent sends a request to the LLM and returns responses.
// Set stream=true for streaming responses, false for a single response.
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

// generate sends a non-streaming request and yields a single response.
func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildResponseParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Responses.New(ctx, params)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := convertResponse(resp)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a streaming request and yields partial responses
// as they arrive, followed by a final aggregated response.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildResponseParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Responses.NewStreaming(ctx, params)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if event.Delta == "" {
					continue
				}
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: event.Delta}},
					},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}

			case "response.reasoning_summary_text.delta":
				if event.Delta == "" {
					continue
				}
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: event.Delta, Thought: true}},
					},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}

			case "response.completed", "response.incomplete":
				resp := &event.Response
				llmResp, err := convertResponse(resp)
				if err != nil {
					// Incomplete with no output → return empty response with
					// the correct finish reason instead of an opaque error.
					llmResp = &model.LLMResponse{
						Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{}},
						UsageMetadata: convertUsageMetadata(resp.Usage),
						FinishReason: convertStatus(resp.Status, resp.IncompleteDetails),
						TurnComplete: true,
					}
				}
				yield(llmResp, nil)
				return

			case "response.failed":
				errMsg := event.Response.Error.Message
				if errMsg == "" {
					errMsg = "response generation failed"
				}
				yield(nil, fmt.Errorf("openai responses api: %s", errMsg))
				return

			case "error":
				yield(nil, fmt.Errorf("openai responses api stream error: %s", event.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}
	}
}

// buildResponseParams converts an LLMRequest into Responses API parameters.
func (m *Model) buildResponseParams(req *model.LLMRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(m.modelName),
	}

	// System instruction → instructions field
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if text := extractText(req.Config.SystemInstruction); text != "" {
			params.Instructions = param.NewOpt(text)
		}
	}

	// Conversation history → input items
	var input responses.ResponseInputParam
	for _, content := range req.Contents {
		items, err := convertContentToInputItems(content)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		input = append(input, items...)
	}
	if len(input) > 0 {
		params.Input.OfInputItemList = input
	}

	// Generation config
	if req.Config != nil {
		applyGenerationConfig(&params, req.Config)
	}

	return params, nil
}

// applyGenerationConfig applies optional generation settings to the request params.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) {
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}

	// Reasoning (native support via Responses API)
	if cfg.ThinkingConfig != nil {
		params.Reasoning = shared.ReasoningParam{
			Effort:  convertThinkingLevel(cfg.ThinkingConfig.ThinkingLevel),
			Summary: shared.ReasoningSummaryAuto,
		}
	}

	// JSON mode
	if cfg.ResponseMIMEType == "application/json" {
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
			},
		}
	}

	// Structured output with schema
	if cfg.ResponseSchema != nil {
		if schemaMap, err := convertSchema(cfg.ResponseSchema); err == nil {
			params.Text = responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
						Name:        "response",
						Description: param.NewOpt(cfg.ResponseSchema.Description),
						Schema:      schemaMap,
						Strict:      param.NewOpt(true),
					},
				},
			}
		}
	}

	// Tools
	if len(cfg.Tools) > 0 {
		if tools, err := convertTools(cfg.Tools); err == nil {
			params.Tools = tools
		}
	}

	// ToolConfig → tool_choice
	if cfg.ToolConfig != nil && cfg.ToolConfig.FunctionCallingConfig != nil {
		fcc := cfg.ToolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case genai.FunctionCallingConfigModeAuto:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
			}
		case genai.FunctionCallingConfigModeNone:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
			}
		case genai.FunctionCallingConfigModeAny:
			if len(fcc.AllowedFunctionNames) == 1 {
				params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
					OfFunctionTool: &responses.ToolChoiceFunctionParam{
						Name: fcc.AllowedFunctionNames[0],
					},
				}
			} else {
				params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
					OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
				}
			}
		}
	}
}

// convertContentToInputItems converts a genai.Content into Responses API input items.
// A single Content may produce multiple items: text/media coalesce into a message,
// while FunctionCall and FunctionResponse become separate typed items.
func convertContentToInputItems(content *genai.Content) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam
	var textParts []string
	var mediaParts []responses.ResponseInputContentUnionParam
	var phase, messageID string
	role := convertRole(content.Role)

	flushMessage := func() {
		if len(textParts) == 0 && len(mediaParts) == 0 {
			return
		}

		// Model output with phase metadata → build OutputMessage manually to
		// preserve phase and message ID for GPT-5.3-Codex+ round-tripping.
		if role == responses.EasyInputMessageRoleAssistant && phase != "" {
			var contentParts []responses.ResponseOutputMessageContentUnionParam
			for _, t := range textParts {
				contentParts = append(contentParts, responses.ResponseOutputMessageContentUnionParam{
					OfOutputText: &responses.ResponseOutputTextParam{Text: t},
				})
			}
			msg := responses.ResponseOutputMessageParam{
				ID:      messageID,
				Content: contentParts,
				Status:  responses.ResponseOutputMessageStatusCompleted,
				Phase:   responses.ResponseOutputMessagePhase(phase),
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &msg,
			})
		} else if len(mediaParts) == 0 {
			items = append(items, responses.ResponseInputItemParamOfMessage(
				joinTexts(textParts), role,
			))
		} else {
			var contentList responses.ResponseInputMessageContentListParam
			for _, t := range textParts {
				contentList = append(contentList, responses.ResponseInputContentParamOfInputText(t))
			}
			contentList = append(contentList, mediaParts...)
			items = append(items, responses.ResponseInputItemParamOfMessage(
				contentList, role,
			))
		}

		textParts = nil
		mediaParts = nil
		phase = ""
		messageID = ""
	}

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			flushMessage()
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(
				part.FunctionResponse.ID, string(responseJSON),
			))

		case part.FunctionCall != nil:
			flushMessage()
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(
				string(argsJSON), part.FunctionCall.ID, part.FunctionCall.Name,
			))

		case part.Thought && part.Text != "":
			flushMessage()
			summary := []responses.ResponseReasoningItemSummaryParam{
				{Text: part.Text},
			}
			id := ""
			if part.PartMetadata != nil {
				if rid, ok := part.PartMetadata["reasoning_id"].(string); ok {
					id = rid
				}
			}
			item := responses.ResponseInputItemParamOfReasoning(id, summary)
			if part.PartMetadata != nil {
				if ec, ok := part.PartMetadata["encrypted_content"].(string); ok && ec != "" {
					item.OfReasoning.EncryptedContent = param.NewOpt(ec)
				}
			}
			items = append(items, item)

		case part.Text != "":
			if part.PartMetadata != nil {
				if p, ok := part.PartMetadata["phase"].(string); ok && p != "" {
					phase = p
				}
				if mid, ok := part.PartMetadata["message_id"].(string); ok && mid != "" {
					messageID = mid
				}
			}
			textParts = append(textParts, part.Text)

		case part.InlineData != nil:
			p, err := convertInlineDataToPart(part.InlineData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)
		}
	}

	flushMessage()
	return items, nil
}

// convertResponse transforms a Responses API response into an LLMResponse.
func convertResponse(resp *responses.Response) (*model.LLMResponse, error) {
	if len(resp.Output) == 0 {
		return nil, ErrNoOutputInResponse
	}

	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	for _, item := range resp.Output {
		switch item.Type {
		case "reasoning":
			for _, summary := range item.Summary {
				if summary.Text != "" {
					part := &genai.Part{Text: summary.Text, Thought: true}
					meta := map[string]any{"reasoning_id": item.ID}
					if item.EncryptedContent != "" {
						meta["encrypted_content"] = item.EncryptedContent
					}
					part.PartMetadata = meta
					content.Parts = append(content.Parts, part)
				}
			}

		case "message":
			for _, cp := range item.Content {
				switch cp.Type {
				case "output_text":
					part := &genai.Part{Text: cp.Text}
					if item.Phase != "" || item.ID != "" {
						meta := map[string]any{}
						if item.Phase != "" {
							meta["phase"] = string(item.Phase)
						}
						if item.ID != "" {
							meta["message_id"] = item.ID
						}
						part.PartMetadata = meta
					}
					content.Parts = append(content.Parts, part)
				case "refusal":
					content.Parts = append(content.Parts, &genai.Part{Text: cp.Refusal})
				}
			}

		case "function_call":
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   item.CallID,
					Name: item.Name,
					Args: parseJSONArgs(item.Arguments.OfString),
				},
			})
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: convertUsageMetadata(resp.Usage),
		FinishReason:  convertStatus(resp.Status, resp.IncompleteDetails),
		TurnComplete:  true,
	}, nil
}

// convertTools transforms genai tools into Responses API function tool format.
func convertTools(genaiTools []*genai.Tool) ([]responses.ToolUnionParam, error) {
	var tools []responses.ToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			tools = append(tools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        funcDecl.Name,
					Description: param.NewOpt(funcDecl.Description),
					Parameters:  convertToStrictFunctionParams(params),
					Strict:      param.NewOpt(true),
				},
			})
		}
	}

	return tools, nil
}

// --- Helper functions ---

// convertInlineDataToPart converts inline data to the appropriate Responses API content part.
// Supports images (as data URI) and generic files (PDF, text, audio).
func convertInlineDataToPart(data *genai.Blob) (*responses.ResponseInputContentUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("inline data is nil")
	}

	mediaType := data.MIMEType
	base64Data := base64.StdEncoding.EncodeToString(data.Data)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data)

	switch {
	case mediaType == "image/jpeg" || mediaType == "image/jpg" || mediaType == "image/png" ||
		mediaType == "image/gif" || mediaType == "image/webp":
		return &responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: param.NewOpt(dataURI),
				Detail:   responses.ResponseInputImageDetailAuto,
			},
		}, nil

	case mediaType == "application/pdf" || strings.HasPrefix(mediaType, "text/") ||
		strings.HasPrefix(mediaType, "audio/"):
		return &responses.ResponseInputContentUnionParam{
			OfInputFile: &responses.ResponseInputFileParam{
				FileData: param.NewOpt(dataURI),
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported inline data MIME type for Responses API: %s", mediaType)
	}
}

// convertUsageMetadata converts Responses API usage stats to genai format.
func convertUsageMetadata(usage responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage.TotalTokens == 0 {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(usage.InputTokens),
		CandidatesTokenCount:    int32(usage.OutputTokens),
		TotalTokenCount:         int32(usage.TotalTokens),
		ThoughtsTokenCount:      int32(usage.OutputTokensDetails.ReasoningTokens),
		CachedContentTokenCount: int32(usage.InputTokensDetails.CachedTokens),
	}
}

// convertRole maps genai roles to Responses API EasyInputMessageRole.
func convertRole(role string) responses.EasyInputMessageRole {
	switch role {
	case "model":
		return responses.EasyInputMessageRoleAssistant
	case "system":
		return responses.EasyInputMessageRoleSystem
	default:
		return responses.EasyInputMessageRoleUser
	}
}

// convertStatus maps Responses API status to genai finish reason.
func convertStatus(status responses.ResponseStatus, details responses.ResponseIncompleteDetails) genai.FinishReason {
	switch status {
	case responses.ResponseStatusCompleted:
		return genai.FinishReasonStop
	case responses.ResponseStatusIncomplete:
		switch details.Reason {
		case "max_output_tokens":
			return genai.FinishReasonMaxTokens
		case "content_filter":
			return genai.FinishReasonSafety
		}
		return genai.FinishReasonUnspecified
	default:
		return genai.FinishReasonUnspecified
	}
}

// convertThinkingLevel maps genai thinking levels to Responses API reasoning effort.
func convertThinkingLevel(level genai.ThinkingLevel) shared.ReasoningEffort {
	switch level {
	case genai.ThinkingLevelLow:
		return shared.ReasoningEffortLow
	case genai.ThinkingLevelHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

// convertToStrictFunctionParams converts various parameter types to a map
// compliant with OpenAI strict mode. Strict mode requires every object to
// have "properties" and "additionalProperties": false. A nil input produces
// a valid empty object schema so parameterless tools are accepted.
func convertToStrictFunctionParams(params any) map[string]any {
	if params == nil {
		return map[string]any{
			"type":                 "object",
			"properties":          map[string]any{},
			"additionalProperties": false,
		}
	}

	var m map[string]any

	if dm, ok := params.(map[string]any); ok {
		m = dm
	} else {
		jsonBytes, err := json.Marshal(params)
		if err != nil {
			return nil
		}
		if json.Unmarshal(jsonBytes, &m) != nil {
			return nil
		}
	}

	lowercaseTypes(m)
	ensureStrictObjectSchema(m)
	return m
}

// ensureStrictObjectSchema recursively ensures all object schemas have a
// "properties" field and "additionalProperties": false, as required by
// OpenAI's strict mode for function tool parameters.
func ensureStrictObjectSchema(schema map[string]any) {
	if schema == nil {
		return
	}

	if t, ok := schema["type"].(string); ok && t == "object" {
		if _, hasProps := schema["properties"]; !hasProps {
			schema["properties"] = map[string]any{}
		}
		schema["additionalProperties"] = false
	}

	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				ensureStrictObjectSchema(propMap)
			}
		}
	}

	if items, ok := schema["items"].(map[string]any); ok {
		ensureStrictObjectSchema(items)
	}
}

// lowercaseTypes recursively lowercases all "type" fields in a JSON schema map.
func lowercaseTypes(m map[string]any) {
	for k, v := range m {
		if k == "type" {
			if s, ok := v.(string); ok {
				m[k] = strings.ToLower(s)
			}
		} else if vMap, ok := v.(map[string]any); ok {
			lowercaseTypes(vMap)
		} else if vList, ok := v.([]any); ok {
			for _, item := range vList {
				if itemMap, ok := item.(map[string]any); ok {
					lowercaseTypes(itemMap)
				}
			}
		}
	}
}

// convertSchema recursively converts a genai.Schema to JSON schema format.
func convertSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}, nil
	}

	result := make(map[string]any)

	if schema.Type != genai.TypeUnspecified {
		result["type"] = schemaTypeToString(schema.Type)
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, propSchema := range schema.Properties {
			converted, err := convertSchema(propSchema)
			if err != nil {
				return nil, err
			}
			props[name] = converted
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		items, err := convertSchema(schema.Items)
		if err != nil {
			return nil, err
		}
		result["items"] = items
	}

	return result, nil
}

// schemaTypeToString converts genai.Type to JSON schema type string.
func schemaTypeToString(t genai.Type) string {
	types := map[genai.Type]string{
		genai.TypeString:  "string",
		genai.TypeNumber:  "number",
		genai.TypeInteger: "integer",
		genai.TypeBoolean: "boolean",
		genai.TypeArray:   "array",
		genai.TypeObject:  "object",
	}
	if s, ok := types[t]; ok {
		return s
	}
	return "string"
}

// extractText extracts all text parts from a Content and joins them.
func extractText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return joinTexts(texts)
}

// joinTexts joins multiple text strings with newlines.
func joinTexts(texts []string) string {
	return strings.Join(texts, "\n")
}

// parseJSONArgs parses a JSON string into a map. Returns empty map on error.
func parseJSONArgs(argsJSON string) map[string]any {
	if argsJSON == "" {
		return make(map[string]any)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return make(map[string]any)
	}
	return args
}
