// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

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

// Package responses provides an OpenAI Responses API (/v1/responses)
// implementation for the ADK.
//
// The Responses API is the interface OpenAI recommends for new applications,
// with native reasoning, built-in tools, and structured output.
//
// This adapter drives the API statelessly to match ADK's model: ADK owns the
// conversation state and passes the full history on every call, so each request
// replays that history as input items instead of chaining server-side state via
// previous_response_id. Requests are sent with store=false so nothing is
// persisted server-side, and reasoning is requested with encrypted_content so
// reasoning items can be replayed across turns: this keeps the model's chain
// of thought available between tool calls, as reasoning models require.
// Reasoning items without encrypted content (e.g. from gateways that do not
// support the include parameter) are skipped on replay, since their bare IDs
// only resolve in the context of the originating response.
//
// Encrypted reasoning content is bound to the credentials that produced it:
// the API encrypts it with the organization's key, and replaying it through a
// different provider, API key, or model fails with a 400
// invalid_encrypted_content error. Sessions may switch channels mid-flight
// (e.g. OpenAI to Azure fallback), so each reasoning part records the origin
// (a fingerprint of base URL, API key, and model) and is replayed only when
// it matches the requesting model; otherwise the encrypted content is
// dropped and the turn degrades gracefully to fresh reasoning.
//
// For OpenAI-compatible gateways (Ollama, vLLM, DeepSeek, Kimi, etc.) that only
// expose the Chat Completions endpoint, use the genai/openai package instead.
package responses

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/v2/model"
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
	// origin fingerprints the channel (base URL, API key, model) this Model
	// talks to. Encrypted reasoning content is only replayed to the channel
	// that produced it; see the package documentation.
	origin string
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

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	return &Model{
		client:    &client,
		modelName: cfg.ModelName,
		origin:    computeOrigin(cfg.BaseURL, apiKey, cfg.ModelName),
	}
}

// computeOrigin fingerprints a channel so encrypted reasoning content is only
// replayed to the channel that produced it. The API encrypts reasoning with
// the organization's key, so any change of provider, API key, or model makes
// previously captured content undecryptable (400 invalid_encrypted_content).
// The fingerprint deliberately covers all three dimensions.
func computeOrigin(baseURL, apiKey, modelName string) string {
	h := sha256.New()
	for _, s := range []string{baseURL, apiKey, modelName} {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
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

		llmResp, err := convertResponse(resp, m.origin)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a streaming request and yields partial responses
// as they arrive, followed by a final aggregated response.
//
// Streamed deltas (Partial=true) are display-only; ADK persists only the final
// non-partial event. The stream must therefore always end with a complete
// aggregated event. Some OpenAI-compatible gateways omit the aggregated output
// from response.completed, or close the connection without any terminal event
// at all. Relying solely on the server-provided output then yields an empty
// final event, so the assistant turn is lost from history on reload even though
// it streamed fine. To stay robust, the deltas are accumulated locally as a
// fallback, mirroring the Chat Completions adapter.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildResponseParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Responses.NewStreaming(ctx, params)

		var acc streamAccumulator
		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if event.Delta == "" {
					continue
				}
				acc.text.WriteString(event.Delta)
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
				acc.reasoning.WriteString(event.Delta)
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

			case "response.output_item.done":
				// Completed function calls are accumulated so the fallback
				// final event includes them: losing a tool call would
				// silently break the agent loop.
				if event.Item.Type == "function_call" {
					acc.functionCalls = append(acc.functionCalls, &genai.FunctionCall{
						ID:   event.Item.CallID,
						Name: event.Item.Name,
						Args: parseJSONArgs(event.Item.Arguments.OfString),
					})
				}

			case "response.completed", "response.incomplete":
				resp := &event.Response
				llmResp, err := convertResponse(resp, m.origin)
				if err != nil || hasNoContent(llmResp) {
					// The terminal event carried no aggregated output: rebuild the
					// final response from the accumulated deltas, otherwise ADK
					// would persist an empty event and the turn would be lost.
					llmResp = acc.finalResponse(
						convertStatus(resp.Status, resp.IncompleteDetails),
						convertUsageMetadata(resp.Usage),
					)
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

		// The stream ended without any terminal event (some gateways just close
		// the connection after the last delta). Synthesize the final event from
		// the accumulated deltas so the turn is persisted and ADK does not raise
		// "last event is not final".
		if acc.hasContent() {
			yield(acc.finalResponse(genai.FinishReasonStop, nil), nil)
		}
	}
}

// streamAccumulator collects streamed deltas so a complete final response
// can be rebuilt when the terminal event lacks the aggregated output.
// reasoning holds the reasoning summary; text holds the final answer;
// functionCalls holds tool calls completed during the stream.
type streamAccumulator struct {
	reasoning     strings.Builder
	text          strings.Builder
	functionCalls []*genai.FunctionCall
}

// hasContent reports whether anything was accumulated that could be used to
// rebuild a final response.
func (a *streamAccumulator) hasContent() bool {
	return a.reasoning.Len() > 0 || a.text.Len() > 0 || len(a.functionCalls) > 0
}

// finalResponse builds a non-partial final response from the accumulated deltas
// (for ADK to persist). The reasoning part precedes the answer part, which
// precedes the tool calls, matching the temporal order in which they streamed.
func (a *streamAccumulator) finalResponse(
	finishReason genai.FinishReason,
	usage *genai.GenerateContentResponseUsageMetadata,
) *model.LLMResponse {
	content := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{}}
	if a.reasoning.Len() > 0 {
		content.Parts = append(content.Parts, &genai.Part{Text: a.reasoning.String(), Thought: true})
	}
	if a.text.Len() > 0 {
		content.Parts = append(content.Parts, &genai.Part{Text: a.text.String()})
	}
	for _, fc := range a.functionCalls {
		content.Parts = append(content.Parts, &genai.Part{FunctionCall: fc})
	}
	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usage,
		FinishReason:  finishReason,
		Partial:       false,
		TurnComplete:  true,
	}
}

// hasNoContent reports whether convertResponse produced no persistable content
// parts, used to decide whether to fall back to the accumulated deltas.
func hasNoContent(resp *model.LLMResponse) bool {
	return resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0
}

// buildResponseParams converts an LLMRequest into Responses API parameters.
func (m *Model) buildResponseParams(req *model.LLMRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(m.modelName),
		// ADK owns the conversation state, so nothing needs to be stored
		// server-side. store=false also makes the API return encrypted
		// reasoning content (requested via include below), which is the only
		// way to replay reasoning items in a stateless flow.
		Store: param.NewOpt(false),
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
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
		items, err := convertContentToInputItems(content, m.origin)
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
		if err := applyGenerationConfig(&params, req.Config); err != nil {
			return responses.ResponseNewParams{}, err
		}
	}

	return params, nil
}

// applyGenerationConfig applies optional generation settings to the request params.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
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

	// Structured output with schema (also strict-normalised). Conversion
	// errors are returned rather than swallowed: silently dropping the
	// schema or the tools would send the request anyway and surface as
	// inexplicable model behaviour.
	if cfg.ResponseSchema != nil {
		schemaMap, err := convertSchema(cfg.ResponseSchema)
		if err != nil {
			return fmt.Errorf("failed to convert response schema: %w", err)
		}
		normalizeStrictSchema(schemaMap)
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

	// Tools
	if len(cfg.Tools) > 0 {
		tools, err := convertTools(cfg.Tools)
		if err != nil {
			return fmt.Errorf("failed to convert tools: %w", err)
		}
		params.Tools = tools
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

	return nil
}

// convertContentToInputItems converts a genai.Content into Responses API input items.
// A single Content may produce multiple items: text/media coalesce into a message,
// while FunctionCall and FunctionResponse become separate typed items. origin
// identifies the requesting channel: encrypted reasoning is only replayed when
// it was captured from the same origin.
func convertContentToInputItems(content *genai.Content, origin string) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam
	var textParts []string
	var mediaParts []responses.ResponseInputContentUnionParam
	var phase string
	role := convertRole(content.Role)

	// A replayed reasoning item must be followed by another item from the
	// same turn (message or function call), or the API rejects it as
	// dangling. Track the last part that produces such an item so trailing
	// thoughts (e.g. from an interrupted turn) are not replayed.
	lastFollower := -1
	for i, part := range content.Parts {
		if part.Thought {
			continue
		}
		if part.FunctionCall != nil || part.FunctionResponse != nil ||
			part.Text != "" || part.InlineData != nil || part.FileData != nil {
			lastFollower = i
		}
	}

	flushMessage := func() {
		if len(textParts) == 0 && len(mediaParts) == 0 {
			return
		}

		// Model output with phase metadata → build OutputMessage manually to
		// preserve the phase field for GPT-5.3-Codex+ round-tripping.
		if role == responses.EasyInputMessageRoleAssistant && phase != "" {
			var contentParts []responses.ResponseOutputMessageContentUnionParam
			for _, t := range textParts {
				contentParts = append(contentParts, responses.ResponseOutputMessageContentUnionParam{
					OfOutputText: &responses.ResponseOutputTextParam{Text: t},
				})
			}
			msg := responses.ResponseOutputMessageParam{
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
	}

	for i, part := range content.Parts {
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

		case part.Thought:
			// Reasoning items carrying encrypted content are replayed so the
			// model keeps its chain of thought across turns; reasoning models
			// require the reasoning item preceding a function_call to be
			// present in the input. Skipped instead of replayed when:
			//   - the content is not an assistant turn: in multi-agent
			//     histories another agent's output (thoughts included) shows
			//     up under other roles, where a reasoning item is invalid;
			//   - there is no encrypted content (e.g. gateways that ignore
			//     the include parameter): bare IDs only resolve in the
			//     originating response and replaying them is a 400;
			//   - the origin does not match: encrypted content is bound to
			//     the provider/API key/model that produced it, and replaying
			//     it elsewhere is a 400 invalid_encrypted_content;
			//   - no item from the same turn follows: the API rejects
			//     dangling reasoning items.
			// Skipping degrades gracefully: the model re-derives its
			// reasoning for the turn instead of the request failing.
			if role != responses.EasyInputMessageRoleAssistant {
				continue
			}
			enc, _ := part.PartMetadata["encrypted_content"].(string)
			id, _ := part.PartMetadata["reasoning_id"].(string)
			if enc == "" || id == "" {
				continue
			}
			if partOrigin, _ := part.PartMetadata["reasoning_origin"].(string); origin == "" || partOrigin != origin {
				continue
			}
			if i > lastFollower {
				continue
			}
			flushMessage()
			reasoning := responses.ResponseReasoningItemParam{
				ID:               id,
				Summary:          []responses.ResponseReasoningItemSummaryParam{},
				EncryptedContent: param.NewOpt(enc),
			}
			if part.Text != "" {
				reasoning.Summary = append(reasoning.Summary,
					responses.ResponseReasoningItemSummaryParam{Text: part.Text})
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfReasoning: &reasoning,
			})

		case part.Text != "":
			if part.PartMetadata != nil {
				if p, ok := part.PartMetadata["phase"].(string); ok && p != "" {
					phase = p
				}
			}
			textParts = append(textParts, part.Text)

		case part.InlineData != nil:
			p, err := convertInlineDataToPart(part.InlineData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)

		case part.FileData != nil:
			p, err := convertFileDataToPart(part.FileData)
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
// origin identifies the channel that produced the response; it is recorded
// alongside encrypted reasoning content so replay can be restricted to the
// same channel.
func convertResponse(resp *responses.Response, origin string) (*model.LLMResponse, error) {
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
			// One part per reasoning item, even when the summary is empty:
			// the encrypted content must survive the round-trip through ADK
			// history so the item can be replayed on the next turn.
			var summaryTexts []string
			for _, summary := range item.Summary {
				if summary.Text != "" {
					summaryTexts = append(summaryTexts, summary.Text)
				}
			}
			if len(summaryTexts) == 0 && item.EncryptedContent == "" {
				continue
			}
			meta := map[string]any{"reasoning_id": item.ID}
			if item.EncryptedContent != "" {
				meta["encrypted_content"] = item.EncryptedContent
				meta["reasoning_origin"] = origin
			}
			content.Parts = append(content.Parts, &genai.Part{
				Text:         joinTexts(summaryTexts),
				Thought:      true,
				PartMetadata: meta,
			})

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
			// Assign only a non-nil *genai.Schema: a nil pointer stored in
			// the interface would defeat the nil check that gives
			// parameterless tools a valid empty object schema.
			if params == nil && funcDecl.Parameters != nil {
				params = funcDecl.Parameters
			}

			strictParams := convertToStrictFunctionParams(params)
			if strictParams == nil {
				return nil, fmt.Errorf("parameters of tool %q cannot be converted to a JSON schema object", funcDecl.Name)
			}

			tools = append(tools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        funcDecl.Name,
					Description: param.NewOpt(funcDecl.Description),
					Parameters:  strictParams,
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

// convertFileDataToPart converts URL-referenced file data to a Responses API content part.
// Only images are supported: the Responses API accepts a remote URL for input_image,
// whereas file inputs require the bytes to be uploaded first.
// Returns an error for unsupported MIME types, mirroring convertInlineDataToPart.
func convertFileDataToPart(data *genai.FileData) (*responses.ResponseInputContentUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("file data is nil")
	}

	switch data.MIMEType {
	case "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp":
		return &responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: param.NewOpt(data.FileURI),
				Detail:   responses.ResponseInputImageDetailAuto,
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported file data MIME type for Responses API: %s", data.MIMEType)
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
// compliant with OpenAI strict mode. A nil input produces a valid empty
// object schema so parameterless tools are accepted. The input is deep-copied
// via JSON round-trip to avoid mutating the caller's schema.
func convertToStrictFunctionParams(params any) map[string]any {
	if params == nil {
		return map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"required":             []any{},
			"additionalProperties": false,
		}
	}

	m := deepCopySchema(params)
	if m == nil {
		return nil
	}

	lowercaseTypes(m)
	normalizeStrictSchema(m)
	return m
}

// normalizeStrictSchema recursively makes a JSON schema compliant with
// OpenAI strict mode. For every object type (whether type is "object" or
// contains "object" in an array like ["object", "null"]) it:
//   - adds "additionalProperties": false
//   - ensures "properties" exists
//   - sets "required" to all property keys
//   - expands originally-optional properties to nullable (["type", "null"])
//
// Child schemas are normalised before the parent marks optional fields as
// nullable, so nested objects get their own required/additionalProperties
// regardless of whether the parent considers them optional.
func normalizeStrictSchema(schema map[string]any) {
	if schema == nil {
		return
	}

	// Recurse into children first so nested objects are fully normalised
	// before we decide which parent-level fields are optional.
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				normalizeStrictSchema(propMap)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		normalizeStrictSchema(items)
	}

	if !isObjectSchema(schema) {
		return
	}

	if _, ok := schema["type"]; !ok {
		schema["type"] = "object"
	}

	if _, hasProps := schema["properties"]; !hasProps {
		schema["properties"] = map[string]any{}
	}
	schema["additionalProperties"] = false

	props, _ := schema["properties"].(map[string]any)
	existing := toStringSet(schema["required"])
	allKeys := make([]any, 0, len(props))
	for key := range props {
		allKeys = append(allKeys, key)
		if !existing[key] {
			makeNullable(props[key])
		}
	}
	// Sort for a deterministic order: map iteration is randomised, and an
	// unstable "required" array changes the serialized tool definition on
	// every request, which breaks OpenAI prompt-cache prefix matching.
	sort.Slice(allKeys, func(i, j int) bool {
		return allKeys[i].(string) < allKeys[j].(string)
	})
	schema["required"] = allKeys
}

// isObjectSchema returns true if the schema represents an object — either
// by explicit type or by having a "properties" field (common in dynamically
// registered tools that omit "type").
func isObjectSchema(schema map[string]any) bool {
	if isObjectType(schema) {
		return true
	}
	_, hasProps := schema["properties"]
	return hasProps
}

// isObjectType returns true if the schema's "type" is "object", either as a
// plain string or inside an array (e.g. ["object", "null"]).
func isObjectType(schema map[string]any) bool {
	switch t := schema["type"].(type) {
	case string:
		return t == "object"
	case []any:
		for _, v := range t {
			if s, ok := v.(string); ok && s == "object" {
				return true
			}
		}
	}
	return false
}

// makeNullable expands a property's type to ["<original>", "null"] so strict
// mode accepts it as an optional (nullable) field.
func makeNullable(prop any) {
	propMap, ok := prop.(map[string]any)
	if !ok {
		return
	}
	switch t := propMap["type"].(type) {
	case string:
		propMap["type"] = []any{t, "null"}
	case []any:
		for _, v := range t {
			if v == "null" {
				return
			}
		}
		propMap["type"] = append(t, "null")
	}
}

// toStringSet builds a set from a "required" field ([]string or []any).
func toStringSet(v any) map[string]bool {
	set := map[string]bool{}
	switch r := v.(type) {
	case []string:
		for _, s := range r {
			set[s] = true
		}
	case []any:
		for _, item := range r {
			if s, ok := item.(string); ok {
				set[s] = true
			}
		}
	}
	return set
}

// deepCopySchema converts any schema representation into a fresh
// map[string]any via JSON round-trip, avoiding mutation of the caller's data.
func deepCopySchema(params any) map[string]any {
	jsonBytes, err := json.Marshal(params)
	if err != nil {
		return nil
	}
	var m map[string]any
	if json.Unmarshal(jsonBytes, &m) != nil {
		return nil
	}
	return m
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
