#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <ATen/Dispatch.h>
#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

enum class SpecVerifyRowState {
    Active,
    Finished,
    Terminated,
};

ErrorInfo preflightSpecVerifyRequest(const SpecLogitsProcessorRequest& request) {
    if (request.bitmask_size_int32 < SpecLogitsProcessor::bitmaskWordCount(request.vocab_size)) {
        return ErrorInfo(ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
                         "grammar MTP verify: bitmask buffer smaller than model vocab (words="
                             + std::to_string(request.bitmask_size_int32)
                             + ", vocab=" + std::to_string(request.vocab_size) + ")");
    }
    return {};
}

ErrorInfo validateSpecVerifyMatcher(RtpGrammarMatcher& matcher, int64_t eos_token_id, size_t W) {
    auto grammar_vocab_size_or = matcher.vocabSize();
    if (!grammar_vocab_size_or.ok()) {
        matcher.markFinished();
        return grammar_vocab_size_or.status();
    }
    const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
    if (grammar_vocab_size <= 0) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "grammar MTP verify: invalid grammar vocab size " + std::to_string(grammar_vocab_size));
    }
    if (SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB,
                         "grammar vocab exceeds model vocab in MTP verify (grammar="
                             + std::to_string(grammar_vocab_size) + ", model_words=" + std::to_string(W) + ")");
    }

    auto token_in_range = [W](int64_t t) { return t >= 0 && static_cast<size_t>(t / 32) < W; };
    if (!token_in_range(eos_token_id)) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                         "grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id)
                             + ") out of model vocab bitmask (words=" + std::to_string(W) + ")");
    }
    return {};
}

ErrorResult<SpecVerifyRowState>
failSpecVerifyRow(RtpGrammarMatcher& matcher, int32_t* row, size_t W, int64_t eos_token_id, const ErrorInfo& error) {
    matcher.markFinished();
    forceTokenInBitmask(row, W, eos_token_id);
    return error;
}

ErrorResult<SpecVerifyRowState>
fillSpecVerifyRow(RtpGrammarMatcher& matcher, int64_t eos_token_id, int32_t* row, size_t W, size_t model_vocab_size) {
    std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
    if (matcher.finished()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Finished;
    }
    auto terminated = matcher.isTerminated();
    if (!terminated.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, terminated.status());
    }
    if (terminated.value()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Terminated;
    }

    auto grammar_vocab_size_or = matcher.vocabSize();
    if (!grammar_vocab_size_or.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, grammar_vocab_size_or.status());
    }
    const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
    const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

    int64_t  dl_shape[2];
    DLTensor dl     = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
    auto     filled = matcher.fillBitmask(&dl, 0);
    if (!filled.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, filled.status());
    }
    // xgrammar returns false when the produced mask is all-true; that is an
    // unconstrained row, not a matcher failure.
    clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(model_vocab_size));
    return SpecVerifyRowState::Active;
}

bool specVerifyRowCanConsumeDraft(SpecVerifyRowState row_state) {
    return row_state == SpecVerifyRowState::Active;
}

bool specVerifyDraftTokenAllowed(const int32_t* row, size_t W, size_t vocab_size, int32_t token) {
    return token >= 0 && static_cast<size_t>(token) < vocab_size && bitmaskAllowsToken(row, W, token);
}

class ProvisionalSpecAcceptGuard {
public:
    explicit ProvisionalSpecAcceptGuard(RtpGrammarMatcher& matcher): matcher_(matcher) {}

    void recordAccepted() {
        ++accepted_prefix_;
    }

    ErrorInfo rollbackAndReport(int32_t* fallback_row, size_t W, int64_t eos_token_id) {
        auto rollback_err = rollback();
        if (!rollback_err.hasError()) {
            return ErrorInfo::OkStatus();
        }
        matcher_.markFinished();
        forceTokenInBitmask(fallback_row, W, eos_token_id);
        return rollback_err;
    }

private:
    ErrorInfo rollback() {
        if (accepted_prefix_ > 0) {
            return matcher_.rollback(accepted_prefix_);
        }
        return ErrorInfo::OkStatus();
    }

    RtpGrammarMatcher& matcher_;
    int                accepted_prefix_ = 0;
};

[[nodiscard]] ErrorResult<int> verifyDraftPrefixAndFillBitmask(RtpGrammarMatcher&                matcher,
                                                               int64_t                           eos_token_id,
                                                               const SpecLogitsProcessorRequest& request,
                                                               ProvisionalSpecAcceptGuard&       provisional) {
    const int  P = request.propose_step;
    const auto W = request.bitmask_size_int32;

    for (int offset = 0; offset <= P; ++offset) {
        int32_t*   row       = request.bitmask_cpu_out + offset * W;
        const auto row_state = fillSpecVerifyRow(matcher, eos_token_id, row, W, request.vocab_size);
        if (!row_state.ok()) {
            return row_state.status();
        }
        if (offset == P) {
            return int(P);
        }
        if (!specVerifyRowCanConsumeDraft(row_state.value())) {
            return int(offset);
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (!specVerifyDraftTokenAllowed(row, W, request.vocab_size, draft_token)) {
            return int(offset);
        }
        auto accepted = matcher.acceptToken(draft_token);
        if (!accepted.ok()) {
            return accepted.status();
        }
        if (!accepted.value()) {
            return int(offset);
        }
        provisional.recordAccepted();
    }

    return int(P);
}

ErrorResult<int> verifySpecDraftAndFillBitmask(RtpGrammarMatcher&                matcher,
                                               int64_t                           eos_token_id,
                                               const SpecLogitsProcessorRequest& request) {
    if (auto err = preflightSpecVerifyRequest(request); err.hasError()) {
        return err;
    }

    const auto W = request.bitmask_size_int32;

    if (auto err = validateSpecVerifyMatcher(matcher, eos_token_id, W); err.hasError()) {
        return err;
    }

    ProvisionalSpecAcceptGuard provisional(matcher);

    auto cap = verifyDraftPrefixAndFillBitmask(matcher, eos_token_id, request, provisional);
    if (!cap.ok()) {
        matcher.markFinished();
        forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id);
        return cap.status();
    }

    auto rollback_err = provisional.rollbackAndReport(request.bitmask_cpu_out, W, eos_token_id);
    if (rollback_err.hasError()) {
        return rollback_err;
    }
    return int(cap.value());
}

ErrorInfo
applyPackedAllowMaskCpu(const torch::Tensor& logits, const torch::Tensor& packed_allow_mask, size_t vocab_size) {
    if (!logits.device().is_cpu() || logits.dim() != 1 || logits.stride(0) != 1) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "grammar packed CPU mask requires contiguous 1D CPU logits");
    }
    if (!packed_allow_mask.device().is_cpu() || packed_allow_mask.dim() != 2 || packed_allow_mask.size(0) != 1
        || packed_allow_mask.scalar_type() != torch::kInt32 || packed_allow_mask.stride(1) != 1) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                         "grammar packed CPU mask requires one contiguous int32 bitmask row");
    }
    const size_t words = static_cast<size_t>(packed_allow_mask.size(1));
    if (words < SpecLogitsProcessor::bitmaskWordCount(vocab_size)) {
        return ErrorInfo(ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
                         "grammar packed CPU mask is smaller than the logits vocab");
    }

    // CPU-only fallback used by non-GPU callers and unit tests. Production
    // decode keeps the packed representation and applies it on the GPU.
    const auto* bits = packed_allow_mask.data_ptr<int32_t>();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(), "applyPackedAllowMaskCpu", [&] {
            auto* data = logits.data_ptr<scalar_t>();
            for (size_t token = 0; token < vocab_size; ++token) {
                if (!bitmaskAllowsToken(bits, words, static_cast<int32_t>(token))) {
                    data[token] = static_cast<scalar_t>(-std::numeric_limits<float>::infinity());
                }
            }
        });
    return ErrorInfo::OkStatus();
}

}  // namespace

class GrammarLogitsProcessor::DecodeMaskBuilder final {
public:
    ErrorInfo
    apply(const torch::Tensor& logits, RtpGrammarMatcher& matcher, int64_t accepted_token_len, int64_t eos_token_id) {
        if (!logits.defined() || logits.dim() != 1 || logits.stride(0) != 1) {
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                             "grammar logits processor requires contiguous 1D logits rows");
        }

        if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len) {
            return applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
        }

        auto state_or = buildState(matcher, accepted_token_len);
        if (!state_or.ok()) {
            device_mask_state_ = finishedState(accepted_token_len);
            applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
            return state_or.status();
        }

        device_mask_state_ = std::move(state_or.value());
        return applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
    }

    ErrorInfo refreshAfterCommit(RtpGrammarMatcher& matcher, int64_t accepted_token_len) {
        auto state_or = buildState(matcher, accepted_token_len);
        if (!state_or.ok()) {
            device_mask_state_ = finishedState(accepted_token_len);
            return state_or.status();
        }

        device_mask_state_ = std::move(state_or.value());
        return {};
    }

private:
    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode mode          = DeviceMaskMode::UNSET;
        int64_t        token_len     = -1;
        bool           mask_required = false;
        torch::Tensor  packed_allow_mask_cpu;
        int32_t        grammar_vocab_size = 0;
    };

    static DeviceMaskState finishedState(int64_t accepted_token_len) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;
        state.mode      = DeviceMaskMode::FINISHED;
        return state;
    }

    ErrorResult<DeviceMaskState> buildState(RtpGrammarMatcher& matcher, int64_t accepted_token_len) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;

        if (matcher.finished()) {
            state.mode = DeviceMaskMode::FINISHED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }
        auto terminated = matcher.isTerminated();
        if (!terminated.ok()) {
            matcher.markFinished();
            return terminated.status();
        }
        if (terminated.value()) {
            state.mode = DeviceMaskMode::TERMINATED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }

        auto grammar_vocab_size_or = matcher.vocabSize();
        if (!grammar_vocab_size_or.ok()) {
            matcher.markFinished();
            return grammar_vocab_size_or.status();
        }
        const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
        if (grammar_vocab_size <= 0) {
            state.mode = DeviceMaskMode::NOOP;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }

        auto bitmask = prepareBitmask(grammar_vocab_size);
        auto filled  = fillMatcherBitmask(matcher, bitmask);
        if (!filled.ok()) {
            matcher.markFinished();
            return filled.status();
        }

        state.mode                  = DeviceMaskMode::MASK;
        state.mask_required         = filled.value();
        state.packed_allow_mask_cpu = filled.value() ? std::move(bitmask) : torch::Tensor{};
        state.grammar_vocab_size    = grammar_vocab_size;
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

    torch::Tensor prepareBitmask(int32_t grammar_vocab_size) {
        const int32_t words = (grammar_vocab_size + 31) / 32;
        if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
            reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
            reusable_mask_words_  = words;
        } else {
            reusable_bitmask_cpu_.fill_(-1);
        }
        return reusable_bitmask_cpu_.narrow(1, 0, words);
    }

    static ErrorResult<bool> fillMatcherBitmask(RtpGrammarMatcher& matcher, const torch::Tensor& bitmask) {
        int64_t  dl_shape[2];
        DLTensor dl =
            makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), static_cast<int32_t>(bitmask.size(1)), dl_shape);
        return matcher.fillBitmask(&dl, 0);
    }

    static ErrorInfo
    applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state, int64_t eos_token_id) {
        switch (state.mode) {
            case DeviceMaskMode::UNSET:
            case DeviceMaskMode::NOOP:
            case DeviceMaskMode::FINISHED:
                return ErrorInfo::OkStatus();
            case DeviceMaskMode::TERMINATED:
                forceToken(logits, eos_token_id);
                return ErrorInfo::OkStatus();
            case DeviceMaskMode::MASK:
                break;
        }

        const size_t logits_vocab_size = static_cast<size_t>(logits.size(0));
        const size_t mask_vocab_size   = std::min(logits_vocab_size, static_cast<size_t>(state.grammar_vocab_size));
        if (state.mask_required && mask_vocab_size > 0) {
            if (!state.packed_allow_mask_cpu.defined()) {
                return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "grammar packed mask state is missing its CPU source");
            }
            if (logits.is_cuda()) {
                auto packed_allow_mask_gpu = state.packed_allow_mask_cpu.to(logits.device(), /*non_blocking=*/true);
                runtimeApplyPackedMaskLogits(logits, packed_allow_mask_gpu, mask_vocab_size);
            } else {
                auto error = applyPackedAllowMaskCpu(logits, state.packed_allow_mask_cpu, mask_vocab_size);
                if (error.hasError()) {
                    return error;
                }
            }
        }
        if (mask_vocab_size < logits_vocab_size) {
            logits
                .narrow(
                    0, static_cast<int64_t>(mask_vocab_size), static_cast<int64_t>(logits_vocab_size - mask_vocab_size))
                .fill_(BaseLogitsProcessor::neg_inf);
        }
        return ErrorInfo::OkStatus();
    }

    static void forceToken(const torch::Tensor& logits, int64_t token_id) {
        if (token_id < 0 || token_id >= logits.size(0)) {
            return;
        }
        logits.fill_(BaseLogitsProcessor::neg_inf);
        logits[token_id] = 0.0f;
    }

    DeviceMaskState device_mask_state_{};
    torch::Tensor   reusable_bitmask_cpu_;
    int32_t         reusable_mask_words_ = 0;
};

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                               int64_t                            eos_token_id,
                                               bool                               replay_each_step):
    matcher_(std::move(matcher)),
    eos_token_id_(eos_token_id),
    replay_each_step_(replay_each_step),
    decode_mask_builder_(std::make_unique<DecodeMaskBuilder>()) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

std::optional<ErrorInfo>
GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return std::nullopt;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return std::nullopt;
    }
    if (!replay_each_step_ && batch_size != 1) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
    }
    if (replay_each_step_) {
        if (!inputs.token_ids.defined() || !inputs.input_lengths.defined() || !inputs.sequence_lengths.defined()
            || inputs.token_ids.dim() != 2 || inputs.token_ids.scalar_type() != torch::kInt32
            || !inputs.token_ids.is_contiguous()) {
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                             "grammar replay requires contiguous int32 token_ids and sequence length inputs");
        }
        if (finish_idx > static_cast<size_t>(inputs.token_ids.size(0))
            || finish_idx > static_cast<size_t>(inputs.input_lengths.numel())
            || finish_idx > static_cast<size_t>(inputs.sequence_lengths.numel())) {
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                             "grammar replay input rows do not cover the logits processor interval");
        }

        for (size_t row = start_idx; row < finish_idx; ++row) {
            if (inputs.finished_mask.defined() && inputs.finished_mask.data_ptr<bool>()[row]) {
                continue;
            }
            auto matcher_or = matcher_->newMatcher();
            if (!matcher_or.ok()) {
                return matcher_or.status();
            }
            auto replay_error = replayRow(inputs, row, *matcher_or.value());
            if (replay_error.hasError()) {
                return replay_error;
            }

            // Two beams can have equal output lengths but different parser
            // states, so their mask caches must not be shared.
            DecodeMaskBuilder row_mask_builder;
            const int64_t     output_len =
                inputs.sequence_lengths.data_ptr<int32_t>()[row] - inputs.input_lengths.data_ptr<int32_t>()[row];
            auto error = row_mask_builder.apply(inputs.logits[row], *matcher_or.value(), output_len, eos_token_id_);
            if (error.hasError()) {
                return error;
            }
        }
        return std::nullopt;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return std::nullopt;
        }
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    auto error = decode_mask_builder_->apply(inputs.logits[start_idx], *matcher_, accepted_token_len_, eos_token_id_);
    if (error.hasError()) {
        return error;
    }
    return std::nullopt;
}

std::optional<ErrorInfo> GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    return commitTokens(new_tokens, num_new_tokens);
}

std::optional<ErrorInfo> GrammarLogitsProcessor::commitTokens(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_) {
        return std::nullopt;
    }
    if (replay_each_step_) {
        // Normal decode rebuilds from GenerateStream's authoritative token
        // history before the next mask, so there is no mutable state to commit.
        return std::nullopt;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int batch_size = static_cast<int>(new_tokens.size(0));
    // Keep parity with process(): this processor owns one matcher state machine,
    // so multi-sequence updates would corrupt parser state.
    if (batch_size != 1) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
    }
    const auto* data = new_tokens.data_ptr<int32_t>();

    std::lock_guard<std::mutex> lock(state_mutex_);
    auto                        error = acceptCommittedLocked(data, static_cast<size_t>(num_new_tokens));
    if (error.hasError()) {
        return error;
    }
    return std::nullopt;
}

ErrorInfo GrammarLogitsProcessor::replayRow(const SamplerInputs& inputs, size_t row, RtpGrammarMatcher& matcher) const {
    const auto*   input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const auto*   sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    const int32_t input_len        = input_lengths[row];
    const int32_t sequence_len     = sequence_lengths[row];
    const int64_t token_capacity   = inputs.token_ids.size(1);
    if (input_len < 0 || sequence_len < input_len || sequence_len > token_capacity) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                         "grammar replay received invalid token bounds for row " + std::to_string(row) + " (input_len="
                             + std::to_string(input_len) + ", sequence_len=" + std::to_string(sequence_len)
                             + ", capacity=" + std::to_string(token_capacity) + ")");
    }

    const auto* tokens = inputs.token_ids.data_ptr<int32_t>() + row * token_capacity + input_len;
    for (int32_t offset = 0; offset < sequence_len - input_len; ++offset) {
        const int32_t token      = tokens[offset];
        auto          terminated = matcher.isTerminated();
        if (!terminated.ok()) {
            return terminated.status();
        }
        if (terminated.value()) {
            if (token == static_cast<int32_t>(eos_token_id_)) {
                return ErrorInfo::OkStatus();
            }
            return ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                             "grammar replay received non-EOS token after terminal state at row "
                                 + std::to_string(row));
        }
        auto accepted = matcher.acceptToken(token);
        if (!accepted.ok()) {
            return accepted.status();
        }
        if (!accepted.value()) {
            return ErrorInfo(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                             "grammar replay rejected token " + std::to_string(token) + " at row "
                                 + std::to_string(row));
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<int> GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return static_cast<int>(request.propose_step);
    }

    int cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto                        cap_or = verifySpecDraftAndFillBitmask(*matcher_, eos_token_id_, request);
        if (!cap_or.ok()) {
            return cap_or.status();
        }
        cap_out = cap_or.value();
    }
    return ErrorResult<int>(std::move(cap_out));
}

ErrorInfo GrammarLogitsProcessor::acceptCommittedLocked(const int32_t* tokens, size_t n) {
    if (!matcher_ || matcher_->finished() || n == 0) {
        return ErrorInfo::OkStatus();
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    auto finish_with_error = [this](const ErrorInfo& error) {
        matcher_->markFinished();
        return error;
    };

    for (size_t i = 0; i < n; ++i) {
        const int32_t tok        = tokens[i];
        auto          terminated = matcher_->isTerminated();
        if (!terminated.ok()) {
            return finish_with_error(terminated.status());
        }
        if (terminated.value()) {
            // Keep the matcher TERMINATED rather than FINISHED. FINISHED makes
            // DecodeMaskBuilder stop applying a mask; a still-live stream must
            // continue to allow only EOS instead of resuming unconstrained
            // generation when min_new_tokens or ignore_eos delays completion.
            if (tok == static_cast<int32_t>(eos_token_id_)) {
                accepted_token_len_ = matcher_->numAcceptedTokens() + 1;
            } else {
                return ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                 "grammar received non-EOS token after terminal state");
            }
            break;
        }
        auto accepted = matcher_->acceptToken(tok);
        if (!accepted.ok()) {
            return finish_with_error(accepted.status());
        }
        if (!accepted.value()) {
            return finish_with_error(ErrorInfo(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                                               "grammar commit error: parser rejected token " + std::to_string(tok)));
        }
        accepted_token_len_ = matcher_->numAcceptedTokens();
    }

    return decode_mask_builder_->refreshAfterCommit(*matcher_, accepted_token_len_);
}

}  // namespace rtp_llm
