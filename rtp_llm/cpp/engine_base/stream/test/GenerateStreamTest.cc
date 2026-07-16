
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
        model_config_.vocab_size  = 128;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int>                input_ids,
                                          std::shared_ptr<GenerateConfig> generate_config = nullptr) {
        std::shared_ptr<GenerateInput> generate_input(new GenerateInput());
        ResourceContext                resource_context;
        if (generate_config == nullptr) {
            generate_config.reset(new GenerateConfig());
        }
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto complete_ids = stream_ptr->completeTokenIds();
        std::memcpy(complete_ids.data_ptr<int32_t>() + stream_ptr->seqLength(),
                    new_token_ids.data(),
                    new_token_ids.size() * sizeof(int));
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testLogprobsHistoryUsesBoundedGeometricGrowth) {
    auto generate_config             = std::make_shared<GenerateConfig>();
    generate_config->return_logprobs = true;
    generate_config->top_logprobs    = 20;
    generate_config->max_new_tokens  = 2000;
    auto stream                      = GenerateStreamBuilder().createContextStream({1}, generate_config);

    ASSERT_EQ(stream->getTokenLogProbs().sizes(), (torch::IntArrayRef{1, 64}));
    ASSERT_EQ(stream->getTopLogprobTokenIds().sizes(), (torch::IntArrayRef{1, 64, 20}));
    ASSERT_EQ(stream->getTopLogProbs().sizes(), (torch::IntArrayRef{1, 64, 20}));
    EXPECT_LT(stream->getTokenLogProbs().size(1), generate_config->max_new_tokens);

    auto selected = torch::arange(65, torch::kFloat32).reshape({1, 65});
    auto top_ids  = torch::arange(65 * 20, torch::kInt32).reshape({1, 65, 20});
    auto top      = torch::arange(65 * 20, torch::kFloat32).reshape({1, 65, 20});
    stream->setLogProbs(selected, top_ids, top, torch::Tensor(), stream->inputLength());

    ASSERT_EQ(stream->getTokenLogProbs().size(1), 128);
    EXPECT_TRUE(torch::equal(stream->getTokenLogProbs().narrow(1, 0, 65), selected));
    EXPECT_TRUE(torch::equal(stream->getTopLogprobTokenIds().narrow(1, 0, 65), top_ids));
    EXPECT_TRUE(torch::equal(stream->getTopLogProbs().narrow(1, 0, 65), top));
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->reuseCache());

    // flip to false and verify
    stream->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(stream->reuseCache());

    // flip back to true and verify
    stream->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(stream->reuseCache());
}

TEST_F(GenerateStreamTest, testMaxTokenNumExcludesThinkingTokens) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 1;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = 5;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);

    ASSERT_EQ(stream->maxTokenNum(), 8);

    auto processors = stream->getAllLogitsProcessorPtr();
    ASSERT_FALSE(processors.empty());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(think_processor, nullptr);

    think_processor->updateStatus(torch::tensor({{8, 9}}, torch::kInt32), 2);
    ASSERT_EQ(think_processor->finishedThinkOutputLen(), 2);
    ASSERT_EQ(stream->maxTokenNum(), 5);
}

TEST_F(GenerateStreamTest, testSpecUpdateZeroCommitEnqueuesTerminalOutput) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 1;
    config->is_streaming          = true;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = 5;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);
    auto sp_buffer                = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buffer->tokens             = torch::tensor({{42, 99}}, torch::kInt32);
    sp_buffer->hidden_states      = torch::tensor({{1.0f}});
    sp_buffer->all_probs          = torch::tensor({{2.0f}});
    stream->setSPOutputBuffer(sp_buffer);

    // The boundary packet is emitted before the think processor publishes its
    // shorter, observed thinking length.
    const int32_t boundary_tokens[] = {8, 9, 42};
    auto          complete_ids      = stream->completeTokenIds();
    std::memcpy(complete_ids.data_ptr<int32_t>() + stream->seqLength(), boundary_tokens, sizeof(boundary_tokens));
    stream->setSeqLength(stream->seqLength() + 3);
    stream->updateOutput(StreamUpdateInfo{});

    auto processors = stream->getAllLogitsProcessorPtr();
    ASSERT_FALSE(processors.empty());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(think_processor, nullptr);
    think_processor->updateStatus(torch::tensor({{8, 9, 42}}, torch::kInt32), 3);

    ASSERT_EQ(stream->seqLength(), 5);
    ASSERT_EQ(stream->maxTokenNum(), 5);
    ASSERT_TRUE(stream->hasOutput());
    auto boundary_output = stream->nextOutput();
    ASSERT_TRUE(boundary_output.ok());
    ASSERT_EQ(boundary_output.value().generate_outputs.size(), 1);
    EXPECT_EQ(boundary_output.value().generate_outputs[0].output_ids.numel(), 3);
    EXPECT_FALSE(boundary_output.value().generate_outputs[0].finished);
    EXPECT_FALSE(stream->hasOutput());

    auto expected_sp_tokens = sp_buffer->tokens.clone();
    auto expected_hidden    = sp_buffer->hidden_states.clone();
    auto expected_probs     = sp_buffer->all_probs.clone();

    // The dynamic cap now equals seqLength(), so CompleteTokenIds commits none
    // of the proposed tokens. This update must only publish the terminal state.
    StreamSpecUpdateInfo zero_commit_update{
        torch::tensor({{50, 51, 52, 53, 54}}, torch::kInt32), 5, 98, torch::tensor({{3.0f}}), torch::tensor({{4.0f}})};
    stream->specUpdate(zero_commit_update);

    EXPECT_EQ(stream->seqLength(), 5);
    EXPECT_TRUE(torch::equal(sp_buffer->tokens, expected_sp_tokens));
    EXPECT_TRUE(torch::equal(sp_buffer->hidden_states, expected_hidden));
    EXPECT_TRUE(torch::equal(sp_buffer->all_probs, expected_probs));
    ASSERT_TRUE(stream->hasOutput());
    auto terminal_output = stream->nextOutput();
    ASSERT_TRUE(terminal_output.ok());
    ASSERT_EQ(terminal_output.value().generate_outputs.size(), 1);
    EXPECT_EQ(terminal_output.value().generate_outputs[0].output_ids.numel(), 0);
    EXPECT_TRUE(terminal_output.value().generate_outputs[0].finished);
    EXPECT_FALSE(stream->hasOutput());

    stream->specUpdate(zero_commit_update);
    EXPECT_FALSE(stream->hasOutput());
}

TEST_F(GenerateStreamTest, testPdFirstTokenDedupPreservedWithEmptyTerminal) {
    auto builder = GenerateStreamBuilder();

    // Decode ingests a first token that prefill already published. Advancing
    // last_output_pos_ before update keeps the token out of the response queue.
    auto running_config            = std::make_shared<GenerateConfig>();
    running_config->max_new_tokens = 2;
    running_config->pd_separation  = true;
    running_config->is_streaming   = false;
    auto running_stream            = builder.createContextStream({1, 2}, running_config);
    running_stream->incLastOutputPos();
    running_stream->completeTokenIds().data_ptr<int32_t>()[running_stream->seqLength()] = 3;
    running_stream->setSeqLength(running_stream->seqLength() + 1);
    StreamUpdateInfo running_update;
    running_update.update_remote_generate = false;
    running_stream->updateOutput(running_update);

    EXPECT_EQ(running_stream->seqLength(), 3);
    EXPECT_FALSE(running_stream->hasOutput());

    // If that already-published token is also the last allowed token, publish
    // only the terminal state. The token itself must not be emitted twice.
    auto terminal_config            = std::make_shared<GenerateConfig>();
    terminal_config->max_new_tokens = 1;
    terminal_config->pd_separation  = true;
    terminal_config->is_streaming   = false;
    auto terminal_stream            = builder.createContextStream({1, 2}, terminal_config);
    terminal_stream->incLastOutputPos();
    terminal_stream->completeTokenIds().data_ptr<int32_t>()[terminal_stream->seqLength()] = 3;
    terminal_stream->setSeqLength(terminal_stream->seqLength() + 1);
    StreamUpdateInfo terminal_update;
    terminal_update.update_remote_generate = false;
    terminal_stream->updateOutput(terminal_update);

    ASSERT_TRUE(terminal_stream->hasOutput());
    auto terminal_output = terminal_stream->nextOutput();
    ASSERT_TRUE(terminal_output.ok());
    ASSERT_EQ(terminal_output.value().generate_outputs.size(), 1);
    EXPECT_EQ(terminal_output.value().generate_outputs[0].output_ids.numel(), 0);
    EXPECT_TRUE(terminal_output.value().generate_outputs[0].finished);
    EXPECT_FALSE(terminal_stream->hasOutput());

    // Re-evaluating the same finished stream must not enqueue another terminal.
    terminal_stream->updateOutput(terminal_update);
    EXPECT_FALSE(terminal_stream->hasOutput());
}

// clearMtpAsyncDeviceState rejects stale epochs. A worker that
// captured epoch N must not clear state that step N+1 already published
// under epoch N+1.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateStaleEpochReject) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // Start: epoch counter is 0, state is default-constructed.
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 1: publish state, capture epoch_1.
    GenerateStream::MtpAsyncDeviceState s1;
    s1.accept_len_gpu      = torch::ones({1}, torch::kInt32);
    const uint64_t epoch_1 = stream->setMtpAsyncDeviceState(std::move(s1));
    ASSERT_EQ(epoch_1, 1u);
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 2: another publish before the worker for epoch_1 ran. Counter
    // bumps; old epoch should now be stale.
    GenerateStream::MtpAsyncDeviceState s2;
    s2.accept_len_gpu      = torch::ones({1}, torch::kInt32) * 2;
    const uint64_t epoch_2 = stream->setMtpAsyncDeviceState(std::move(s2));
    ASSERT_EQ(epoch_2, 2u);
    ASSERT_NE(epoch_1, epoch_2);

    // Stale worker for epoch_1 attempts to clear: must be rejected, state
    // for epoch_2 must remain intact.
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_1));
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, epoch_2);

    // Worker for epoch_2 clears successfully.
    ASSERT_TRUE(stream->clearMtpAsyncDeviceState(epoch_2));
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);

    // Repeated stale clear after the live state is gone is also a no-op
    // (epoch 0 != epoch_2 since state was reset to default).
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_2));
}

TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateTracksRealAndUpperBoundSeqLen) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    GenerateStream::MtpAsyncDeviceState state;
    state.last_real_seq_len = stream->seqLength();
    state.next_real_seq_len = state.last_real_seq_len + 2;
    stream->setMtpAsyncDeviceState(std::move(state));

    ASSERT_EQ(stream->getMtpAsyncDeviceState().last_real_seq_len, stream->seqLength());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, stream->seqLength() + 2);
}

// setSpecDecodeDeviceState / clearSpecDecodeDeviceState
// continue to work as wrappers around the new struct API.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateBackCompatWrappers) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    auto accept_len     = torch::ones({1}, torch::kInt32);
    auto accept_tokens  = torch::ones({1, 2}, torch::kInt32);
    auto next_seq_len   = torch::ones({1}, torch::kInt32) * 7;
    auto propose_tokens = torch::ones({1, 4}, torch::kInt32);

    stream->setSpecDecodeDeviceState(accept_len, accept_tokens, next_seq_len, propose_tokens);
    ASSERT_TRUE(stream->getAcceptLenGpu().defined());
    ASSERT_TRUE(stream->getAcceptTokensGpu().defined());
    ASSERT_TRUE(stream->getNextSeqLenGpu().defined());
    ASSERT_TRUE(stream->getProposeTokensGpu().defined());

    stream->clearSpecDecodeDeviceState();
    ASSERT_FALSE(stream->getAcceptLenGpu().defined());
    ASSERT_FALSE(stream->getAcceptTokensGpu().defined());
    ASSERT_FALSE(stream->getNextSeqLenGpu().defined());
    ASSERT_FALSE(stream->getProposeTokensGpu().defined());
}

}  // namespace rtp_llm
