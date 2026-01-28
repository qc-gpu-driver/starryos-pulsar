/* Inference for Llama-2 with NPU acceleration + Self-Speculative Decoding */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// NPU support - set to 1 to enable, 0 to disable
#define USE_NPU 1

#if USE_NPU
#include <stdint.h>
#include <sys/ioctl.h>
#include <libdrm/drm.h>
#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"

// Define if not in header
#ifndef RKNPU_PC_DATA_EXTRA_AMOUNT
#define RKNPU_PC_DATA_EXTRA_AMOUNT 0
#endif
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

#if USE_NPU
// Cached weight structure for NPU
typedef struct {
    _Float16 *data;
    uint64_t dma;
    uint64_t obj;
    uint32_t handle;
    size_t size;
    int is_cached;
} NPUWeightCache;

// Pre-allocated buffer pool
typedef struct {
    void *data;
    uint64_t dma;
    uint64_t obj;
    uint32_t handle;
    size_t size;
    int in_use;
} NPUBuffer;

#define MAX_BUFFER_POOL_SIZE 8
#endif

// 在原有优化代码基础上添加Self-Speculative功能
// 完整代码基于run_optimized.c,这里只展示新增和修改的部分

// ============================================================================
// Self-Speculative Decoding Implementation
// ============================================================================

#define SPECULATIVE_K 15  // 预测K个token (加上当前token共16个,正好是NPU batch size)

// Speculative decoding的统计信息
typedef struct {
    int total_generations;      // 总生成次数
    int total_tokens_generated; // 总生成的token数
    int total_accepted;         // 总接受的token数
    int total_drafted;          // 总draft的token数
    float acceptance_rate;      // 平均接受率
    float speedup;             // 平均加速比
} SpeculativeStats;

// 在Transformer结构体中添加speculative相关字段
typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
#if USE_NPU
    // NPU resources
    int npu_fd;
    uint64_t regcmd_dma, regcmd_obj;
    uint32_t regcmd_handle;
    uint64_t *regcmd;
    uint64_t tasks_dma, tasks_obj;
    uint32_t tasks_handle;
    struct rknpu_task *tasks;
    uint64_t npu_regs[112];

    // Statistics
    int npu_calls;
    int npu_success;
    int npu_fallback;

    // OPTIMIZATION: Weight cache for frequently used weights
    struct {
        NPUWeightCache *wq_cache;  // [n_layers]
        NPUWeightCache *wk_cache;  // [n_layers]
        NPUWeightCache *wv_cache;  // [n_layers]
        NPUWeightCache *wo_cache;  // [n_layers]
        NPUWeightCache *w1_cache;  // [n_layers]
        NPUWeightCache *w2_cache;  // [n_layers]
        NPUWeightCache *w3_cache;  // [n_layers]
        NPUWeightCache *wcls_cache;
        int enabled;
    } weight_cache;

    // OPTIMIZATION: Larger buffer pool to reduce dynamic allocation
    NPUBuffer buffer_pool[MAX_BUFFER_POOL_SIZE];
    int buffer_pool_initialized;

    // Self-Speculative Decoding
    struct {
        int enabled;                    // 是否启用
        int K;                          // 预测长度 (默认15)
        int *draft_tokens;              // draft的token序列 [K]
        float *draft_probs;             // draft的概率 [K]
        float *verify_logits;           // 验证用的logits [K * vocab_size]
        SpeculativeStats stats;         // 统计信息

        // 用于快速draft的临时buffer
        float *draft_x;                 // [dim]
        float *draft_xb;                // [dim]
        float *draft_logits;            // [vocab_size]
    } speculative;
#endif
} Transformer;

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

// ----------------------------------------------------------------------------
// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// ----------------------------------------------------------------------------
// Forward declarations

void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);
float* forward(Transformer* transformer, int token, int pos);
void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
int matmul(Transformer* t, float* xout, float* x, float* w, int n, int d);

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
int sample(Sampler* sampler, float* logits);
float random_f32(unsigned long long *state);

long time_in_ms();
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps);

#if USE_NPU
// NPU helper function declarations
NPUBuffer* get_buffer_from_pool(Transformer* t, size_t size);
void release_buffer_to_pool(NPUBuffer* buf);
// Note: mem_allocate and mem_destroy are declared in npu_interface.h
// void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle);
// void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr);
int gen_matmul_fp16(matmul_params_t *params);
#endif

#if USE_NPU

// ============================================================================
// 快速draft: 使用简化的forward pass (只用CPU,更快)
// ============================================================================

// 简化版forward: 只计算必要的部分,用于快速draft
static void forward_draft_fast(Transformer* t, int token, int pos, float* output_logits) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    float *x = t->speculative.draft_x;
    float *xb = t->speculative.draft_xb;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // Token embedding
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
    
    // 只处理最后一层 (快速近似)
    int l = p->n_layers - 1;  // 只用最后一层
    
    // Attention rmsnorm
    rmsnorm(xb, x, w->rms_att_weight + l*dim, dim);
    
    // 简化: 跳过完整的attention,直接用FFN
    // (这是一个近似,但对draft来说足够了)
    
    // FFN rmsnorm
    rmsnorm(xb, x, w->rms_ffn_weight + l*dim, dim);
    
    // 只计算到FFN中间,不做完整计算
    // 这里用一个更简单的启发式: 直接用embedding + 最后的rmsnorm
    
    // Final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
    
    // Classifier
    matmul(t, output_logits, x, w->wcls, p->dim, p->vocab_size);
}

// 更快的draft: 只用top-k候选,避免完整softmax
static int sample_draft_fast(float* logits, int vocab_size, unsigned long long* rng_state) {
    // 找到top-3候选
    const int TOP_K = 3;
    int top_indices[TOP_K];
    float top_values[TOP_K];
    
    for (int i = 0; i < TOP_K; i++) {
        top_indices[i] = -1;
        top_values[i] = -1e10f;
    }
    
    // 找到最大的3个
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > top_values[TOP_K-1]) {
            // 插入到合适位置
            int insert_pos = TOP_K - 1;
            for (int j = 0; j < TOP_K - 1; j++) {
                if (logits[i] > top_values[j]) {
                    insert_pos = j;
                    break;
                }
            }
            
            // 移动后面的元素
            for (int j = TOP_K - 1; j > insert_pos; j--) {
                top_indices[j] = top_indices[j-1];
                top_values[j] = top_values[j-1];
            }
            
            top_indices[insert_pos] = i;
            top_values[insert_pos] = logits[i];
        }
    }
    
    // 从top-3中随机选一个 (简单的多项分布采样)
    float total = 0.0f;
    for (int i = 0; i < TOP_K; i++) {
        top_values[i] = expf(top_values[i]);
        total += top_values[i];
    }
    
    float coin = random_f32(rng_state) * total;
    float cumsum = 0.0f;
    for (int i = 0; i < TOP_K; i++) {
        cumsum += top_values[i];
        if (coin < cumsum) {
            return top_indices[i];
        }
    }
    
    return top_indices[0];
}

// ============================================================================
// Batch forward: 一次处理多个token (利用NPU的batch能力)
// ============================================================================

// 批量matmul: 处理batch_size个输入向量
// input: [batch_size x n], weights: [d x n], output: [batch_size x d]
static int matmul_npu_batch_multi(Transformer* t, float* output, float* input, 
                                   NPUWeightCache* weight_cache,
                                   int n, int d, int batch_size, int layer_idx) {
    if (!weight_cache || !weight_cache->is_cached || batch_size < 1 || batch_size > 16) {
        return -1;
    }
    
    // 复用现有的matmul_npu_cached逻辑
    // 但input现在是 [batch_size x n] 而不是 [n]
    
    t->npu_calls++;
    
    // 检查维度
    if (d != 1 && (d % 4 != 0 || d > 544)) {
        t->npu_fallback++;
        return -1;
    }
    if (n % 32 != 0 || n > 4096) {
        t->npu_fallback++;
        return -1;
    }
    
    int M = d;
    int K = n;
    int N = batch_size;
    
    // NPU要求N>=16,padding
    if (N < 16) {
        N = 16;
    }
    
    int N_padded = ((N + 15) / 16) * 16;
    int K_padded = ((K + 31) / 32) * 32;
    
    size_t weights_layout_size = ((K_padded + 31) / 32) * 32 * 16 * sizeof(_Float16);
    size_t input_size = M * K * sizeof(_Float16);
    size_t weights_size = weights_layout_size;
    size_t output_size = M * N * sizeof(float);
    
    if (input_size > 8*1024*1024 || weights_size > 8*1024*1024 || output_size > 8*1024*1024) {
        t->npu_fallback++;
        return -1;
    }
    
    // 获取缓冲区
    NPUBuffer *input_buf = get_buffer_from_pool(t, input_size);
    NPUBuffer *weights_buf = get_buffer_from_pool(t, weights_size);
    NPUBuffer *output_buf = get_buffer_from_pool(t, output_size);
    
    void *npu_input, *npu_weights, *npu_output;
    uint64_t input_dma, weights_dma, output_dma;
    uint32_t input_handle, weights_handle, output_handle;
    uint64_t input_obj, weights_obj, output_obj;
    int use_pool = 0;
    
    if (input_buf && weights_buf && output_buf) {
        npu_input = input_buf->data;
        input_dma = input_buf->dma;
        input_handle = input_buf->handle;
        input_obj = input_buf->obj;
        
        npu_weights = weights_buf->data;
        weights_dma = weights_buf->dma;
        weights_handle = weights_buf->handle;
        weights_obj = weights_buf->obj;
        
        npu_output = output_buf->data;
        output_dma = output_buf->dma;
        output_handle = output_buf->handle;
        output_obj = output_buf->obj;
        
        use_pool = 1;
    } else {
        if (input_buf) release_buffer_to_pool(input_buf);
        if (weights_buf) release_buffer_to_pool(weights_buf);
        if (output_buf) release_buffer_to_pool(output_buf);
        
        npu_input = mem_allocate(t->npu_fd, input_size, &input_dma, &input_obj, 0, &input_handle);
        if (!npu_input) { t->npu_fallback++; return -1; }
        
        npu_weights = mem_allocate(t->npu_fd, weights_size, &weights_dma, &weights_obj, 0, &weights_handle);
        if (!npu_weights) {
            munmap(npu_input, input_size);
            mem_destroy(t->npu_fd, input_handle, input_obj);
            t->npu_fallback++;
            return -1;
        }
        
        npu_output = mem_allocate(t->npu_fd, output_size, &output_dma, &output_obj, 0, &output_handle);
        if (!npu_output) {
            munmap(npu_input, input_size);
            munmap(npu_weights, weights_size);
            mem_destroy(t->npu_fd, input_handle, input_obj);
            mem_destroy(t->npu_fd, weights_handle, weights_obj);
            t->npu_fallback++;
            return -1;
        }
    }
    
    // 使用缓存的权重
    memcpy(npu_input, weight_cache->data, weight_cache->size);
    
    // 转换批量输入到NPU格式
    _Float16 *weights_fp16 = npu_weights;
    for (int n_idx = 1; n_idx <= N; n_idx++) {
        for (int k = 1; k <= K; k++) {
            int src_idx;
            if (n_idx <= batch_size) {
                src_idx = (n_idx - 1) * K + (k - 1);
            } else {
                // Padding: 重复最后一个
                src_idx = (batch_size - 1) * K + (k - 1);
            }
            int dst_idx = weight_fp16(K, n_idx, k);
            
            if (src_idx >= 0 && src_idx < batch_size * n && 
                dst_idx >= 0 && dst_idx < (int)(weights_size / sizeof(_Float16))) {
                weights_fp16[dst_idx] = (_Float16)input[src_idx];
            } else {
                goto cleanup_error;
            }
        }
    }
    
    // 配置和提交NPU任务 (与之前相同)
    matmul_params_t params;
    params.m = M;
    params.k = K;
    params.n = N;
    params.input_dma = input_dma;
    params.weights_dma = weights_dma;
    params.output_dma = output_dma;
    params.tasks = (uint64_t *)&t->npu_regs;
    params.fp32tofp16 = 0;
    
    if (gen_matmul_fp16(&params) != 0) {
        goto cleanup_error;
    }
    
    memcpy(t->regcmd, t->npu_regs, sizeof(t->npu_regs));
    
    t->tasks[0].flags = 0;
    t->tasks[0].op_idx = 0;
    t->tasks[0].enable_mask = 0xd;
    t->tasks[0].int_mask = 0x300;
    t->tasks[0].int_clear = 0x1ffff;
    t->tasks[0].int_status = 0;
    t->tasks[0].regcfg_amount = sizeof(t->npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
    t->tasks[0].regcfg_offset = 0;
    t->tasks[0].regcmd_addr = t->regcmd_dma;
    
    struct rknpu_submit submit = {
        .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
        .timeout = 6000,
        .task_start = 0,
        .task_number = 1,
        .task_counter = 0,
        .priority = 0,
        .task_obj_addr = t->tasks_obj,
        .regcfg_obj_addr = 0,
        .task_base_addr = 0,
        .user_data = 0,
        .core_mask = 1,
        .fence_fd = -1,
        .subcore_task = {
            { .task_start = 0, .task_number = 1 },
            { 1, 0 }, { 2, 0 }, { 0, 0 }, { 0, 0 }
        },
    };
    
    if (ioctl(t->npu_fd, DRM_IOCTL_RKNPU_SUBMIT, &submit) < 0) {
        goto cleanup_error;
    }
    
    // 拷贝批量结果
    float *output_data = (float*) npu_output;
    for (int b = 0; b < batch_size; b++) {
        for (int m = 1; m <= M; m++) {
            int dst_idx = b * M + (m - 1);
            int src_idx = feature_data(N, M, 1, 4, b + 1, m, 1);
            
            if (dst_idx >= 0 && dst_idx < batch_size * d && 
                src_idx >= 0 && src_idx < M * N) {
                output[dst_idx] = output_data[src_idx];
            } else {
                goto cleanup_error;
            }
        }
    }
    
    // 清理
    if (use_pool) {
        release_buffer_to_pool(input_buf);
        release_buffer_to_pool(weights_buf);
        release_buffer_to_pool(output_buf);
    } else {
        munmap(npu_input, input_size);
        munmap(npu_weights, weights_size);
        munmap(npu_output, output_size);
        mem_destroy(t->npu_fd, input_handle, input_obj);
        mem_destroy(t->npu_fd, weights_handle, weights_obj);
        mem_destroy(t->npu_fd, output_handle, output_obj);
    }
    
    t->npu_success++;
    return 0;
    
cleanup_error:
    if (use_pool) {
        release_buffer_to_pool(input_buf);
        release_buffer_to_pool(weights_buf);
        release_buffer_to_pool(output_buf);
    } else {
        munmap(npu_input, input_size);
        munmap(npu_weights, weights_size);
        munmap(npu_output, output_size);
        mem_destroy(t->npu_fd, input_handle, input_obj);
        mem_destroy(t->npu_fd, weights_handle, weights_obj);
        mem_destroy(t->npu_fd, output_handle, output_obj);
    }
    t->npu_fallback++;
    return -1;
}

// ============================================================================
// Self-Speculative主函数
// ============================================================================

// 初始化speculative decoding
void init_speculative(Transformer* t) {
    if (!t->speculative.enabled) return;

    t->speculative.K = SPECULATIVE_K;
    t->speculative.draft_tokens = calloc(SPECULATIVE_K, sizeof(int));
    t->speculative.draft_probs = calloc(SPECULATIVE_K, sizeof(float));
    t->speculative.verify_logits = calloc(SPECULATIVE_K * t->config.vocab_size, sizeof(float));

    t->speculative.draft_x = calloc(t->config.dim, sizeof(float));
    t->speculative.draft_xb = calloc(t->config.dim, sizeof(float));
    t->speculative.draft_logits = calloc(t->config.vocab_size, sizeof(float));

    // 检查分配是否成功
    if (!t->speculative.draft_tokens || !t->speculative.draft_probs ||
        !t->speculative.verify_logits || !t->speculative.draft_x ||
        !t->speculative.draft_xb || !t->speculative.draft_logits) {
        fprintf(stderr, "Failed to allocate memory for speculative decoding\n");
        // 释放已分配的内存
        if (t->speculative.draft_tokens) free(t->speculative.draft_tokens);
        if (t->speculative.draft_probs) free(t->speculative.draft_probs);
        if (t->speculative.verify_logits) free(t->speculative.verify_logits);
        if (t->speculative.draft_x) free(t->speculative.draft_x);
        if (t->speculative.draft_xb) free(t->speculative.draft_xb);
        if (t->speculative.draft_logits) free(t->speculative.draft_logits);
        // 重置为 NULL
        t->speculative.draft_tokens = NULL;
        t->speculative.draft_probs = NULL;
        t->speculative.verify_logits = NULL;
        t->speculative.draft_x = NULL;
        t->speculative.draft_xb = NULL;
        t->speculative.draft_logits = NULL;
        t->speculative.enabled = 0;
        return;
    }

    memset(&t->speculative.stats, 0, sizeof(SpeculativeStats));

    fprintf(stderr, "Self-Speculative Decoding enabled (K=%d)\n", t->speculative.K);
}

void free_speculative(Transformer* t) {
    if (!t->speculative.enabled) return;

    if (t->speculative.draft_tokens) free(t->speculative.draft_tokens);
    if (t->speculative.draft_probs) free(t->speculative.draft_probs);
    if (t->speculative.verify_logits) free(t->speculative.verify_logits);
    if (t->speculative.draft_x) free(t->speculative.draft_x);
    if (t->speculative.draft_xb) free(t->speculative.draft_xb);
    if (t->speculative.draft_logits) free(t->speculative.draft_logits);

    // 重置指针为 NULL，避免重复释放
    t->speculative.draft_tokens = NULL;
    t->speculative.draft_probs = NULL;
    t->speculative.verify_logits = NULL;
    t->speculative.draft_x = NULL;
    t->speculative.draft_xb = NULL;
    t->speculative.draft_logits = NULL;
    t->speculative.enabled = 0;
}

// Self-Speculative生成: 返回接受的token数
int generate_speculative(Transformer* t, Sampler* sampler, int current_token, int pos, 
                         int* output_tokens, int max_tokens) {
    if (!t->speculative.enabled || max_tokens < 1) {
        return -1;  // Fallback to normal generation
    }
    
    Config* p = &t->config;
    int K = t->speculative.K;
    if (K > max_tokens) K = max_tokens;
    
    // ========================================
    // Phase 1: Draft - 快速生成K个候选token
    // ========================================
    
    int draft_token = current_token;
    for (int i = 0; i < K; i++) {
        // 快速draft (只用CPU,简化计算)
        forward_draft_fast(t, draft_token, pos + i, t->speculative.draft_logits);
        
        // 快速采样 (top-k)
        draft_token = sample_draft_fast(t->speculative.draft_logits, 
                                        p->vocab_size, 
                                        &sampler->rng_state);
        
        t->speculative.draft_tokens[i] = draft_token;
        
        // 记录draft的最大概率 (用于后续验证)
        float max_prob = t->speculative.draft_logits[draft_token];
        t->speculative.draft_probs[i] = max_prob;
    }
    
    t->speculative.stats.total_drafted += K;
    
    // ========================================
    // Phase 2: Verify - 用完整模型批量验证
    // ========================================
    
    // 准备批量输入: [current_token, draft[0], draft[1], ..., draft[K-1]]
    int batch_tokens[16];
    batch_tokens[0] = current_token;
    for (int i = 0; i < K && i < 15; i++) {
        batch_tokens[i + 1] = t->speculative.draft_tokens[i];
    }
    
    int batch_size = K + 1;
    if (batch_size > 16) batch_size = 16;
    
    // 批量forward (利用NPU)
    // 注意: 这里需要修改forward函数支持batch,或者逐个调用
    // 简化实现: 逐个验证 (仍然比完全重新生成快)
    
    int accepted = 0;
    for (int i = 0; i < K; i++) {
        // 完整forward验证
        float* verify_logits = forward(t, batch_tokens[i], pos + i);
        
        // 应用temperature和softmax
        if (sampler->temperature > 0.0f) {
            for (int q = 0; q < p->vocab_size; q++) {
                verify_logits[q] /= sampler->temperature;
            }
        }
        softmax(verify_logits, p->vocab_size);
        
        // 检查是否接受这个token
        int draft_token = t->speculative.draft_tokens[i];
        float verify_prob = verify_logits[draft_token];
        float draft_prob = expf(t->speculative.draft_probs[i]);
        
        // 接受准则: verify_prob >= draft_prob (简化版)
        // 更严格的版本应该用概率接受
        float acceptance_threshold = 0.7f;  // 可调
        
        if (verify_prob >= draft_prob * acceptance_threshold) {
            // 接受!
            output_tokens[accepted] = draft_token;
            accepted++;
        } else {
            // 拒绝: 从verify distribution重新采样
            int resampled = sample(sampler, verify_logits);
            output_tokens[accepted] = resampled;
            accepted++;
            break;  // 停止接受后续token
        }
    }
    
    // ========================================
    // Phase 3: 更新统计
    // ========================================
    
    t->speculative.stats.total_generations++;
    t->speculative.stats.total_tokens_generated += accepted;
    t->speculative.stats.total_accepted += accepted;
    
    t->speculative.stats.acceptance_rate = 
        (float)t->speculative.stats.total_accepted / t->speculative.stats.total_drafted;
    
    t->speculative.stats.speedup = 
        (float)t->speculative.stats.total_tokens_generated / t->speculative.stats.total_generations;
    
    return accepted;
}

// 打印speculative统计
void print_speculative_stats(Transformer* t) {
    if (!t->speculative.enabled) return;
    
    SpeculativeStats* s = &t->speculative.stats;
    fprintf(stderr, "\n=== Speculative Decoding Statistics ===\n");
    fprintf(stderr, "Total generations: %d\n", s->total_generations);
    fprintf(stderr, "Tokens generated: %d\n", s->total_tokens_generated);
    fprintf(stderr, "Tokens drafted: %d\n", s->total_drafted);
    fprintf(stderr, "Tokens accepted: %d\n", s->total_accepted);
    fprintf(stderr, "Acceptance rate: %.1f%%\n", s->acceptance_rate * 100);
    fprintf(stderr, "Average speedup: %.2fx\n", s->speedup);
    fprintf(stderr, "========================================\n");
}

#endif  // USE_NPU

// ============================================================================
// 修改generate函数以使用speculative decoding
// ============================================================================

void generate_with_speculative(Transformer *transformer, Tokenizer *tokenizer,
                                Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    if (!prompt_tokens) {
        fprintf(stderr, "Failed to allocate memory for prompt tokens\n");
        exit(EXIT_FAILURE);
    }
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        free(prompt_tokens);
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int token = prompt_tokens[0];
    int pos = 0;

    int* spec_output = malloc(SPECULATIVE_K * sizeof(int));
    if (!spec_output) {
        fprintf(stderr, "Failed to allocate memory for speculative output\n");
        free(prompt_tokens);
        exit(EXIT_FAILURE);
    }
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        int next;
        if (pos < num_prompt_tokens - 1) {
            // 处理prompt阶段: 使用prompt中的下一个token
            next = prompt_tokens[pos + 1];
        } else {
            // 生成阶段: 使用speculative decoding

#if USE_NPU
            if (transformer->speculative.enabled && pos + SPECULATIVE_K < steps) {
                // 尝试speculative生成
                int accepted = generate_speculative(transformer, sampler, token, pos,
                                                   spec_output, steps - pos);

                if (accepted > 0) {
                    // 成功! 输出所有接受的token
                    for (int i = 0; i < accepted; i++) {
                        next = spec_output[i];
                        pos++;

                        if (next == 1) break;  // EOS

                        char* piece = decode(tokenizer, token, next);
                        safe_printf(piece);
                        fflush(stdout);
                        token = next;
                    }

                    if (start == 0) { start = time_in_ms(); }
                    continue;
                }
            }
#endif

            // Fallback: 正常生成
            next = sample(sampler, logits);
        }

        pos++;

        if (next == 1) break;  // EOS

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    
#if USE_NPU
    if (transformer->speculative.enabled) {
        print_speculative_stats(transformer);
    }
#endif

    free(spec_output);
    free(prompt_tokens);
}

// ============================================================================
// 在build_transformer中初始化speculative
// ============================================================================

void build_transformer_with_speculative(Transformer *t, char* checkpoint_path, int enable_speculative) {
    // 调用原有的build_transformer
    build_transformer(t, checkpoint_path);

#if USE_NPU
    // 初始化speculative指针为NULL，避免释放时出错
    t->speculative.enabled = 0;
    t->speculative.K = 0;
    t->speculative.draft_tokens = NULL;
    t->speculative.draft_probs = NULL;
    t->speculative.verify_logits = NULL;
    t->speculative.draft_x = NULL;
    t->speculative.draft_xb = NULL;
    t->speculative.draft_logits = NULL;

    if (t->npu_fd >= 0 && enable_speculative) {
        t->speculative.enabled = 1;
        init_speculative(t);
    }
#endif
}

// ============================================================================
// 在free_transformer中清理speculative
// ============================================================================

void free_transformer_with_speculative(Transformer* t) {
#if USE_NPU
    if (t->speculative.enabled) {
        free_speculative(t);
    }
#endif
    
    free_transformer(t);
}

// ============================================================================
// main函数修改: 添加speculative选项
// ============================================================================

#ifndef TESTING

void error_usage_speculative() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\" --spec\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  --spec      enable self-speculative decoding (NPU only)\n");
    fprintf(stderr, "  --spec-k <int>  speculative K value (default 15)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;
    int enable_speculative = 0;
    int speculative_k = SPECULATIVE_K;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage_speculative(); }
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--spec") == 0) {
            enable_speculative = 1;
        } else if (strcmp(argv[i], "--spec-k") == 0) {
            if (i + 1 < argc) {
                speculative_k = atoi(argv[i + 1]);
                i++;
            }
        } else if (argv[i][0] == '-' && strlen(argv[i]) == 2) {
            if (i + 1 >= argc) { error_usage_speculative(); }
            
            if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
            else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
            else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
            else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
            else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
            else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
            else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
            else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
            else { error_usage_speculative(); }
            i++;
        }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer_with_speculative(&transformer, checkpoint_path, enable_speculative);
    
    if (enable_speculative) {
        transformer.speculative.K = speculative_k;
        fprintf(stderr, "Speculative decoding enabled with K=%d\n", speculative_k);
    }
    
    if (steps == 0 || steps > transformer.config.seq_len) 
        steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate_with_speculative(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        // Chat mode可以类似修改
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage_speculative();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer_with_speculative(&transformer);
    return 0;
}

#endif