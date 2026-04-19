/* Inference for Llama-2 Transformer model in pure C with RK3588 NPU acceleration - OPTIMIZED */

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
#endif
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

#if USE_NPU
// OPTIMIZATION: Pre-convert and cache weight matrix to NPU format
NPUWeightCache* create_weight_cache(int npu_fd, float* weights, int M, int K) {
    // Check if dimensions are NPU-compatible
    if ((M != 1 && (M % 4 != 0 || M > 544)) || K % 32 != 0 || K > 4096) {
        return NULL;  // Can't use NPU for this weight
    }
    
    NPUWeightCache* cache = calloc(1, sizeof(NPUWeightCache));
    if (!cache) return NULL;
    
    // Calculate required size for NPU weight layout
    int K_padded = ((K + 31) / 32) * 32;
    cache->size = M * K * sizeof(_Float16);
    
    // Allocate NPU memory
    cache->data = mem_allocate(npu_fd, cache->size, &cache->dma, &cache->obj, 0, &cache->handle);
    if (!cache->data) {
        free(cache);
        return NULL;
    }
    
    // Convert weights to FP16 format
    _Float16* fp16_data = cache->data;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int src_idx = m * K + k;
            int dst_idx = feature_data(K, M, 1, 8, k + 1, m + 1, 1);
            if (dst_idx >= 0 && dst_idx < M * K) {
                fp16_data[dst_idx] = (_Float16)weights[src_idx];
            }
        }
    }
    
    cache->is_cached = 1;
    return cache;
}

void free_weight_cache(int npu_fd, NPUWeightCache* cache) {
    if (cache && cache->data) {
        munmap(cache->data, cache->size);
        mem_destroy(npu_fd, cache->handle, cache->obj);
        cache->data = NULL;
        cache->is_cached = 0;
    }
    // NOTE: Do not free(cache) here because cache is usually part of an array
    // The array itself will be freed separately
}

// OPTIMIZATION: Initialize larger buffer pool
void init_buffer_pool(Transformer* t) {
    if (t->buffer_pool_initialized) return;
    
    // Pre-allocate buffers for common sizes (reduced to avoid memory exhaustion)
    // Prioritize smaller buffers which are more commonly needed
    size_t sizes[] = {
        512 * 1024,       // 512KB
        1024 * 1024,      // 1MB
        2 * 1024 * 1024,  // 2MB
        512 * 1024,       // 512KB (duplicate for parallel use)
        256 * 1024,       // 256KB
        256 * 1024,       // 256KB (duplicate)
        128 * 1024,       // 128KB
        128 * 1024        // 128KB (duplicate)
    };
    
    int successful = 0;
    for (int i = 0; i < MAX_BUFFER_POOL_SIZE; i++) {
        t->buffer_pool[i].size = sizes[i];
        t->buffer_pool[i].data = mem_allocate(t->npu_fd, sizes[i],
                                              &t->buffer_pool[i].dma,
                                              &t->buffer_pool[i].obj,
                                              0,
                                              &t->buffer_pool[i].handle);
        t->buffer_pool[i].in_use = 0;
        
        if (!t->buffer_pool[i].data) {
            fprintf(stderr, "Warning: Failed to allocate buffer pool entry %d (size=%zu KB)\n", 
                    i, sizes[i] / 1024);
            // Mark as unavailable
            t->buffer_pool[i].size = 0;
        } else {
            successful++;
        }
    }
    
    t->buffer_pool_initialized = 1;
    fprintf(stderr, "Buffer pool initialized: %d/%d buffers allocated successfully\n", 
            successful, MAX_BUFFER_POOL_SIZE);
}

// OPTIMIZATION: Get buffer from pool
NPUBuffer* get_buffer_from_pool(Transformer* t, size_t required_size) {
    for (int i = 0; i < MAX_BUFFER_POOL_SIZE; i++) {
        if (!t->buffer_pool[i].in_use && 
            t->buffer_pool[i].data && 
            t->buffer_pool[i].size > 0 &&  // Check that buffer was successfully allocated
            t->buffer_pool[i].size >= required_size) {
            t->buffer_pool[i].in_use = 1;
            return &t->buffer_pool[i];
        }
    }
    return NULL;  // No suitable buffer found
}

void release_buffer_to_pool(NPUBuffer* buffer) {
    if (buffer) {
        buffer->in_use = 0;
    }
}

void free_buffer_pool(Transformer* t) {
    for (int i = 0; i < MAX_BUFFER_POOL_SIZE; i++) {
        if (t->buffer_pool[i].data && t->buffer_pool[i].size > 0) {
            munmap(t->buffer_pool[i].data, t->buffer_pool[i].size);
            mem_destroy(t->npu_fd, t->buffer_pool[i].handle, t->buffer_pool[i].obj);
            t->buffer_pool[i].data = NULL;
            t->buffer_pool[i].size = 0;
        }
    }
}
#endif

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    
#if USE_NPU
    // Initialize NPU
    fprintf(stderr, "Initializing NPU with optimizations...\n");
    t->npu_fd = npu_open();
    if (t->npu_fd < 0) {
        fprintf(stderr, "Warning: Failed to open NPU device, will use CPU only\n");
        t->npu_fd = -1;
        return;
    }
    
    // Allocate NPU control structures
    t->regcmd = mem_allocate(t->npu_fd, 1024, &t->regcmd_dma, &t->regcmd_obj, 0, &t->regcmd_handle);
    t->tasks = mem_allocate(t->npu_fd, 1024, &t->tasks_dma, &t->tasks_obj, 
                           RKNPU_MEM_KERNEL_MAPPING, &t->tasks_handle);
    
    if (t->regcmd == NULL || t->tasks == NULL) {
        fprintf(stderr, "Warning: Failed to allocate NPU memory, will use CPU only\n");
        if (t->regcmd) {
            munmap(t->regcmd, 1024);
            mem_destroy(t->npu_fd, t->regcmd_handle, t->regcmd_obj);
        }
        if (t->tasks) {
            munmap(t->tasks, 1024);
            mem_destroy(t->npu_fd, t->tasks_handle, t->tasks_obj);
        }
        npu_close(t->npu_fd);
        t->npu_fd = -1;
        return;
    }
    
    // Reset NPU
    npu_reset(t->npu_fd);
    
    // OPTIMIZATION: Initialize larger buffer pool
    t->buffer_pool_initialized = 0;
    init_buffer_pool(t);
    
    // OPTIMIZATION: Pre-convert weights to NPU format
    fprintf(stderr, "Pre-converting weights to NPU format...\n");
    t->weight_cache.enabled = 1;
    
    int dim = t->config.dim;
    int hidden_dim = t->config.hidden_dim;
    int n_layers = t->config.n_layers;
    int kv_dim = (dim * t->config.n_kv_heads) / t->config.n_heads;
    
    // Allocate cache arrays
    t->weight_cache.wq_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.wk_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.wv_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.wo_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.w1_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.w2_cache = calloc(n_layers, sizeof(NPUWeightCache));
    t->weight_cache.w3_cache = calloc(n_layers, sizeof(NPUWeightCache));
    
    int cached_count = 0;
    int total_count = 0;
    
    // Pre-convert layer weights
    for (int l = 0; l < n_layers; l++) {
        // Attention weights
        NPUWeightCache* wq = create_weight_cache(t->npu_fd, t->weights.wq + l*dim*dim, dim, dim);
        if (wq) { t->weight_cache.wq_cache[l] = *wq; free(wq); cached_count++; }
        total_count++;
        
        NPUWeightCache* wk = create_weight_cache(t->npu_fd, t->weights.wk + l*dim*kv_dim, kv_dim, dim);
        if (wk) { t->weight_cache.wk_cache[l] = *wk; free(wk); cached_count++; }
        total_count++;
        
        NPUWeightCache* wv = create_weight_cache(t->npu_fd, t->weights.wv + l*dim*kv_dim, kv_dim, dim);
        if (wv) { t->weight_cache.wv_cache[l] = *wv; free(wv); cached_count++; }
        total_count++;
        
        NPUWeightCache* wo = create_weight_cache(t->npu_fd, t->weights.wo + l*dim*dim, dim, dim);
        if (wo) { t->weight_cache.wo_cache[l] = *wo; free(wo); cached_count++; }
        total_count++;
        
        // FFN weights
        NPUWeightCache* w1 = create_weight_cache(t->npu_fd, t->weights.w1 + l*dim*hidden_dim, hidden_dim, dim);
        if (w1) { t->weight_cache.w1_cache[l] = *w1; free(w1); cached_count++; }
        total_count++;
        
        NPUWeightCache* w2 = create_weight_cache(t->npu_fd, t->weights.w2 + l*hidden_dim*dim, dim, hidden_dim);
        if (w2) { t->weight_cache.w2_cache[l] = *w2; free(w2); cached_count++; }
        total_count++;
        
        NPUWeightCache* w3 = create_weight_cache(t->npu_fd, t->weights.w3 + l*dim*hidden_dim, hidden_dim, dim);
        if (w3) { t->weight_cache.w3_cache[l] = *w3; free(w3); cached_count++; }
        total_count++;
    }
    
    // Classifier weights
    t->weight_cache.wcls_cache = calloc(1, sizeof(NPUWeightCache));
    NPUWeightCache* wcls = create_weight_cache(t->npu_fd, t->weights.wcls, t->config.vocab_size, dim);
    if (wcls) { 
        *t->weight_cache.wcls_cache = *wcls; 
        free(wcls); 
        cached_count++; 
    }
    total_count++;
    
    // Initialize statistics
    t->npu_calls = 0;
    t->npu_success = 0;
    t->npu_fallback = 0;
    
    fprintf(stderr, "NPU initialized successfully:\n");
    fprintf(stderr, "  - Cached %d/%d weight matrices (%.1f%%)\n", 
            cached_count, total_count, 100.0 * cached_count / total_count);
    fprintf(stderr, "  - Buffer pool: %d buffers\n", MAX_BUFFER_POOL_SIZE);
#endif
}

void free_transformer(Transformer* t) {
#if USE_NPU    
    // Clean up NPU resources
    if (t->npu_fd >= 0) {
        // Free weight cache
        if (t->weight_cache.enabled) {
            for (int l = 0; l < t->config.n_layers; l++) {
                if (t->weight_cache.wq_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.wq_cache[l]);
                if (t->weight_cache.wk_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.wk_cache[l]);
                if (t->weight_cache.wv_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.wv_cache[l]);
                if (t->weight_cache.wo_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.wo_cache[l]);
                if (t->weight_cache.w1_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.w1_cache[l]);
                if (t->weight_cache.w2_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.w2_cache[l]);
                if (t->weight_cache.w3_cache[l].is_cached)
                    free_weight_cache(t->npu_fd, &t->weight_cache.w3_cache[l]);
            }

            // wcls_cache is separately allocated, so handle it specially
            if (t->weight_cache.wcls_cache) {
                if (t->weight_cache.wcls_cache->is_cached) {
                    // Free NPU resources
                    if (t->weight_cache.wcls_cache->data) {
                        munmap(t->weight_cache.wcls_cache->data, t->weight_cache.wcls_cache->size);
                        mem_destroy(t->npu_fd, t->weight_cache.wcls_cache->handle, t->weight_cache.wcls_cache->obj);
                    }
                }
                // Free the wcls_cache structure itself
                free(t->weight_cache.wcls_cache);
            }

            // Free the arrays themselves
            free(t->weight_cache.wq_cache);
            free(t->weight_cache.wk_cache);
            free(t->weight_cache.wv_cache);
            free(t->weight_cache.wo_cache);
            free(t->weight_cache.w1_cache);
            free(t->weight_cache.w2_cache);
            free(t->weight_cache.w3_cache);
        }
        
        // Free buffer pool
        if (t->buffer_pool_initialized) {
            free_buffer_pool(t);
        }
        
        if (t->regcmd != NULL) {
            munmap(t->regcmd, 1024);
            mem_destroy(t->npu_fd, t->regcmd_handle, t->regcmd_obj);
        }
        if (t->tasks != NULL) {
            munmap(t->tasks, 1024);
            mem_destroy(t->npu_fd, t->tasks_handle, t->tasks_obj);
        }
        npu_close(t->npu_fd);
    }
#endif
    
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

#if USE_NPU
// OPTIMIZED: NPU accelerated matrix-vector multiplication with cached weights
int matmul_npu_cached(Transformer* t, float* xout, float* x, NPUWeightCache* weight_cache, 
                      int n, int d, int layer_idx) {
    // Use pre-converted cached weights if available
    if (!weight_cache || !weight_cache->is_cached) {
        return -1;  // No cache available
    }
    
    t->npu_calls++;
    
    // Check dimension constraints for NPU
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
    int N = 16;  // NPU requires N >= 16
    
    // Calculate buffer sizes
    int N_padded = ((N + 15) / 16) * 16;
    int K_padded = ((K + 31) / 32) * 32;
    
    size_t weights_layout_size = ((K_padded + 31) / 32) * 32 * 16 * sizeof(_Float16);
    size_t input_size = M * K * sizeof(_Float16);
    size_t weights_size = weights_layout_size;
    size_t output_size = M * N * sizeof(float);
    
    // Sanity check
    if (input_size > 8*1024*1024 || weights_size > 8*1024*1024 || output_size > 8*1024*1024) {
        t->npu_fallback++;
        return -1;
    }
    
    // OPTIMIZATION: Try to get buffers from pool
    NPUBuffer *input_buf = get_buffer_from_pool(t, input_size);
    NPUBuffer *weights_buf = get_buffer_from_pool(t, weights_size);
    NPUBuffer *output_buf = get_buffer_from_pool(t, output_size);
    
    void *input, *weights, *output;
    uint64_t input_dma, weights_dma, output_dma;
    uint32_t input_handle, weights_handle, output_handle;
    uint64_t input_obj, weights_obj, output_obj;
    int use_pool = 0;
    
    if (input_buf && weights_buf && output_buf) {
        // Use pool buffers
        input = input_buf->data;
        input_dma = input_buf->dma;
        input_handle = input_buf->handle;
        input_obj = input_buf->obj;
        
        weights = weights_buf->data;
        weights_dma = weights_buf->dma;
        weights_handle = weights_buf->handle;
        weights_obj = weights_buf->obj;
        
        output = output_buf->data;
        output_dma = output_buf->dma;
        output_handle = output_buf->handle;
        output_obj = output_buf->obj;
        
        use_pool = 1;
    } else {
        // Fallback to dynamic allocation
        if (input_buf) release_buffer_to_pool(input_buf);
        if (weights_buf) release_buffer_to_pool(weights_buf);
        if (output_buf) release_buffer_to_pool(output_buf);
        
        input = mem_allocate(t->npu_fd, input_size, &input_dma, &input_obj, 0, &input_handle);
        if (!input) { t->npu_fallback++; return -1; }
        
        weights = mem_allocate(t->npu_fd, weights_size, &weights_dma, &weights_obj, 0, &weights_handle);
        if (!weights) {
            munmap(input, input_size);
            mem_destroy(t->npu_fd, input_handle, input_obj);
            t->npu_fallback++;
            return -1;
        }
        
        output = mem_allocate(t->npu_fd, output_size, &output_dma, &output_obj, 0, &output_handle);
        if (!output) {
            munmap(input, input_size);
            munmap(weights, weights_size);
            mem_destroy(t->npu_fd, input_handle, input_obj);
            mem_destroy(t->npu_fd, weights_handle, weights_obj);
            t->npu_fallback++;
            return -1;
        }
    }
    
    // OPTIMIZATION: Removed unnecessary memset calls - data will be overwritten
    
    // OPTIMIZATION: Use cached weight data directly (already in NPU format)
    memcpy(input, weight_cache->data, weight_cache->size);
    
    // Convert input vector x[n] to NPU weights format
    _Float16 *weights_fp16 = weights;
    for (int n_idx = 1; n_idx <= N; n_idx++) {
        for (int k = 1; k <= K; k++) {
            int src_idx = k - 1;  // Repeat same input for all N
            int dst_idx = weight_fp16(K, n_idx, k);
            
            if (src_idx >= 0 && src_idx < n && dst_idx >= 0 && dst_idx < (int)(weights_size / sizeof(_Float16))) {
                weights_fp16[dst_idx] = (_Float16)x[src_idx];
            } else {
                goto cleanup_error;
            }
        }
    }
    
    // Configure NPU
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
    
    // Setup task
    t->tasks[0].flags = 0;
    t->tasks[0].op_idx = 0;
    t->tasks[0].enable_mask = 0xd;
    t->tasks[0].int_mask = 0x300;
    t->tasks[0].int_clear = 0x1ffff;
    t->tasks[0].int_status = 0;
    t->tasks[0].regcfg_amount = sizeof(t->npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
    t->tasks[0].regcfg_offset = 0;
    t->tasks[0].regcmd_addr = t->regcmd_dma;
    
    // Submit to NPU
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
    
    // Copy results back
    float *output_data = (float*) output;
    for (int m = 1; m <= M; m++) {
        int dst_idx = m - 1;
        int src_idx = feature_data(N, M, 1, 4, 1, m, 1);  // Only extract first result
        
        if (dst_idx >= 0 && dst_idx < d && src_idx >= 0 && src_idx < M * N) {
            xout[dst_idx] = output_data[src_idx];
        } else {
            goto cleanup_error;
        }
    }
    
    // Clean up
    if (use_pool) {
        release_buffer_to_pool(input_buf);
        release_buffer_to_pool(weights_buf);
        release_buffer_to_pool(output_buf);
    } else {
        munmap(input, input_size);
        munmap(weights, weights_size);
        munmap(output, output_size);
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
        munmap(input, input_size);
        munmap(weights, weights_size);
        munmap(output, output_size);
        mem_destroy(t->npu_fd, input_handle, input_obj);
        mem_destroy(t->npu_fd, weights_handle, weights_obj);
        mem_destroy(t->npu_fd, output_handle, output_obj);
    }
    t->npu_fallback++;
    return -1;
}

// Wrapper function for matmul with automatic cache lookup
int matmul_npu_with_cache(Transformer* t, float* xout, float* x, float* w, 
                          int n, int d, int layer_idx, const char* weight_type) {
    if (!t->weight_cache.enabled) {
        return -1;
    }
    
    NPUWeightCache* cache = NULL;
    
    // Lookup cached weight
    if (strcmp(weight_type, "wq") == 0) {
        cache = &t->weight_cache.wq_cache[layer_idx];
    } else if (strcmp(weight_type, "wk") == 0) {
        cache = &t->weight_cache.wk_cache[layer_idx];
    } else if (strcmp(weight_type, "wv") == 0) {
        cache = &t->weight_cache.wv_cache[layer_idx];
    } else if (strcmp(weight_type, "wo") == 0) {
        cache = &t->weight_cache.wo_cache[layer_idx];
    } else if (strcmp(weight_type, "w1") == 0) {
        cache = &t->weight_cache.w1_cache[layer_idx];
    } else if (strcmp(weight_type, "w2") == 0) {
        cache = &t->weight_cache.w2_cache[layer_idx];
    } else if (strcmp(weight_type, "w3") == 0) {
        cache = &t->weight_cache.w3_cache[layer_idx];
    } else if (strcmp(weight_type, "wcls") == 0) {
        cache = t->weight_cache.wcls_cache;
    }
    
    return matmul_npu_cached(t, xout, x, cache, n, d, layer_idx);
}
#endif

void matmul(Transformer* transformer, float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
#if USE_NPU
    // Try NPU acceleration first
    if (transformer && transformer->npu_fd >= 0) {
        // Note: For general matmul without layer context, we can't use cache
        // This is mainly for classifier or cases where we don't have layer info
        
        // Check if this is a valid NPU operation
        if ((d == 1 || (d % 4 == 0 && d <= 544)) && n % 32 == 0 && n <= 4096) {
            // For now, fallback to CPU for non-cached operations
            // In a future optimization, we could cache on-the-fly
        }
    }
#endif

    // CPU fallback
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// OPTIMIZATION: Wrapper for matmul with layer and weight type information
void matmul_layer(Transformer* transformer, float* xout, float* x, float* w, 
                  int n, int d, int layer_idx, const char* weight_type) {
#if USE_NPU
    // Try cached NPU acceleration
    if (transformer && transformer->npu_fd >= 0 && transformer->weight_cache.enabled) {
        if (matmul_npu_with_cache(transformer, xout, x, w, n, d, layer_idx, weight_type) == 0) {
            return;  // NPU succeeded
        }
    }
#endif
    
    // CPU fallback
    matmul(transformer, xout, x, w, n, d);
}

float* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // OPTIMIZATION: qkv matmuls with cached weights
        matmul_layer(transformer, s->q, s->xb, w->wq + l*dim*dim, dim, dim, l, "wq");
        matmul_layer(transformer, s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim, l, "wk");
        matmul_layer(transformer, s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim, l, "wv");

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul_layer(transformer, s->xb2, s->xb, w->wo + l*dim*dim, dim, dim, l, "wo");

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
        matmul_layer(transformer, s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim, l, "w1");
        matmul_layer(transformer, s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim, l, "w3");

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul_layer(transformer, s->xb, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim, l, "w2");

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul_layer(transformer, s->logits, x, w->wcls, p->dim, p->vocab_size, 0, "wcls");
    
    return s->logits;
}

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

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

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

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

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

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    int8_t user_turn = 1;
    int next;
    int token;
    int prev_token;
    int pos = 0;
    
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        
        if (token == 2) { user_turn = 1; }

        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI

#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
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

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif