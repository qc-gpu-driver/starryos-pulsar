// Copyright (c) 2026
//
// 3-thread concurrent YOLOv8 inference demo:
// - thread0 binds RKNN_NPU_CORE_0
// - thread1 binds RKNN_NPU_CORE_1
// - thread2 binds RKNN_NPU_CORE_2
// Each thread scans its own image folder and runs person detection.

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include "image_drawing.h"
#include "image_utils.h"
#include "yolov8.h"

namespace {

constexpr int kThreadCount = 3;
constexpr int kPersonClassId = 0;
constexpr size_t kPathMax = 512;

pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int thread_id;
    rknn_core_mask core_mask;
    const char *model_path;
    char input_dir[kPathMax];
    char output_dir[kPathMax];

    int image_count;
    int person_image_count;
    int total_person_boxes;
    int failed_images;
    uint64_t elapsed_us;
    int status;
} thread_ctx_t;

static uint64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static void thread_log(int thread_id, const char *fmt, ...) {
    pthread_mutex_lock(&g_log_mutex);
    printf("[thread-%d] ", thread_id);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    pthread_mutex_unlock(&g_log_mutex);
}

static int ensure_dir_exists(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return 0;
        }
        fprintf(stderr, "path exists but is not directory: %s\n", path);
        return -1;
    }

    if (mkdir(path, 0755) == 0) {
        return 0;
    }

    if (errno == EEXIST) {
        return 0;
    }

    fprintf(stderr, "mkdir failed: %s, errno=%d (%s)\n", path, errno, strerror(errno));
    return -1;
}

static bool has_supported_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) {
        return false;
    }

    return strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 ||
           strcmp(ext, ".JPG") == 0 || strcmp(ext, ".JPEG") == 0 ||
           strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0 ||
           strcmp(ext, ".data") == 0;
}

static int collect_images(const char *dir_path, std::vector<std::string> *paths) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "opendir failed: %s, errno=%d (%s)\n", dir_path, errno, strerror(errno));
        return -1;
    }

    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!has_supported_image_ext(entry->d_name)) {
            continue;
        }

        std::string full_path = std::string(dir_path) + "/" + entry->d_name;

        struct stat st;
        if (stat(full_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }

        paths->emplace_back(full_path);
    }

    closedir(dir);
    std::sort(paths->begin(), paths->end());
    return 0;
}

static const char *base_name(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? (slash + 1) : path;
}

static void strip_extension(const char *name, char *out, size_t out_size) {
    snprintf(out, out_size, "%s", name);
    char *dot = strrchr(out, '.');
    if (dot) {
        *dot = '\0';
    }
}

static void draw_person_detections(image_buffer_t *image, const object_detect_result_list *results, int *person_boxes) {
    *person_boxes = 0;
    char text[128];

    for (int i = 0; i < results->count; ++i) {
        const object_detect_result *det = &results->results[i];
        if (det->cls_id != kPersonClassId) {
            continue;
        }

        int x1 = det->box.left;
        int y1 = det->box.top;
        int x2 = det->box.right;
        int y2 = det->box.bottom;
        int w = x2 - x1;
        int h = y2 - y1;

        draw_rectangle(image, x1, y1, w, h, COLOR_BLUE, 3);
        snprintf(text, sizeof(text), "person %.1f%%", det->prop * 100.0f);
        draw_text(image, text, x1, y1 - 20, COLOR_RED, 10);
        (*person_boxes)++;
    }
}

static void *worker_entry(void *arg) {
    thread_ctx_t *ctx = reinterpret_cast<thread_ctx_t *>(arg);
    uint64_t start_us = now_us();
    rknn_app_context_t app_ctx;
    std::vector<std::string> image_paths;
    int ret = 0;

    memset(&app_ctx, 0, sizeof(app_ctx));
    ctx->status = -1;

    ret = init_yolov8_model(ctx->model_path, &app_ctx);
    if (ret != 0) {
        thread_log(ctx->thread_id, "init_yolov8_model failed: ret=%d model=%s\n", ret, ctx->model_path);
        goto out;
    }

    ret = rknn_set_core_mask(app_ctx.rknn_ctx, ctx->core_mask);
    if (ret != RKNN_SUCC) {
        thread_log(ctx->thread_id, "rknn_set_core_mask failed: ret=%d core_mask=%d\n", ret, (int)ctx->core_mask);
        goto out_release;
    }

    thread_log(ctx->thread_id, "bind core mask success: %d\n", (int)ctx->core_mask);

    if (collect_images(ctx->input_dir, &image_paths) != 0) {
        thread_log(ctx->thread_id, "collect_images failed: %s\n", ctx->input_dir);
        goto out_release;
    }
    if (image_paths.empty()) {
        thread_log(ctx->thread_id, "no image found in %s\n", ctx->input_dir);
    }

    for (size_t i = 0; i < image_paths.size(); ++i) {
        const std::string &image_path = image_paths[i];
        image_buffer_t src_image;
        object_detect_result_list od_results;
        int person_boxes = 0;
        char output_stem[kPathMax];

        memset(&src_image, 0, sizeof(src_image));
        memset(&od_results, 0, sizeof(od_results));

        ret = read_image(image_path.c_str(), &src_image);
        if (ret != 0) {
            ctx->failed_images++;
            thread_log(ctx->thread_id, "read_image failed: %s\n", image_path.c_str());
            continue;
        }

        ret = inference_yolov8_model(&app_ctx, &src_image, &od_results);
        if (ret != 0) {
            ctx->failed_images++;
            thread_log(ctx->thread_id, "inference failed: %s ret=%d\n", image_path.c_str(), ret);
            if (src_image.virt_addr) {
                free(src_image.virt_addr);
            }
            continue;
        }

        draw_person_detections(&src_image, &od_results, &person_boxes);
        if (person_boxes > 0) {
            ctx->person_image_count++;
            ctx->total_person_boxes += person_boxes;
        }
        ctx->image_count++;

        strip_extension(base_name(image_path.c_str()), output_stem, sizeof(output_stem));
        std::string output_path = std::string(ctx->output_dir) + "/" + output_stem +
                                  "_thread" + std::to_string(ctx->thread_id) + ".png";
        write_image(output_path.c_str(), &src_image);

        thread_log(ctx->thread_id,
                   "done %s -> %s, persons=%d/%d\n",
                   image_path.c_str(),
                   output_path.c_str(),
                   person_boxes,
                   od_results.count);

        if (src_image.virt_addr) {
            free(src_image.virt_addr);
        }
    }

    ctx->status = 0;

out_release:
    release_yolov8_model(&app_ctx);
out:
    ctx->elapsed_us = now_us() - start_us;
    return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "model/yolov8.rknn";
    const rknn_core_mask core_masks[kThreadCount] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    pthread_t threads[kThreadCount];
    thread_ctx_t thread_ctx[kThreadCount];
    uint64_t total_start_us;
    uint64_t total_wall_us;
    int overall_status = 0;

    printf("yolov8 3-thread core-bind demo start\n");
    printf("model: %s\n", model_path);
    printf("input dirs: model/wait_process_0,1,2\n");

    if (init_post_process() != 0) {
        fprintf(stderr, "init_post_process failed\n");
        return 1;
    }

    for (int i = 0; i < kThreadCount; ++i) {
        memset(&thread_ctx[i], 0, sizeof(thread_ctx[i]));
        thread_ctx[i].thread_id = i;
        thread_ctx[i].core_mask = core_masks[i];
        thread_ctx[i].model_path = model_path;
        snprintf(thread_ctx[i].input_dir, sizeof(thread_ctx[i].input_dir), "model/wait_process_%d", i);
        snprintf(thread_ctx[i].output_dir, sizeof(thread_ctx[i].output_dir), "model/wait_process_%d_out", i);

        if (ensure_dir_exists(thread_ctx[i].output_dir) != 0) {
            overall_status = 1;
            goto out;
        }
    }

    total_start_us = now_us();
    for (int i = 0; i < kThreadCount; ++i) {
        int err = pthread_create(&threads[i], nullptr, worker_entry, &thread_ctx[i]);
        if (err != 0) {
            fprintf(stderr, "pthread_create failed: thread=%d err=%d (%s)\n", i, err, strerror(err));
            overall_status = 1;
            for (int j = 0; j < i; ++j) {
                pthread_join(threads[j], nullptr);
            }
            goto out;
        }
    }

    for (int i = 0; i < kThreadCount; ++i) {
        pthread_join(threads[i], nullptr);
    }
    total_wall_us = now_us() - total_start_us;

    {
        int total_images = 0;
        int total_failed = 0;
        int total_person_images = 0;
        int total_person_boxes = 0;
        uint64_t accumulated_thread_us = 0;

        printf("\n================ per-thread summary ================\n");
        for (int i = 0; i < kThreadCount; ++i) {
            const thread_ctx_t *ctx = &thread_ctx[i];
            double fps = ctx->elapsed_us > 0 ? ((double)ctx->image_count * 1000000.0 / (double)ctx->elapsed_us) : 0.0;

            if (ctx->status != 0) {
                overall_status = 1;
            }

            total_images += ctx->image_count;
            total_failed += ctx->failed_images;
            total_person_images += ctx->person_image_count;
            total_person_boxes += ctx->total_person_boxes;
            accumulated_thread_us += ctx->elapsed_us;

            printf("thread-%d core_mask=%d images=%d person_images=%d person_boxes=%d failed=%d "
                   "elapsed=%.3f ms fps=%.2f\n",
                   i,
                   (int)ctx->core_mask,
                   ctx->image_count,
                   ctx->person_image_count,
                   ctx->total_person_boxes,
                   ctx->failed_images,
                   (double)ctx->elapsed_us / 1000.0,
                   fps);
        }

        printf("================ overall summary ===================\n");
        printf("total_wall=%.3f ms accumulated_thread=%.3f ms overlap=%.3f x\n",
               (double)total_wall_us / 1000.0,
               (double)accumulated_thread_us / 1000.0,
               total_wall_us > 0 ? (double)accumulated_thread_us / (double)total_wall_us : 0.0);
        printf("total_images=%d person_images=%d person_boxes=%d failed=%d throughput=%.2f img/s\n",
               total_images,
               total_person_images,
               total_person_boxes,
               total_failed,
               total_wall_us > 0 ? ((double)total_images * 1000000.0 / (double)total_wall_us) : 0.0);
    }

out:
    deinit_post_process();
    return overall_status;
}
