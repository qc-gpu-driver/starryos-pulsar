#include "libnpu.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

static void init_empty_submit(libnpu_submit *submit) {
  memset(submit, 0, sizeof(*submit));
  submit->timeout = 1000;
  submit->core_mask = 0x7;
}

int main(int argc, char **argv) {
  libnpu_submit submit;
  const libnpu_submit *submit_ptr = &submit;

  if (argc > 1 && strcmp(argv[1], "--null") == 0) {
    submit_ptr = NULL;
  } else {
    init_empty_submit(&submit);
  }

  printf("npu_dump_status demo\n");
  printf("  submit_ptr = %s\n", submit_ptr == NULL ? "NULL" : "empty submit");

  if (npu_dump_status(submit_ptr) != 0) {
    fprintf(stderr, "npu_dump_status failed: %s\n", strerror(errno));
    return 1;
  }

  puts("npu_dump_status ok");
  puts("check kernel log/console for \"Process NPU State\"");
  return 0;
}
