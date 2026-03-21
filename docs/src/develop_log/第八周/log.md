# 第八周开发日志（3.15-3.21）

## 工作总结

- **阶段性代码提交** — 本周完成一次阶段性代码提交 `1d4e8fe`，提交内容包括 DMA syscall、`StarryOS/demo` 用户态测试、`demo/` 下的多进程与混合负载测例、`rknpu` 驱动的单 task 边界调度，以及进程级 NPU 状态导出接口；同时将“硬件寄存器级保存/恢复、毒值写坏、读回校验”实验路径从当前热路径中注释保留，不纳入本次可运行提交。提交链接：[Commit](https://github.com/qc-gpu-driver/starryos-pulsar/commit/1d4e8fea8a5459bef7cff3d9eb0ec2a7dadfa472)
- **NPU 提交路径支持单 task 边界切换** — 将 `submit_ioctrl` 改造成可步进推进的提交路径，在每个 task 完成 IRQ 边界主动让出 CPU，使多个进程可以轮流推进各自未完成的 submit
- **DMA syscall 与用户态辅助库打通** — 新增 `sys_dma_malloc`、`sys_dma_free`、`sys_dump_npu_status`，补齐 DMA 分配、释放和驱动状态导出的最小用户态链路
- **新增多进程/混合工作负载验证** — 新增 `matmul_multi_process` 与 `matmul_llama_concurrent` 两个测试程序，板端验证多进程交错提交和 llama 推理并发场景
- **中断边界寄存器实验先收敛** — 在验证中发现部分 task-window 配置寄存器在“IRQ 完成边界”后并不保证按原值稳定读回，因此暂不把寄存器级恢复实验纳入当前主路径

---

## 单 task 边界调度改造

本周核心工作是把原来“一次 ioctl 一口气跑完整批任务”的同步提交逻辑，改造成“每次只推进一个当前 task-batch，完成后返回调度器”的步进式路径。

驱动侧新增了两类关键状态：

- **resident owner** — 记录当前真正驻留在 NPU 硬件上的 owner，以及它最近一次进入驱动时的 submit 快照
- **task/binding 状态** — 记录某个 `(owner, core)` 当前跑到哪个 task、期望什么中断、最近看到了什么 IRQ 状态

新的路径大致如下：

1. `card1` 在进入 `RKNPU_SUBMIT` 时，用“当前线程 ID + 当前进程 ID + 当前地址空间指针”组装 owner
2. 驱动执行 `submit_ioctrl_step_with_owner()`，本次最多只推进一个 task-batch
3. 每个核心收到预期完成中断后，更新对应 task 的 `int_status`
4. 如果整次 submit 还没结束，就 `yield_now()` 主动让出，允许别的 owner 进来推进它自己的 submit
5. 下次同一个 owner 再进来时，根据 `task_counter` 从上次稳定完成的位置继续

这样之后，进程级抢占点就落在了“单个 task 完成 IRQ 边界”上，而不是“整个 submit 全部完成”之后。

板端日志已经可以观察到这种轮转：

```text
[NPU] owner switch at task boundary: prev(task=16, process=16, aspace=0xffff...) -> next(task=18, process=18, aspace=0xffff...)
[NPU]   batch done: int_status=0x300
[NPU] owner switch at task boundary: prev(task=18, process=18, aspace=0xffff...) -> next(task=15, process=15, aspace=0xffff...)
...
matmul_multi_process: pass
```

日志表明当前已经不是“一个进程一次性独占跑完 20 个 task”，而是在 task 边界发生了真实的 owner 切换。

---

## DMA syscall 与用户态辅助库

为后续调度和状态观察打基础，本周把最小 DMA syscall 链路补齐了。

新增 syscall：

- `sys_dma_malloc` — 为当前进程分配一段 coherent DMA 内存，同时映射到用户地址空间，并返回 DMA/bus 地址
- `sys_dma_free` — 按 `dma_malloc` 返回的原始用户虚拟地址释放 DMA 映射
- `sys_dump_npu_status` — 按当前 owner 视角读取并打印一份完整的 NPU 驱动态快照

为了让这些 syscall 能直接在板端试验，又补了一套轻量用户态库和 demo：

- `StarryOS/demo/libnpu.c` / `libnpu.h`
- `dma_malloc_demo.c`
- `dump_npu_status_demo.c`

此外，`ProcessData` 里增加了 DMA 分配记录表和一个预留的 `npu_isdirty` 标志，方便后续做更完整的 NPU 上下文切换时继续扩展。

---

## 多进程与混合负载测试

为了验证当前这套“单 task 边界切换”不是只在单个 demo 里自娱自乐，本周新增了两个更贴近真实使用方式的测试程序。

### 1. `matmul_multi_process`

这个测试会同时启动 4 个子进程。每个子进程只发起 **1 次 submit**，但这次 submit 里串了 20 个相关联的 matmul task：

- task 0 的输出作为 task 1 的输入
- task 1 的输出再作为 task 2 的输入
- …
- 最终做逐 task、逐元素 CPU 参考校验

它验证的不是“多个短任务抢一个锁”，而是：

- 单个 submit 内还有剩余 task 时，驱动能否在 IRQ 边界切出去
- 之后切回来时，能否从正确的 `task_counter` 继续推进
- 多个进程交错执行后，中间状态和最终输出是否仍然正确

最终板端结果为：

```text
matmul_multi_process: pass
```
---

## 中断边界寄存器实验的收敛

本周原本也同步尝试了更激进的路线：在 IRQ 完成边界读取 live 寄存器快照，把准备恢复的寄存器先写入固定毒值，再按镜像恢复，最后读回校验。

但在板端验证时发现，至少有一部分 **task-window 配置寄存器** 在“任务刚完成、IRQ 已到达”的这个边界之后，并不保证还能按提交时的原值稳定读回。例如曾经观察到：

```text
first_task_shadow_mismatch={ offset=0x100c, expected=0x120, got=0x0 }
first_task_shadow_mismatch={ offset=0x4058, expected=0xf, got=0x0 }
```

这说明两个事实：

- 当前边界对“task 已完成”是稳定的
- 但它不一定对“所有任务窗口寄存器还能按原值读回”稳定

因此本周做了一个明确收敛：

- 保留寄存器级保存/恢复、毒值验证的代码结构
- 但把它们从当前主执行路径中注释掉
- 当前主路径只保留已经能稳定跑通的 **单 task 边界调度**

也就是说，本周提交解决的是“可运行、可验证、可继续扩展”的调度基础设施；**任意寄存器级/算子级抢占** 留到下一轮继续推进。

---

## 本周结论

本周最的进展，是把 **单任务级边界(核irq中断边界)切换 + 多进程交错推进 + 结果正确”** 这一层跑通了。

当前已经可以确认：

- 抢占/切换粒度已经从“整次 submit”下沉到了“单个 task 完成 IRQ 边界”
- 多个进程提交各自含有多 task 的 submit 时，驱动可以轮流推进
- 当前提交内容已经整理为一次独立 commit，后续继续做寄存器级恢复实验时不会和这部分稳定代码混在一起

下一步的重点，将是在当前 task 粒度调度稳定的基础上，再继续推进更细的寄存器级、算子级抢占与恢复。
