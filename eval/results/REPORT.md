# 长期记忆智能体实验报告

> 最后运行：2026 年 11 月 3 日。主口径为 `data/dataset_large.jsonl` 的 219 条记录与 `eval/benchmark_large.jsonl` 的 48 题。

## 1. 三架构对照

| 方案 | 近期 | 矛盾 | 总结 | 时间 | 时间线 | 诱导拒答 | 综合 | 记忆条目 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A 最近 3 条 | 0% | 0% | 17% | 0% | 11% | 0% | 3.8% | 0 |
| B 全量历史，不治理 | 96% | 90% | 61% | 100% | 33% | 0% | 79.2% | 219 |
| **C 分层治理与三路检索** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100.0%** | 活跃 48，档案 143 |

C 相比 B 提高 **20.8pt**。最终状态为 active 48、compressed 132、superseded 11、discarded 28；13 组冲突全部解决，形成 10 条结论。

原始结果：`large_metrics.json`、`large_evidence.json`。

## 2. 检索消融

| 配置 | 综合 | 相对全开 |
|---|---:|---:|
| BM25 | 92.7% | -7.3pt |
| 词法 + 时间 | 96.9% | -3.1pt |
| 词法 + 主题图 | 94.8% | -5.2pt |
| **三路 RRF** | **100.0%** | - |

| 关键机制 | 综合 | 变化 |
|---|---:|---:|
| 全开 | 100.0% | - |
| 关闭日历窗口 | 95.8% | -4.2pt |
| 关闭档案回溯 | 82.3% | -17.7pt |
| 关闭多轮改写 | 100.0% | 0pt |

结论：时间语义、主题关系和档案回溯是当前准确率的主要来源。多轮改写在现有确定性检索已充分时不是必要条件，仅保留为在线深度模式。

## 3. 任务分工对照

| 方案 | 综合 | 冲突发现/解决 |
|---|---:|---:|
| 关闭独立冲突治理 | 97.9% | 无独立账本 |
| **记录抽取、冲突治理、证据检索、回答审核完整分工** | **100.0%** | **13/13** |

完整分工比对照高 2.1pt，并生成 13 组已解决冲突及对应取代链。两组实验使用相同数据、题集和检索机制，变量为是否执行独立冲突治理。

## 4. 其他验证

| 验证 | 结果 |
|---|---:|
| 固定语义压力集 | 76/76，路由、目标召回和干扰隔离均 100% |
| 扩展语义压力集 | 126/126，其中记忆题 90、通用题 36 |
| 拟真自然笔记回放 | 3/3 |
| 小集架构回归 | C 98.6% |
| 40 用例端到端评测 | 98.8% |
| 自动测试 | 141/141 |
| 本体导出 | 1233 条 RDF 三元组 |

## 5. 复现命令

```powershell
python scripts\run_large_eval.py
python scripts\run_large_evidence.py
python scripts\run_notes_replay.py
python scripts\run_ablation.py
python scripts\run_agent_compare.py
python scripts\run_forget_curve.py
python scripts\run_memory_evolution.py
python scripts\run_ontology_demo.py
python scripts\run_model_stress.py --fixed
python scripts\run_model_stress.py --reuse
python scripts\run_eval.py
python eval.py
python -m unittest discover -s tests -v
```

所有脚本均从项目代码目录执行。固定实验脚本强制 `MEMAGENT_LLM=0`；在线模型属于可选语义增强能力，不计入本报告的确定性指标。
