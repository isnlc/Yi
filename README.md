# 长期记忆智能体「忆」

「忆」面向个人长期记录，包含记忆准入、版本治理、分层存储、证据检索和多模态交互。系统通过 OpenAI 兼容接口接入文字模型，用于语义分类、通用问答、个性化建议和附件综合；正式记忆写入、版本关系和证据范围仍由本地机制控制。

代码仓库：<https://github.com/isnlc/Yi>

## 核心架构

系统采用“统一接入、统一编排、任务分流、共享状态”的结构。多模态内容先被解析为带来源标签的统一请求，再由编排器识别任务类型、执行记忆准入，并选择仅检索、检索后推理或通用问答。记录抽取、冲突治理、证据检索和回答审核是统一编排器中的四项智能体任务，共享同一记忆状态和流程日志。

### 记忆写入与治理任务

```text
统一编排器
  → 来源与安全检查
  → 明确记录 / 高置信事实 / 待确认候选 / 普通对话 / 附件资料
  → 结构化语义
  → 冲突裁决与取代链
  → active / compressed / superseded / discarded / conclusion
```

- 模型不能直接写记忆；候选必须经过原文依据和本地策略校验。
- 文档正文与 OCR 默认进入资料库，不自动成为用户事实。
- 冲突先校验记录是否指向同一对象；同日更新再使用全局写入序判断先后，旧版本保留用于历史回溯。

### 检索与作答任务

```text
统一编排器
  → 是否依赖个人记忆
  → 是否需要 LLM 综合
  → retrieval_only / retrieval_llm / general_llm
```

- `retrieval_only`：当前事实、时间、历史状态和配置查询，使用本地确定性回答。
- `retrieval_llm`：先检索个人证据，再由 LLM 完成建议、比较或总结。
- `general_llm`：普通知识和创作，不读取个人记忆。

检索由 BM25、时间窗和主题图三路组成，使用 RRF 融合。现状查询裁决当前版本，历史查询保留旧状态，演变查询保留前后链路。

“我最近做了什么”“这几天有哪些活动”等宽泛时间回顾统一使用时间窗查询，不再被当作当前状态压缩为单条记录。未指定天数时默认七天；若当前七天处于数据静默期，系统回退到最近有记录的同长度窗口，并在回答中明确标注实际日期范围。带具体对象的问法仍同时执行对象过滤；“最近一次”与“现在还在”分别保留事件查询和当前状态语义。

## 产品能力

- 文字、语音输入与回答朗读。
- 多图片、多文件和文字组合发送。
- TXT、MD、CSV、JSON、LOG、PDF、DOCX、XLSX、PPTX 解析。
- 图片与扫描 PDF 的 OCR；配置视觉模型时可分析物体和场景。
- 记忆历书、记忆库、取代链、淡忘与恢复。
- 待确认候选、独立资料库、折叠查看附件内容。
- 处理过程、证据、记忆洞察和 JSON 导出。

旧版 `.doc/.xls/.ppt` 不直接解析，需转换为现代格式。

## 快速运行

```powershell
pip install -r requirements.txt

# 复制配置文件，并在 .env 中填写接口地址、密钥和模型名称
Copy-Item .env.example .env
python web/app.py --port 8765
```

打开 `http://127.0.0.1:8765`。

系统会自动读取项目目录下的 `.env`。模型接口暂时不可用时，系统使用本地降级路径，保留记忆治理、确定性事实检索、文档解析和 Windows OCR；完整交互与正式展示应启用文字模型接口。图片中的物体、场景和图文关系分析需要另行配置视觉模型接口。

## 最新实验

| 项目 | 结果 |
|---|---:|
| 219 条、48 题：最近 3 条 | 3.8% |
| 219 条、48 题：全量历史 | 79.2% |
| 219 条、48 题：本方案 | 100.0% |
| 活跃 / 档案 / 淡忘 | 49 / 142 / 28 |
| BM25 / 词法+时间 / 词法+主题 / 三路 | 92.7% / 96.9% / 94.8% / 100.0% |
| 关闭日历窗口 / 关闭档案回溯 | 95.8% / 80.2% |
| 固定 / 扩展语义压力集 | 76/76 / 126/126 |
| 拟真笔记回放 | 3/3 |
| 关闭独立冲突治理 / 完整任务分工 | 97.9% / 100.0% |
| 完整方案冲突账本 | 12/12 |
| 40 用例端到端评测 | 98.8% |
| 自动测试 | 157/157 |

## 实验命令

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

固定实验脚本锁定数据、时间基准、检索配置和评分条件，用于检查路由、目标召回、干扰隔离和机制贡献；开放式表达另通过端到端用例与实际交互检查。

## 主要文件

| 路径 | 内容 |
|---|---|
| `web/app.py` | API、对话路由、附件处理和记忆准入 |
| `web/index.html` | 单页产品界面 |
| `memagent/router.py` | 记忆依赖与回答模式分类 |
| `memagent/interaction.py` | 事件、建议、偏好和确认策略 |
| `memagent/memory_policy.py` | 准入与敏感信息边界 |
| `memagent/memory.py` | 抽取、冲突、结论和保留策略 |
| `memagent/semantics.py` | 对象、属性、取值与查询范围 |
| `memagent/retrieve.py` | 三路检索和 RRF |
| `memagent/answer.py` | 当前、历史、演变和 LLM 证据回答 |
| `tools/document_tool.py` | 文档、OCR 和 Office 解析 |
| `eval/results/REPORT.md` | 最新实验总表 |
| `docs/机理与实验对照.md` | 机制、主实验与消融结果对照 |
| `docs/失败分析.md` | 结构性问题、修复方法与最终回归 |

## 提交材料

`submission/` 包含最新版项目说明 DOCX/PDF、9 页汇报 PPT、现场展示方案和固定演示附件。PPT 的演讲者备注已写入文件，材料中的架构、功能、实验口径和测试结果与当前代码一致。

## 配置说明

- `.env.example` 仅包含变量名和示例地址。
- `.env` 不属于提交内容。
- 视觉理解只在显式配置视觉模型、端点和密钥后启用；未配置时使用本地 OCR。
