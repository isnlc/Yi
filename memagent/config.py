# 全局配置：路径、时间基准、遗忘策略参数
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

DIARY_FILE = DATA_DIR / "sample_diary.jsonl"
BENCHMARK_FILE = EVAL_DIR / "benchmark.jsonl"
STORE_FILE = DATA_DIR / "memory_store.json"

# 评测与"现在"的时间基准（固定，保证实验可复现）
NOW = "2026-10-15"

# 遗留策略：score = w_recency * exp(-days/tau) + w_importance * importance + w_freq * min(freq,5)/5
RETENTION = {
    "w_recency": 0.5,
    "w_importance": 0.3,
    "w_freq": 0.2,
    "tau_days": 90.0,
    "discard_below": 0.34,   # 低于：丢弃（留 tombstone）
    "compress_below": 0.45,  # 低于：压缩归档，高于：保留
}

# 检索融合：RRF 常数
RRF_K = 60
TOP_K = 8   # 语料扩到 300 条后 top-6 不够：时间线与日历窗口需要更宽的证据面

# 档案层开关（消融实验用）：False 时被遗忘/折叠的记忆彻底不可检索，
# 用来验证"档案层让遗忘变安全"这一设计结论
ALLOW_ARCHIVE = True

# 检索通道开关（消融实验用）：三路检索可独立启停
RETRIEVE_CHANNELS = {"lex": True, "time": True, "graph": True}

# 取代链 / 关键事件强制入证（消融实验用）：关闭后证据集只含融合排名 topk
EVIDENCE_EXPANSION = True

# ReAct 查询循环的最大轮数（规划 + 档案层重试 + LLM 改写重查）
MAX_QUERY_ROUNDS = 3
