# -*- coding: utf-8 -*-
# 数据扩容：对话历史（chat 摘记）+ 脱敏资料（note 笔记）两路来源
# - 日常流水账：模板笛卡尔积确定性生成（seed=42，可复现，不花 API）
# - 新剧情线关键事件 / 脱敏笔记：调 LLM 渲染成自然日记（关键词缺失回退事实原文）
# 输出 data/dataset_large.jsonl = 原 100 条 + 新增记录；原文件不动
import sys, os, json, random
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memagent import llm, config

SEED = 42
rng = random.Random(SEED)
OUT = config.DATA_DIR / "dataset_large.jsonl"
OLD = config.DIARY_FILE


def load_old():
    with open(OLD, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def wednesdays(start="2026-03-04", end="2026-10-14"):
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= e:
        if d.weekday() == 2:
            out.append(d.isoformat())
        d += timedelta(days=7)
    return out


def daterange(start, end):
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def cross(openers, remarks):
    """开场白 × 事实 -> 不重复文案池"""
    return [o + r for o in openers for r in remarks]


OPENERS = ["和AI助手聊日常：", "对话记录：", "聊天摘记：", "语音备忘：", "和助手同步：", "随手记："]

# 各主题流水账池（笛卡尔积；注意避开会污染既有基准答案的词：
# 不出现 熬夜/放弃/更改/浅烘深烘偏好/课程表/选题 等敏感表达）
POOLS = {
    "badminton": cross(OPENERS, [
        "周三羽毛球照常，出了不少汗。", "周三羽毛球，练了半小时高远球。",
        "周三羽毛球，今天手感不错赢了一局。", "周三羽毛球，和球友复盘了接发球。",
        "周三羽毛球，状态一般但坚持去了。", "周三羽毛球，场地人不多打得很爽。",
        "周三羽毛球，打完和球友吃了顿宵夜。", "周三羽毛球，今天主要练步伐。",
    ]),
    "running": cross(OPENERS, [
        "今天跑步五公里，配速和上次差不多。", "早上跑步完成，状态比想象中好。",
        "夜跑完成，跑步时听了两期播客。", "今天跑步按计划完成，膝盖没有不适。",
        "跑步打卡，最后两公里提了点速。", "慢跑放松跑，跑完拉伸了十分钟。",
        "今天跑了五公里，中途走了半分钟。", "晨跑完成，回来做了组拉伸。",
    ]),
    "coffee": cross(OPENERS[:4], [
        "手冲的水温这次控制得很稳。", "豆子库存还够两周，先不补货。",
        "今天手冲了一杯，酸香味比上次明显。", "磨豆机该清洁了，明天记得处理。",
        "滤杯洗好收进柜子了。", "研究了一下新豆子的产地信息。",
    ]),
    "sleep": cross(OPENERS[:4], [
        "这周有三天做到了十二点前睡。", "今天早起成功，白天精神不错。",
        "午休二十分钟，下午效率高了很多。", "晚上定了十一点半的闹钟，准备早睡。",
        "睡前把手机放远了，入睡快了些。", "中午眯了一会儿，晚上状态在线。",
    ]),
    "reading": cross(OPENERS[:4], [
        "睡前读了半小时书，比刷手机踏实。", "把书单整理了一下，下半年想读的还有几本。",
        "今天读到喜欢的段落，摘抄了两句。", "图书馆借的书看完了一半。",
        "通勤路上听书代替了刷短视频。", "给读过的书写了两行短评。",
    ]),
    "trip": cross(OPENERS[:4], [
        "收藏了两个古城的攻略，回头对比一下。", "看了旅行博主的行程模板，记了几个可复用的思路。",
        "把假期的车票比价截图发给了朋友。", "列了旅行装备清单，放在备忘录里。",
        "研究了当地的公共交通线路。", "查了目的地的天气趋势。",
    ]),
    "finance": cross(OPENERS[:4], [
        "这个月的记账做完了，大头还是吃饭。", "把订阅服务清理了一遍，省了一笔小钱。",
        "看了一眼基金净值，忍住没动，按计划来。", "把生活费按周做了预算。",
        "把这个月的支出分了类，心里有数了。", "研究了学生优惠的存款产品。",
    ]),
    "health": cross(OPENERS[:4], [
        "这周外面吃饭的次数降下来了。", "晚上散步四十分钟，代替了刷手机。",
        "称了一下体重，和上周持平。", "多喝了水，含糖饮料少了一半。",
        "十一点前放下了手头的事。", "晚饭吃到了七分饱就停了。",
    ]),
    "course": cross(OPENERS[:4], [
        "今天两节课听下来，把疑问点记到了笔记里。", "晚自习把下周的预习提纲列出来了。",
        "实验课的数据先存了备份，回宿舍再整理。", "图书馆复习了两小时，效率还行。",
        "这周的作业截止日标到了日历上。", "课后找老师问了两个问题，都解决了。",
    ]),
    "social": cross(OPENERS[:4], [
        "和室友晚饭后散步聊了会儿天。", "帮同学带了东西，顺路的事。",
        "和同学约好周末一起去球场。", "班级群里接龙了团建时间。",
        "和老朋友视频了半小时。", "社团例会讨论了下学期的安排。",
    ]),
    "life": cross(OPENERS[:4], [
        "今天天气不错，心情也好。", "把桌面收拾了一遍，效率高了点。",
        "路上看到花开得不错，拍了张照。", "晚饭后走了走，消消食。",
        "整理了相册，删了不少截图。", "下班路上的云特别好看。",
    ]),
}


def build_fillers(old):
    """确定性模板流水账：同一天允许多条不同主题记录（真实日记形态），
    避开一切会污染既有基准答案的事实。十月只允许 10-01/10-13/10-14 三天，
    保证"十月以来"类日历问题的时间线完整。"""
    seen_texts = {r["text"] for r in old}
    october_guard = {"2026-10-01", "2026-10-13", "2026-10-14"}
    out = []

    def add(date_str, text, source="chat"):
        if text in seen_texts:
            return False
        if date_str.startswith("2026-10") and date_str not in october_guard:
            return False
        out.append({"date": date_str, "text": text, "source": source})
        seen_texts.add(text)
        return True

    def draw(pool, k):
        """从池中不重复抽 k 条；不够时允许循环使用（仍受文本去重保护）"""
        extended = (pool * (k // len(pool) + 1))[:max(k, len(pool))]
        return rng.sample(extended, len(extended))

    # 羽毛球：补齐所有周三
    i = 0
    for ds in wednesdays():
        if add(ds, draw(POOLS["badminton"], 33)[i]):
            i += 1
    # 跑步：活跃期的周二/周六
    run_days = [d for d in daterange("2026-03-10", "2026-05-16") if date.fromisoformat(d).weekday() in (1, 5)]
    run_days += [d for d in daterange("2026-09-08", "2026-10-13") if date.fromisoformat(d).weekday() in (1, 5)]
    for i, ds in enumerate(run_days):
        add(ds, draw(POOLS["running"], len(run_days))[i])
    # 其余主题：散点（十月只落在守卫日期上）
    scatter = [
        ("coffee", ["2026-03-20", "2026-04-25", "2026-05-26", "2026-06-21", "2026-07-19", "2026-08-16", "2026-09-13", "2026-10-13"]),
        ("sleep", ["2026-03-24", "2026-04-28", "2026-05-24", "2026-06-22", "2026-09-15", "2026-10-14"]),
        ("reading", ["2026-03-16", "2026-05-11", "2026-07-20", "2026-09-08"]),
        ("trip", ["2026-04-26", "2026-08-09", "2026-10-01"]),
        ("finance", ["2026-04-30", "2026-06-30", "2026-08-31", "2026-09-30"]),
        ("health", ["2026-05-21", "2026-06-11", "2026-09-22"]),
        ("course", ["2026-09-14", "2026-09-18", "2026-09-21", "2026-09-25", "2026-09-30", "2026-10-13"]),
        ("social", ["2026-03-28", "2026-04-15", "2026-06-28", "2026-09-12", "2026-10-14"]),
        ("life", ["2026-03-30", "2026-04-05", "2026-05-09", "2026-06-14", "2026-07-11", "2026-07-25", "2026-08-15", "2026-08-28", "2026-09-26"]),
    ]
    for key, dates in scatter:
        picks = draw(POOLS[key], len(dates))
        for ds, text in zip(dates, picks):
            add(ds, text)
    return out


# ---------------- 2. 剧情线关键事件（LLM 渲染，带必须关键词） ----------------
STORY_EVENTS = [
    # 阅读：纸质 -> Kindle -> 回纸质 -> 读完12本 -> 在读活着
    ("2026-03-10", "开始读《三体》第一部，纸质书", ["三体"], "note"),
    ("2026-04-15", "为了通勤方便，改成用Kindle看书", ["Kindle"], "chat"),
    ("2026-06-20", "Kindle吃灰了，还是改回看纸质书，翻页有感觉", ["纸质"], "chat"),
    ("2026-09-15", "今年读完的书累计到第十二本", ["第十二本"], "note"),
    ("2026-09-28", "最近在读《活着》，进度过半", ["活着"], "chat"),
    # 旅行：约好西安 -> 改泉州 -> 成行
    ("2026-05-01", "和朋友约好国庆去西安玩，先mark", ["西安"], "chat"),
    ("2026-07-20", "看攻略后把国庆行程改成去泉州，古城更对口味", ["泉州"], "chat"),
    ("2026-10-02", "泉州之行完成，开元寺和面线糊都体验了", ["泉州"], "note"),
    # 理财：开始定投 -> 停止 -> 恢复并改额度
    ("2026-04-01", "开始基金定投，每月500，发薪日自动扣", ["定投"], "note"),
    ("2026-08-10", "行情太差，停止了基金定投，先观望", ["停止"], "chat"),
    ("2026-09-20", "想通了，恢复定投，改成每月300，跌了更能摊低成本", ["恢复"], "chat"),
    # 健康：戒奶茶 -> 失败 -> 体重变化
    ("2026-05-05", "和室友说好一起戒掉奶茶，为期一个月", ["戒掉"], "chat"),
    ("2026-06-18", "放弃戒奶茶的计划，又喝上了奶茶，一周两杯封顶", ["又喝"], "chat"),
    ("2026-09-10", "体检和体重秤都向好：体重比暑假轻了三斤", ["体重"], "note"),
]

NOTE_FACTS = [
    ("2026-03-22", "整理《数据结构》第3章笔记（成同学课堂笔记，姓名学号已脱敏）", ["笔记"]),
    ("2026-04-12", "把 SRTP 项目会议记录归档，参会人用小A小B代替", ["归档"]),
    ("2026-05-18", "支教备课资料入库：教案模板一式两份（脱敏处理）", ["备课"]),
    ("2026-06-08", "课程表助手项目的需求文档存档，作者信息脱敏", ["存档"]),
    ("2026-07-06", "期末复习资料打包：三份提纲（来源：同学共享，已匿名）", ["提纲"]),
    ("2026-08-18", "实习周报归档，公司名与人名均已脱敏", ["周报"]),
    ("2026-09-06", "新学期课表与教材清单入库（学生信息脱敏）", ["教材"]),
    ("2026-10-06", "整理十月待办：把散落在三个便签的事项合并成清单", ["待办"]),
]


def render_batch(events):
    """events: [(date, fact, keywords, source)] -> 记录列表。LLM 渲染，关键词缺失回退原文。"""
    out = []
    BATCH = 6
    for i in range(0, len(events), BATCH):
        chunk = events[i:i + BATCH]
        listing = "\n".join(f"{j+1}. 日期{e[0]}：{e[1]}（必须出现：{('、'.join(e[2]))}）" for j, e in enumerate(chunk))
        try:
            texts = llm.chat_json([
                {"role": "system",
                 "content": "你是日记改写助手。把每条事实改写成一句 30-55 字的第一人称日记，"
                            "口语自然、信息完整，必须原样包含指定的关键词。"
                            '只输出 JSON 数组：[{"id": 1, "text": "..."}]。'},
                {"role": "user", "content": listing},
            ])
            by_id = {int(t.get("id", 0)): str(t.get("text", "")) for t in texts if isinstance(t, dict)}
        except Exception:
            by_id = {}
        for j, (ds, fact, kws, src) in enumerate(chunk, start=1):
            text = by_id.get(j, "").strip()
            if not text or any(k not in text for k in kws):
                text = fact   # 关键词缺失 -> 回退事实原文，保证基准答案可追溯
            out.append({"date": ds, "text": text, "source": src})
    return out


def main():
    old = load_old()
    fillers = build_fillers(old)
    print(f"模板流水账：{len(fillers)} 条（seed={SEED}）")
    rendered = render_batch(STORY_EVENTS)
    print(f"剧情线关键事件（LLM 渲染）：{len(rendered)} 条")
    notes = render_batch([(d, f, k, "note") for d, f, k in NOTE_FACTS])
    print(f"脱敏资料笔记（LLM 渲染）：{len(notes)} 条")

    records = old + fillers + rendered + notes
    records.sort(key=lambda r: r["date"])
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"date": r["date"], "text": r["text"],
                                "source": r.get("source", "diary")}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\n合计 {len(records)} 条 -> {OUT}")
    print("来源分布：", dict(Counter(r.get("source", "diary") for r in records)))
    print("月份分布：", dict(sorted(Counter(r["date"][:7] for r in records).items())))


if __name__ == "__main__":
    main()
