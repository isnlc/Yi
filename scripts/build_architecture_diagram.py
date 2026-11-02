from __future__ import annotations

import html
import subprocess
from pathlib import Path


WIDTH = 2400
HEIGHT = 1080

BG = "#F7F3E9"
PAPER = "#FFFEFA"
INK = "#28241F"
MUTED = "#6F675E"
LINE = "#D6CAB9"
RED = "#9B382B"
RED_SOFT = "#F5E9E4"
BLUE = "#356B85"
BLUE_SOFT = "#EAF1F4"
GOLD = "#987441"
GOLD_SOFT = "#F4EFE3"


def rect(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    fill: str = PAPER,
    stroke: str = LINE,
    radius: int = 3,
    width: int = 2,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def text(
    x: int,
    y: int,
    value: str,
    *,
    size: int,
    fill: str = INK,
    weight: int = 400,
    anchor: str = "start",
    family: str = "Microsoft YaHei, sans-serif",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def multiline(
    x: int,
    y: int,
    lines: list[str],
    *,
    size: int,
    fill: str = MUTED,
    weight: int = 400,
    anchor: str = "start",
    gap: int = 28,
) -> str:
    spans = []
    for index, value in enumerate(lines):
        spans.append(
            f'<tspan x="{x}" dy="{0 if index == 0 else gap}">'
            f"{html.escape(value)}</tspan>"
        )
    return (
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{"".join(spans)}</text>'
    )


def line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: str,
    width: int = 4,
    marker: str | None = None,
    dash: str | None = None,
) -> str:
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'fill="none"{marker_attr}{dash_attr}/>'
    )


def path(
    value: str,
    *,
    color: str,
    width: int = 4,
    marker: str | None = None,
    dash: str | None = None,
) -> str:
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{value}" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" fill="none"'
        f'{marker_attr}{dash_attr}/>'
    )


def definitions() -> str:
    markers = []
    for name, color in (
        ("arrow-red", RED),
        ("arrow-blue", BLUE),
        ("arrow-gold", GOLD),
    ):
        markers.append(
            f'<marker id="{name}" markerWidth="16" markerHeight="16" '
            f'refX="14" refY="8" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M 0 0 L 16 8 L 0 16 z" fill="{color}"/></marker>'
        )
    return "<defs>" + "".join(markers) + "</defs>"


def section_header(
    x: int,
    y: int,
    width: int,
    title: str,
    subtitle: str,
    *,
    color: str,
) -> str:
    return (
        text(x, y, title, size=31, fill=color, weight=700)
        + text(x + 210, y - 1, subtitle, size=18, fill=MUTED)
        + line(x, y + 22, x + width, y + 22, color=LINE, width=2)
    )


def node(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    detail: str,
    *,
    color: str,
    fill: str = PAPER,
) -> str:
    return (
        rect(x, y, w, h, fill=fill, stroke=LINE)
        + rect(x, y, 7, h, fill=color, stroke=color, radius=2, width=0)
        + text(x + 30, y + 39, title, size=25, fill=color, weight=700)
        + text(x + 30, y + 72, detail, size=18, fill=MUTED)
    )


def build_svg() -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        definitions(),
        rect(0, 0, WIDTH, HEIGHT, fill=BG, stroke=BG, radius=0, width=0),
        text(70, 60, "交互接入", size=28, fill=RED, weight=700),
        text(70, 91, "输入与来源边界", size=16, fill=MUTED),
        line(220, 72, 2320, 72, color=LINE, width=2),
    ]

    out += [
        node(
            260,
            30,
            470,
            94,
            "多模态组合输入",
            "文字、语音、图片、多个文件",
            color=RED,
        ),
        node(
            870,
            30,
            470,
            94,
            "解析与来源隔离",
            "正文、OCR、元数据、附件边界",
            color=RED,
        ),
        node(
            1480,
            30,
            470,
            94,
            "统一请求对象",
            "内容、附件与来源标签",
            color=RED,
        ),
        line(730, 77, 870, 77, color=RED, marker="arrow-red"),
        line(1340, 77, 1480, 77, color=RED, marker="arrow-red"),
        line(1715, 124, 1715, 164, color=RED, marker="arrow-red"),
    ]

    out += [
        text(70, 198, "统一编排", size=28, fill=RED, weight=700),
        text(70, 229, "任务识别与读写边界", size=16, fill=MUTED),
        line(220, 210, 2320, 210, color=LINE, width=2),
        rect(500, 164, 1400, 142, fill=PAPER, stroke=RED, width=3),
        text(540, 207, "统一编排器", size=28, fill=RED, weight=700),
        text(
            540,
            246,
            "识别任务类型，执行记忆准入，选择仅检索 / 检索后推理 / 通用问答",
            size=20,
            fill=INK,
        ),
        text(
            540,
            277,
            "规则和结构化状态控制正式写入、版本关系与证据范围",
            size=17,
            fill=MUTED,
        ),
        rect(1950, 164, 350, 142, fill=BG, stroke=BLUE, width=2, dash="9 8"),
        text(
            2125,
            207,
            "可选模型服务",
            size=22,
            fill=BLUE,
            weight=700,
            anchor="middle",
        ),
        multiline(
            2125,
            242,
            ["语义分类、歧义补全、建议推理与表达", "不直接写记忆，不改变证据范围"],
            size=16,
            fill=MUTED,
            weight=600,
            anchor="middle",
            gap=28,
        ),
        line(
            1950,
            235,
            1900,
            235,
            color=BLUE,
            width=3,
            marker="arrow-blue",
            dash="7 7",
        ),
        path("M 925 306 L 925 350 L 560 350 L 560 388", color=RED, marker="arrow-red"),
        path("M 1475 306 L 1475 350 L 1780 350 L 1780 388", color=BLUE, marker="arrow-blue"),
    ]

    out += [
        text(70, 416, "任务执行", size=28, fill=RED, weight=700),
        text(70, 447, "四项智能体任务", size=16, fill=MUTED),
        line(220, 428, 2320, 428, color=LINE, width=2),
        text(310, 374, "记忆写入与治理", size=22, fill=RED, weight=700),
        text(1530, 374, "检索与作答", size=22, fill=BLUE, weight=700),
        node(
            260,
            388,
            600,
            100,
            "Recorder：记录抽取",
            "准入结果结构化为主题、对象、属性、值、时间和来源",
            color=RED,
        ),
        line(560, 488, 560, 522, color=RED, marker="arrow-red"),
        node(
            260,
            522,
            600,
            100,
            "Reconciler：冲突治理",
            "依据对象、事实日期与全局写入序裁决，保留旧版本和取代关系",
            color=RED,
        ),
        node(
            1480,
            388,
            600,
            100,
            "Retriever：证据检索",
            "BM25、时间窗和主题图三路召回，使用 RRF 融合",
            color=BLUE,
        ),
        line(1780, 488, 1780, 522, color=BLUE, marker="arrow-blue"),
        node(
            1480,
            522,
            600,
            100,
            "范围裁决",
            "按当前、历史、演变或画像确定可用证据",
            color=BLUE,
        ),
        line(1780, 622, 1780, 656, color=BLUE, marker="arrow-blue"),
        node(
            1480,
            656,
            600,
            100,
            "Auditor：回答审核",
            "检查证据充分性、版本时效和拒答边界",
            color=BLUE,
        ),
        line(1780, 756, 1780, 786, color=BLUE, marker="arrow-blue"),
        rect(1480, 786, 600, 52, fill=BLUE_SOFT, stroke=BLUE, width=3),
        text(
            1780,
            822,
            "回答 + 证据",
            size=24,
            fill=BLUE,
            weight=700,
            anchor="middle",
        ),
    ]

    out += [
        text(70, 680, "状态与证据", size=28, fill=RED, weight=700),
        text(70, 711, "共享记忆状态", size=16, fill=MUTED),
        line(220, 692, 1320, 692, color=LINE, width=2),
        line(560, 622, 560, 656, color=RED, marker="arrow-red"),
        rect(260, 656, 920, 182, fill=PAPER, stroke=RED, width=3),
        text(292, 699, "分层记忆与资料状态", size=26, fill=RED, weight=700),
        text(
            1148,
            698,
            "正式状态由规则维护",
            size=16,
            fill=MUTED,
            weight=600,
            anchor="end",
        ),
        line(292, 720, 1148, 720, color=LINE, width=2),
        text(310, 760, "当前", size=19, fill=INK, weight=700),
        text(475, 760, "归档", size=19, fill=INK, weight=700),
        text(640, 760, "淡化", size=19, fill=INK, weight=700),
        text(805, 760, "长期结论", size=19, fill=INK, weight=700),
        text(1015, 760, "附件资料", size=19, fill=BLUE, weight=700),
        line(292, 781, 1148, 781, color=LINE, width=2),
        text(310, 814, "取代链", size=17, fill=RED, weight=700),
        text(475, 814, "来源证据", size=17, fill=BLUE, weight=700),
        text(690, 814, "可恢复状态", size=17, fill=GOLD, weight=700),
        path(
            "M 1180 748 L 1335 748 L 1335 438 L 1480 438",
            color=BLUE,
            width=4,
            marker="arrow-blue",
        ),
        rect(1200, 560, 270, 64, fill=BG, stroke=BG, width=0),
        multiline(
            1335,
            585,
            ["按问题读取", "受版本约束的证据"],
            size=16,
            fill=BLUE,
            weight=700,
            anchor="middle",
            gap=22,
        ),
    ]

    out += [
        text(70, 902, "验证与回归", size=28, fill=GOLD, weight=700),
        text(245, 901, "运行日志和实验结果约束规则，不直接改写记忆结论", size=17, fill=MUTED),
        line(70, 922, 2320, 922, color=LINE, width=2),
        rect(120, 948, 510, 86, fill=PAPER, stroke=LINE),
        text(150, 984, "统一事件日志", size=22, fill=GOLD, weight=700),
        text(150, 1013, "准入、冲突、检索、审核", size=16, fill=MUTED),
        line(630, 991, 710, 991, color=GOLD, marker="arrow-gold"),
        rect(710, 948, 720, 86, fill=PAPER, stroke=LINE),
        text(740, 984, "固定题集、消融实验、自动回归", size=22, fill=GOLD, weight=700),
        text(740, 1013, "统一数据与时间基准，验证各机制贡献", size=16, fill=MUTED),
        line(1430, 991, 1510, 991, color=GOLD, marker="arrow-gold"),
        rect(1510, 948, 790, 86, fill=GOLD_SOFT, stroke=GOLD),
        text(1540, 984, "失败样本回灌", size=22, fill=GOLD, weight=700),
        text(1540, 1013, "进入规则、题集和回归测试，保持结果可复现", size=16, fill=INK),
    ]

    out.append("</svg>")
    return "".join(out)


def find_browser() -> Path:
    candidates = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 Edge 或 Chrome，无法渲染架构图。")


def build_architecture(output_png: Path, output_svg: Path | None = None) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg = output_svg or output_png.with_suffix(".svg")
    output_svg.write_text(build_svg(), encoding="utf-8")

    command = [
        str(find_browser()),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={WIDTH},{HEIGHT}",
        f"--screenshot={output_png}",
        output_svg.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_png


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    target = (
        root
        / "02_项目代码与文档"
        / "docs"
        / "figures"
        / "architecture_latest.png"
    )
    build_architecture(target)
    print(target)
