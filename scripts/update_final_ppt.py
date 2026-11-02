from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from build_architecture_diagram import build_architecture


ROOT = Path(__file__).resolve().parents[2]
PPT = (
    ROOT
    / "01_汇报"
    / "3240102321_成可心_大作业汇报"
    / "3240102321_成可心_期末大作业汇报_9页最终版.pptx"
)
ARCH = (
    ROOT
    / "02_项目代码与文档"
    / "docs"
    / "figures"
    / "architecture_latest.png"
)

ARCHITECTURE_NOTES = (
    "系统首先把文字、语音、图片和多个文件解析为带来源标签的统一请求，再交给统一编排器。"
    "编排器识别任务类型，执行记忆准入，并在仅检索、检索后推理和通用问答之间选择处理方式。"
    "规则和结构化状态始终负责正式写入、版本关系和证据范围。\n"
    "需要写入时，Recorder 把准入结果抽取为主题、对象、属性、取值、时间和来源；"
    "Reconciler 再根据事实日期和全局写入顺序处理冲突，保留旧版本和取代关系。"
    "治理结果进入分层记忆与资料状态，包括当前、归档、淡化、长期结论、附件资料、取代链和来源证据。\n"
    "需要查询时，Retriever 从共享状态中读取受版本约束的证据，组合 BM25、时间窗和主题图，"
    "通过 RRF 融合后按当前、历史、演变或画像裁决范围。Auditor 最后检查证据充分性、"
    "版本时效和拒答边界，再输出回答与证据。在线模型只提供语义分类、歧义补全、建议推理和表达，"
    "不能直接写入记忆或改变证据范围。所有关键事件进入统一日志，并由固定题集、消融实验和自动回归验证。"
)


def replace_architecture_picture(slide) -> None:
    pictures = [
        shape for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    ]
    if len(pictures) != 1:
        raise RuntimeError(
            f"第 3 页应有且仅有一张架构图，实际找到 {len(pictures)} 张。"
        )
    picture = pictures[0]
    left, top, width, height = picture.left, picture.top, picture.width, picture.height
    picture._element.getparent().remove(picture._element)
    slide.shapes.add_picture(str(ARCH), left, top, width, height)


def set_notes(slide, value: str) -> None:
    frame = slide.notes_slide.notes_text_frame
    frame.clear()
    frame.paragraphs[0].text = value


def update_final_deck() -> None:
    if not PPT.exists():
        raise FileNotFoundError(f"未找到最终版 PPT：{PPT}")

    build_architecture(ARCH, ARCH.with_suffix(".svg"))
    presentation = Presentation(PPT)
    if len(presentation.slides) != 9:
        raise RuntimeError(
            f"PPT 页数校验失败：应为 9 页，实际为 {len(presentation.slides)} 页。"
        )

    architecture_slide = presentation.slides[2]
    replace_architecture_picture(architecture_slide)
    set_notes(architecture_slide, ARCHITECTURE_NOTES)

    temporary = PPT.with_name(f"{PPT.stem}.tmp.pptx")
    presentation.save(temporary)

    check = Presentation(temporary)
    if len(check.slides) != 9:
        raise RuntimeError("保存后 PPT 页数异常。")
    if not check.slides[2].notes_slide.notes_text_frame.text.strip():
        raise RuntimeError("第 3 页演讲者备注写入失败。")
    if sum(
        1 for shape in check.slides[2].shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    ) != 1:
        raise RuntimeError("第 3 页架构图替换失败。")

    os.replace(temporary, PPT)


if __name__ == "__main__":
    update_final_deck()
    print(PPT)
