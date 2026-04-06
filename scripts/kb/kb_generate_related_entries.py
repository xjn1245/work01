import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import sys

# Ensure project root is importable no matter where you run the script from.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from knowledge_base import StudyAbroadKnowledgeBase  # noqa: E402


COUNTRIES_10 = ["美国", "英国", "加拿大", "澳大利亚", "德国", "法国", "荷兰", "新加坡", "日本", "韩国"]


SUBTOPICS_BY_CATEGORY_10 = {
    "院校信息": [
        "排名与评估维度解读",
        "申请周期与关键节点",
        "学费与总成本预算方法",
        "住宿选择与生活环境评估",
        "课程结构与专业设置要点",
        "地理位置与通勤/签证便利",
        "科研资源与实验室匹配",
        "校内设施与学术支持",
        "入学门槛与常见差异",
        "校园体验与适应建议",
    ],
    "专业介绍": [
        "课程方向选择思路",
        "核心课程内容与学习路径",
        "选课与学分/要求概览",
        "研究方向与论文/项目要求",
        "实习与项目实践机会",
        "适合背景与能力画像",
        "就业技能与能力清单",
        "证书/课程与能力提升",
        "学术写作与研究方法要求",
        "跨学科选择的权衡点",
    ],
    "申请要求": [
        "学历背景与GPA常见要求",
        "语言成绩（托福/雅思）要点",
        "标准化成绩（GRE/GMAT等）选择",
        "推荐信准备与人选策略",
        "个人陈述/文书（SOP/PS）写法框架",
        "简历结构化与亮点提炼",
        "科研/项目经历如何呈现",
        "申请材料时间线（倒推法）",
        "面试准备与高频问题",
        "常见误区与避坑清单",
    ],
    "签证政策": [
        "资金证明材料清单与证明方式",
        "签证申请流程步骤拆解",
        "入境前准备与行程规划要点",
        "在读期间合规事项概览",
        "面签/材料审核常见关注点",
        "拒签原因排查与修正思路",
        "学费与奖学金证明的使用",
        "签证延期/换签的准备原则",
        "风险控制与材料一致性",
        "学生签证常见口径差异",
    ],
    "奖学金": [
        "奖学金类型对比（校内/外部）",
        "申请策略与竞争力构建",
        "文书在奖学金中的作用",
        "费用覆盖与资助范围解读",
        "经济背景/资助证明如何准备",
        "申请时间线（倒推）与截止点",
        "筛选标准与关键材料",
        "常见误区与成功要点",
        "优秀案例的结构复盘方法",
        "提升竞争力的行动清单",
    ],
    "生活指南": [
        "住宿选择与成本对比",
        "每月预算与弹性规划方法",
        "交通出行与通勤策略",
        "医疗/保险与就医流程",
        "手机卡/银行开户与日常工具",
        "社团融入与文化适应",
        "饮食与健康习惯调整",
        "安全与应急准备清单",
        "打工/兼职规则的合规提醒",
        "学习生活平衡与时间管理",
    ],
    "就业前景": [
        "行业岗位趋势与技能需求",
        "简历与作品/项目展示方法",
        "实习机会获取路径",
        "求职时间线（从申请到面试）",
        "面试准备框架与案例法",
        "岗位匹配度评估与选择",
        "薪资与生活成本的关系",
        "工签/身份衔接准备原则",
        "职业技能清单与积累策略",
        "校友网络与资源利用方法",
    ],
    "最新政策": [
        "如何追踪政策更新时间与来源",
        "官方渠道信息筛选方法",
        "变化影响评估（对申请/签证）",
        "时间敏感事项的优先级",
        "口径差异与常见误读",
        "材料准备适配策略",
        "风险提示与合规检查清单",
        "跨国家政策对比的做法",
        "信息更新记录模板",
        "遇到不确定时的核验路径",
    ],
}


def build_entry(category: str, country: str, subtopic: str, idx: int, source: str) -> tuple[str, str, list[str]]:
    # idx 从 1 开始用于标题编号，保持唯一与可读。
    num = idx + 1
    keyword = subtopic.split("（")[0][:10]
    tags = [category, country, keyword, "留学问答"]

    if category == "最新政策":
        title = f"{country}{category}获取与核验要点第{num}条"
        content = (
            f"在{country}的{category}信息检索中，建议你按以下步骤操作：\n"
            f"1) 明确信息来源：优先使用官方部门/学校公告/权威媒体的原文。\n"
            f"2) 形成核验清单：把关键字段（时间、对象、适用范围、材料要求）逐项对照。\n"
            f"3) 评估影响范围：判断变化主要影响申请阶段、签证阶段还是在读阶段。\n"
            f"4) 风险控制：若遇到口径不一致，以原文或官方问答为准，并留存记录。\n"
            f"5) 更新节奏：为每个关键节点设定复查时间，避免“过期信息”。\n"
            f"\n提示：最终口径请以官方发布为准。本条为通用方法模板。"
        )
        return title, content, tags

    title = f"{country}{category}—{subtopic}要点汇总第{num}条"
    content = (
        f"围绕“{category}”在{country}的实际准备，针对“{subtopic}”，建议你这样组织信息与行动：\n"
        f"1) 先拆解目标：把问题拆成材料/流程/时间线/证据四类。\n"
        f"2) 再匹配要点：列出与{subtopic}直接相关的关键条目（尽量用可核验的证据）。\n"
        f"3) 落地执行：按倒推法制定截止时间，并在每一步保留记录/截图/文件。\n"
        f"4) 复盘优化：对照要求检查差距，必要时调整措辞与材料结构。\n"
        f"5) 核对来源：建议以官方渠道或权威媒体为准，结合“{source}”的写作思路完善内容。\n"
        f"\n备注：本条为通用知识条目模板，可根据你的目标学校/项目再定制。"
    )
    return title, content, tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="study_abroad_kb.db", help="sqlite db path")
    parser.add_argument("--per_category", type=int, default=100, help="entries per category")
    parser.add_argument("--source", default="用户贡献", help="entry source (e.g. 官方渠道/权威媒体/用户贡献)")
    parser.add_argument("--expiration_days", type=int, default=365, help="expiration days")
    parser.add_argument("--skip_if_title_exists", action="store_true", help="skip if category+title exists")
    parser.add_argument("--commit", action="store_true", help="actually write into DB (without this: dry-run)")
    args = parser.parse_args()

    kb = StudyAbroadKnowledgeBase(args.db)

    categories = kb.knowledge_categories
    if not categories:
        raise ValueError("No categories found in knowledge base.")

    # Ensure we have 10 subtopics for deterministic 100 entries (10x10).
    for c in categories:
        if c not in SUBTOPICS_BY_CATEGORY_10:
            raise ValueError(f"Missing subtopics for category: {c}")
        if len(SUBTOPICS_BY_CATEGORY_10[c]) != 10:
            raise ValueError(f"SUBTOPICS_BY_CATEGORY_10[{c}] must have exactly 10 items.")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    existing_titles = set()
    if args.skip_if_title_exists:
        qs = ",".join(["?"] * len(categories))
        cur.execute(f"SELECT category, title FROM knowledge_entries WHERE category IN ({qs})", categories)
        for cat, title in cur.fetchall():
            existing_titles.add((cat, title))

    credibility_weight = kb.knowledge_sources.get(args.source, {}).get("权重", 0.5)
    expiration_date = datetime.now() + timedelta(days=int(args.expiration_days))

    to_insert = []
    for category in categories:
        subtopics = SUBTOPICS_BY_CATEGORY_10[category]
        for i in range(int(args.per_category)):
            country = COUNTRIES_10[i // len(subtopics) % len(COUNTRIES_10)]
            subtopic = subtopics[i % len(subtopics)]

            title, content, tags = build_entry(category, country, subtopic, i, args.source)
            if args.skip_if_title_exists and (category, title) in existing_titles:
                continue

            to_insert.append(
                (
                    category,
                    title,
                    content,
                    args.source,
                    float(credibility_weight),
                    expiration_date,
                    json.dumps(tags, ensure_ascii=False),
                )
            )

    if not args.commit:
        print(f"[dry-run] to_insert={len(to_insert)} (db={args.db})")
        print("Run again with --commit to write into DB.")
        return

    cur.executemany(
        """
        INSERT INTO knowledge_entries
        (category, title, content, source, credibility_score, expiration_date, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        to_insert,
    )
    conn.commit()
    conn.close()

    print(f"[done] inserted={len(to_insert)} into {args.db}")


if __name__ == "__main__":
    main()

