from __future__ import annotations

import os
import time
import re

from logger import logger
from performance_config import get_performance_config, optimize_llm_response, optimize_tts_config
from config_loader import get_config_value


def _strip_citations_for_speech(text: str) -> str:
    """
    去掉类似 [E1] 的证据编号，避免 TTS 把引用读出来。
    """
    return re.sub(r"\[E\d+\]", "", text).strip()


def _clean_text_for_tts(text: str) -> str:
    """
    清理 LLM 输出中的 Markdown/符号，避免 TTS 读出诸如 `*`、`#` 等特殊字符。
    """
    # 最常见问题：Markdown 的强调/列表用到的 * 会被读出来
    text = text.replace("*", "")
    # 轻量清理常见 Markdown 标记（不做激进替换，避免误伤正文）
    text = text.replace("`", "")
    text = re.sub(r"(?m)^\s*#+\s*", "", text)  # 标题行：# xxx
    text = re.sub(r"(?m)^\s*>\s*", "", text)  # 引用块：> xxx
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_first_chunk_replay_guard(text: str, enabled: bool, replay_chars: int) -> str:
    """
    首段保底：把开头若干字符重复一次，缓解播放链路首段被吞的问题。
    例如：'您好，我是顾问' -> '您好，您好，我是顾问'
    """
    if not enabled:
        return text
    if not text:
        return text
    n = max(0, int(replay_chars))
    if n <= 0:
        return text
    prefix = text[:n].strip()
    if not prefix:
        return text
    # 避免过短文本重复后过于突兀
    if len(text) <= n + 1:
        return text
    return f"{prefix}，{text}"


def _normalize_ui_lang(ui_lang: str | None) -> str:
    raw = (ui_lang or "").strip().lower()
    if raw.startswith("zh"):
        return "zh-CN"
    if raw.startswith("en"):
        return "en"
    if raw.startswith("ja"):
        return "ja"
    if raw.startswith("ko"):
        return "ko"
    return "zh-CN"


def _language_instruction(ui_lang: str) -> str:
    if ui_lang == "en":
        return "请使用英文回答用户，除非用户明确要求其他语言。"
    if ui_lang == "ja":
        return "日本語で回答してください。ユーザーが明示的に別言語を要求した場合のみ従ってください。"
    if ui_lang == "ko":
        return "한국어로 답변하세요. 사용자가 다른 언어를 명시적으로 요청한 경우에만 따르세요."
    return "请使用简体中文回答用户，除非用户明确要求其他语言。"


def llm_response_with_identity(message, nerfreal, identity=None, is_current=None, rag_evidence=None, ui_lang=None):
    """
    支持身份信息的LLM响应函数（从 app.py 抽离，便于模块化）。
    """
    perf_config = get_performance_config()

    start_total = time.perf_counter()
    logger.info(f"开始处理用户消息: {message[:50]}... (优化级别: {perf_config.level})")

    llm_config = optimize_llm_response(perf_config)
    tts_config = optimize_tts_config(perf_config)
    logger.info(f"TTS引擎: {tts_config['engine']}, LLM模型: {llm_config['model']}")

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 表情=C：把分段的文本映射为 facial_expression（将其作为 expression_id 传给 TTS -> eventpoint）
    try:
        from audio_video_sync import EmotionalExpressionGenerator  # project local

        emotion_generator = EmotionalExpressionGenerator()
    except Exception:
        emotion_generator = None

    init_end = time.perf_counter()
    logger.info(f"LLM初始化时间: {init_end - start_total:.3f}s")

    if identity and identity.strip():
        system_message = identity.strip()
        logger.info(f"使用预设身份: {system_message[:100]}...")
    else:
        system_message = "您是一位专业的留学顾问，拥有丰富的留学咨询经验，擅长解答留学申请、院校选择、专业规划等问题。"

    resolved_lang = _normalize_ui_lang(ui_lang)
    system_message += "\n\n" + _language_instruction(resolved_lang)
    logger.info(f"本次对话语言: {resolved_lang}")

    # Keyword RAG evidence injection (MVP)
    if rag_evidence:
        evidence_blocks = []
        for i, ev in enumerate(rag_evidence, start=1):
            title = ev.get("title", "")
            source = ev.get("source", "")
            last_updated = str(ev.get("last_updated", ""))
            content_excerpt = (ev.get("content_excerpt") or ev.get("content", "") or "").strip()

            evidence_blocks.append(
                f"[E{i}] {title}\n来源: {source}\n更新时间: {last_updated}\n证据内容: {content_excerpt}"
            )

        evidence_text = "\n\n".join(evidence_blocks)
        system_message = (
            system_message
            + "\n\n你必须严格基于以下证据回答用户问题，禁止引入证据之外的信息。\n"
            + "每个关键结论句末尾必须添加证据编号，例如：[E1]。[E2]。\n"
            + "如果证据不足以回答，请明确说明不足并提出最关键的 1-3 个澄清问题，或建议用户以官网为准。\n\n"
            + "【证据库】\n"
            + evidence_text
        )
    else:
        # 无命中时允许自主回答，但避免编造具体政策条款
        system_message = (
            system_message
            + "\n\n如果你在证据库中未检索到与用户问题高度相关的信息，你可以基于一般留学经验与常识进行回答。\n"
            + "但涉及具体政策、截止日期、费用、签证材料清单等“可能随时间变化”的细节时，务必提示“建议以官网为准”，避免编造无法核实的条款。\n"
        )

    completion = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": message}],
        stream=True,
        max_tokens=llm_config["max_tokens"],
        temperature=llm_config["temperature"],
        stream_options={"include_usage": True},
    )

    if is_current is not None and not is_current():
        return ""

    result = ""
    full_result = ""
    first = True
    first_speech_chunk = True
    chunk_count = 0
    start_llm = time.perf_counter()
    first_chunk_replay_enabled = bool(get_config_value("tts.first_chunk_replay_enabled", True))
    first_chunk_replay_chars = int(get_config_value("tts.first_chunk_replay_chars", 6))

    for chunk in completion:
        if is_current is not None and not is_current():
            return full_result
        chunk_count += 1
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            if first:
                first_chunk_end = time.perf_counter()
                logger.info(f"LLM首字节时间: {first_chunk_end - start_llm:.3f}s")
                first = False

            msg = chunk.choices[0].delta.content
            result += msg
            full_result += msg

            chunk_size = llm_config.get("text_chunk_size", 30)
            enable_smart_seg = llm_config.get("enable_smart_segmentation", True)

            # 分段策略：尽量在“断句符”处切，保证一句话尽可能完整。
            # 同时修复：当 LLM 输出 Markdown（例如 * 列表）时，避免把 `*` 读出来。
            boundary_chars = "，。！？；\n.!?;"

            should_send = False
            send_text = ""

            if len(result) >= chunk_size:
                # 优先切到最后一个断句符，避免半句就播报
                last_boundary_idx = -1
                for ch in boundary_chars:
                    idx = result.rfind(ch)
                    if idx > last_boundary_idx:
                        last_boundary_idx = idx

                if last_boundary_idx != -1:
                    should_send = True
                    send_text = result[: last_boundary_idx + 1]
                    result = result[last_boundary_idx + 1 :]
                else:
                    # 没找到断句符：是否允许直接播报，取决于智能分段开关
                    if (not enable_smart_seg) or len(result) >= chunk_size * 2:
                        should_send = True
                        send_text = result
                        result = ""

            if should_send:
                if is_current is None or is_current():
                    speech_text = _strip_citations_for_speech(send_text)
                    speech_text = _clean_text_for_tts(speech_text)
                    if first_speech_chunk and speech_text:
                        speech_text = _apply_first_chunk_replay_guard(
                            speech_text,
                            enabled=first_chunk_replay_enabled,
                            replay_chars=first_chunk_replay_chars,
                        )
                    if speech_text:
                        expression_id = None
                        if emotion_generator is not None:
                            try:
                                scores = emotion_generator.analyze_text_emotion(speech_text)
                                params = emotion_generator.generate_facial_expression(scores, speech_text)
                                expression_id = params.get("facial_expression")
                            except Exception:
                                expression_id = None
                        nerfreal.put_msg_txt(
                            speech_text,
                            {
                                "expression_mode": "C",
                                "expression_id": expression_id or "neutral",
                                "ui_lang": resolved_lang,
                            },
                        )
                        first_speech_chunk = False
                else:
                    return

    if result and (is_current is None or is_current()):
        speech_text = _strip_citations_for_speech(result)
        speech_text = _clean_text_for_tts(speech_text)
        if first_speech_chunk and speech_text:
            speech_text = _apply_first_chunk_replay_guard(
                speech_text,
                enabled=first_chunk_replay_enabled,
                replay_chars=first_chunk_replay_chars,
            )
        if speech_text:
            expression_id = None
            if emotion_generator is not None:
                try:
                    scores = emotion_generator.analyze_text_emotion(speech_text)
                    params = emotion_generator.generate_facial_expression(scores, speech_text)
                    expression_id = params.get("facial_expression")
                except Exception:
                    expression_id = None
            nerfreal.put_msg_txt(
                speech_text,
                {
                    "expression_mode": "C",
                    "expression_id": expression_id or "neutral",
                    "ui_lang": resolved_lang,
                },
            )
            first_speech_chunk = False
            result = ""

    # Log which evidence ids were cited (useful for debugging RAG)
    try:
        cited_ids = sorted(set(re.findall(r"\[E\d+\]", full_result)))
        if cited_ids:
            logger.info(f"RAG cited evidence ids: {', '.join(cited_ids)}")
    except Exception:
        pass

    llm_end = time.perf_counter()
    total_time = llm_end - start_total
    logger.info(f"LLM处理完成 - 总时间: {total_time:.3f}s, 处理chunks: {chunk_count}")

    return full_result
