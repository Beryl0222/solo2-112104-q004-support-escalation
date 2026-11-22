"""Text-level helpers: deterministic splitting, quote locating and small
Chinese-language recognisers.

The recognisers are intentionally conservative.  When they are not certain,
they report a clarification instead of guessing (for example: which order does
"那个" refer to?).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

CLAUSE_DELIMITERS = "，,。；;！!？?\n"
_ORDER_PATTERN = re.compile(r"订单\s*([A-Za-z0-9]+)")
_ORDER_TOPIC_PATTERN = re.compile(r"([A-Za-z0-9]+)\s*订单")
_CLAUSE_TRAILING = "，,。；;！!？?、：:\n "
_CJK = re.compile(r"[一-鿿]")

_AGENT_PREFIXES = (
    "客服：", "客服:", "坐席：", "坐席:", "专员：", "专员:",
    "支持：", "支持:", "Agent：", "Agent:",
)
_CUSTOMER_PREFIXES = (
    "客户：", "客户:", "用户：", "用户:", "Customer：", "Customer:",
)

# An annotation may be attached to either side of a quote: 「...」 or (...)
_ANNOTATION_PATTERNS = (
    re.compile(r"「([^」]+)」"),
    re.compile(r"（([^）]+)）"),
    re.compile(r"\(([^)]+)\)"),
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def speaker_of(text: str) -> str:
    for prefix in _AGENT_PREFIXES:
        if text.startswith(prefix):
            return "agent"
    for prefix in _CUSTOMER_PREFIXES:
        if text.startswith(prefix):
            return "customer"
    return "unknown"


def strip_speaker_prefix(text: str) -> Tuple[str, int]:
    """Return (remaining text, offset of its first character in ``text``)."""
    for prefix in _AGENT_PREFIXES + _CUSTOMER_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):], len(prefix)
    return text, 0


def split_clauses(text: str) -> list[Tuple[str, int, int]]:
    """Split ``text`` on punctuation while keeping exact offsets.

    Returns ``(clause, start, end)`` with whitespace trimmed only from the
    reported clause; ``start``/``end`` always point into the original string.
    """
    pieces: list[Tuple[str, int, int]] = []
    cursor = 0
    for index, char in enumerate(text):
        if char in CLAUSE_DELIMITERS:
            if cursor < index:
                pieces.append((text[cursor:index], cursor, index))
            cursor = index + 1
    if cursor < len(text):
        pieces.append((text[cursor:], cursor, len(text)))

    result: list[Tuple[str, int, int]] = []
    for piece, start, end in pieces:
        stripped = piece.strip()
        if not stripped or not CLEAN_CLAUSE.fullmatch(stripped.replace(" ", "")):
            continue
        offset = piece.find(stripped)
        result.append((stripped, start + offset, start + offset + len(stripped)))
    return result


# A clause must contain at least one letter/digit/CJK; punctuation and quoted
# annotations are allowed because they are part of the original utterance.
CLEAN_CLAUSE = re.compile(r".*[0-9A-Za-z一-鿿].*")


def short_quote(quote: str, limit: int = 36) -> str:
    quote = quote.strip()
    if len(quote) <= limit:
        return quote
    return quote[: limit - 1] + "…"


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def find_order_ids(text: str) -> list[str]:
    found = [match.group(1) for match in _ORDER_PATTERN.finditer(text)]
    found.extend(match.group(1) for match in _ORDER_TOPIC_PATTERN.finditer(text))
    # De-duplicate while preserving order; ignore empty/single-letter noise later.
    return unique(found)


def mentions_order(text: str) -> bool:
    return bool(_ORDER_PATTERN.search(text) or _ORDER_TOPIC_PATTERN.search(text))


_ANAPHORS = (
    "那个订单", "这个订单", "那笔订单", "这笔订单",
    "那个", "这个", "那笔", "这笔", "它", "该订单",
)


def find_anaphor(text: str) -> Optional[str]:
    # Check longer phrases first so "那个订单" is not reported as "那个".
    for phrase in _ANAPHORS:
        if phrase in text:
            return phrase
    return None


def annotations(text: str) -> list[str]:
    """Return quoted annotations such as 「退款」 or （取消） found in text."""
    result: list[str] = []
    for pattern in _ANNOTATION_PATTERNS:
        result.extend(match.group(1).strip() for match in pattern.finditer(text))
    return unique(result)


def remove_annotations(text: str) -> str:
    cleaned = text
    for pattern in _ANNOTATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# --- intent cues ---------------------------------------------------------
# Order is significant: refund before logistics, complaint is separate.
REFUND_CUES = (
    "退款", "退钱", "退回", "退费", "赔付", "赔偿", "补偿",
    "重复扣款", "多扣", "扣错", "退一赔", "退差价", "售后",
)
LOGISTICS_CUES = (
    "物流", "快递", "承运", "配送", "发货", "签收", "到货", "没到",
    "送达", "运单", "包裹", "派件", "查件",
)
COMPLAINT_CUES = (
    "投诉", "太差", "太慢", "很慢", "这么慢", "处理慢", "没人管", "拖延",
    "失望", "不满意", "不满", "生气", "差评", "不能接受", "无法接受",
    "很不满", "什么态度", "气死", "离谱", "等太久",
)

# A customer may explicitly disclaim an action.  "我不是要退款" must not be
# turned into a refund request.
REFUND_NEGATIONS = (
    "不是要退款", "不是退款", "不要退款", "不申请退款", "不需要退款",
    "不用退款", "不是来退款", "并非要退款",
)
CANCEL_CUES = ("取消订单", "取消购买", "不要了", "退订")

# A clause may only label a request when it actually asks/claims something.
_QUESTION_MARKERS = (
    "吗", "呢", "?", "？", "是否", "有没有", "查", "麻烦", "请", "帮我",
    "何时", "什么时候", "怎么", "为什么", "哪一步", "到哪", "多久",
)
_DEMAND_MARKERS = ("要求", "必须", "投诉", "申请", "我要", "需要", "希望", "请")


def clause_has_claim(text: str) -> bool:
    return any(marker in text for marker in _QUESTION_MARKERS + _DEMAND_MARKERS)


def refund_is_negated(text: str) -> bool:
    return any(phrase in text for phrase in REFUND_NEGATIONS)


def detect_intents(text: str) -> list[Tuple[str, str]]:
    """Return ``[(intent_code, matched_cue)]`` for one clause.

    Actionable asks come first.  Dissatisfaction is deliberately *not* an
    actionable intent: "我很失望" must not turn into a refund or compensation
    request.  A complaint request is only created for a clause that expresses
    dissatisfaction and has no actionable ask.
    """
    if not refund_is_negated(text) and any(cue in text for cue in REFUND_CUES):
        return [("refund", next(cue for cue in REFUND_CUES if cue in text))]
    if any(cue in text for cue in CANCEL_CUES):
        return [("cancel", next(cue for cue in CANCEL_CUES if cue in text))]
    if any(cue in text for cue in LOGISTICS_CUES):
        return [("logistics", next(cue for cue in LOGISTICS_CUES if cue in text))]
    if has_dissatisfaction(text):
        return [("complaint", next(cue for cue in COMPLAINT_CUES if cue in text))]
    return []


def has_dissatisfaction(text: str) -> bool:
    return any(cue in text for cue in COMPLAINT_CUES)


# --- agent commitment cues ----------------------------------------------
# The utterance has to look like the agent speaking, either through a speaker
# prefix or through a commitment phrase.  A bare "可以退款" in a customer
# question must never become a promise.
# These markers describe a commitment by the agent.  Bare process words such
# as "升级" or "处理中" are intentionally absent: "已升级退款审核团队处理中"
# reports escalation progress, it does not promise a new deadline.
AGENT_PROMISE_MARKERS = (
    "给您", "帮您", "为您", "我们会", "我会", "我们将在", "将在", "会在",
    "已为您", "已帮您", "承诺", "保证", "安排", "答复", "回复",
    "确认后", "核对", "跟进", "联系", "发送", "告知", "登记",
    "提交", "加急", "优先",
)
_DEADLINE_PATTERN = re.compile(
    r"(\d+\s*个工作日内|\d+\s*个工作日内|\d+\s*小时内|"
    r"今天下班前|今天(?:上午|下午|中午|晚上)?|"
    r"明天(?:上午|下午|中午|晚上)?|"
    r"本周[^，。；;！!？?]{0,8}|"
    r"周[一二三四五六日天](?:上午|下午|中午|晚上)?|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日[^，。；;！!？?]{0,8})"
)


def find_deadline(text: str) -> Optional[str]:
    match = _DEADLINE_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).strip()


def is_agent_utterance(text: str, known_agent_labels: Sequence[str] = ()) -> bool:
    stripped = text.strip()
    if speaker_of(stripped) == "agent":
        return True
    for label in known_agent_labels:
        if label and stripped.startswith(label):
            return True
    return False


def looks_like_commitment(text: str, known_agent_labels: Sequence[str] = ()) -> bool:
    if not is_agent_utterance(text, known_agent_labels):
        return False
    if has_dissatisfaction(text):
        return False
    return any(marker in text for marker in AGENT_PROMISE_MARKERS)


def agent_label_of(text: str, known_agent_labels: Sequence[str] = ()) -> Optional[str]:
    stripped = text.strip()
    for prefix in _AGENT_PREFIXES:
        if stripped.startswith(prefix):
            return prefix.rstrip("：:")
    for label in known_agent_labels:
        if label and stripped.startswith(label):
            return label
    return None


# --- completion cues -----------------------------------------------------
COMPLETION_CUES = ("已退款", "退款完成", "已退回", "已到账", "已经到账", "到账了",
                   "已经退回", "已经退款", "已完成", "已处理", "已经处理",
                   "已解决", "已签收", "已送达")


def finds_completion(text: str) -> bool:
    return any(cue in text for cue in COMPLETION_CUES)


def is_question(text: str) -> bool:
    return "?" in text or "？" in text or any(marker in text for marker in ("吗", "呢", "是否", "有没有"))
