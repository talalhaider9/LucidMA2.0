"""Optional evidence-bounded AutoGen review of an already computed profile."""

import re

import autogen

from .config import SETTINGS, Settings


def _verified_counts(context: str) -> set[str]:
    """Extract every number that legitimately appears in the verified data."""
    return set(re.findall(r"\d[\d,]*", context))


def _counts_in(text: str) -> set[str]:
    """Extract the numeric tokens mentioned in a reply or briefing."""
    return set(re.findall(r"\d[\d,]*", text))


def _unsupported_counts(text: str, allowed: set[str]) -> set[str]:
    """Return briefing counts that do not trace to verified data.

    A number passes if it appears in the verified context (comma-insensitive) or is
    at most 100 (derived shares, percents, hours, district codes, calendar parts).
    """
    bad = set()
    for raw in _counts_in(text):
        value = raw.replace(",", "")
        if value in allowed or raw in allowed:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if number > 100:
            bad.add(raw)
    return bad


def _verified_hours(context: str) -> set[str]:
    """Collect every hour referenced by a verified timestamp in the data."""
    hours = set()
    for hour, _minute in re.findall(r"\b(\d{1,2}):(\d{2})\b", context):
        hours.add(hour.lstrip("0") or "0")
    return hours


def _unsupported_times(text: str, verified: set[str]) -> set[str]:
    """Return time-of-day references in a briefing that contradict verified data.

    A 24-hour reference passes if its hour matches a verified timestamp; a 12-hour
    reference is converted to 24-hour before comparing.
    """
    bad = set()
    for hour, minute in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
        if (hour.lstrip("0") or "0") not in verified:
            bad.add(f"{hour}:{minute}")
    remainder = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    for hour, meridiem in re.findall(r"\b(\d{1,2})\s*(AM|PM)\b", remainder, flags=re.IGNORECASE):
        converted = int(hour) % 12 + (12 if meridiem.upper() == "PM" else 0)
        if str(converted) not in verified:
            bad.add(f"{hour} {meridiem.upper()}")
    return bad


# Reasoning the verified data cannot support. These stems never occur in a verified
# profile, so a briefing containing any is rejected deterministically regardless of
# the model: recommendation, causation, speculation, behavioural claims, institutional
# confabulation, location-of-area nonsense, and the specific hallucinated tokens seen.
_UNSUPPORTED_PATTERNS = (
    r"should\b", r"\bmust\b", r"recommend\w*",
    r"\bresident[s]?\b", r"\bprone\b", r"more likely\b", r"engage\w* in violent",
    r"at risk\b", r"counsel\w*", r"safe space\w*", r"\bstigma\b",
    r"intervention\w*", r"law enforcement", r"first points of contact",
    r"located in (?:a )?schools?", r"school setting", r"schools are\b",
    r"non-?a swing", r"handgunes", r"effective law enforcement",
    r"positive sign",
    r"community program\w*", r"outreach", r"stakeholder\w*",
    r"\bprograms\b", r"initiatives?\b", r"expand\w*", r"prevent\w*",
    r"targeted intervention\w*", r"localized intervention\w*", r"high-risk",
    r"support services", r"root cause\w*", r"track progress",
    r"evaluate and adjust", r"prioritiz\w*", r"engagement and support",
)


def _unsupported_claims(text: str) -> set[str]:
    """Return confabulated reasoning phrases present in a briefing.

    The verified data cannot support causal, behavioural, recommendation, or
    'located in' claims; these are always rejected regardless of the model.
    """
    return {match.group(0) for pattern in _UNSUPPORTED_PATTERNS
            for match in [re.search(pattern, text, flags=re.IGNORECASE)] if match}


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"\b{token}\b", text) is not None


def _has_verdict(text: str, verdict: str) -> bool:
    """Check for an explicit verdict statement like 'VERDICT: VALID' or 'Conclusion: ... Valid'."""
    pattern = rf"(?:VERDICT\s*:\s*{verdict}\b|\bVERDICT\s+IS\s+{verdict}\b|^\s*{verdict}\b|\bCONCLUSION\s*:\s*(?:THE\s+BRIEFING\s+IS\s+)?{verdict}\b)"
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _clean_thought_tags(text: str) -> str:
    """Strip DeepSeek-R1 <think>...</think> reasoning traces to keep turns clean."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


_CONTEXT_CHAR_BUDGET = 12_000


def _trim_history(groupchat) -> None:
    """Bound the conversation to the anchor message plus a recent character window."""
    messages = groupchat.messages
    if len(messages) <= 2:
        return
    anchor = messages[0]
    kept: list[dict] = []
    used = len(str(anchor.get("content") or ""))
    for message in reversed(messages[1:]):
        length = len(str(message.get("content") or ""))
        if used + length > _CONTEXT_CHAR_BUDGET:
            break
        kept.append(message)
        used += length
    groupchat.messages[:] = [anchor, *reversed(kept)]


def run_agent_analysis(context: str, settings: Settings = SETTINGS):
    """Iteratively draft and review a briefing, up to the configured epoch limit."""
    config = settings.llm_config
    state = {"epoch": 0, "briefing_errors": None, "briefing_soft": None, "reviewer_retries": 0}
    allowed = _verified_counts(context)
    verified_hours = _verified_hours(context)
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message=(
            "You are the Crime Data Analyst. Analyze the VERIFIED DATA provided at the start.\n"
            "Present 3-5 substantive insights on temporal patterns, category distribution, "
            "and geography, each backed strictly by its figures.\n\n"
            "Rules:\n"
            "- Every figure cited must appear verbatim in VERIFIED DATA (or be a simple share of two verified figures).\n"
            "- Never invent counts, cross-tabulations, causes, population risk, or rates absent from the data.\n"
            "- Do not attribute causes or behaviour to residents or areas, and give NO recommendations.\n"
            "- Do not re-paste the raw VERIFIED DATA block.\n"
            "- If previous feedback flagged unverified values, remove them. Never repeat previous messages verbatim."
        ),
        llm_config=config,
    )
    briefing_writer = autogen.AssistantAgent(
        name="BriefingWriter",
        system_message=(
            "You are the BriefingWriter. Synthesize the Analyst's insights into a concise, "
            "neutral briefing containing: Headline Summary, Scope, Temporal Patterns, Geography, and Categories.\n\n"
            "Rules:\n"
            "- Every count, percentage, and peak hour must match VERIFIED DATA verbatim; add no new figures.\n"
            "- When listing entities (districts, community areas, descriptions), preserve their exact paired counts from VERIFIED DATA.\n"
            "- Never label counts from a single crime category as 'combined' or 'for both types'. Only use 'combined' if you explicitly sum verified numbers from each category.\n"
            "- Describe rankings strictly as reported counts, not rates or risk.\n"
            "- Give NO recommendations, advice, causal claims, or behavioural claims about residents or areas.\n"
            "- Apply all feedback from the EvidenceReviewer to improve clarity without altering the data.\n"
            "- Never repeat your previous briefing verbatim."
        ),
        llm_config=config,
    )
    reviewer = autogen.AssistantAgent(
        name="EvidenceReviewer",
        system_message=(
            "You are the EvidenceReviewer. Your sole responsibility is to evaluate the "
            "latest briefing produced by BriefingWriter against the VERIFIED DATA provided at the start.\n\n"
            "Audit the briefing rigorously against these criteria:\n"
            "1. FACTUAL VALIDITY:\n"
            "   - Cross-check every entity-count pair (e.g. check each District, Community Area, and Description count against VERIFIED DATA). If the writer writes 'District 006: 14' when VERIFIED DATA states '006 (15)', that is a factual discrepancy and MUST be rejected with NEEDS_REVISION.\n"
            "   - Check that numbers are not falsely claimed as 'combined' or 'for both types' when they only belong to one category.\n"
            "   - Verify peak hours, date ranges, and percentages match verbatim.\n"
            "2. REPORT QUALITY:\n"
            "   - Ensure the briefing is factual, objective, concise, and structured.\n"
            "   - Reject any causal explanations, resident behavior speculation, demographic stereotyping, or policy recommendations.\n\n"
            "CRITICAL: Do NOT write a new briefing, summary, or report. Judge only the text written by BriefingWriter.\n\n"
            "You MUST reply using this exact format:\n"
            "VERDICT: VALID (or NEEDS_REVISION)\n"
            "FACTUAL VALIDITY: <Line-by-line verification of cited counts and entity pairings against VERIFIED DATA>\n"
            "REPORT QUALITY: <Evaluation of neutrality, tone, and accuracy of aggregate descriptions>\n"
            "SPECIFIC DEFECTS: <Bulleted list of errors, or 'None'>\n"
            "REQUIRED REVISIONS: <Actionable fixes for BriefingWriter, or 'None'>"
        ),
        llm_config=config,
    )
    proxy = autogen.UserProxyAgent(
        name="Orchestrator", human_input_mode="NEVER", code_execution_config=False
    )

    def select_next(last_speaker, groupchat):
        # Strip DeepSeek-R1 <think>...</think> tags so history stays clean
        if groupchat.messages and groupchat.messages[-1].get("content"):
            cleaned = _clean_thought_tags(str(groupchat.messages[-1]["content"]))
            if cleaned:
                groupchat.messages[-1]["content"] = cleaned

        _trim_history(groupchat)
        if last_speaker is proxy:
            state["epoch"] = 1
            print(f"Agent review epoch {state['epoch']}/{settings.epoch_limit}")
            return analyst
        if last_speaker is analyst:
            return briefing_writer
        if last_speaker is briefing_writer:
            briefing = str(groupchat.messages[-1].get("content") or "")
            errors = _unsupported_counts(briefing, allowed) | _unsupported_times(briefing, verified_hours)
            soft = _unsupported_claims(briefing)
            state["briefing_errors"] = errors or None
            state["briefing_soft"] = soft or None
            if errors:
                print(f"Rejected briefing: unverified value(s) {sorted(errors)}")
            if soft:
                print(f"Flagged confabulated reasoning: {sorted(soft)}")

            check_hints = []
            if errors:
                check_hints.append(f"Automated verification flagged unverified figures: {sorted(errors)}.")
            if soft:
                check_hints.append(f"Automated verification flagged unsupported claims: {sorted(soft)}.")
            hint_str = f" NOTE: {' '.join(check_hints)}" if check_hints else ""

            groupchat.messages.append({
                "role": "user",
                "content": (
                    f"EvidenceReviewer: Audit the latest briefing above against VERIFIED DATA.{hint_str}\n"
                    "1. Verify every entity-count pair matches VERIFIED DATA (do not let altered counts pass).\n"
                    "2. Check that single-category counts are not mislabeled as 'combined'.\n"
                    "Start your reply with 'VERDICT: VALID' or 'VERDICT: NEEDS_REVISION'. "
                    "Do NOT write a briefing yourself."
                ),
            })
            return reviewer
        if last_speaker is reviewer:
            review_msg = str(groupchat.messages[-1].get("content") or "").strip()
            review = review_msg.upper()
            errors = state.get("briefing_errors")
            soft = state.get("briefing_soft")
            has_valid = _has_verdict(review_msg, "VALID") or _has_token(review, "VALID")
            has_needs = _has_verdict(review_msg, "NEEDS_REVISION") or _has_token(review, "NEEDS_REVISION")

            # If reviewer didn't provide a verdict token, retry reviewer directly
            if not has_valid and not has_needs:
                reviewer_retries = state.get("reviewer_retries", 0)
                if reviewer_retries < 2:
                    state["reviewer_retries"] = reviewer_retries + 1
                    groupchat.messages.append({
                        "role": "user",
                        "content": (
                            "EvidenceReviewer: Your response did not contain a valid verdict token. "
                            "Start your reply with 'VERDICT: VALID' or 'VERDICT: NEEDS_REVISION' "
                            "and critique the briefing's factual validity and quality."
                        ),
                    })
                    return reviewer

            state["reviewer_retries"] = 0

            # Only a clean VALID verdict terminates the loop; anything else keeps it going
            if has_valid and not has_needs and not errors and not soft:
                print(f"Briefing passed evidence review in {state['epoch']} epoch(s).")
                return None
            if state["epoch"] >= settings.epoch_limit:
                print(f"Reached the {settings.epoch_limit}-epoch limit; keeping deterministic reports authoritative.")
                return None
            state["epoch"] += 1
            print(f"Agent review epoch {state['epoch']}/{settings.epoch_limit}")
            if errors and soft:
                groupchat.messages.append({
                    "role": "user",
                    "content": (
                        f"Briefing REJECTED. Remove unverified figures {sorted(errors)} and "
                        f"unsupported claims {sorted(soft)}. Reviewer notes: {review_msg[:300]}. "
                        "Rewrite using ONLY verified figures and neutral statements."
                    ),
                })
            elif errors:
                groupchat.messages.append({
                    "role": "user",
                    "content": "The verified-data check REJECTED this briefing. These figures do "
                               "NOT appear in VERIFIED DATA and must be removed: "
                               f"{sorted(errors)}. Rewrite using only verified figures.",
                })
            elif soft:
                groupchat.messages.append({
                    "role": "user",
                    "content": "Remove these unsupported claims; the data cannot support them: "
                               f"{sorted(soft)}. Rewrite as neutral statements of reported counts "
                               "only, with no causal, behavioural, or policy claims.",
                })
            elif has_needs:
                groupchat.messages.append({
                    "role": "user",
                    "content": f"The reviewer rejected the briefing with feedback:\n{review_msg[:400]}\n"
                               "Address every issue and rewrite the briefing.",
                })
            else:
                groupchat.messages.append({
                    "role": "user",
                    "content": "Rewrite the briefing ensuring complete factual validity and neutrality.",
                })
            return analyst
        return None

    groupchat = autogen.GroupChat(
        agents=[proxy, analyst, briefing_writer, reviewer],
        messages=[],
        # Leave headroom for the opening message, the final reviewer handoff, and the
        # code-driven rejection notes; the selector enforces the stricter epoch limit.
        max_round=4 * settings.epoch_limit + 2,
        speaker_selection_method=select_next,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
    return proxy.initiate_chat(
        manager,
        message=(
            "VERIFIED DATA follows. The deterministic JSON and Markdown reports are the "
            "source of truth. Use only this data. Analyst, begin.\n\n"
            f"{context}"
        ),
    )
