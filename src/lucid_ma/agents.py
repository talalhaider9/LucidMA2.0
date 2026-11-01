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


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"\b{token}\b", text) is not None


def run_agent_analysis(context: str, settings: Settings = SETTINGS):
    """Iteratively draft and review a briefing, up to the configured epoch limit."""
    config = settings.llm_config
    state = {"epoch": 0, "briefing_invalid": None}
    allowed = _verified_counts(context)
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message=(
            "Analyze VERIFIED DATA: every figure you cite must appear verbatim in it "
            "(commas optional). You may compute only shares of two verified numbers "
            "(e.g., 1998/2918 = 68%); never invent absolute counts or cross-tabulations "
            "that are not in the data. Present 3-5 substantive insights - temporal shape, "
            "category concentration, geography - each backed by its figures. Never infer "
            "causes, trends, population, risk, recommendations, or district-specific rates "
            "absent from the data. If your previous message contained an unverified count, "
            "remove it. Never repeat your previous message verbatim."
        ),
        llm_config=config,
    )
    briefing_writer = autogen.AssistantAgent(
        name="BriefingWriter",
        system_message=(
            "Synthesize the Analyst's analysis into a short neutral briefing: headline "
            "summary plus sections for scope, temporal patterns, geography, and categories. "
            "Every number must appear verbatim in VERIFIED DATA (commas optional); add no "
            "new counts. Describe rankings as reported counts, not rates or risk. Give no "
            "recommendations, advice, implications, or unsupported claims. Apply "
            "NEEDS_REVISION and QUALITY_SUGGESTIONS feedback; improve clarity without "
            "altering the data, and never repeat your previous briefing verbatim."
        ),
        llm_config=config,
    )
    reviewer = autogen.AssistantAgent(
        name="EvidenceReviewer",
        system_message=(
            "Review the latest briefing against VERIFIED DATA on two grounds: (1) factual "
            "support - every count must appear verbatim in the verified data or be a share "
            "of verified numbers; reject invented counts, causes, population, trends, risk, "
            "and policy recommendations; (2) quality - accurate, complete, clear, neutral, "
            "and a substantive analysis rather than a restated list. Reply with your verdict "
            "token on the first line by itself: VALID only if both grounds pass, otherwise "
            "NEEDS_REVISION, then list the weak claims and add QUALITY_SUGGESTIONS with "
            "specific changes for the BriefingWriter to apply."
        ),
        llm_config=config,
    )
    proxy = autogen.UserProxyAgent(
        name="Orchestrator", human_input_mode="NEVER", code_execution_config=False
    )

    def select_next(last_speaker, groupchat):
        if last_speaker is proxy:
            state["epoch"] = 1
            print(f"Agent review epoch {state['epoch']}/{settings.epoch_limit}")
            return analyst
        if last_speaker is analyst:
            return briefing_writer
        if last_speaker is briefing_writer:
            briefing = str(groupchat.messages[-1].get("content") or "")
            bad = _unsupported_counts(briefing, allowed)
            state["briefing_invalid"] = bad or None
            if bad:
                print(f"Rejected briefing: unverified count(s) {sorted(bad)}")
            return reviewer
        if last_speaker is reviewer:
            review = str(groupchat.messages[-1].get("content") or "").upper()
            forced = bool(state.get("briefing_invalid"))
            passed = (not forced) and not _has_token(review, "NEEDS_REVISION")
            if passed:
                print(f"Briefing passed evidence review in {state['epoch']} epoch(s).")
                return None
            if state["epoch"] >= settings.epoch_limit:
                print(f"Reached the {settings.epoch_limit}-epoch limit; keeping deterministic reports authoritative.")
                return None
            state["epoch"] += 1
            print(f"Agent review epoch {state['epoch']}/{settings.epoch_limit}")
            return analyst
        return None

    groupchat = autogen.GroupChat(
        agents=[proxy, analyst, briefing_writer, reviewer],
        messages=[],
        # Leave headroom for the opening message and final reviewer handoff;
        # the selector enforces the stricter epoch limit.
        max_round=3 * settings.epoch_limit + 2,
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
