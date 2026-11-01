"""Optional evidence-bounded AutoGen review of an already computed profile."""

import autogen

from .config import SETTINGS, Settings


def run_agent_analysis(context: str, settings: Settings = SETTINGS):
    """Iteratively draft and review a briefing, up to the configured epoch limit."""
    config = settings.llm_config
    state = {"epoch": 0}
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message=(
            "Perform a genuine analysis of VERIFIED DATA, not just a list of findings. "
            "Quote supplied counts, percentages, dates, categories, and coarse hotspot "
            "cells accurately. Interpret the data using only arithmetic derivable from it: "
            "compute each category's share of the total, compare categories, and describe "
            "distribution shape (e.g., the hourly peak, weekday spread, concentration in "
            "the top districts). Present 3-5 substantive analytical insights, each backed "
            "by its supporting figures. When given NEEDS_REVISION feedback, correct the "
            "specific unsupported claims. Never infer unreported crime, population, "
            "causes, trends, safety risk, or district-specific rates absent from the data."
        ),
        llm_config=config,
    )
    briefing_writer = autogen.AssistantAgent(
        name="BriefingWriter",
        system_message=(
            "Write an actual briefing that synthesizes the Analyst's analysis into a short "
            "neutral narrative: a headline summary plus clearly labelled sections for "
            "scope, temporal patterns, geography, and categories. Preserve every number "
            "and category label exactly as supplied. Describe rankings as reported counts, "
            "not rates or risk. Do not add interpretation or unsupported claims beyond "
            "what the Analyst established. When given NEEDS_REVISION feedback, correct the "
            "specific unsupported claims; when given QUALITY_SUGGESTIONS, improve clarity, "
            "structure, and precision without altering the underlying data."
        ),
        llm_config=config,
    )
    reviewer = autogen.AssistantAgent(
        name="EvidenceReviewer",
        system_message=(
            "Review the latest briefing on two grounds. Ground one, factual support: every "
            "claim and number must trace to VERIFIED DATA; reject causes, population, "
            "rates, unreported crime, trends, or risk absent from the data, and reject any "
            "statistic not derivable from it. Ground two, report quality: the briefing "
            "must be a substantive, analysis-based narrative - accurate, complete, clear, "
            "neutral, precise - and not a bare restated list. Begin your reply with the "
            "verdict on its own line: VALID if and only if BOTH grounds pass; otherwise "
            "NEEDS_REVISION followed by the unsupported or weak claims, then add "
            "QUALITY_SUGGESTIONS with specific, actionable changes to the report for the "
            "BriefingWriter to apply."
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
            return reviewer
        if last_speaker is reviewer:
            review = str(groupchat.messages[-1].get("content") or "").strip().upper()
            if review.startswith("VALID"):
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
