"""Optional evidence-bounded AutoGen review of an already computed profile."""

import autogen

from .config import SETTINGS, Settings


def run_agent_analysis(context: str, settings: Settings = SETTINGS):
    """Run one analyst, briefing writer, and evidence reviewer sequence."""
    config = settings.llm_config
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message=(
            "Write four concise factual findings from VERIFIED DATA. Quote only supplied "
            "counts, percentages, time periods, categories, and coarse hotspot cells. "
            "Do not infer unreported crime, population, causes, trends, safety risk, or "
            "district-specific rates that are not in the data."
        ),
        llm_config=config,
    )
    briefing_writer = autogen.AssistantAgent(
        name="BriefingWriter",
        system_message=(
            "Turn the Analyst's findings into a short neutral briefing. Preserve every "
            "number exactly. Use 'highest reported count' for ranked areas. Do not add "
            "interpretations, predictions, or claims that cannot be traced to VERIFIED DATA."
        ),
        llm_config=config,
    )
    reviewer = autogen.AssistantAgent(
        name="EvidenceReviewer",
        system_message=(
            "Review the briefing against VERIFIED DATA. Respond with VALID if all claims "
            "are supported. Otherwise, list unsupported claims only. Reject statements "
            "about causes, population, unreported crime, or risk when those values are absent."
        ),
        llm_config=config,
    )
    proxy = autogen.UserProxyAgent(
        name="Orchestrator", human_input_mode="NEVER", code_execution_config=False
    )

    def select_next(last_speaker, _groupchat):
        if last_speaker is proxy:
            return analyst
        if last_speaker is analyst:
            return briefing_writer
        if last_speaker is briefing_writer:
            return reviewer
        return None

    groupchat = autogen.GroupChat(
        agents=[proxy, analyst, briefing_writer, reviewer],
        messages=[],
        max_round=settings.max_rounds,
        speaker_selection_method=select_next,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
    return proxy.initiate_chat(
        manager,
        message=(
            "VERIFIED DATA follows. The deterministic JSON and Markdown reports are the "
            "source of truth. Use only this data.\n\n"
            f"{context}"
        ),
    )
