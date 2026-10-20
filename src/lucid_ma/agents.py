"""AutoGen agent setup and semantic early-stopping conversation."""

from sentence_transformers import util
import autogen

from .config import SETTINGS, Settings


def run_agent_analysis(context: str, embedder, settings: Settings = SETTINGS):
    history = []
    config = settings.llm_config
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message=("You are a Crime Analyst. Review the DATA and FEEDBACK. "
                        "If feedback says INVALID, refine your analysis. If VALID, confirm hotspot details."),
        llm_config=config,
    )
    predictor = autogen.AssistantAgent(
        name="Predictor",
        system_message=('Output ONLY: "PREDICTION: [Crime] at [Lat/Lon] around [Time]." '
                        "Use exact coordinates from the Analyst."),
        llm_config=config,
    )
    feedback = autogen.AssistantAgent(
        name="Feedback",
        system_message='Check whether the Predictor used specific Lat/Lon coordinates. Say "VALID" or "INVALID".',
        llm_config=config,
    )
    proxy = autogen.UserProxyAgent(
        name="Orchestrator", human_input_mode="NEVER", code_execution_config=False
    )

    def select_next(last_speaker, groupchat):
        if last_speaker is proxy or last_speaker is feedback:
            return analyst
        if last_speaker is analyst:
            return predictor
        if last_speaker is predictor:
            message = groupchat.messages[-1].get("content") or ""
            embedding = embedder.encode(message, convert_to_tensor=True)
            if history:
                score = util.pytorch_cos_sim(embedding, history[-1]).item()
                print(f"Convergence score: {score:.4f}")
                history.append(embedding)
                if score > settings.convergence_threshold:
                    print("System converged; stopping the agent loop.")
                    return None
            else:
                history.append(embedding)
            return feedback
        return None

    groupchat = autogen.GroupChat(
        agents=[proxy, analyst, predictor, feedback], messages=[],
        max_round=settings.max_rounds, speaker_selection_method=select_next,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
    return proxy.initiate_chat(
        manager, message=f"Here is the verified data:\n{context}\n\nAnalyst, begin report."
    )
