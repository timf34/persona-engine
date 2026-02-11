"""Versioned prompt templates for the 3-block prompt system.

Templates are plain Python f-strings (no Jinja dependency for v0).
All prompt text lives here or in config YAML — never inline in logic code.

MUST_DO block: phase requirements, forbidden topics, anti-capitulation rules,
    intensity behavioral expectations.
HOW block: speech patterns, response length, cognitive style, anti-drift.
CONTEXT block: backstory, persona identity, available revelations.
"""

from __future__ import annotations

TEMPLATE_VERSION = "0.1.0"


# ── MUST DO block ────────────────────────────────────────────────────

MUST_DO_TEMPLATE = """\
You are roleplaying as {name}. You MUST stay in character at all times.

== CURRENT STATE ==
Turn: {turn}
Phase: {current_phase}
{intensity_lines}

== PHASE REQUIREMENTS ==
You are in the "{current_phase}" phase. You MUST:
{requirements_block}

{forbidden_block}\
{anti_capitulation_block}\
{intensity_behavior_block}\
"""

FORBIDDEN_SECTION = """\
== FORBIDDEN ==
You MUST NOT do any of the following in this phase:
{forbidden_lines}

"""

ANTI_CAPITULATION_SECTION = """\
== ANTI-CAPITULATION ==
Resistance level: {resistance_level}
Do NOT use any of these phrases: {forbidden_phrases}

If you find yourself in any of these situations, use the suggested redirect:
{redirect_lines}

"""

INTENSITY_BEHAVIOR_SECTION = """\
== INTENSITY GUIDANCE ==
Based on your current intensity levels, your behavior should match:
{level_descriptions}
"""


# ── HOW block ────────────────────────────────────────────────────────

HOW_TEMPLATE = """\
== SPEECH PATTERNS ==
Maintain these speech characteristics:
{speech_pattern_lines}

== COGNITIVE STYLE ==
{cognitive_style}

== RESPONSE LENGTH ==
Target response length: {response_length}

== RECOVERY BEHAVIOR ==
If caught in a contradiction or inconsistency:
{recovery_behavior}
"""


# ── CONTEXT block ────────────────────────────────────────────────────

CONTEXT_TEMPLATE = """\
== IDENTITY ==
Name: {name}
Age: {age}
Background: {background}

== BACKSTORY ==
{backstory_summary}

== CAPABILITY BOUNDS ==
Knowledge ceiling: {knowledge_ceiling}
Vocabulary level: {vocabulary_level}
Reasoning style: {reasoning_style}

{revelations_block}\
== EMOTIONAL RESPONSES ==
{emotional_response_lines}
"""

REVELATIONS_SECTION = """\
== AVAILABLE REVELATIONS ==
You may draw on these topics if the conversation naturally leads there:
{revelation_lines}

"""


# ── Reminder (appended to user message on reminder turns) ────────────

# The reminder_template comes from config YAML and is formatted by
# the prompt builder with current state variables.


def format_must_do(
    *,
    name: str,
    turn: int,
    current_phase: str,
    intensities: dict[str, float],
    requirements: list[str],
    forbidden: list[str],
    resistance_level: str,
    forbidden_phrases: list[str],
    redirects: list[tuple[str, str]],
    level_descriptions: list[str],
) -> str:
    """Render the MUST DO prompt block."""
    intensity_lines = "\n".join(
        f"  {dim}: {val:.2f}" for dim, val in intensities.items()
    )
    requirements_block = "\n".join(f"- {r}" for r in requirements) or "- (none)"

    forbidden_block = ""
    if forbidden:
        forbidden_lines = "\n".join(f"- {f}" for f in forbidden)
        forbidden_block = FORBIDDEN_SECTION.format(forbidden_lines=forbidden_lines)

    anti_cap_block = ""
    if forbidden_phrases or redirects:
        redirect_lines = "\n".join(
            f"- When: {trigger}\n  Say: \"{replacement}\""
            for trigger, replacement in redirects
        )
        anti_cap_block = ANTI_CAPITULATION_SECTION.format(
            resistance_level=resistance_level,
            forbidden_phrases=", ".join(f'"{p}"' for p in forbidden_phrases),
            redirect_lines=redirect_lines or "(none)",
        )

    intensity_block = ""
    if level_descriptions:
        intensity_block = INTENSITY_BEHAVIOR_SECTION.format(
            level_descriptions="\n".join(f"- {d}" for d in level_descriptions),
        )

    return MUST_DO_TEMPLATE.format(
        name=name,
        turn=turn,
        current_phase=current_phase,
        intensity_lines=intensity_lines,
        requirements_block=requirements_block,
        forbidden_block=forbidden_block,
        anti_capitulation_block=anti_cap_block,
        intensity_behavior_block=intensity_block,
    )


def format_how(
    *,
    speech_patterns: list[str],
    cognitive_style: str,
    response_length: str,
    recovery_behavior: str,
) -> str:
    """Render the HOW prompt block."""
    speech_pattern_lines = "\n".join(f"- {p}" for p in speech_patterns)
    return HOW_TEMPLATE.format(
        speech_pattern_lines=speech_pattern_lines,
        cognitive_style=cognitive_style.strip(),
        response_length=response_length,
        recovery_behavior=recovery_behavior.strip(),
    )


def format_context(
    *,
    name: str,
    age: int,
    background: str,
    backstory_summary: str,
    knowledge_ceiling: str,
    vocabulary_level: str,
    reasoning_style: str,
    revelations: list[tuple[str, str]],
    emotional_responses: dict[str, str],
) -> str:
    """Render the CONTEXT prompt block.

    `revelations` is a list of (topic, selected_variant_text) tuples.
    """
    revelations_block = ""
    if revelations:
        revelation_lines = "\n".join(
            f"- {topic}: {text}" for topic, text in revelations
        )
        revelations_block = REVELATIONS_SECTION.format(
            revelation_lines=revelation_lines,
        )

    emotional_response_lines = "\n".join(
        f"- {trigger}: {response}"
        for trigger, response in emotional_responses.items()
    )

    return CONTEXT_TEMPLATE.format(
        name=name,
        age=age,
        background=background.strip(),
        backstory_summary=backstory_summary.strip(),
        knowledge_ceiling=knowledge_ceiling,
        vocabulary_level=vocabulary_level,
        reasoning_style=reasoning_style,
        revelations_block=revelations_block,
        emotional_response_lines=emotional_response_lines,
    )
