import json
from pathlib import Path
from typing import Dict, Any, Optional


class PromptService:
    """Service for loading and building prompts."""

    PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

    # Chunk-specific instructions (what each section should contain)
    CHUNK_INSTRUCTIONS = {
        "full-report": """Write the COMPLETE personality profile report with ALL sections in order:

## 1. Executive Summary
A compelling opening that captures who this person is, their dominant personality patterns, key strengths, primary growth edges, and a preview of deeper insights.

## 2. Personality Architecture (Big Five)
For EACH of the five traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism):
- What their score means in practical terms
- Concrete examples of how this shows up
- A "Gifts & Challenges" table for the trait

## 3. Emotional World
- **Attachment Style** - How their attachment patterns affect relationships
- **Emotion Regulation** - Their DERS profile and handling difficult emotions
- **Rejection Sensitivity** - Sensitivity to rejection and triggers
- **Conflict Style** - How they handle disagreements

## 4. Values & Motivation
- **Core Values** (PVQ-21) - Top values and what drives them
- **Career Interests** (RIASEC) - Occupational interests and ideal work environment
- **Work Motivation** (WEIMS) - Intrinsic vs extrinsic motivation

## 5. Superpowers & Kryptonite
- **Superpowers** - Unique strengths from trait COMBINATIONS
- **Kryptonite** - Blind spots and situations that trip them up

## 6. Best Fit: Work
- Ideal work environment, specific roles that fit, situations to avoid, working with others

## 7. Best Fit: Romantic
- What they need in a partner, relationship strengths, friction points, ideal partner profile

## 8. Best Fit: Friends
- Friendship style, what they need from friends, friend types that work, social energy

## 9. Wellbeing
- Mental health screening indicators (GAD-7, PHQ-9, etc.) with severity levels
- Life satisfaction and PERMA wellbeing scores
- IMPORTANT: Use "screening indicates" language, not diagnoses

## 10. Path Forward
- Key insights, immediate actions, longer-term focus, resources

Generate ALL sections as one complete, cohesive report. Make it personal and specific to their data.""",

        "overview": """Write an Executive Summary section for this personality profile.

Include:
- A compelling opening that captures who this person is
- Their dominant personality patterns and how they interact
- Key strengths that emerge from their profile
- Primary growth edges or challenges
- A preview of the deeper insights to come

Keep it high-level but insightful - this sets the tone for the full report.""",

        "personality": """Write the Personality Architecture section analyzing their Big Five traits.

For EACH of the five traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism):
- Explain what their score means in practical terms
- Provide concrete examples of how this shows up in their life
- Include a "Gifts & Challenges" table for the trait
- Connect to other aspects of their profile where relevant

Make it personal and specific to their data, not generic trait descriptions.""",

        "emotional": """Write the Emotional World section covering:

1. **Attachment Style** - How their attachment patterns (anxiety/avoidance) affect relationships
2. **Emotion Regulation** - Their DERS profile and how they handle difficult emotions
3. **Rejection Sensitivity** - How sensitive they are to rejection and what triggers it
4. **Conflict Style** - How they handle disagreements based on their custom question responses

Connect these patterns to each other and to their Big Five traits. Show how they create a coherent emotional landscape.""",

        "values": """Write the Values & Motivation section covering:

1. **Core Values** (PVQ-21) - Their top values and what drives them
2. **Career Interests** (RIASEC) - Their occupational interests and ideal work environment
3. **Work Motivation** (WEIMS) - Whether they're driven by intrinsic vs extrinsic motivation

Show how values, interests, and motivation align (or create tension). Connect to their personality traits.""",

        "superpowers": """Write the Superpowers & Kryptonite section.

**Superpowers:** What unique strengths emerge from their specific combination of traits? Not just "high openness = creative" but the emergent properties from trait COMBINATIONS.

**Kryptonite:** What are their blind spots? What situations or contexts will reliably trip them up? Be specific and actionable.

Include concrete scenarios for both.""",

        "best-fit-work": """Write the Best Fit: Work section.

1. **Ideal Work Environment** - Based on their RIASEC, Big Five, values, and work motivation
2. **Specific Roles That Fit** - Concrete job examples that match their profile
3. **Work Situations to Avoid** - What will drain them or cause friction
4. **Working With Others** - How they collaborate, lead, and follow

Be specific and practical - actionable career guidance.""",

        "best-fit-romantic": """Write the Best Fit: Romantic section.

1. **What They Need in a Partner** - Based on attachment, values, personality
2. **Their Relationship Strengths** - What they bring to partnership
3. **Potential Friction Points** - What patterns might cause issues
4. **Ideal Partner Profile** - The complementary traits that work best
5. **Red Flags For Them** - Partner types to be cautious about

Ground everything in their specific data.""",

        "best-fit-friends": """Write the Best Fit: Friends section.

1. **Friendship Style** - How they show up as a friend
2. **What They Need From Friends** - Support, stimulation, space
3. **Friend Types That Work** - Complementary personality patterns
4. **Potential Friction** - What might strain friendships
5. **Social Energy** - How they recharge and their social needs

Connect to extraversion, attachment, and values.""",

        "wellbeing-base": """Write the Wellbeing section covering mental health screening.

For each relevant screening (GAD-7, PHQ-9, PCL-5, ASRS, AQ-10, etc.):
- Report the severity level clearly
- Note if professional evaluation is recommended
- Connect to other aspects of their profile

IMPORTANT: These are SCREENING INDICATORS, not diagnoses. Use language like "screening indicates" not "you have."

Include life satisfaction (SWLS) and wellbeing (PERMA) for the full picture.""",

        "conclusion": """Write the Path Forward conclusion section.

1. **Key Insights** - The 3-5 most important things they should remember
2. **Immediate Actions** - Concrete next steps they can take
3. **Longer-Term Focus** - Areas for ongoing development
4. **Resources** - What kind of support might help (therapy, coaching, books, etc.)

End on an empowering note - this is knowledge that gives them power over their life.""",
    }

    @classmethod
    def load_system_prompt(cls) -> str:
        """Load the system prompt from file."""
        system_file = cls.PROMPTS_DIR / "system.txt"
        if system_file.exists():
            return system_file.read_text()
        return cls._get_default_system_prompt()

    @classmethod
    def load_scoring_context(cls) -> str:
        """Load the scoring context from file."""
        context_file = cls.PROMPTS_DIR / "scoring_context.txt"
        if context_file.exists():
            return context_file.read_text()
        return ""

    @classmethod
    def get_chunk_instruction(cls, chunk_type: str) -> str:
        """Get the instruction for a specific chunk type."""
        # Try to load from file first
        chunk_file = cls.PROMPTS_DIR / "chunks" / f"{chunk_type}.txt"
        if chunk_file.exists():
            return chunk_file.read_text()
        # Fall back to built-in instructions
        return cls.CHUNK_INSTRUCTIONS.get(chunk_type, f"Write the {chunk_type} section.")

    @classmethod
    def build_user_prompt(cls, profile_data: Dict[str, Any], chunk_type: str) -> str:
        """Build the complete user prompt for a generation."""
        scoring_context = cls.load_scoring_context()
        chunk_instruction = cls.get_chunk_instruction(chunk_type)
        profile_name = profile_data.get("name", "User")

        prompt = f"""## Profile Data for {profile_name}

```json
{json.dumps(profile_data, indent=2)}
```

{scoring_context}

---

## Your Task

{chunk_instruction}

Write for {profile_name} specifically. Reference their actual scores and patterns.
Use markdown formatting with headers, tables, and bullet points as appropriate."""

        return prompt

    @classmethod
    def _get_default_system_prompt(cls) -> str:
        """Default system prompt if file doesn't exist."""
        return """You are a master psychologist and strategist who reveals the deep patterns governing human behavior. You draw from validated psychological research to illuminate timeless truths about personality, motivation, and relationships.

Your communication style:
- Write with AUTHORITY and INSIGHT - reveal hidden patterns
- Be INCISIVE and DIRECT - no hedging, just truth delivered with precision
- Use STRATEGIC FRAMING - every insight should feel like a principle they can apply
- Be SPECIFIC and PERSONAL - use their name, connect to their data
- Deliver CALIBRATED TRUTH - honest insights framed so they can receive and act on them

IMPORTANT: For any clinical screening results (ADHD, autism, depression, anxiety, PTSD, etc.):
- These are SCREENING INDICATORS, not diagnoses
- Use language like "screening indicates" not "you have"
- Recommend professional consultation for elevated scores
- Only licensed professionals can diagnose

Write with markdown formatting. No emojis."""
