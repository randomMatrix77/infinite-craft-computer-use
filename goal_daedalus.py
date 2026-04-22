import json
import logging
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from vision_system import VisionSystem
from hand_system import HandSystem

from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
# Consider moving these runtime constants into a dedicated config.py module.
# MODEL_NAME = "huggingface.co/unsloth/gemma-4-e4b-it-gguf:UD-Q8_K_XL"
MODEL_NAME = "huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL"
LOCAL_BASE_URL = "https://butterfly-pci-laughing-executed.trycloudflare.com/v1/"
MAX_STEPS: int = 50  # <--- THE BUDGET

MAX_TOKENS: int = 2048
LLM_TIMEOUT_SECONDS: int = 300
LLM_TEMPERATURE: float = 0.6

STARTUP_DELAY_SECONDS: float = 3.0
POST_SUCCESS_INVENTORY_DELAY_SECONDS: float = 1.5
LOOP_DELAY_SECONDS: float = 0.5
MAX_CONTEXT_INVENTORY_ITEMS: int = 40
BASIC_ITEMS_COUNT: int = 4
RECENT_ITEMS_COUNT: int = 20
MIDDLE_SAMPLE_COUNT: int = 10
HISTORY_WINDOW_SIZE: int = 40
CRITICAL_STEPS_THRESHOLD: int = 10
WARNING_STEPS_THRESHOLD: int = 25

LOG_LEVEL_DEFAULT: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logger = logging.getLogger(__name__)


class GoalVoyager:
    """Run LLM-guided planning and UI execution for Infinite Craft."""

    def __init__(self, target_element: str) -> None:
        """Initialize the voyager with target state and subsystems."""
        self.target_element = target_element
        self.eyes = VisionSystem()
        self.hand = HandSystem()

        # Swapped to local ChatOpenAI wrapper for Gemma-4
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            openai_api_key="dummy_key",
            openai_api_base=LOCAL_BASE_URL,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        self.history: Set[Tuple[str, str]] = set()
        self.inventory: List[str] = []
        self.step_count: int = 0

    def _clean_json(self, text: str) -> Dict[str, Any]:
        """Parse a JSON object from raw or markdown-wrapped model output."""
        if not text:
            return {}
        if not isinstance(text, str):
            text = str(text)

        try:
            candidates: List[str] = []
            candidates.extend(
                re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
            )
            candidates.append(text)

            for candidate in candidates:
                cleaned = candidate.strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

                parsed_direct = self._parse_json_object(cleaned)
                if parsed_direct is not None:
                    return parsed_direct

                for match in re.finditer(r"\{[\s\S]*?\}", cleaned):
                    parsed_match = self._parse_json_object(match.group(0))
                    if parsed_match is not None:
                        return parsed_match

            logger.warning("Agent output did not contain a parseable JSON object.")
            logger.debug("Raw model output for failed JSON parse: %s", text)
            return {}
        except Exception as error:
            logger.exception("JSON parse pipeline failed: %s", error)
            return {}

    @staticmethod
    def _parse_json_object(candidate: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse a candidate string as a JSON dictionary."""
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def update_inventory(self) -> None:
        """Refresh inventory from visible sidebar labels."""
        logger.info("Vision scan: reading visible inventory.")
        try:
            visible_items = self.eyes.read_sidebar_text()
        except Exception as error:
            logger.exception("Inventory scan failed: %s", error)
            return

        if not visible_items:
            logger.warning("Vision returned no sidebar items. Sidebar may be closed.")
            return

        added_count: int = 0
        for item in visible_items:
            if item not in self.inventory:
                self.inventory.append(item)
                added_count += 1

        if added_count > 0:
            logger.info(
                "Discovered %s new items. Inventory size: %s",
                added_count,
                len(self.inventory),
            )
        else:
            logger.info("No new items. Inventory size: %s", len(self.inventory))

    def format_history(self) -> str:
        """Return a bounded history string for prompt context."""
        if not self.history:
            return "None"
        recent = list(self.history)[-HISTORY_WINDOW_SIZE:]
        return ", ".join([f"({e1} + {e2})" for e1, e2 in recent])

    def _build_inventory_context(self) -> List[str]:
        """Build bounded inventory context for the planner prompt."""
        if len(self.inventory) <= MAX_CONTEXT_INVENTORY_ITEMS:
            return self.inventory

        basics = self.inventory[:BASIC_ITEMS_COUNT]
        recent = self.inventory[-RECENT_ITEMS_COUNT:]
        middle_pool = self.inventory[BASIC_ITEMS_COUNT:-RECENT_ITEMS_COUNT]
        middle = (
            random.sample(middle_pool, MIDDLE_SAMPLE_COUNT)
            if len(middle_pool) > MIDDLE_SAMPLE_COUNT
            else middle_pool
        )
        return list(set(basics + recent + middle))

    def _build_urgency_note(self, steps_remaining: int) -> str:
        """Create urgency guidance based on remaining budget."""
        if steps_remaining < CRITICAL_STEPS_THRESHOLD:
            return "CRITICAL WARNING: YOU ARE ALMOST OUT OF MOVES. DO NOT EXPERIMENT. TAKE THE MOST DIRECT PATH POSSIBLE."
        if steps_remaining < WARNING_STEPS_THRESHOLD:
            return "WARNING: You are halfway through your budget. Stop exploring random concepts and focus strictly on the goal."
        return ""

    def plan_next_move(self) -> Optional[Dict[str, Any]]:
        """Request the next combine action from the language model."""
        steps_remaining = MAX_STEPS - self.step_count
        context_inv = self._build_inventory_context()
        history_str = self.format_history()
        urgency_note = self._build_urgency_note(steps_remaining)

        # Gemma 4 System Prompt with <|think|> tag
        sys_prompt = "<|think|> You are a logical planning agent playing Infinite Craft. Always reason step-by-step before outputting JSON."

        prompt = f"""
        GOAL: Create "{self.target_element}".
        BUDGET: {MAX_STEPS} Steps Total.
        STEPS REMAINING: {steps_remaining}
        
        CURRENT INVENTORY: {', '.join(context_inv)}
        
        ALREADY TRIED (DO NOT REPEAT): 
        {history_str}
        
        {urgency_note}

        CRITICAL GAME MECHANIC:
        You can combine the same element twice (e.g., Water + Water -> Lake). The order of elements does not matter (e.g., Water + Fire is the same as Fire + Water).
        
        TASK:
        1. Analyze the inventory.
        2. Reason backwards from "{self.target_element}".
        3. Select the SINGLE most efficient combination to get closer to the goal. You must ONLY use elements that currently exist in the inventory.
        
        FORMAT (Strict JSON at the very end):
        {{
            "thought": "I have {steps_remaining} steps left. To get {self.target_element}, I need X and Y.",
            "element_1": "ElementX",
            "element_2": "ElementY"
        }}
        """

        try:
            logger.info("Agent strategizing. Moves left: %s", steps_remaining)

            messages = [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]

            response = self.llm.invoke(messages)
            plan = self._clean_json(response.content)

            if not plan or "element_1" not in plan:
                logger.warning("Agent returned invalid JSON. Retrying next step.")
                return None

            return plan
        except KeyboardInterrupt:
            logger.info("Planning interrupted by user.")
            raise
        except Exception as error:
            logger.exception("Planner invocation failed: %s", error)
            return None

    def _goal_reached(self) -> bool:
        """Check whether the target element is already in inventory."""
        return self.target_element in self.inventory

    def _extract_plan_fields(
        self, plan: Dict[str, Any]
    ) -> Optional[Tuple[str, str, str]]:
        """Validate and normalize plan fields from the model output."""
        e1 = str(plan.get("element_1", "")).strip()
        e2 = str(plan.get("element_2", "")).strip()
        thought = str(plan.get("thought", "No thought provided.")).strip()

        if not e1 or not e2:
            logger.warning("Plan missing required element fields.")
            return None
        return e1, e2, thought

    def _execute_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate a plan and execute the UI combine action."""
        parsed = self._extract_plan_fields(plan)
        if not parsed:
            return False
        e1, e2, thought = parsed

        if e1 not in self.inventory or e2 not in self.inventory:
            logger.warning(
                "Agent hallucinated invalid pair: (%s, %s). Skipping.", e1, e2
            )
            self.step_count -= 1  # Don't penalize hallucinations
            return False

        pair = tuple(sorted((e1, e2)))
        if pair in self.history:
            logger.info("Duplicate pair %s. Penalty applied.", pair)
            return False

        logger.debug('Planner thought: "%s"', thought)
        logger.info("Action: combining %s + %s", e1, e2)

        success = self.hand.combine(e1, e2)
        self.history.add(pair)

        if success:
            time.sleep(POST_SUCCESS_INVENTORY_DELAY_SECONDS)
            self.update_inventory()

        return True

    def run(self) -> None:
        """Run the main planning and execution loop."""
        logger.info("Voyager online. Target: %s", self.target_element)
        logger.info("Time limit: %s steps.", MAX_STEPS)
        time.sleep(STARTUP_DELAY_SECONDS)

        self.update_inventory()
        if not self.inventory:
            logger.error("Cannot read inventory. Exiting.")
            return

        try:
            while self.step_count < MAX_STEPS:
                self.step_count += 1
                logger.info("--- STEP %s/%s ---", self.step_count, MAX_STEPS)

                if self._goal_reached():
                    logger.info(
                        "VICTORY: '%s' created in %s steps.",
                        self.target_element,
                        self.step_count,
                    )
                    return

                plan = self.plan_next_move()
                if not plan:
                    continue

                step_completed = self._execute_plan(plan)
                if step_completed:
                    time.sleep(LOOP_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.info("Run interrupted by user. Exiting safely.")
            return

        logger.info(
            "GAME OVER. Failed to create '%s' within %s steps.",
            self.target_element,
            MAX_STEPS,
        )


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL_DEFAULT.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


if __name__ == "__main__":
    configure_logging()
    logger.info("=======================================")
    logger.info("   INFINITE CRAFT AUTONOMOUS AGENT     ")
    logger.info("=======================================")

    # The interactive prompt
    user_target = input("🎯 Please provide the required element: ").strip()

    if not user_target:
        logger.error("No element provided. Shutting down.")
        sys.exit(1)

    bot = GoalVoyager(target_element=user_target)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
