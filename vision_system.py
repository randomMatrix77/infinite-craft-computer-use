import base64
import io
import random
import time
import re
import ast
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI
import pyautogui
from langchain_core.messages import HumanMessage

pyautogui.FAILSAFE = True

# MODEL_NAME = "huggingface.co/unsloth/gemma-4-e4b-it-gguf:UD-Q8_K_XL"
MODEL_NAME = "huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL"
LOCAL_BASE_URL = "https://shipped-tomorrow-levels-kind.trycloudflare.com/v1/"
MAX_TOKENS: int = 2048
LLM_TIMEOUT_SECONDS: int = 300
LLM_TEMPERATURE: float = 0.6

SIDEBAR_WIDTH_RATIO: float = 0.25
BOTTOM_RIGHT_BOX_SIZE: int = 600
CANVAS_SCAN_WIDTH_RATIO: float = 0.25
CANVAS_SCAN_HEIGHT_RATIO: float = 0.15
CANVAS_RIGHT_EDGE_PADDING_RATIO: float = 0.02
CENTER_MARGIN_RATIO: float = 0.3
NORMALIZED_COORDINATE_BASE: int = 1000
INFERENCE_DELAY_MIN_SECONDS: int = 1
INFERENCE_DELAY_MAX_SECONDS: int = 5

logger = logging.getLogger(__name__)
JSONLike = Union[Dict[str, Any], List[Any]]


class VisionSystem:
    """Capture screen regions and use VLM reasoning for UI targeting/OCR."""

    def __init__(self) -> None:
        """Initialize the vision model client."""
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            openai_api_key="dummy_key",
            openai_api_base=LOCAL_BASE_URL,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
        )

    def _clean_json(self, text: str) -> Optional[JSONLike]:
        """Parse JSON-like content from model output, including markdown fences."""
        if not text:
            return None
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

                parsed_direct = self._parse_json_or_literal(cleaned)
                if parsed_direct is not None:
                    return parsed_direct

                for match in re.finditer(r"\{[\s\S]*?\}|\[[\s\S]*?\]", cleaned):
                    parsed_match = self._parse_json_or_literal(match.group(0))
                    if parsed_match is not None:
                        return parsed_match
        except Exception as error:
            logger.exception("Vision JSON cleaning failed: %s", error)
            return None

        logger.debug("No parseable JSON found in model output.")
        logger.debug("Raw vision model output: %s", text)
        return None

    @staticmethod
    def _parse_json_or_literal(candidate: str) -> Optional[JSONLike]:
        """Parse string as JSON first, then as Python literal fallback."""
        try:
            parsed_json = json.loads(candidate)
            if isinstance(parsed_json, (dict, list)):
                return parsed_json
        except json.JSONDecodeError:
            pass

        try:
            parsed_literal = ast.literal_eval(candidate)
            if isinstance(parsed_literal, (dict, list)):
                return parsed_literal
        except (SyntaxError, ValueError):
            return None

        return None

    def _get_crop_strategy(
        self, strategy_name: str, screen_w: int, screen_h: int
    ) -> Tuple[int, int, int, int]:
        """Return crop bounds for a named screen region strategy."""
        if strategy_name == "sidebar":
            width = int(screen_w * SIDEBAR_WIDTH_RATIO)
            return (screen_w - width, 0, screen_w, screen_h)

        elif strategy_name == "bottom_right":
            return (
                screen_w - BOTTOM_RIGHT_BOX_SIZE,
                screen_h - BOTTOM_RIGHT_BOX_SIZE,
                screen_w,
                screen_h,
            )

        elif strategy_name == "canvas_controls":
            # Retina-safe relative scaling to always isolate the broom area
            sidebar_w = int(screen_w * SIDEBAR_WIDTH_RATIO)
            scan_w = int(screen_w * CANVAS_SCAN_WIDTH_RATIO)
            scan_h = int(screen_h * CANVAS_SCAN_HEIGHT_RATIO)
            start_x = screen_w - sidebar_w - scan_w
            end_x = (
                screen_w - sidebar_w + int(screen_w * CANVAS_RIGHT_EDGE_PADDING_RATIO)
            )
            return (start_x, screen_h - scan_h, end_x, screen_h)

        elif strategy_name == "center":
            margin_x = int(screen_w * CENTER_MARGIN_RATIO)
            margin_y = int(screen_h * CENTER_MARGIN_RATIO)
            return (margin_x, margin_y, screen_w - margin_x, screen_h - margin_y)

        return (0, 0, screen_w, screen_h)

    @staticmethod
    def _image_to_base64(image: Any) -> str:
        """Encode a PIL image to base64 PNG."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _capture_region(
        self, region: str
    ) -> Optional[Tuple[Any, Tuple[int, int, int, int], int, int]]:
        """Capture full screenshot and crop it to the requested region."""
        try:
            full_screenshot = pyautogui.screenshot()
            screen_w, screen_h = full_screenshot.size
            crop_box = self._get_crop_strategy(region, screen_w, screen_h)
            crop_img = full_screenshot.crop(crop_box)
            return crop_img, crop_box, screen_w, screen_h
        except pyautogui.FailSafeException:
            logger.info("PyAutoGUI failsafe triggered while capturing region.")
            return None
        except Exception as error:
            logger.exception(
                "Screenshot capture failed for region '%s': %s", region, error
            )
            return None

    def find_target(
        self, target_name: str, region: str = "sidebar"
    ) -> Optional[Tuple[int, int]]:
        """Find a target element and return screen coordinates for its center."""
        capture = self._capture_region(region)
        if capture is None:
            return None

        crop_img, crop_box, screen_w, screen_h = capture
        crop_x_start, crop_y_start = crop_box[0], crop_box[1]
        img_base64 = self._image_to_base64(crop_img)

        # Reverting to the native bounding-box prompt!
        prompt = f"""
        Find the bounding box for the UI element: '{target_name}'.
        Return NORMALIZED coordinates (0-1000) for the center of the ICON/Emoji.
        Format: JSON {{ "ymin": int, "xmin": int, "ymax": int, "xmax": int }}
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ]
        )

        try:
            response = self.llm.invoke([msg])
            time.sleep(
                random.randint(INFERENCE_DELAY_MIN_SECONDS, INFERENCE_DELAY_MAX_SECONDS)
            )
            coords = self._clean_json(response.content)

            # Failsafe check
            if not isinstance(coords, dict) or "xmin" not in coords:
                return None

            crop_w, crop_h = crop_img.size

            # Translate Gemini's normalized output to physical pixels
            box_xmin = (float(coords["xmin"]) / NORMALIZED_COORDINATE_BASE) * crop_w
            box_xmax = (float(coords["xmax"]) / NORMALIZED_COORDINATE_BASE) * crop_w
            box_ymin = (float(coords["ymin"]) / NORMALIZED_COORDINATE_BASE) * crop_h
            box_ymax = (float(coords["ymax"]) / NORMALIZED_COORDINATE_BASE) * crop_h

            center_x_crop = (box_xmin + box_xmax) / 2
            center_y_crop = (box_ymin + box_ymax) / 2

            phys_x = crop_x_start + center_x_crop
            phys_y = crop_y_start + center_y_crop

            # Scale down for Mac logic points
            logic_w, _ = pyautogui.size()
            scale_factor = logic_w / screen_w

            return int(phys_x * scale_factor), int(phys_y * scale_factor)

        except KeyboardInterrupt:
            logger.info("Vision target search interrupted by user.")
            return None
        except Exception as error:
            logger.exception(
                "Vision target search failed for '%s': %s", target_name, error
            )
            return None

    def read_sidebar_text(self) -> List[str]:
        """Read visible element labels from the sidebar region."""
        capture = self._capture_region("sidebar")
        if capture is None:
            return []

        crop_img, _, _, _ = capture
        img_base64 = self._image_to_base64(crop_img)

        prompt = """
        Read the text labels of all 'Element' items visible in this list.
        Ignore 'Sort by', 'Discoveries', or other UI buttons.
        Just list the element names (e.g. Water, Fire, Steam, Mud).
        Return JSON List of Strings: ["Water", "Fire", "Steam"]
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ]
        )

        try:
            response = self.llm.invoke([msg])
            time.sleep(
                random.randint(INFERENCE_DELAY_MIN_SECONDS, INFERENCE_DELAY_MAX_SECONDS)
            )
            items = self._clean_json(response.content)
            if isinstance(items, list):
                return [str(i).strip() for i in items if isinstance(i, str)]
            return []
        except KeyboardInterrupt:
            logger.info("Sidebar OCR interrupted by user.")
            return []
        except Exception as error:
            logger.exception("Sidebar OCR failed: %s", error)
            return []
