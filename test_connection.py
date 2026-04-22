import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pyautogui
from openai import OpenAI
from PIL import ImageDraw

pyautogui.FAILSAFE = True

# --- CONFIG ---
# Consider moving these constants into config.py if shared across scripts.
# Swap this based on which model you are currently running on the A100
MODEL_NAME = "huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL"
TUNNEL_URL = "https://moral-day-soviet-chubby.trycloudflare.com/v1/"
TARGET_ELEMENT = "Wind"
# ------------

STARTUP_DELAY_SECONDS: int = 3
SIDEBAR_WIDTH_RATIO: float = 0.25
GRID_SIZE: int = 10
NORMALIZED_COORDINATE_BASE: int = 1000
MARKER_RADIUS: int = 8

LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logger = logging.getLogger(__name__)


def draw_intersection_grid(img_orig: Any, grid_size: int = GRID_SIZE) -> Any:
    """Overlay a coordinate grid on a copy of a PIL image."""
    img = img_orig.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    cell_w = width / grid_size
    cell_h = height / grid_size

    for i in range(1, grid_size):
        x = i * cell_w
        y = i * cell_h
        draw.line([(x, 0), (x, height)], fill="red", width=1)
        draw.line([(0, y), (width, y)], fill="red", width=1)

    for col in range(1, grid_size):
        for row in range(1, grid_size):
            x = col * cell_w
            y = row * cell_h
            pct_x = int((col / grid_size) * 100)
            pct_y = int((row / grid_size) * 100)
            draw.text((x + 2, y + 2), f"({pct_x},{pct_y})", fill="blue")
    return img


def img_to_base64(img_pil: Any) -> str:
    """Encode a PIL image into base64 PNG string."""
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def clean_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a JSON object from potentially wrapped model output."""
    if not text:
        return None
    if not isinstance(text, str):
        text = str(text)

    try:
        candidates = re.findall(
            r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE
        )
        candidates.append(text)
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            match = re.search(r"\{[\s\S]*?\}", cleaned)
            if not match:
                continue
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    except Exception as error:
        logger.exception("JSON cleanup failed: %s", error)
    return None


def main() -> None:
    """Run the connection test for vision-guided coordinate localization."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    client = OpenAI(api_key="dummy_key", base_url=TUNNEL_URL)

    logger.info("Get ready. Taking screenshot in %s seconds...", STARTUP_DELAY_SECONDS)
    time.sleep(STARTUP_DELAY_SECONDS)

    logger.info("Processing screenshots.")
    try:
        full_img = pyautogui.screenshot()
    except pyautogui.FailSafeException:
        logger.info("PyAutoGUI failsafe triggered before screenshot.")
        return
    except Exception as error:
        logger.exception("Failed to take screenshot: %s", error)
        return

    screen_w, screen_h = full_img.size
    crop_w = int(screen_w * SIDEBAR_WIDTH_RATIO)
    crop_box = (screen_w - crop_w, 0, screen_w, screen_h)
    clean_crop_img = full_img.crop(crop_box)
    gridded_crop_img = draw_intersection_grid(clean_crop_img, grid_size=GRID_SIZE)

    clean_base64 = img_to_base64(clean_crop_img)
    grid_base64 = img_to_base64(gridded_crop_img)

    prompt = f"""
You are looking at two images of the exact same UI.
Image 1 is the clean UI screenshot. Use this to read the text perfectly.
Image 2 has a red grid overlaid on it. Every grid intersection is labeled with its (X, Y) percentage coordinates in blue text, like (10, 10), (20, 10), etc.

Step 1: Locate the '{TARGET_ELEMENT}' element in the clean UI (Image 1).
Step 2: Look at the exact same location in the gridded UI (Image 2).
Step 3: Read the blue (X, Y) intersection labels closest to the center of the '{TARGET_ELEMENT}' element. 
Step 4: Multiply those percentage numbers by 10 to estimate the exact NORMALIZED coordinates (0-1000). 

After your thought process, return ONLY a JSON object in this exact format. 
{{"reasoning": "your thoughts", "x": 120, "y": 200}}
"""

    logger.info("Sending payload to A100 (%s).", MODEL_NAME)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "<|think|> You are a precise vision-language agent.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{clean_base64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{grid_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
            top_p=0.95,
        )
    except Exception as error:
        logger.exception("Model request failed: %s", error)
        return

    output_text = response.choices[0].message.content or ""
    logger.info("Model output received.")
    logger.debug("Raw output: %s", output_text)

    data = clean_json_block(output_text)
    if not data:
        logger.error("Could not find a valid JSON block.")
        return

    try:
        norm_x = float(data.get("x", 0))
        norm_y = float(data.get("y", 0))

        crop_width, crop_height = gridded_crop_img.size
        center_x = (norm_x / NORMALIZED_COORDINATE_BASE) * crop_width
        center_y = (norm_y / NORMALIZED_COORDINATE_BASE) * crop_height

        fig, ax = plt.subplots(1)
        ax.imshow(gridded_crop_img)
        circle = patches.Circle(
            (center_x, center_y),
            radius=MARKER_RADIUS,
            edgecolor="lime",
            facecolor="lime",
        )
        ax.add_patch(circle)
        plt.title(f"Target: {TARGET_ELEMENT} | Guess: ({norm_x}, {norm_y})")
        plt.axis("off")
        plt.show()
    except Exception as error:
        logger.exception("Error parsing or rendering coordinates: %s", error)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Test stopped by user.")
