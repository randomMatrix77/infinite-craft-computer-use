import logging
import time
from typing import Optional, Tuple

import pyautogui
from vision_system import VisionSystem

pyautogui.FAILSAFE = True

ANVIL_X_RATIO: float = 0.4
ANVIL_Y_RATIO: float = 0.5
CLICK_DURATION_SECONDS: float = 0.5
DRAG_DURATION_SECONDS: float = 0.6
DRAG_HOLD_DELAY_SECONDS: float = 0.1
MIX_WAIT_SECONDS: float = 1.5
CANVAS_SETTLE_SECONDS: float = 1.0
CANVAS_WIPE_DELAY_SECONDS: float = 0.5
CANVAS_BLIND_FALLBACK_OFFSET_X: int = 90
CANVAS_BLIND_FALLBACK_OFFSET_Y: int = 50
RESET_MOUSE_OFFSET_X: int = 200
POST_DRAG_MOUSE_NUDGE_X: int = -100
MICRO_ADJUST_DURATION_SECONDS: float = 0.2
WIGGLE_OFFSETS: Tuple[Tuple[int, int], ...] = ((-40, 0), (40, 0), (0, -40), (0, 40))

logger = logging.getLogger(__name__)


class HandSystem:
    """Execute UI interactions for dragging and combining elements."""

    def __init__(self) -> None:
        """Initialize UI executor and derive key canvas coordinates."""
        self.vision = VisionSystem()

        screen_w, screen_h = pyautogui.size()
        self.anvil_x: int = int(screen_w * ANVIL_X_RATIO)
        self.anvil_y: int = int(screen_h * ANVIL_Y_RATIO)

    def _click_target(self, name: str, region: str = "sidebar") -> bool:
        """Locate and click a target element by vision label."""
        coords = self.vision.find_target(name, region=region)
        if coords:
            try:
                pyautogui.moveTo(coords[0], coords[1], duration=CLICK_DURATION_SECONDS)
                pyautogui.click()
                return True
            except pyautogui.FailSafeException:
                logger.info("PyAutoGUI failsafe triggered while clicking target.")
                raise KeyboardInterrupt
            except Exception as error:
                logger.exception("Click target failed for '%s': %s", name, error)
                return False
        return False

    def drag_to_anvil(self, element_name: str) -> bool:
        """Drag an inventory element to the anvil area."""
        logger.info("Dragging '%s' to anvil.", element_name)
        coords: Optional[Tuple[int, int]] = self.vision.find_target(
            element_name, region="sidebar"
        )

        if not coords:
            logger.warning("Vision failed to locate '%s'.", element_name)
            return False

        try:
            pyautogui.moveTo(
                coords[0],
                coords[1],
                duration=DRAG_DURATION_SECONDS,
                tween=pyautogui.easeInOutQuad,
            )
            pyautogui.mouseDown()
            time.sleep(DRAG_HOLD_DELAY_SECONDS)
            pyautogui.moveTo(
                self.anvil_x,
                self.anvil_y,
                duration=DRAG_DURATION_SECONDS,
                tween=pyautogui.easeInOutQuad,
            )
            pyautogui.mouseUp()
            pyautogui.moveRel(POST_DRAG_MOUSE_NUDGE_X, 0)
            return True
        except pyautogui.FailSafeException:
            logger.info("PyAutoGUI failsafe triggered during drag.")
            raise KeyboardInterrupt
        except Exception as error:
            logger.exception("Drag failed for '%s': %s", element_name, error)
            return False

    def _reset_mouse(self) -> None:
        """Moves mouse to a neutral spot so it doesn't block vision."""
        try:
            pyautogui.moveTo(self.anvil_x - RESET_MOUSE_OFFSET_X, self.anvil_y)
        except pyautogui.FailSafeException:
            logger.info("PyAutoGUI failsafe triggered while resetting mouse.")
            raise KeyboardInterrupt
        except Exception as error:
            logger.exception("Failed to reset mouse position: %s", error)

    def clear_canvas(self) -> None:
        """Clear the crafting canvas and confirm the dialog when present."""
        logger.info("Clearing canvas.")

        # Using the actual emoji grounds the LLM's visual search
        coords = self.vision.find_target("Broom (🧹) icon", region="canvas_controls")

        hit_confirmed: bool = False

        if coords:
            try:
                pyautogui.moveTo(coords[0], coords[1], duration=CLICK_DURATION_SECONDS)
                pyautogui.click()
                time.sleep(CANVAS_SETTLE_SECONDS)
                if self.vision.find_target("the word 'Yes'", region="center"):
                    hit_confirmed = True
                else:
                    logger.warning(
                        "Missed broom click or popup did not load. Applying micro-adjustments."
                    )
                    for ox, oy in WIGGLE_OFFSETS:
                        new_x = coords[0] + ox
                        new_y = coords[1] + oy
                        pyautogui.moveTo(
                            new_x,
                            new_y,
                            duration=MICRO_ADJUST_DURATION_SECONDS,
                        )
                        pyautogui.click()
                        time.sleep(CANVAS_SETTLE_SECONDS)
                        if self.vision.find_target("the word 'Yes'", region="center"):
                            logger.info("Recovered popup with wiggle adjustment.")
                            hit_confirmed = True
                            break
            except pyautogui.FailSafeException:
                logger.info("PyAutoGUI failsafe triggered while clearing canvas.")
                raise KeyboardInterrupt
            except Exception as error:
                logger.exception("Canvas clear click path failed: %s", error)

        # ATTEMPT 3: DYNAMIC BLIND FALLBACK
        if not hit_confirmed:
            logger.warning("Vision/wiggle failed. Deploying dynamic fallback.")
            try:
                screen_w, screen_h = pyautogui.size()
                sidebar_w = int(screen_w * 0.25)
                blind_x = screen_w - sidebar_w - CANVAS_BLIND_FALLBACK_OFFSET_X
                blind_y = screen_h - CANVAS_BLIND_FALLBACK_OFFSET_Y
                pyautogui.moveTo(blind_x, blind_y, duration=CLICK_DURATION_SECONDS)
                pyautogui.click()
                time.sleep(CANVAS_SETTLE_SECONDS)
            except pyautogui.FailSafeException:
                logger.info("PyAutoGUI failsafe triggered during fallback clear.")
                raise KeyboardInterrupt
            except Exception as error:
                logger.exception("Fallback canvas clear failed: %s", error)

        # 2. CONFIRMATION
        # Explicitly target the text to avoid clicking the empty space of the dialog box
        if self._click_target("the word 'Yes'", region="center"):
            logger.info("Canvas cleared.")
            time.sleep(CANVAS_WIPE_DELAY_SECONDS)
        else:
            logger.debug("No confirmation needed or confirmation popup missed.")

        self._reset_mouse()

    def combine(self, item1: str, item2: str) -> bool:
        """Combine two elements by dragging each to the anvil."""
        logger.info("Combining: %s + %s", item1, item2)

        try:
            success: bool = True
            if not self.drag_to_anvil(item1):
                success = False
            if success and not self.drag_to_anvil(item2):
                success = False

            if success:
                logger.info("Mixing...")
                time.sleep(MIX_WAIT_SECONDS)

            self.clear_canvas()
            time.sleep(CANVAS_WIPE_DELAY_SECONDS)

            return success
        except KeyboardInterrupt:
            logger.info("Combine interrupted by user.")
            raise
        except Exception as error:
            logger.exception("Combine operation failed: %s", error)
            return False
