# Infinite Craft Computer Use Agent 🤖⚒️

An autonomous, zero-shot AI agent that plays [Infinite Craft](https://neal.fun/infinite-craft/) using pure visual reasoning and spatial execution. 

**Key Constraints Solved:** Zero DOM/HTML access, no backend API hooking, and no hardcoded game logic. The agent relies entirely on a local vision-language model to read the screen, calculate exact `(X,Y)` coordinates, and execute deterministic mouse movements.

📺 **[Watch the 60-Second Architecture Breakdown on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID_HERE)**

---

## 🧠 System Architecture

The system operates on a continuous, closed-loop pipeline mimicking human computer use:

1. **Eyes (`VisionSystem`):** Takes a screenshot of the active UI, crops to relevant regions, and uses zero-shot VLM capabilities to parse dynamic inventory states and locate target coordinates (translating normalized bounding boxes to physical screen pixels).
2. **Brain (`GoalVoyager`):** Powered by a local **Gemma-4 26B** model. It maintains contextual memory of discovered items and uses chain-of-thought backward reasoning to deduce the most logical elemental combinations to reach the user's target.
3. **Hands (`HandSystem`):** A deterministic execution layer using `PyAutoGUI` to translate the Brain's spatial intent into physical mouse movements. Handles dragging, dropping, and dynamic UI state management (auto-clearing the canvas when cluttered).

## 🛠️ Project Structure

- `goal_daedalus.py`: Main orchestrator. Handles the planning loop, prompt calls, history tracking, and step-budget management.
- `vision_system.py`: Screenshot capture, region cropping, OCR, and target localization via VLM.
- `hand_system.py`: UI interactions (drag, click, clear canvas) with built-in micro-adjustments and blind fallbacks.
- `test_connection.py`: *(Optional)* Manual vision endpoint sanity check and coordinate visualization.

## 📋 Requirements

- **Python:** `>= 3.12`
- **OS:** macOS / Linux / Windows desktop with GUI and mouse control permissions enabled.
- **Environment:** A browser window with Infinite Craft open and fully visible on the screen.
- **LLM:** A local OpenAI-compatible inference endpoint serving your Gemma-4 model (or equivalent VLM).

**Dependencies:**
- `langchain-openai`
- `langchain-google-genai`
- `pyautogui`
- `pillow`
- `matplotlib`
- `python-dotenv`

## 🚀 Setup & Execution

1. **Install dependencies:**
   ```bash
   uv sync
   ```
2. Configure your environment:
LOCAL_BASE_URL (Your inference endpoint). MODEL_NAME (e.g., `huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL`)

3. Open Infinite Craft in a browser window

4. Run the Agent:
```bash
python goal_daedalus.py
```
You will be prompted in the terminal to provide a target element (e.g., Life, Human, Dragon).

5. Switch to browser window

## ⚠️ Troubleshooting
- Agent cannot find items / Mouse clicks the wrong spot:
  - Ensure the browser is at 100% zoom and fully visible.
  - Check your OS display scaling (Retina displays may require tweaking the scale_factor in vision_system.py).

- Invalid/Missing JSON Errors in Terminal:
  - Check your local inference endpoint health.
  - Enable DEBUG logging in goal_daedalus.py to inspect the raw model output for formatting errors.

- Mouse behavior seems erratic:
  - Trigger the PyAutoGUI failsafe (move mouse to the corner) and restart the run.