import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://chronicles28-paper-review-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
HF_TOKEN = os.getenv("HF_TOKEN")

def run():
    session_id = "eval-session-1"
    
    # RESET
    reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"session_id": session_id, "difficulty": "easy"})
    obs = reset_resp.json()["observation"]
    print("START", flush=True)
    
    # STEP
    step_resp = requests.post(f"{API_BASE_URL}/step", json={
        "session_id": session_id,
        "decision": "reject",
        "identified_flaws": ["no baseline comparison", "single dataset"],
        "justification": "The paper lacks baseline comparisons and only evaluates on a single dataset which limits generalizability.",
        "confidence": 0.85
    })
    result = step_resp.json()
    print("STEP", result["reward"], flush=True)
    print("END", flush=True)

if __name__ == "__main__":
    run()
