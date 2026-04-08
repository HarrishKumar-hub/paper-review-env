from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
import uvicorn

from env.core import PaperReviewEnv
from env.models import Action, Observation, Reward

app = FastAPI(
    title="PaperReviewEnv API",
    description="OpenEnv-compliant Research Paper Review RL Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: Dict[str, PaperReviewEnv] = {}


class ResetRequest(BaseModel):
    session_id: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"


class StepRequest(BaseModel):
    session_id: str
    decision: Literal["accept", "reject", "revise"]
    identified_flaws: list[str] = []
    justification: str
    confidence: float = 0.5
    requested_changes: Optional[list[str]] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


@app.get("/")
def root():
    return {
        "name": "PaperReviewEnv",
        "version": "1.0.0",
        "description": "RL environment for training agents to peer-review research papers",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/ground_truth", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    env = PaperReviewEnv(difficulty=request.difficulty)
    obs = env.reset()
    sessions[request.session_id] = env
    return ResetResponse(
        session_id=request.session_id,
        observation=obs.model_dump(),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    env = sessions.get(request.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    action = Action(
        decision=request.decision,
        identified_flaws=request.identified_flaws,
        justification=request.justification,
        confidence=request.confidence,
        requested_changes=request.requested_changes,
    )

    obs, reward, done, info = env.step(action)

    if done:
        sessions.pop(request.session_id, None)

    return StepResponse(
        session_id=request.session_id,
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/ground_truth/{session_id}")
def ground_truth(session_id: str):
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.get_ground_truth()


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
