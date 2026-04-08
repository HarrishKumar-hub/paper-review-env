# 📄 PaperReviewEnv

**An OpenEnv-compliant Reinforcement Learning environment for training agents to peer-review AI research papers.**

> *"I published a research paper. 45 people read it. Zero reviewed it. So I built an RL environment to fix that."*

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**.

---

## 🧠 What is this?

PaperReviewEnv is an RL environment where an agent acts as a peer reviewer. The agent reads a research paper's abstract, methodology, results, and claims — then makes structured decisions: **accept**, **reject**, or **request revisions**.

Papers contain **planted flaws** of varying subtlety and difficulty. The agent is rewarded for:
- Making the correct editorial decision
- Identifying the actual planted flaws (not red herrings)
- Providing a well-reasoned justification

**All rewards are fully deterministic. No LLM in graders.**

---

## 🎯 Why this problem?

The peer review system is under severe strain:
- 5 million papers published per year
- Reviewer shortage and burnout
- 6–12 month average review wait time
- AI-generated papers flooding journals

An RL agent that learns to do expert first-pass triage — flagging fatal flaws, identifying good work — directly addresses one of the biggest bottlenecks in modern science.

---

## 🗂️ Project Structure

```
paper-review-env/
├── env/
│   ├── core.py              # Main PaperReviewEnv class
│   ├── models.py            # Observation, Action, Reward (Pydantic)
│   ├── tasks/
│   │   ├── task_easy.py     # Obvious fatal flaws
│   │   ├── task_medium.py   # Borderline papers, nuance required
│   │   └── task_hard.py     # Adversarial papers, red herrings, multi-turn
│   └── graders/
│       ├── grader_easy.py   # Deterministic grader (easy)
│       ├── grader_medium.py # Deterministic grader (medium, rewards nuance)
│       └── grader_hard.py   # Deterministic grader (hard, penalises red herrings)
├── server/
│   └── app.py               # FastAPI server (OpenEnv-compliant)
├── inference.py             # Baseline keyword-matching agent demo
├── openenv.yaml             # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline agent (no server needed)
python inference.py --difficulty easy
python inference.py --difficulty medium
python inference.py --difficulty hard
```

### Start the API server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

API docs available at: `http://localhost:7860/docs`

### Docker

```bash
docker build -t paper-review-env .
docker run -p 7860:7860 paper-review-env
```

---

## 📡 API Reference

### `POST /reset`

Start a new episode.

```json
{
  "session_id": "agent-001",
  "difficulty": "easy"
}
```

**Response:** Initial observation (paper title, abstract, methodology, results, claims).

---

### `POST /step`

Submit a review action.

```json
{
  "session_id": "agent-001",
  "decision": "reject",
  "identified_flaws": [
    "no baseline comparison",
    "results on single dataset only"
  ],
  "justification": "The paper claims universal generalisation but evaluates on a single dataset with no baseline comparison and no confidence intervals. The results are not reproducible.",
  "confidence": 0.9,
  "requested_changes": null
}
```

**Response:** Next observation, reward breakdown, done flag, info.

---

### `GET /ground_truth/{session_id}`

Retrieve planted flaws and correct decision (evaluation only).

---

## 🏆 Tasks

| Task | Description | Max Steps |
|------|-------------|-----------|
| **Easy** | Paper with obvious fatal flaws (no baselines, single seed, impossible claims) | 3 |
| **Medium** | Borderline paper — real strengths AND hidden weaknesses. Agent must not over-reject good work. | 5 |
| **Hard** | Adversarially crafted paper. Impressive-sounding methodology hides subtle fatal flaws. Red herrings present. New information revealed over multiple turns. | 5 |

---

## 📊 Reward Structure

All graders are **fully deterministic** — no LLM involved.

| Component | Weight | Description |
|-----------|--------|-------------|
| `decision_score` | 40% | Correct accept/reject/revise decision |
| `flaw_detection_score` | 35% | Fraction of planted flaws identified |
| `justification_score` | 15% | Length and specificity of justification |
| `efficiency_bonus` | 10% | High-confidence correct decisions |
| `red_herring_penalty` | −up to 30% | [Hard only] Penalises citing red herrings as decisive flaws |

---

## 🔬 Example Episode (Easy)

```
Paper: "Transformer-based Sentiment Analysis Achieves 99.8% Accuracy on All Datasets"

Agent identifies:
  - "no baseline comparison"
  - "single dataset overgeneralisation"
  - "no ablation study"
  - "single seed evaluation"

Decision: reject | Confidence: 0.9

Reward:
  decision_score: 1.0
  flaw_detection_score: 0.8
  justification_score: 0.74
  efficiency_bonus: 0.1
  TOTAL: 0.94
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** — REST API server
- **Pydantic v2** — typed observation/action/reward models
- **OpenEnv** — environment interface compliance
- **PyTorch-ready** — observation tensors can be derived from text embeddings

---

## 📬 Contact

Built with ❤️ for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology.

For questions: [your email here]
