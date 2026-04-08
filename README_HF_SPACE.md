---
title: PaperReviewEnv
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RL environment for training agents to peer-review research papers
---

# PaperReviewEnv 🔬

**OpenEnv-compliant Research Paper Review RL Environment**

An RL environment where an agent acts as a peer reviewer — reading research papers with planted flaws, making accept/reject/revise decisions, and receiving deterministic rewards.

## Live API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit a review action |
| `/ground_truth/{session_id}` | GET | Get ground truth (post-episode) |

## Quick Start

```bash
# 1. Reset (start episode)
curl -X POST https://YOUR-SPACE-URL/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "difficulty": "easy"}'

# 2. Step (submit review)
curl -X POST https://YOUR-SPACE-URL/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo",
    "decision": "reject",
    "identified_flaws": ["no baseline", "single seed evaluation"],
    "justification": "The paper lacks baseline comparisons and uses only a single seed.",
    "confidence": 0.85
  }'
```

## Tasks

| Difficulty | Description | Max Steps |
|-----------|-------------|-----------|
| `easy` | Obvious fatal flaws in a single paper | 3 |
| `medium` | Borderline paper — real strengths + hidden flaws | 3 |
| `hard` | Adversarial paper with red herrings, multi-turn reveals | 5 |

## Reward Signal (fully deterministic — no LLM)

- **Decision score** (0 or 1): Correct accept/reject/revise
- **Flaw detection score** (0–1): Fraction of planted flaws identified
- **Justification score** (0–1): Completeness of written justification
- **Efficiency bonus** (0 or 0.1): High confidence + correct decision

Built for the **Meta PyTorch OpenEnv Hackathon**.
