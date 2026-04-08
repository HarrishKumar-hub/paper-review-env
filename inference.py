"""
inference.py — Baseline agent that interacts with PaperReviewEnv directly (no server needed).

Demonstrates a simple keyword-matching baseline agent.
Replace with your RL agent for actual training.

Usage:
    python inference.py --difficulty easy
    python inference.py --difficulty medium
    python inference.py --difficulty hard
"""

import argparse
from env.core import PaperReviewEnv
from env.models import Action


FLAW_SIGNALS = [
    "single seed", "no ablation", "no baseline", "no confidence interval",
    "overgeneralised", "single dataset", "not reproducible", "cherry-picked",
    "p=0.07", "not significant", "appendix", "insufficient episodes",
    "single site", "one hospital", "assumption not validated",
]


def baseline_agent(obs) -> Action:
    """
    Simple keyword-matching baseline agent.
    Reads the paper text and flags any known flaw signals.
    """
    full_text = f"{obs.abstract} {obs.methodology} {obs.results} {' '.join(obs.claims)}".lower()

    found_flaws = [signal for signal in FLAW_SIGNALS if signal in full_text]

    # Decision logic: reject if many flaws, revise if some, accept if clean
    if len(found_flaws) >= 3:
        decision = "reject"
        confidence = 0.85
    elif len(found_flaws) >= 1:
        decision = "revise"
        confidence = 0.65
    else:
        decision = "accept"
        confidence = 0.6

    justification = (
        f"Baseline agent identified {len(found_flaws)} potential issue(s): {found_flaws}. "
        f"Decision: {decision} based on flaw count heuristic."
        if found_flaws
        else "No obvious flaws detected by baseline keyword scan. Recommending accept."
    )

    return Action(
        decision=decision,
        identified_flaws=found_flaws,
        justification=justification,
        confidence=confidence,
        requested_changes=[f"Please address: {f}" for f in found_flaws] if decision == "revise" else None,
    )


def run_episode(difficulty: str):
    print(f"\n{'='*60}")
    print(f"Running baseline agent | Difficulty: {difficulty.upper()}")
    print(f"{'='*60}")

    env = PaperReviewEnv(difficulty=difficulty)
    obs = env.reset()

    print(f"\nPaper: {obs.title}")
    print(f"Task: {obs.task_id} | Step: {obs.step}/{obs.max_steps}")
    print(f"\nAbstract: {obs.abstract[:200]}...")

    total_reward = 0.0
    step = 0

    while True:
        action = baseline_agent(obs)
        print(f"\n[Step {step+1}] Action: decision={action.decision}, confidence={action.confidence}")
        print(f"  Flaws identified: {action.identified_flaws}")
        print(f"  Justification: {action.justification[:150]}...")

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        step += 1

        print(f"\n[Reward] total={reward.total:.4f} | decision={reward.decision_score} | flaws={reward.flaw_detection_score:.4f}")
        print(f"  Breakdown: {reward.breakdown}")

        if done:
            print(f"\n{'='*60}")
            print(f"Episode complete | Total reward: {total_reward:.4f}")
            print(f"Ground truth: {info['ground_truth']}")
            print(f"{'='*60}")
            break

    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agent on PaperReviewEnv")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    args = parser.parse_args()
    run_episode(args.difficulty)
