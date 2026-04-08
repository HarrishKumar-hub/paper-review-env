"""
tests/test_env.py

Run with: python -m pytest tests/ -v
Or directly: python tests/test_env.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.core import PaperReviewEnv
from env.models import Action, Observation, Reward


def make_action(decision="reject", flaws=None, justification=None, confidence=0.85):
    return Action(
        decision=decision,
        identified_flaws=flaws or ["no baseline", "single seed", "no ablation"],
        justification=justification or (
            "This paper has no baseline comparison, uses a single seed evaluation, "
            "no ablation study, and overgeneralised claims from a single dataset. "
            "The results have no confidence intervals and are not reproducible."
        ),
        confidence=confidence,
        requested_changes=["Add baseline comparisons", "Run multiple seeds"],
    )


def test_easy_task_reset():
    env = PaperReviewEnv(difficulty="easy")
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.task_id == "easy"
    assert obs.step == 0
    assert obs.title != ""
    assert len(obs.claims) > 0
    print(f"  [easy] Paper: {obs.title[:60]}...")


def test_easy_task_step_correct():
    env = PaperReviewEnv(difficulty="easy")
    env.reset()
    action = make_action(decision="reject", confidence=0.9)
    obs, reward, done, info = env.step(action)
    assert isinstance(reward, Reward)
    assert 0.0 <= reward.total <= 1.2
    assert reward.decision_score == 1.0  # reject is always correct for easy
    assert done is True
    print(f"  [easy] Reward: {reward.total:.4f} | decision={reward.decision_score} | flaws={reward.flaw_detection_score:.4f}")


def test_easy_task_wrong_decision():
    env = PaperReviewEnv(difficulty="easy")
    env.reset()
    action = make_action(decision="accept", confidence=0.9)
    obs, reward, done, info = env.step(action)
    assert reward.decision_score == 0.0
    print(f"  [easy wrong] Reward: {reward.total:.4f}")


def test_medium_task_reset():
    env = PaperReviewEnv(difficulty="medium")
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.task_id == "medium"
    assert "strengths_present" in obs.metadata
    print(f"  [medium] Paper: {obs.title[:60]}...")


def test_medium_task_step():
    env = PaperReviewEnv(difficulty="medium")
    env.reset()
    action = Action(
        decision="revise",
        identified_flaws=[
            "no out-of-distribution evaluation",
            "hyperparameters tuned on val set",
            "overgeneralised applicability claim",
        ],
        justification=(
            "The paper makes overgeneralised applicability claims but only evaluates on one domain. "
            "Hyperparameters were tuned on the validation set introducing leakage. "
            "No out-of-distribution evaluation provided. However, the paper has multiple seeds "
            "and clear baselines with statistically significant results, which are genuine strengths."
        ),
        confidence=0.75,
        requested_changes=["Add OOD evaluation", "Report val-set tuning as limitation"],
    )
    obs, reward, done, info = env.step(action)
    assert reward.decision_score == 1.0
    assert reward.flaw_detection_score > 0.0
    print(f"  [medium] Reward: {reward.total:.4f} | flaws={reward.flaw_detection_score:.4f} | breakdown={reward.breakdown}")


def test_hard_task_multi_turn():
    env = PaperReviewEnv(difficulty="hard")
    obs = env.reset()
    assert obs.task_id == "hard"
    assert obs.metadata.get("multi_turn_reveals") is True

    total_reward = 0.0
    for turn in range(1, 4):
        action = Action(
            decision="reject",
            identified_flaws=[
                "single seed evaluation",
                "only 10 episodes insufficient",
                "poor transfer hidden in appendix",
                "results contradict generalisation claim",
                "p=0.07 not statistically significant",
            ],
            justification=(
                "The paper uses a single seed (seed=42) for MuJoCo — insufficient for reliable results. "
                "Atari evaluation used only 10 episodes which is far below the standard 100. "
                "Transfer results show 40% performance drop, buried in appendix C. "
                "The universality claim is directly contradicted by these results. "
                "The GLUE improvement is not statistically significant (p=0.07 > 0.05). "
                "The formal theorem assumes i.i.d. distribution not validated on target domains."
            ),
            confidence=0.8,
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        print(f"  [hard turn {turn}] reward={reward.total:.4f} | done={done} | revealed={info.get('revealed_this_turn', '')[:50]}")
        if done:
            break

    assert total_reward > 0.0
    print(f"  [hard] Total reward across turns: {total_reward:.4f}")


def test_ground_truth():
    env = PaperReviewEnv(difficulty="easy")
    env.reset()
    gt = env.get_ground_truth()
    assert "decision" in gt
    assert "flaws" in gt
    assert gt["decision"] in ["accept", "reject", "revise"]
    print(f"  [ground_truth] decision={gt['decision']} | flaws={gt['flaws']}")


def test_invalid_difficulty():
    try:
        env = PaperReviewEnv(difficulty="impossible")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [invalid] Correctly raised ValueError")


def test_step_before_reset():
    env = PaperReviewEnv(difficulty="easy")
    try:
        env.step(make_action())
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  [step_before_reset] Correctly raised RuntimeError")


if __name__ == "__main__":
    tests = [
        test_easy_task_reset,
        test_easy_task_step_correct,
        test_easy_task_wrong_decision,
        test_medium_task_reset,
        test_medium_task_step,
        test_hard_task_multi_turn,
        test_ground_truth,
        test_invalid_difficulty,
        test_step_before_reset,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n▶ {test.__name__}")
        try:
            test()
            print(f"  ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed ✅")
    sys.exit(0 if failed == 0 else 1)
