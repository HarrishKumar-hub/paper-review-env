import random
from typing import Tuple, Dict, Any, List
from env.tasks._base import BaseTask
from env.models import Observation, Action, Reward
from env.graders.grader_hard import HardGrader

PAPERS = [
    {
        "paper_id": "hard_001",
        "title": "UniRL: A Universal Reinforcement Learning Algorithm for All Continuous Control Tasks",
        "abstract": (
            "We propose UniRL, a model-free RL algorithm that achieves state-of-the-art "
            "performance on MuJoCo, Atari, and continuous control benchmarks simultaneously. "
            "UniRL generalises across reward structures and demonstrates robust transfer to "
            "unseen environments without additional fine-tuning."
        ),
        "methodology": (
            "UniRL combines PPO with a learned world model and adaptive entropy regularisation. "
            "We evaluate on 12 MuJoCo tasks (seed=42), 6 Atari games (10 evaluation episodes each), "
            "and 3 transfer tasks. Compute: 4x TPU v3; total FLOPs methodology stated in appendix. "
            "Code will be released on acceptance. Ablation study conducted on 4 components. "
            "Multiple statistical tests applied (Wilcoxon signed-rank)."
        ),
        "results": (
            "MuJoCo: +12% over SAC (seed=42 only). Atari: +8% mean score over DQN (10 episodes). "
            "Transfer: 40% performance drop on held-out tasks (reported in appendix C). "
            "GLUE language proxy task: 0.3 point improvement over baseline (p=0.07). "
            "Formal theorem proving generalisation guarantee holds under i.i.d. assumption "
            "(not validated on target domains)."
        ),
        "claims": [
            "Universal across all continuous control tasks",
            "Robust zero-shot transfer without fine-tuning",
            "State-of-the-art on MuJoCo, Atari, and GLUE simultaneously",
            "Theoretical generalisation guarantee",
        ],
        "ground_truth_decision": "reject",
        "planted_flaws": [
            "mujoco_single_seed",
            "atari_insufficient_episodes",
            "poor_transfer_hidden_in_appendix",
            "generalisation_claim_contradicted_by_results",
            "glue_difference_not_statistically_significant",
            "theoretical_assumption_not_validated_on_target_domains",
        ],
        "red_herrings": [
            "detailed_compute_disclosure",
            "code_release_promised",
            "ablation_study_present",
            "multiple_statistical_tests",
            "formal_theorem_with_proof",
        ],
        # Multi-turn: reveal appendix content only after step 2
        "turn_reveals": {
            2: "Appendix C (now visible): Transfer task results show 40% performance drop. "
               "The formal theorem assumes i.i.d. data distribution which does not hold for target domains.",
        },
    },
    {
        "paper_id": "hard_002",
        "title": "FlashSSM: Linear-Time Sequence Models Match Transformers on All Long-Range Tasks",
        "abstract": (
            "We present FlashSSM, a structured state space model with hardware-aware kernels "
            "achieving O(N) complexity. FlashSSM matches transformer performance on Long Range Arena "
            "and shows significant wall-clock speedups across all sequence lengths."
        ),
        "methodology": (
            "FlashSSM implements S4 with fused CUDA kernels. Evaluated on Long Range Arena (6 tasks), "
            "language modelling (WikiText-103), and image classification (CIFAR-10). "
            "Multiple strong baselines included. FLOPs counted for all experiments. "
            "TPU compute budget disclosed. Code released on GitHub."
        ),
        "results": (
            "LRA average: 85.2 vs Transformer 85.0 (p=0.07, not significant). "
            "WikiText-103 perplexity: 19.8 vs 18.9 for Transformer. "
            "Speed: 3.1x faster than Transformer at N=4096. At N=512, no speedup observed. "
            "CIFAR-10: 95.1% vs 95.3% for ViT (within noise). "
            "All speedup claims hold only for sequence length N > 2048."
        ),
        "claims": [
            "Matches transformers on all long-range tasks",
            "Significant wall-clock speedups across all sequence lengths",
            "State-of-the-art on language modelling",
        ],
        "ground_truth_decision": "revise",
        "planted_flaws": [
            "glue_difference_not_statistically_significant",
            "speed_benefit_only_at_long_sequences_not_highlighted",
            "overgeneralised_applicability_claim",
        ],
        "red_herrings": [
            "multiple_baselines",
            "tpu_compute_disclosed",
            "code_release_promised",
            "flops_methodology_stated",
        ],
        "turn_reveals": {
            2: "Supplementary Table S3 (now visible): Speed benchmarks broken down by sequence length. "
               "At N=512 (most real-world NLP tasks), FlashSSM shows 0% speedup over standard Transformer. "
               "The 3.1x speedup is only achieved at N >= 2048.",
        },
    },
]


class HardTask(BaseTask):
    task_id = "hard"
    difficulty = "hard"

    def __init__(self):
        self.grader = HardGrader()
        self.paper = None
        self.step_count = 0
        self.max_steps = 5
        self._extra_context: List[str] = []

    def reset(self) -> Observation:
        self.paper = random.choice(PAPERS)
        self.step_count = 0
        self._extra_context = []
        return self._make_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1

        # Reveal hidden info at specific turns
        reveal = self.paper.get("turn_reveals", {}).get(self.step_count)
        if reveal:
            self._extra_context.append(reveal)

        reward = self.grader.grade(action, self.paper, turn=self.step_count)
        done = self.step_count >= self.max_steps or (
            reward.total >= 0.75 and action.decision == self.paper["ground_truth_decision"]
        )

        obs = self._make_obs()
        info = {"ground_truth": self.get_ground_truth(), "revealed_this_turn": reveal}
        return obs, reward, done, info

    def _make_obs(self) -> Observation:
        extra = " ".join(self._extra_context)
        methodology = self.paper["methodology"]
        if extra:
            methodology = methodology + "\n\n[NEWLY REVEALED]: " + extra

        return Observation(
            task_id=self.task_id,
            paper_id=self.paper["paper_id"],
            title=self.paper["title"],
            abstract=self.paper["abstract"],
            methodology=methodology,
            results=self.paper["results"],
            claims=self.paper["claims"],
            step=self.step_count,
            max_steps=self.max_steps,
            metadata={
                "red_herrings_present": True,
                "multi_turn_reveals": True,
                "note": "Some information is revealed progressively. Re-read methodology each step.",
            },
        )

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "decision": self.paper["ground_truth_decision"],
            "flaws": self.paper["planted_flaws"],
            "red_herrings": self.paper["red_herrings"],
        }
