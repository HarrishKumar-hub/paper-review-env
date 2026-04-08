import random
from typing import Tuple, Dict, Any
from env.tasks._base import BaseTask
from env.models import Observation, Action, Reward
from env.graders.grader_easy import EasyGrader

PAPERS = [
    {
        "paper_id": "easy_001",
        "title": "Transformer-based Sentiment Analysis Achieves 99.8% Accuracy on All Datasets",
        "abstract": (
            "We present SentiFormer, a transformer model for sentiment analysis. "
            "Our model achieves 99.8% accuracy across all benchmark datasets, "
            "surpassing all prior work by a significant margin with no exceptions."
        ),
        "methodology": (
            "We fine-tuned BERT-base on the SST-2 training set for 3 epochs "
            "with default hyperparameters. No ablation studies were conducted. "
            "We evaluated only on SST-2 test set but claim results generalise universally."
        ),
        "results": (
            "SST-2 accuracy: 99.8%. We report no confidence intervals, no standard deviations, "
            "and no comparison baselines. The model was tested on a single random seed."
        ),
        "claims": [
            "Achieves 99.8% accuracy on ALL datasets",
            "Surpasses all prior work universally",
            "Results generalise across all domains without further training",
        ],
        "ground_truth_decision": "reject",
        "planted_flaws": [
            "no_baseline_comparison",
            "single_dataset_overgeneralisation",
            "no_ablation_study",
            "no_confidence_intervals",
            "single_seed_evaluation",
        ],
    },
    {
        "paper_id": "easy_002",
        "title": "A Novel CNN Outperforms GPT-4 on All NLP Benchmarks",
        "abstract": (
            "We introduce MiniCNN-NLP, a 3-layer CNN trained on Wikipedia text. "
            "It outperforms GPT-4 on GLUE, SuperGLUE, and all language reasoning tasks."
        ),
        "methodology": (
            "MiniCNN-NLP uses 3 convolutional layers with kernel size 3. "
            "Training was done on a 10,000 sentence subset of Wikipedia. "
            "GPT-4 was evaluated using a different prompt format than the one recommended."
        ),
        "results": (
            "MiniCNN-NLP: 91.2 GLUE score. GPT-4 (our eval): 47.3 GLUE score. "
            "No external reproduction attempted. Evaluation code not released."
        ),
        "claims": [
            "3-layer CNN beats GPT-4 on all NLP benchmarks",
            "Evaluation is fair and reproducible",
        ],
        "ground_truth_decision": "reject",
        "planted_flaws": [
            "unfair_baseline_evaluation",
            "no_code_release",
            "cherry_picked_comparison",
            "implausible_claim",
        ],
    },
]


class EasyTask(BaseTask):
    task_id = "easy"
    difficulty = "easy"

    def __init__(self):
        self.grader = EasyGrader()
        self.paper = None
        self.step_count = 0
        self.max_steps = 3

    def reset(self) -> Observation:
        self.paper = random.choice(PAPERS)
        self.step_count = 0
        return Observation(
            task_id=self.task_id,
            paper_id=self.paper["paper_id"],
            title=self.paper["title"],
            abstract=self.paper["abstract"],
            methodology=self.paper["methodology"],
            results=self.paper["results"],
            claims=self.paper["claims"],
            step=self.step_count,
            max_steps=self.max_steps,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = self.grader.grade(action, self.paper)
        done = True  # easy task resolves in one review action
        obs = Observation(
            task_id=self.task_id,
            paper_id=self.paper["paper_id"],
            title=self.paper["title"],
            abstract=self.paper["abstract"],
            methodology=self.paper["methodology"],
            results=self.paper["results"],
            claims=self.paper["claims"],
            step=self.step_count,
            max_steps=self.max_steps,
        )
        info = {"ground_truth": self.get_ground_truth()}
        return obs, reward, done, info

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "decision": self.paper["ground_truth_decision"],
            "flaws": self.paper["planted_flaws"],
        }
