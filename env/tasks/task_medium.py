import random
from typing import Tuple, Dict, Any
from env.tasks._base import BaseTask
from env.models import Observation, Action, Reward
from env.graders.grader_medium import MediumGrader

PAPERS = [
    {
        "paper_id": "medium_001",
        "title": "LoRA-Lite: Parameter-Efficient Fine-Tuning with 0.01% Trainable Parameters",
        "abstract": (
            "We propose LoRA-Lite, an extension of LoRA that reduces trainable parameters "
            "to 0.01% of total model size while maintaining competitive downstream performance. "
            "Experiments on GLUE show promising results with significant efficiency gains."
        ),
        "methodology": (
            "LoRA-Lite introduces structured sparsity in the low-rank matrices. "
            "We evaluate on GLUE benchmark using RoBERTa-base as backbone. "
            "Hyperparameters were tuned on the validation set of each GLUE task separately. "
            "We compare against standard LoRA and full fine-tuning. "
            "Experiments run on 3 seeds; average reported."
        ),
        "results": (
            "LoRA-Lite achieves 84.1 average GLUE score vs LoRA's 85.3 and full FT's 86.0. "
            "Training time reduced by 40%. Memory reduced by 60%. "
            "Performance drop of 1.2 points is statistically significant (p<0.05). "
            "No evaluation on larger models or out-of-distribution datasets."
        ),
        "claims": [
            "Competitive performance with 0.01% parameters",
            "Significant efficiency gains over LoRA",
            "Broadly applicable to parameter-efficient fine-tuning",
        ],
        "ground_truth_decision": "revise",
        "planted_flaws": [
            "no_ood_evaluation",
            "hyperparams_tuned_on_val_set",
            "overgeneralised_applicability_claim",
        ],
        "genuine_strengths": [
            "multi_seed_evaluation",
            "clear_baselines",
            "statistically_significant_results",
        ],
    },
    {
        "paper_id": "medium_002",
        "title": "Contrastive Pretraining Improves Medical Image Segmentation",
        "abstract": (
            "We apply contrastive self-supervised pretraining to chest X-ray segmentation. "
            "Our method improves Dice score by 4.2% over supervised baselines "
            "in low-data regimes (100 labelled samples)."
        ),
        "methodology": (
            "SimCLR pretraining on 50,000 unlabelled chest X-rays. "
            "Fine-tuned U-Net on 100/500/1000 labelled samples. "
            "Compared against supervised U-Net from scratch and ImageNet pretrained. "
            "Single hospital dataset used for all experiments. "
            "IRB approval obtained. Data preprocessing described in detail."
        ),
        "results": (
            "100 labels: +4.2% Dice. 500 labels: +1.8% Dice. 1000 labels: +0.3% Dice (not significant). "
            "No external validation on other hospital datasets. "
            "Results averaged over 5 seeds."
        ),
        "claims": [
            "Contrastive pretraining improves medical segmentation",
            "Effective in low-data regimes",
            "Generalisable across medical imaging tasks",
        ],
        "ground_truth_decision": "revise",
        "planted_flaws": [
            "single_site_dataset",
            "overgeneralised_applicability_claim",
            "benefit_diminishes_with_more_data",
        ],
        "genuine_strengths": [
            "multi_seed_evaluation",
            "multiple_data_regimes_tested",
            "irb_approved",
            "detailed_preprocessing",
        ],
    },
]


class MediumTask(BaseTask):
    task_id = "medium"
    difficulty = "medium"

    def __init__(self):
        self.grader = MediumGrader()
        self.paper = None
        self.step_count = 0
        self.max_steps = 5

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
            metadata={"strengths_present": self.paper["genuine_strengths"]},
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = self.grader.grade(action, self.paper)
        done = self.step_count >= self.max_steps or action.decision != "revise" and reward.total > 0.7
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
            "strengths": self.paper["genuine_strengths"],
        }
