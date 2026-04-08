from typing import Tuple, Dict, Any, Literal
from env.models import Observation, Action, Reward
from env.tasks.task_easy import EasyTask
from env.tasks.task_medium import MediumTask
from env.tasks.task_hard import HardTask


class PaperReviewEnv:
    """
    Research Paper Review Environment for OpenEnv.

    An RL environment where an agent acts as a peer reviewer:
    - Reads research paper sections (abstract, methodology, results, claims)
    - Identifies planted flaws of varying subtlety
    - Makes accept/reject/revise decisions
    - Receives deterministic rewards based on decision correctness and flaw detection

    Tasks:
        easy   - Single paper, one or more obvious fatal flaws
        medium - Borderline paper, hidden flaws + genuine strengths (nuance required)
        hard   - Adversarially written paper, red herrings, multi-turn information reveal
    """

    TASK_MAP = {
        "easy": EasyTask,
        "medium": MediumTask,
        "hard": HardTask,
    }

    def __init__(self, difficulty: Literal["easy", "medium", "hard"] = "easy"):
        if difficulty not in self.TASK_MAP:
            raise ValueError(f"difficulty must be one of {list(self.TASK_MAP.keys())}")
        self.difficulty = difficulty
        self.task = self.TASK_MAP[difficulty]()
        self._current_obs: Observation = None

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._current_obs = self.task.reset()
        return self._current_obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Submit a review action.

        Args:
            action: An Action object containing decision, identified_flaws, justification, confidence.

        Returns:
            (observation, reward, done, info)
        """
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")
        obs, reward, done, info = self.task.step(action)
        self._current_obs = obs
        return obs, reward, done, info

    def get_ground_truth(self) -> Dict[str, Any]:
        """Return ground truth (for evaluation only, not available to agent during episode)."""
        return self.task.get_ground_truth()

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "task_id": "str",
            "paper_id": "str",
            "title": "str",
            "abstract": "str",
            "methodology": "str",
            "results": "str",
            "claims": "List[str]",
            "step": "int",
            "max_steps": "int",
            "metadata": "dict",
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "decision": "Literal['accept', 'reject', 'revise']",
            "identified_flaws": "List[str]",
            "justification": "str",
            "confidence": "float [0.0, 1.0]",
            "requested_changes": "Optional[List[str]]",
        }
