from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from env.models import Observation, Action, Reward


class BaseTask(ABC):
    task_id: str
    difficulty: str

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Take action, return (next_obs, reward, done, info)."""
        pass

    @abstractmethod
    def get_ground_truth(self) -> Dict[str, Any]:
        """Return ground truth for grading."""
        pass
