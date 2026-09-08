from transformers import LlavaConfig
from typing import Dict, List, Optional

class Bitvla_Config(LlavaConfig):
    model_type: str = "bitvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins
        super().__init__(**kwargs)
