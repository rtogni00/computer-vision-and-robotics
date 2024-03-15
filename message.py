from dataclasses import dataclass, field
from typing import List

@dataclass
class Message:
    id: int = 0
    timestamp: float = 0
    start: bool = False
    landmark_ids: List = field(default_factory=lambda: [])
    landmark_rs: List = field(default_factory=lambda: [])
    landmark_alphas: List = field(default_factory=lambda: [])
    landmark_xs: List = field(default_factory=lambda: [])
    landmark_ys: List = field(default_factory=lambda: [])
    landmark_stdevs: List = field(default_factory=lambda: [])

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    stdev: List = field(default_factory=lambda: [])
