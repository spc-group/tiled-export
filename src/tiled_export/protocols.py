from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    notebook: str
