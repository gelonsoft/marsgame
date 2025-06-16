from typing import Optional
from .ElapsedCpuTimer import ElapsedCpuTimer

class ElapsedCpuChessTimer(ElapsedCpuTimer):
    def __init__(self, max_time_minutes: float, increment_action: float, 
                 increment_turn: float, increment_round: float, 
                 increment_milestone: float):
        super().__init__()
        self.set_max_time_millis(max_time_minutes * 60000)
        self.increment_action = increment_action * 1_000_000_000
        self.increment_turn = increment_turn * 1_000_000_000
        self.increment_round = increment_round * 1_000_000_000
        self.increment_milestone = increment_milestone * 1_000_000_000
        self.time_remaining: Optional[int] = None
        self.reset()

    def reset(self):
        super().reset()
        self.time_remaining = self.max_time

    def pause(self):
        self.time_remaining -= self.elapsed()

    def increment_action(self):
        self.time_remaining += self.increment_action

    def increment_turn(self):
        self.time_remaining += self.increment_turn

    def increment_round(self):
        self.time_remaining += self.increment_round

    def increment_milestone(self):
        self.time_remaining += self.increment_milestone

    def resume(self):
        self.old_time = self.get_time()

    def remaining_time(self) -> int:
        return self.time_remaining

    def remaining_time_millis(self) -> int:
        return int(self.time_remaining / 1_000_000.0)

    def exceeded_max_time(self) -> bool:
        return (self.time_remaining - self.increment_action) <= 0

    def copy(self) -> 'ElapsedCpuChessTimer':
        new_timer = ElapsedCpuChessTimer(
            self.max_time, 
            self.increment_action / 1_000_000_000,
            self.increment_turn / 1_000_000_000,
            self.increment_round / 1_000_000_000,
            self.increment_milestone / 1_000_000_000
        )
        new_timer.old_time = self.old_time
        new_timer.bean = self.bean
        new_timer.n_iters = self.n_iters
        new_timer.time_remaining = self.time_remaining
        return new_timer

    def __str__(self) -> str:
        return f"{self.remaining_time_millis()} ms remaining ({self.increment_action/1_000_000.0} ms) increment act"