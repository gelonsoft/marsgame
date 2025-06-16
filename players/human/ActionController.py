from queue import Queue
from typing import Optional, Deque
from core.actions import AbstractAction

class ActionController:
    def __init__(self):
        self.debug = False
        self.actions_queue = Queue(maxsize=1)
        self.last_action_played: Optional[AbstractAction] = None

    def add_action(self, candidate: AbstractAction) -> None:
        if candidate is not None and self.actions_queue.qsize() < self.actions_queue.maxsize:
            self.actions_queue.put(candidate)
            if self.debug:
                print(f"Action {candidate} added to ActionController")

    def get_action(self) -> AbstractAction:
        self.last_action_played = self.actions_queue.get()
        if self.debug:
            print(f"Action {self.last_action_played} taken via get_action()")
        return self.last_action_played

    def copy(self) -> 'ActionController':
        new_controller = ActionController()
        if not self.actions_queue.empty():
            # Peek at the first item without removing it
            item = self.actions_queue.queue[0]
            new_controller.actions_queue.put(item)
        return new_controller

    def reset(self) -> None:
        if self.debug and not self.actions_queue.empty():
            print(f"Action Queue being cleared with {self.actions_queue.qsize()} actions")
        # Clear the queue by getting all items
        while not self.actions_queue.empty():
            self.actions_queue.get()

    def get_last_action_played(self) -> Optional[AbstractAction]:
        return self.last_action_played

    def get_actions_queue(self) -> Queue:
        return self.actions_queue

    def set_last_action_played(self, a: AbstractAction) -> None:
        self.last_action_played = a

    def has_action(self) -> bool:
        return not self.actions_queue.empty()