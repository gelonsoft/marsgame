from typing import Dict, Optional
from core import AbstractForwardModel, AbstractGameState, AbstractGameStateWithTurnOrder
from core.CoreConstants import GameResult
from core.actions import AbstractAction
from core.rules.nodetypes import BranchingRuleNode, ConditionNode, RuleNode
from .Node import Node


class AbstractRuleBasedForwardModel(AbstractForwardModel):
    def __init__(self):
        self.last_rule: Optional[Node] = None
        self.next_rule: Optional[Node] = None
        self.root: Optional[Node] = None

    def __init__(self, root: Node):
        self.root = root
        self.next_rule = root

    def abstract_setup(self, first_state: AbstractGameState) -> None:
        super().abstract_setup(first_state)
        self.next_rule = self.root
        self.last_rule = None

    def _next(self, state: AbstractGameState, action: AbstractAction) -> None:
        if state.get_game_status() != GameResult.GAME_ONGOING:
            return
        if not isinstance(state, AbstractGameStateWithTurnOrder):
            raise AssertionError("Rules Based Forward Model requires AbstractGameStateWithTurnOrder")

        current_state = state
        if self.next_rule is None:
            self.next_rule = self.last_rule.get_next()
            if self.next_rule is None:
                self.next_rule = self.root
            return

        while True:
            if self.next_rule.require_action():
                if action is not None:
                    self.next_rule.set_action(action)
                    action = None
                else:
                    return  # Wait for action

            self.last_rule = self.next_rule
            self.next_rule = self.next_rule.execute(current_state)

            if self.next_rule is None:
                break

        self.next_rule = self.last_rule.get_next()
        if self.next_rule is None:
            self.next_rule = self.root

    def copy_root(self) -> Node:
        return self._copy_node_graph({}, self.root)

    def _copy_node_graph(self, visited_nodes: Dict[int, Node], node: Node) -> Node:
        if node is None:
            raise AssertionError("Can't copy rule graph containing null nodes!")
        if node.get_id() in visited_nodes:
            return visited_nodes[node.get_id()]

        node_copy = node.copy()
        visited_nodes[node.get_id()] = node_copy

        if isinstance(node, RuleNode):
            if isinstance(node, BranchingRuleNode):
                children = node.get_children()
                copies = [self._copy_node_graph(visited_nodes, child.get_next()) 
                         for child in children]
                for copy in copies:
                    if copy.get_parent() is not None:
                        copy.set_parent(node_copy)
                node_copy.set_next(copies)
            else:
                child = self._copy_node_graph(visited_nodes, node.get_next())
                if node.get_next().get_parent() is not None:
                    child.set_parent(node_copy)
                node_copy.set_next(child)

        elif isinstance(node, ConditionNode):
            child_yes = self._copy_node_graph(visited_nodes, node.get_yes_no()[0])
            child_no = self._copy_node_graph(visited_nodes, node.get_yes_no()[1])
            if node.get_yes_no()[0].get_parent() is not None:
                child_yes.set_parent(node_copy)
            if node.get_yes_no()[1].get_parent() is not None:
                child_no.set_parent(node_copy)
            node_copy.set_yes_no(child_yes, child_no)

        return node_copy