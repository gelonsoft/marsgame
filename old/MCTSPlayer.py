from typing import List, Dict, Optional, Any
import time
from core import AbstractForwardModel
from core import AbstractGameState
from core import AbstractPlayer
from core.actions import AbstractAction
from core.interfaces import IStateHeuristic, IGameListener, IMASTUser, ITreeProcessor
from evaluation.metrics import Event
from players import IAnyTimePlayer
from players.mcts import MCTSParams, MCTSEnums, SingleTreeNode, MultiTreeNode, OMATreeNode, MCGSNode
from utilities import Pair, Utils

class MCTSPlayer(AbstractPlayer, IAnyTimePlayer):
    def __init__(self, params: Optional[MCTSParams] = None, name: str = "MCTSPlayer"):
        params = params if params is not None else MCTSParams()
        super().__init__(params, name)
        self.debug = False
        self.root: Optional[SingleTreeNode] = None
        self.last_action: Optional[Pair[int, AbstractAction]] = None
        self.MAST_stats: Optional[List[Dict[object, Pair[int, float]]]] = None
        self.old_graph_keys: Dict[object, int] = {}
        self.recently_removed_keys: List[object] = []
        
    def get_parameters(self) -> MCTSParams:
        return self.parameters

    def initialize_player(self, state: AbstractGameState) -> None:
        params = self.get_parameters()
        if params.reset_seed_each_game:
            self.rnd = random.Random(params.get_random_seed())
            params.rollout_policy = None
            params.get_rollout_strategy()
            params.opponent_model = None
            params.get_opponent_model()
        
        if isinstance(params.action_heuristic, AbstractPlayer):
            params.action_heuristic.initialize_player(state)
        
        self.MAST_stats = None
        self.root = None
        self.old_graph_keys = {}
        params.get_rollout_strategy().initialize_player(state)
        params.get_opponent_model().initialize_player(state)

    def get_factory(self):
        params = self.get_parameters()
        if params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.OMA or \
           params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.OMA_All:
            return lambda: OMATreeNode()
        elif params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MCGS or \
             params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MCGSSelfOnly:
            return lambda: MCGSNode()
        else:
            return lambda: SingleTreeNode()

    def register_updated_observation(self, game_state: AbstractGameState) -> None:
        super().register_updated_observation(game_state)
        if not self.get_parameters().reuse_tree:
            self.root = None

    def new_multi_tree_root_node(self, state: AbstractGameState) -> MultiTreeNode:
        mt_root = self.root
        new_roots = [None] * state.get_n_players()
        
        for p in range(state.get_n_players()):
            old_root = mt_root.roots[p]
            if old_root is None:
                continue
            if self.debug:
                print(f"\tBacktracking for player {mt_root.roots[p].decision_player}")
            new_roots[p] = self.backtrack(mt_root.roots[p], state)
            if new_roots[p] is not None:
                new_roots[p].rootify(old_root, None)
                new_roots[p].reset_depth(new_roots[p])
                new_roots[p].state = state.copy()
        
        mt_root.roots = new_roots
        mt_root.state = state.copy()
        return mt_root

    def new_root_node(self, game_state: AbstractGameState) -> Optional[SingleTreeNode]:
        params = self.get_parameters()
        self.recently_removed_keys.clear()
        
        if params.reuse_tree and (params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MCGS or 
                                 params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MCGSSelfOnly):
            mcgs_root = self.root
            for key in self.old_graph_keys.keys():
                old_visits = self.old_graph_keys[key]
                node = mcgs_root.get_transposition_map().get(key)
                if node is not None:
                    new_visits = node.n_visits
                    if new_visits == old_visits:
                        mcgs_root.get_transposition_map().pop(key)
                        self.recently_removed_keys.append(key)
                    elif new_visits < old_visits:
                        raise AssertionError("Unexpectedly fewer visits to a state than before")
            
            if mcgs_root is None:
                self.old_graph_keys = {}
                return None
            
            self.old_graph_keys = {k: v.n_visits for k, v in mcgs_root.get_transposition_map().items()}
            ret_value = mcgs_root.get_transposition_map().get(params.MCGS_state_key.get_key(game_state))
            if ret_value is None:
                self.old_graph_keys = {}
                return None
            
            ret_value.set_transposition_map(mcgs_root.get_transposition_map())
            ret_value.rootify(self.root, game_state)
            return ret_value
        
        new_root = None
        if params.reuse_tree and self.root is not None:
            if params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MultiTree:
                return self.new_multi_tree_root_node(game_state)
            
            new_root = self.backtrack(self.root, game_state)
            
            if self.root == new_root:
                raise AssertionError("Root node should not be the same as the new root node")
            if self.debug and new_root is None:
                print("No matching node found")
        
        if new_root is not None:
            if self.debug:
                print("Matching node found")
            if new_root.turn_owner != game_state.get_current_player() or \
               new_root.decision_player != game_state.get_current_player():
                raise AssertionError("Current player does not match decision player in tree")
            new_root.rootify(self.root, game_state)
        
        return new_root

    def backtrack(self, starting_root: SingleTreeNode, game_state: AbstractGameState) -> Optional[SingleTreeNode]:
        history = game_state.get_history()
        last_expected = self.last_action
        params = self.get_parameters()
        self_only = params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.SelfOnly or \
                    params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MultiTree
        new_root = starting_root
        root_player = starting_root.decision_player
        found_point_in_history = False
        
        for backward_loop in range(len(history) - 1, -1, -1):
            if history[backward_loop] == last_expected:
                found_point_in_history = True
                if self.debug:
                    print(f"Matching action found at {backward_loop} of {len(history)} - tracking forward")
                
                for forward_loop in range(backward_loop, len(history)):
                    if self_only and history[forward_loop].a != root_player:
                        continue
                    
                    action = history[forward_loop].b
                    next_action_player = history[forward_loop + 1].a if forward_loop < len(history) - 1 else game_state.get_current_player()
                    next_action_player = root_player if self_only else next_action_player
                    
                    if self.debug:
                        print(f"\tAction: {action}\t Next Player: {next_action_player}")
                    
                    if new_root.children is not None and action in new_root.children:
                        new_root = new_root.children[action][next_action_player]
                    else:
                        new_root = None
                    
                    if new_root is None:
                        break
                break
        
        if not found_point_in_history:
            raise AssertionError(f"Unable to find matching action in history: {last_expected}")
        
        if self.debug:
            print("\tBacktracking complete: " + ("no matching node found" if new_root is None else "node found"))
        return new_root

    def create_root_node(self, game_state: AbstractGameState) -> None:
        new_root = self.new_root_node(game_state)
        params = self.get_parameters()
        
        if new_root is None:
            if params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MultiTree:
                self.root = MultiTreeNode(self, game_state, self.rnd)
            else:
                self.root = SingleTreeNode.create_root_node(self, game_state, self.rnd, self.get_factory())
        else:
            self.root = new_root
        
        if self.MAST_stats is not None and params.MAST_gamma > 0.0:
            self.root.MAST_statistics = [Utils.decay(m, params.MAST_gamma) for m in self.MAST_stats]
        
        rollout_strategy = params.get_rollout_strategy()
        if isinstance(rollout_strategy, IMASTUser):
            rollout_strategy.set_stats(self.root.MAST_statistics)
        
        opponent_model = params.get_opponent_model()
        if isinstance(opponent_model, IMASTUser):
            opponent_model.set_stats(self.root.MAST_statistics)

    def _get_action(self, game_state: AbstractGameState, actions: List[AbstractAction]) -> AbstractAction:
        start_time = time.time_ns()
        self.create_root_node(game_state)
        time_taken = (time.time_ns() - start_time) // 1_000_000
        
        self.root.mcts_search(time_taken)
        
        params = self.get_parameters()
        for processor in [
            params.action_heuristic,
            params.get_rollout_strategy(),
            params.heuristic,
            params.get_opponent_model()
        ]:
            if isinstance(processor, ITreeProcessor):
                processor.process(self.root)
        
        if self.debug:
            if params.opponent_tree_policy == MCTSEnums.OpponentTreePolicy.MultiTree:
                print(self.root.get_root(game_state.get_current_player()))
            else:
                print(self.root)
        
        self.MAST_stats = self.root.MAST_statistics
        
        if len(self.root.children) > 3 * len(actions) and \
           not isinstance(self.root, MCGSNode) and \
           not params.reuse_tree and \
           params.action_space != game_state.get_core_game_parameters().action_space:
            raise AssertionError(
                f"Unexpectedly large number of children: {len(self.root.children)} with action size of {len(actions)}"
            )
        
        self.last_action = Pair(game_state.get_current_player(), self.root.best_action())
        return self.last_action.b.copy()

    def finalize_player(self, state: AbstractGameState) -> None:
        params = self.get_parameters()
        game_over = Event.create_event(Event.GameEvent.GAME_OVER, state)
        
        params.get_rollout_strategy().on_event(game_over)
        params.get_opponent_model().on_event(game_over)
        
        for listener in [params.heuristic, params.action_heuristic]:
            if isinstance(listener, IGameListener):
                listener.on_event(game_over)

    def copy(self) -> 'MCTSPlayer':
        ret_value = MCTSPlayer(self.get_parameters().copy(), str(self))
        if self.forward_model is not None:
            ret_value.set_forward_model(self.forward_model)
        return ret_value

    def set_forward_model(self, model: AbstractForwardModel) -> None:
        super().set_forward_model(model)
        params = self.get_parameters()
        if params.get_rollout_strategy() is not None:
            params.get_rollout_strategy().set_forward_model(model)
        if params.get_opponent_model() is not None:
            params.get_opponent_model().set_forward_model(model)

    def get_decision_stats(self) -> Dict[AbstractAction, Dict[str, Any]]:
        ret_value = {}
        
        if self.root is not None and self.root.get_visits() > 1:
            for action, stats in self.root.action_values.items():
                visits = stats.n_visits if stats is not None else 0
                visit_proportion = visits / self.root.get_visits()
                mean_value = stats.tot_value[self.root.decision_player] / visits if stats is not None and visits > 0 else 0.0
                heuristic_value = self.get_parameters().heuristic.evaluate_state(self.root.state, self.root.decision_player)
                action_value = self.get_parameters().action_heuristic.evaluate_action(
                    action, self.root.state, self.root.actions_from_open_loop_state
                )
                
                action_values = {
                    "visits": visits,
                    "visit_proportion": visit_proportion,
                    "mean_value": mean_value,
                    "heuristic": heuristic_value,
                    "action_value": action_value
                }
                ret_value[action] = action_values
        
        return ret_value

    def set_budget(self, budget: int) -> None:
        self.parameters.budget = budget
        self.parameters.set_parameter_value("budget", budget)

    def get_budget(self) -> int:
        return self.parameters.budget

    def set_state_heuristic(self, heuristic: IStateHeuristic) -> None:
        self.get_parameters().set_parameter_value("heuristic", heuristic)

    def get_state_heuristic(self) -> IStateHeuristic:
        return self.get_parameters().heuristic