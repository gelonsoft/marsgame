from typing import Optional
import random
from core import AbstractPlayer
from core.interfaces import IActionHeuristic, IStateHeuristic, IActionKey, IStateKey, ITunableParameters
from players import PlayerParameters
from players.mcts.MCTSPlayer import MCTSPlayer
from players.simple import RandomPlayer
from players.mcts import MCTSEnums
import json

class MCTSParams(PlayerParameters):
    def __init__(self):
        super().__init__()
        
        # Default parameter values
        self.K = 1.0
        self.rollout_length = 1000
        self.rollout_length_per_player = False
        self.max_tree_depth = 1000
        self.information = MCTSEnums.Information.Information_Set
        self.MAST = MCTSEnums.MASTType.None
        self.use_MAST = False
        self.MAST_gamma = 0.0
        self.MAST_default_value = 0.0
        self.MAST_boltzmann = 0.1
        self.exp3_boltzmann = 0.1
        self.use_MAST_as_action_heuristic = False
        self.selection_policy = MCTSEnums.SelectionPolicy.SIMPLE
        self.tree_policy = MCTSEnums.TreePolicy.UCB
        self.opponent_tree_policy = MCTSEnums.OpponentTreePolicy.OneTree
        self.paranoid = False
        self.rollout_increment_type = MCTSEnums.RolloutIncrement.TICK
        self.rollout_type = MCTSEnums.Strategies.RANDOM
        self.opp_model_type = MCTSEnums.Strategies.DEFAULT
        self.rollout_class = ""
        self.opp_model_class = ""
        self.rollout_policy: Optional[AbstractPlayer] = None
        self.rollout_policy_params: Optional[ITunableParameters] = None
        self.opponent_model: Optional[AbstractPlayer] = None
        self.opponent_model_params: Optional[ITunableParameters] = None
        self.explore_epsilon = 0.1
        self.oma_visits = 30
        self.normalise_rewards = True
        self.maintain_master_state = False
        self.discard_state_after_each_iteration = True
        self.rollout_termination = MCTSEnums.RolloutTermination.DEFAULT
        self.heuristic: IStateHeuristic = lambda gs: gs.get_heuristic_score()
        self.MAST_action_key: Optional[IActionKey] = None
        self.MCGS_state_key: Optional[IStateKey] = None
        self.MCGS_expand_after_clash = True
        self.first_play_urgency = 1e6
        self.action_heuristic: IActionHeuristic = lambda a, gs, al: 0.0
        self.action_heuristic_recalculation_threshold = 20
        self.pUCT = False
        self.pUCT_temperature = 0.0
        self.initialise_visits = 0
        self.progressive_widening_constant = 0.0
        self.progressive_widening_exponent = 0.0
        self.progressive_bias = 0.0
        self.reuse_tree = False
        self.backup_policy = MCTSEnums.BackupPolicy.MonteCarlo
        self.backup_lambda = 1.0
        self.max_backup_threshold = 1000000
        self.instantiation_class = "players.mcts.MCTSPlayer"
        
        # Initialize tunable parameters
        self._initialize_tunable_parameters()
    
    def _initialize_tunable_parameters(self):
        self.add_tunable_parameter("K", 1.0, [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        self.add_tunable_parameter("MASTBoltzmann", 0.1)
        self.add_tunable_parameter("exp3Boltzmann", 0.1)
        self.add_tunable_parameter("rolloutLength", 1000, [0, 3, 10, 30, 100, 1000])
        self.add_tunable_parameter("rolloutLengthPerPlayer", False)
        self.add_tunable_parameter("maxTreeDepth", 1000, [1, 3, 10, 30, 100, 1000])
        self.add_tunable_parameter("rolloutIncrementType", MCTSEnums.RolloutIncrement.TICK, list(MCTSEnums.RolloutIncrement))
        self.add_tunable_parameter("rolloutType", MCTSEnums.Strategies.RANDOM, list(MCTSEnums.Strategies))
        self.add_tunable_parameter("oppModelType", MCTSEnums.Strategies.RANDOM, list(MCTSEnums.Strategies))
        self.add_tunable_parameter("rolloutClass", "")
        self.add_tunable_parameter("oppModelClass", "")
        self.add_tunable_parameter("rolloutPolicyParams", ITunableParameters)
        self.add_tunable_parameter("rolloutTermination", MCTSEnums.RolloutTermination.DEFAULT, list(MCTSEnums.RolloutTermination))
        self.add_tunable_parameter("opponentModelParams", ITunableParameters)
        self.add_tunable_parameter("opponentModel", RandomPlayer())
        self.add_tunable_parameter("information", MCTSEnums.Information.Information_Set, list(MCTSEnums.Information))
        self.add_tunable_parameter("selectionPolicy", MCTSEnums.SelectionPolicy.SIMPLE, list(MCTSEnums.SelectionPolicy))
        self.add_tunable_parameter("treePolicy", MCTSEnums.TreePolicy.UCB, list(MCTSEnums.TreePolicy))
        self.add_tunable_parameter("opponentTreePolicy", MCTSEnums.OpponentTreePolicy.OneTree, list(MCTSEnums.OpponentTreePolicy))
        self.add_tunable_parameter("exploreEpsilon", 0.1)
        self.add_tunable_parameter("heuristic", IStateHeuristic, lambda gs: gs.get_heuristic_score())
        self.add_tunable_parameter("MAST", MCTSEnums.MASTType.None, list(MCTSEnums.MASTType))
        self.add_tunable_parameter("MASTGamma", 0.0, [0.0, 0.5, 0.9, 1.0])
        self.add_tunable_parameter("useMASTAsActionHeuristic", False)
        self.add_tunable_parameter("progressiveWideningConstant", 0.0, [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.add_tunable_parameter("progressiveWideningExponent", 0.0, [0.0, 0.1, 0.2, 0.3, 0.5])
        self.add_tunable_parameter("normaliseRewards", True)
        self.add_tunable_parameter("maintainMasterState", False)
        self.add_tunable_parameter("discardStateAfterEachIteration", True)
        self.add_tunable_parameter("omaVisits", 30)
        self.add_tunable_parameter("paranoid", False)
        self.add_tunable_parameter("MASTActionKey", IActionKey)
        self.add_tunable_parameter("MASTDefaultValue", 0.0)
        self.add_tunable_parameter("MCGSStateKey", IStateKey)
        self.add_tunable_parameter("MCGSExpandAfterClash", True)
        self.add_tunable_parameter("FPU", 1e6)
        self.add_tunable_parameter("actionHeuristic", IActionHeuristic, lambda a, gs, al: 0.0)
        self.add_tunable_parameter("progressiveBias", 0.0)
        self.add_tunable_parameter("pUCT", False)
        self.add_tunable_parameter("pUCTTemperature", 0.0)
        self.add_tunable_parameter("initialiseVisits", 0)
        self.add_tunable_parameter("actionHeuristicRecalculation", 20)
        self.add_tunable_parameter("reuseTree", False)
        self.add_tunable_parameter("backupPolicy", MCTSEnums.BackupPolicy.MonteCarlo, list(MCTSEnums.BackupPolicy))
        self.add_tunable_parameter("backupLambda", 1.0)
        self.add_tunable_parameter("maxBackupThreshold", 1000000)
        self.add_tunable_parameter("instantiationClass", "players.mcts.MCTSPlayer")

    def _reset(self):
        super()._reset()
        self.K = self.get_parameter_value("K")
        self.rollout_length = self.get_parameter_value("rolloutLength")
        self.rollout_length_per_player = self.get_parameter_value("rolloutLengthPerPlayer")
        self.max_tree_depth = self.get_parameter_value("maxTreeDepth")
        self.rollout_increment_type = self.get_parameter_value("rolloutIncrementType")
        self.rollout_type = self.get_parameter_value("rolloutType")
        self.rollout_termination = self.get_parameter_value("rolloutTermination")
        self.opp_model_type = self.get_parameter_value("oppModelType")
        self.information = self.get_parameter_value("information")
        self.tree_policy = self.get_parameter_value("treePolicy")
        self.selection_policy = self.get_parameter_value("selectionPolicy")
        self.opponent_tree_policy = self.get_parameter_value("opponentTreePolicy")
        self.explore_epsilon = self.get_parameter_value("exploreEpsilon")
        self.MAST_boltzmann = self.get_parameter_value("MASTBoltzmann")
        self.MAST = self.get_parameter_value("MAST")
        self.MAST_gamma = self.get_parameter_value("MASTGamma")
        self.exp3_boltzmann = self.get_parameter_value("exp3Boltzmann")
        self.rollout_class = self.get_parameter_value("rolloutClass")
        self.opp_model_class = self.get_parameter_value("oppModelClass")
        self.progressive_bias = self.get_parameter_value("progressiveBias")
        self.oma_visits = self.get_parameter_value("omaVisits")
        self.progressive_widening_constant = self.get_parameter_value("progressiveWideningConstant")
        self.progressive_widening_exponent = self.get_parameter_value("progressiveWideningExponent")
        self.normalise_rewards = self.get_parameter_value("normaliseRewards")
        self.maintain_master_state = self.get_parameter_value("maintainMasterState")
        self.paranoid = self.get_parameter_value("paranoid")
        self.discard_state_after_each_iteration = self.get_parameter_value("discardStateAfterEachIteration")
        self.pUCT = self.get_parameter_value("pUCT")
        self.pUCT_temperature = self.get_parameter_value("pUCTTemperature")
        
        if self.information == MCTSEnums.Information.Closed_Loop:
            self.discard_state_after_each_iteration = False

        self.MAST_action_key = self.get_parameter_value("MASTActionKey")
        self.MAST_default_value = self.get_parameter_value("MASTDefaultValue")
        self.action_heuristic = self.get_parameter_value("actionHeuristic")
        self.heuristic = self.get_parameter_value("heuristic")
        self.MCGS_state_key = self.get_parameter_value("MCGSStateKey")
        self.MCGS_expand_after_clash = self.get_parameter_value("MCGSExpandAfterClash")
        self.rollout_policy_params = self.get_parameter_value("rolloutPolicyParams")
        self.opponent_model_params = self.get_parameter_value("opponentModelParams")
        self.first_play_urgency = self.get_parameter_value("FPU")
        self.initialise_visits = self.get_parameter_value("initialiseVisits")
        self.action_heuristic_recalculation_threshold = self.get_parameter_value("actionHeuristicRecalculation")
        self.reuse_tree = self.get_parameter_value("reuseTree")
        self.backup_policy = self.get_parameter_value("backupPolicy")
        self.backup_lambda = self.get_parameter_value("backupLambda")
        self.max_backup_threshold = self.get_parameter_value("maxBackupThreshold")
        
        try:
            self.instantiation_class = self.get_parameter_value("instantiationClass")
        except Exception as e:
            raise RuntimeError(e)
        
        self.opponent_model = None
        self.rollout_policy = None
        self.use_MAST_as_action_heuristic = self.get_parameter_value("useMASTAsActionHeuristic")
        self.use_MAST = self.MAST != MCTSEnums.MASTType.None
        
        if not self.use_MAST and (self.rollout_type == MCTSEnums.Strategies.MAST or
                                 self.opp_model_type == MCTSEnums.Strategies.MAST or
                                 self.use_MAST_as_action_heuristic):
            print("Setting MAST to Both instead of None given use of MAST in rollout or action heuristic")
            self.use_MAST = True
            self.MAST = MCTSEnums.MASTType.Both

    def _copy(self) -> 'MCTSParams':
        return MCTSParams()

    def get_opponent_model(self) -> AbstractPlayer:
        if self.opponent_model is None:
            if self.opp_model_type == MCTSEnums.Strategies.PARAMS:
                self.opponent_model = self.opponent_model_params.instantiate()
            elif self.opp_model_type == MCTSEnums.Strategies.DEFAULT:
                self.opponent_model = self.get_rollout_strategy()
            else:
                self.opponent_model = self.construct_strategy(self.opp_model_type, self.opp_model_class)
            self.opponent_model.get_parameters().action_space = self.action_space
        return self.opponent_model

    def get_rollout_strategy(self) -> AbstractPlayer:
        if self.rollout_policy is None:
            if self.rollout_type == MCTSEnums.Strategies.PARAMS:
                self.rollout_policy = self.rollout_policy_params.instantiate()
            else:
                self.rollout_policy = self.construct_strategy(self.rollout_type, self.rollout_class)
            self.rollout_policy.get_parameters().action_space = self.action_space
        return self.rollout_policy

    def construct_strategy(self, strategy_type: MCTSEnums.Strategies, details: str) -> AbstractPlayer:
        if strategy_type == MCTSEnums.Strategies.RANDOM:
            return RandomPlayer(random.Random(self.get_random_seed()))
        elif strategy_type == MCTSEnums.Strategies.MAST:
            return MASTPlayer(self.MAST_action_key, self.MAST_boltzmann, 0.0, self.get_random_seed(), self.MAST_default_value)
        elif strategy_type == MCTSEnums.Strategies.CLASS:
            return json.loads(details)  # Assuming JSON utils equivalent
        elif strategy_type == MCTSEnums.Strategies.PARAMS:
            raise AssertionError("PolicyParameters have not been set")
        else:
            raise AssertionError(f"Unknown strategy type: {strategy_type}")

    def get_heuristic(self) -> IStateHeuristic:
        return self.heuristic

    def instantiate(self) -> 'MCTSPlayer':
        if not self.use_MAST and (self.use_MAST_as_action_heuristic or self.rollout_type == MCTSEnums.Strategies.MAST):
            raise AssertionError("MAST data not being collected, but MAST is being used as the rollout policy or action heuristic")
        
        if self.instantiation_class is None or self.instantiation_class == "players.mcts.MCTSPlayer":
            return MCTSPlayer(self.copy())
        else:
            try:
                cls = globals()[self.instantiation_class.split('.')[-1]]
                return cls(self)
            except Exception as e:
                raise AssertionError(f"Could not instantiate class: {self.instantiation_class}")