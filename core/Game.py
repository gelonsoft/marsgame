import sys
import time
import os
sys.path.append(os.getcwd())
from typing import List, Optional
import streamlit as st
from copy import deepcopy

from core import AbstractForwardModel, AbstractGameState, AbstractGameStateWithTurnOrder, AbstractParameters, CoreParameters, GameResult
from core.actions import AbstractAction, DoNothing
from core.interfaces import IPrintable
from core.turnorders import ReactiveTurnOrder
from evaluation.listeners import IGameListener
from evaluation.metrics import Event
from evaluation.summarisers import TAGNumericStatSummary
from games import GameType
from players import AbstractPlayer
from players.human import ActionController,HumanGUIPlayer, HumanConsolePlayer
from players.simple import RandomPlayer
from utilities import Pair, Utils

class Game:
    id_fountain = 0
    
    def __init__(self, game_type: GameType, players: List[AbstractPlayer], 
                 forward_model: AbstractForwardModel, game_state: AbstractGameState):
        self.game_type = game_type
        self.game_state = game_state
        self.forward_model = forward_model
        self.listeners = []
        self.paused = False
        self.stop = False
        self.debug = False
        self.turn_pause = 0
        self.reset(players)
    
    def reset(self, players: List[AbstractPlayer], new_random_seed: Optional[int] = None):
        if new_random_seed is None:
            new_random_seed = self.game_state.game_parameters.random_seed
            
        if self.debug:
            print(f"Game Seed: {new_random_seed}")
            
        self.game_state.reset(new_random_seed)
        self.forward_model.abstract_setup(self.game_state)
        
        # Set forward models for all players
        for player in players:
                player.set_forward_model(self.forward_model)
                
        if len(players) == self.game_state.get_n_players():
            self.players = players
        elif not players:
            # Keep existing players
            pass
        elif len(players) == self.game_state.n_teams:
            self.players = []
            for i in range(self.game_state.get_n_players()):
                team = self.game_state.get_team(i)
                player = players[team]
                self.players.append(player.copy())
        else:
            raise ValueError("PlayerList provided to Game.reset() must be empty, or have the same number of entries as there are players")
            
        # Initialize players
        player_id = 0
        if self.players is not None:
            for player in self.players:
                player.player_id = player_id
                player_id += 1
                observation = self.game_state.copy(player.player_id)
                player.initialize_player(observation)
                
        Game.id_fountain += 1
        self.game_state.set_game_id(Game.id_fountain)
        self.reset_stats()
        
    def reset_stats(self):
        self.next_time = 0
        self.copy_time = 0
        self.agent_time = 0
        self.action_compute_time = 0
        self.n_decisions = 0
        self.action_space_size = []
        self.n_actions_per_turn_sum = 0
        self.n_actions_per_turn = 1
        self.n_actions_per_turn_count = 0
        self.last_player = -1
        
    def run(self):
        for listener in self.listeners:
            listener.on_event(Event.create_event(Event.GameEvent.ABOUT_TO_START, self.game_state))
            
        first_end = True
        
        while self.game_state.is_not_terminal() and not self.stop:
            # Synchronization would be handled differently in Python/Streamlit
            # Here we just check the pause state
            while self.paused and not self.is_human_to_move():
                time.sleep(0.1)
                
            active_player = self.game_state.get_current_player()
            if self.debug:
                print(f"Entered synchronized block in Game for player {active_player}")
                
            current_player = self.players[active_player]
            
            # Check if reacting
            reacting = (isinstance(self.game_state, AbstractGameStateWithTurnOrder) and 
                       isinstance(self.game_state.get_turn_order(), ReactiveTurnOrder) and 
                       len(self.game_state.get_turn_order().get_reactive_players()) > 0)
                       
            # Count actions per turn
            if not reacting:
                if current_player is not None and active_player == self.last_player:
                    self.n_actions_per_turn += 1
                else:
                    self.n_actions_per_turn_sum += self.n_actions_per_turn
                    self.n_actions_per_turn = 1
                    self.n_actions_per_turn_count += 1
                    
            if self.game_state.is_not_terminal():
                if self.debug:
                    print(f"Invoking one_action from Game for player {active_player}")
                self.one_action()
            else:
                if first_end:
                    if self.game_state.core_game_parameters.verbose:
                        print("Ended")
                    self.terminate()
                    first_end = False
                    
        if first_end:
            if self.game_state.core_game_parameters.verbose:
                print("Ended")
            self.terminate()
            
    def is_human_to_move(self) -> bool:
        active_player = self.game_state.get_current_player()
        return isinstance(self.players[active_player], HumanGUIPlayer)
        
    def one_action(self) -> AbstractAction:
        # Pause before each action if needed
        if self.turn_pause > 0:
            time.sleep(self.turn_pause / 1000)
            
        active_player = self.game_state.get_current_player()
        if not self.game_state.is_not_terminal_for_player(active_player):
            raise AssertionError(f"Player {active_player} is not allowed to move")
            
        current_player = self.players[active_player]
        if self.debug:
            print(f"Starting one_action for player {active_player}")
            
        # Get player observation
        start_time = time.time()
        observation = self.game_state.copy(active_player)
        self.copy_time = (time.time() - start_time) * 1e9  # Convert to nanoseconds for consistency
        
        # Get available actions
        start_time = time.time()
        observed_actions = self.forward_model.compute_available_actions(
            observation, current_player.get_parameters().action_space)
            
        if not observed_actions:
            actions_in_progress = self.game_state.get_actions_in_progress()
            top_of_stack = actions_in_progress[-1] if actions_in_progress else None
            last_action = self.game_state.get_history()[-1][1] if len(self.game_state.get_history()) > 1 else None
            
            if self.debug:
                print("---\nActions in progress:")
                for action in actions_in_progress:
                    print(action)
                print("---\nRecent History:")
                history = self.game_state.get_history()
                for i in range(max(0, len(history) - 10), len(history)):
                    print(history[i])
                    
            raise AssertionError(f"No actions available for player {active_player}" + 
                               (f". Last action: {type(last_action).__name__} ({last_action})" if last_action else ". No actions in history") +
                               f". Actions in progress: {len(actions_in_progress)}" +
                               (f". Top of stack: {type(top_of_stack).__name__} ({top_of_stack.get_string(self.game_state) if isinstance(top_of_stack, AbstractAction) else top_of_stack})" if top_of_stack else ""))
                               
        self.action_compute_time = (time.time() - start_time) * 1e9
        self.action_space_size.append(Pair(active_player, len(observed_actions)))
        
        if self.game_state.core_game_parameters.verbose:
            print(f"Round: {self.game_state.get_round_counter()}")
            
        if isinstance(observation, IPrintable) and self.game_state.core_game_parameters.verbose:
            observation.print_to_console()
            
        # Start timer for this decision
        self.game_state.player_timer[active_player].resume()
        
        # Get action from player
        action = None
        if observed_actions:
            if len(observed_actions) == 1 and (not isinstance(current_player, (HumanGUIPlayer, HumanConsolePlayer)) or isinstance(observed_actions[0], DoNothing)):
                action = observed_actions[0]
                current_player.register_updated_observation(observation)
            else:
                start_time = time.time()
                if self.debug:
                    print(f"About to get action for player {self.game_state.get_current_player()}")
                action = current_player.get_action(observation, observed_actions)
                
                if action not in observed_actions:
                    raise AssertionError("Action played that was not in the list of available actions: {action}")
                    
                if self.debug:
                    print(f"Game: {self.game_state.get_game_id():2d} Tick: {self.get_tick():3d}\t{action.get_string(self.game_state)}")
                    
                self.agent_time = (time.time() - start_time) * 1e9
                self.n_decisions += 1
                
            if self.game_state.core_game_parameters.competition_mode and action and action not in observed_actions:
                print(f"Action played that was not in the list of available actions: {action.get_string(self.game_state)}")
                action = None
                
            # Notify listeners about chosen action
            for listener in self.listeners:
                listener.on_event(Event.create_event(Event.GameEvent.ACTION_CHOSEN, self.game_state, action, active_player))
        else:
            current_player.register_updated_observation(observation)
            
        # End timer for this decision
        self.game_state.player_timer[active_player].pause()
        self.game_state.player_timer[active_player].increment_action()
        
        if self.game_state.core_game_parameters.verbose and action:
            print(action)
            
        if not action:
            raise AssertionError("We have a NULL action in the Game loop")
            
        # Check player timeout
        if observation.player_timer[active_player].exceeded_max_time():
            action = self.forward_model.disqualify_or_random_action(
                self.game_state.core_game_parameters.disqualify_player_on_timeout, self.game_state)
        else:
            # Resolve action and game rules
            start_time = time.time()
            self.forward_model.next(self.game_state, deepcopy(action))
            self.next_time = (time.time() - start_time) * 1e9
            
        self.last_player = active_player
        
        # Notify listeners about taken action
        for listener in self.listeners:
            listener.on_event(Event.create_event(Event.GameEvent.ACTION_TAKEN, self.game_state, deepcopy(action), active_player))
            
        if self.debug:
            print(f"Finishing one_action for player {active_player}")
        return action
        
    def terminate(self):
        # Print last state
        if isinstance(self.game_state, IPrintable) and self.game_state.core_game_parameters.verbose:
            self.game_state.print_to_console()
            
        # Perform end of game computations
        self.forward_model.end_game(self.game_state)
        for listener in self.listeners:
            listener.on_event(Event.create_event(Event.GameEvent.GAME_OVER, self.game_state))
            
        if self.game_state.core_game_parameters.record_event_history:
            self.game_state.record_history(Event.GameEvent.GAME_OVER.name)
            for i in range(self.game_state.get_n_players()):
                self.game_state.record_history(
                    f"Player {i} finishes at position {self.game_state.get_ordinal_position(i)} with score: {self.game_state.get_game_score(i):.0f}")
                    
        if self.game_state.core_game_parameters.verbose:
            print("Game Over")
            
        # Allow players to terminate
        for player in self.players:
            player.finalize_player(self.game_state.copy(player.get_player_id()))
            
    # Getters and setters
    def get_game_state(self) -> AbstractGameState:
        return self.game_state
        
    def get_forward_model(self) -> AbstractForwardModel:
        return self.forward_model
        
    def get_agent_time(self) -> float:
        return self.agent_time
        
    def get_copy_time(self) -> float:
        return self.copy_time
        
    def get_next_time(self) -> float:
        return self.next_time
        
    def get_action_compute_time(self) -> float:
        return self.action_compute_time
        
    def get_tick(self) -> int:
        return self.game_state.get_game_tick()
        
    def get_n_decisions(self) -> int:
        return self.n_decisions
        
    def get_n_actions_per_turn(self) -> int:
        return self.n_actions_per_turn_sum
        
    def get_action_space_size(self) -> List[Pair[int, int]]:
        return self.action_space_size
        
    def get_game_type(self) -> GameType:
        return self.game_type
        
    def add_listener(self, listener: IGameListener):
        if listener not in self.listeners:
            self.listeners.append(listener)
            self.game_state.add_listener(listener)
            listener.set_game(self)
            
    def get_listeners(self) -> List[IGameListener]:
        return self.listeners
        
    def clear_listeners(self):
        self.listeners.clear()
        self.get_game_state().clear_listeners()
        
    def get_players(self) -> List[AbstractPlayer]:
        return self.players
        
    def is_paused(self) -> bool:
        return self.paused
        
    def set_paused(self, paused: bool):
        self.paused = paused
        
    def flip_paused(self):
        self.paused = not self.paused
        
    def is_stopped(self) -> bool:
        return self.stop
        
    def set_stopped(self, stopped: bool):
        self.stop = stopped
        
    def get_core_parameters(self) -> CoreParameters:
        return self.game_state.core_game_parameters
        
    def set_core_parameters(self, core_parameters: CoreParameters):
        self.game_state.set_core_game_parameters(core_parameters)
        
    def __str__(self) -> str:
        return str(self.game_type)
        
    @staticmethod
    def run_one(game_type: GameType, parameter_config_file: Optional[str], 
                players: List[AbstractPlayer], seed: int, randomize_parameters: bool,
                listeners: Optional[List[IGameListener]], ac: Optional[ActionController], 
                turn_pause: int) -> 'Game':
        # Creating game instance
        if parameter_config_file:
            params = AbstractParameters.create_from_file(game_type, parameter_config_file)
            game = game_type.create_game_instance(len(players), seed, params)
        else:
            game = game_type.create_game_instance(len(players), seed)
            
        if not game:
            print(f"Error game: {game_type}")
            return None
            
        if listeners:
            agent_names = {str(player) for player in players}
            for listener in listeners:
                listener.init(game, len(players), agent_names)
                game.add_listener(listener)
                
        # Randomize parameters
        if randomize_parameters:
            game_parameters = game.get_game_state().get_game_parameters()
            game_parameters.randomize()
            print(f"Parameters: {game_parameters}")
            
        # Reset game with players
        game.reset(players)
        game.set_turn_pause(turn_pause)
        
        if ac:
            # Streamlit GUI setup would go here
            # This would be quite different from the Java Swing implementation
            st.title(f"{game_type} Game")
            game_placeholder = st.empty()
            
            def update_gui():
                current_player = game.get_game_state().get_current_player()
                player = game.get_players()[current_player]
                # Render game state using Streamlit components
                game_placeholder.write(game.get_game_state().to_html(), unsafe_allow_html=True)
                
            # Run game with GUI updates
            while game.get_game_state().is_not_terminal() and not game.is_stopped():
                game.one_action()
                update_gui()
                time.sleep(0.1)  # Small delay for GUI updates
                
            # Final GUI update
            update_gui()
        else:
            # Run without GUI
            game.run()
            
        return game
        
    @staticmethod
    def run_many(games_to_play: List[GameType], players: List[AbstractPlayer], seed: Optional[int],
                 n_repetitions: int, randomize_parameters: bool, detailed_statistics: bool,
                 listeners: Optional[List[IGameListener]], turn_pause: int):
        n_players = len(players)
        
        # Save win rate statistics
        overall = [TAGNumericStatSummary(f"Overall Player {i}") for i in range(n_players)]
        agent_names = [f"{type(player).__name__}-{i}" for i, player in enumerate(players)]
        
        for game_type in games_to_play:
            # Save win rate statistics for this game
            stats = [TAGNumericStatSummary(f"{{Game: {game_type.name}; Player: {agent_names[i]}}}") 
                    for i in range(n_players)]
                    
            # Play n repetitions
            game = None
            offset = 0
            for i in range(n_repetitions):
                current_seed = seed if seed is not None else int(time.time())
                current_seed += offset
                game = Game.run_one(game_type, None, players, current_seed, 
                                    randomize_parameters, listeners, None, turn_pause)
                if game:
                    Game.record_player_results(stats, game)
                    offset = game.get_game_state().get_round_counter() * game.get_game_state().get_n_players()
                else:
                    break
                    
            if game:
                print("---------------------")
                for i in range(n_players):
                    if detailed_statistics:
                        print(stats[i])
                    else:
                        print(f"{stats[i].name}: {stats[i].mean()} (n={stats[i].n()})")
                    overall[i].add(stats[i])
                    
        # Print final statistics
        print("\n=====================\n")
        for i in range(n_players):
            if detailed_statistics:
                print(overall[i])
            else:
                print(f"{overall[i].name}: {overall[i].mean()}")
                
    @staticmethod
    def record_player_results(stat_summaries: List[TAGNumericStatSummary], game: 'Game'):
        results = game.get_game_state().get_player_results()
        for p in range(len(stat_summaries)):
            if results[p] in {GameResult.WIN_GAME, 
                             GameResult.LOSE_GAME, 
                             GameResult.DRAW_GAME}:
                stat_summaries[p].add(results[p].value)
                
if __name__ == "__main__":


    args=sys.argv[1:]
    game_type = Utils.get_arg(args, "game", "Chess")
    use_gui = Utils.get_arg(args, "gui", False)
    turn_pause = Utils.get_arg(args, "turnPause", 0)
    seed = Utils.get_arg(args, "seed", int(time.time() * 1000))
    ac = ActionController()

    # Set up players for the game
    players: List[AbstractPlayer] = []
    players.append(RandomPlayer())

    #mcts_params = MCTSParams()
    #players.append(MCTSPlayer(mcts_params))
    players.append(RandomPlayer())

    # Game parameter configuration. Set to None to ignore and use default parameters
    game_params = None

    # Run!
    Game.run_one(
        GameType[game_type],
        game_params,
        players,
        seed,
        False,
        None,
        ac if use_gui else None,
        turn_pause
    )