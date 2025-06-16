from enum import Enum
import time
from typing import List, Set, Type, Optional, Dict, Any
from core import AbstractForwardModel, AbstractGameState, AbstractParameters, Game
from games.terraformingmars import TMForwardModel, TMGameParameters, TMGameState
from games.terraformingmars.gui import TMGUI
from gui import AbstractGUIManager, GamePanel
from players.human import ActionController, HumanGUIPlayer
from core.rules import AbstractRuleBasedForwardModel

class GameType:
    """
    Encapsulates all games available in the framework, with minimum and maximum number of players as per game rules.
    All games further include a list of categories and mechanics, which can be used to filter the game collection.
    Additionally: classes where the game state, forward model, parameters and GUI manager (optional, can be null) are implemented,
    and path to where JSON data for the game is stored (optional).
    """
    
    class Category(Enum):
        Strategy = "Strategy"
        Simple = "Simple"
        Abstract = "Abstract"
        Animals = "Animals"
        Cards = "Cards"
        ComicBook = "ComicBook"
        Dice = "Dice"
        Humour = "Humour"
        Medical = "Medical"
        Deduction = "Deduction"
        Renaissance = "Renaissance"
        MoviesTVRadio = "MoviesTVRadio"
        Number = "Number"
        AmericanWest = "AmericanWest"
        Fighting = "Fighting"
        Trains = "Trains"
        CityBuilding = "CityBuilding"
        Medieval = "Medieval"
        TerritoryBuilding = "TerritoryBuilding"
        Adventure = "Adventure"
        Exploration = "Exploration"
        Fantasy = "Fantasy"
        Miniatures = "Miniatures"
        Bluffing = "Bluffing"
        Economic = "Economic"
        Environmental = "Environmental"
        Manufacturing = "Manufacturing"
        Wargame = "Wargame"
        Civilization = "Civilization"
        Ancient = "Ancient"
        CodeBreaking = "CodeBreaking"

        def get_all_games(self) -> List['GameType']:
            """Returns a list of all games within this category."""
            return [gt for gt in GameType.__subclasses__() if self in gt.categories]

        def get_all_games_excluding(self) -> List['GameType']:
            """Returns a list of all games that are NOT within this category."""
            return [gt for gt in GameType.__subclasses__() if self not in gt.categories]

    class Mechanic(Enum):
        Cooperative = "Cooperative"
        ActionPoints = "ActionPoints"
        HandManagement = "HandManagement"
        PointToPointMovement = "PointToPointMovement"
        SetCollection = "SetCollection"
        Trading = "Trading"
        VariablePlayerPowers = "VariablePlayerPowers"
        HotPotato = "HotPotato"
        PlayerElimination = "PlayerElimination"
        PushYourLuck = "PushYourLuck"
        TakeThat = "TakeThat"
        LoseATurn = "LoseATurn"
        CardDrafting = "CardDrafting"
        ActionQueue = "ActionQueue"
        Memory = "Memory"
        SimultaneousActionSelection = "SimultaneousActionSelection"
        ProgrammedEvent = "ProgrammedEvent"
        Influence = "Influence"
        MapAddition = "MapAddition"
        TilePlacement = "TilePlacement"
        PatternBuilding = "PatternBuilding"
        GameMaster = "GameMaster"
        DiceRolling = "DiceRolling"
        GridMovement = "GridMovement"
        WorkerPlacement = "WorkerPlacement"
        EngineBuilding = "EngineBuilding"
        LineOfSight = "LineOfSight"
        ModularBoard = "ModularBoard"
        MovementPoints = "MovementPoints"
        MultipleMaps = "MultipleMaps"
        Campaign = "Campaign"
        Enclosure = "Enclosure"
        DeckManagement = "DeckManagement"
        Drafting = "Drafting"
        EndGameBonus = "EndGameBonus"
        HexagonGrid = "HexagonGrid"
        Income = "Income"
        ProgressiveTurnOrder = "ProgressiveTurnOrder"
        TableauBuilding = "TableauBuilding"
        BattleCardDriven = "BattleCardDriven"
        CommandCards = "CommandCards"
        MoveThroughDeck = "MoveThroughDeck"
        TrickTaking = "TrickTaking"
        RoleSelection = "RoleSelection"
        ClosedDrafting = "ClosedDrafting"
        NeighbourScope = "NeighbourScope"
        ActionRetrieval = "ActionRetrieval"
        AreaMajority = "AreaMajority"
        AreaMovement = "AreaMovement"
        Race = "Race"
        SuddenDeathEnding = "SuddenDeathEnding"
        MultiUseCards = "MultiUseCards"
        Negotiation = "Negotiation"
        VariableSetup = "VariableSetup"
        NetworkAndRouteBuilding = "NetworkAndRouteBuilding"
        RandomProduction = "RandomProduction"

        def get_all_games(self) -> List['GameType']:
            """Returns a list of all games using this mechanic."""
            return [gt for gt in GameType.__subclasses__() if self in gt.mechanics]

        def get_all_games_excluding(self) -> List['GameType']:
            """Returns a list of all games that do NOT use this mechanic."""
            return [gt for gt in GameType.__subclasses__() if self not in gt.mechanics]

    def __init__(self, 
                 min_players: int, 
                 max_players: int, 
                 categories: List[Category], 
                 mechanics: List[Mechanic],
                 game_state_class: Type[AbstractGameState],
                 forward_model_class: Type[AbstractForwardModel],
                 parameter_class: Type[AbstractParameters],
                 gui_manager_class: Type[AbstractGUIManager],
                 data_path: Optional[str] = None):
        self.min_players = min_players
        self.max_players = max_players
        self.categories = categories
        self.mechanics = mechanics
        self.game_state_class = game_state_class
        self.forward_model_class = forward_model_class
        self.parameter_class = parameter_class
        self.gui_manager_class = gui_manager_class
        self.data_path = data_path

    def load_rulebook(self) -> str:
        """Load the rulebook for this game."""
        return "not implemented"

    def create_game_state(self, params: AbstractParameters, n_players: int) -> AbstractGameState:
        """Create a new game state instance."""
        if self.game_state_class is None:
            raise AssertionError(f"No game state class declared for the game: {self}")
        return self.game_state_class(params, n_players)

    def create_forward_model(self, params: AbstractParameters, n_players: int) -> AbstractForwardModel:
        """Create a new forward model instance."""
        if self.forward_model_class is None:
            raise AssertionError(f"No forward model class declared for the game: {self}")
        
        if issubclass(self.forward_model_class, AbstractRuleBasedForwardModel):
            return self.forward_model_class(params, n_players)
        else:
            return self.forward_model_class()

    def create_parameters(self, seed: int) -> AbstractParameters:
        """Create a new parameters instance."""
        if self.parameter_class is None:
            raise AssertionError(f"No parameter class declared for the game: {self}")
        
        if self.data_path is not None:
            try:
                return self.parameter_class(self.data_path, seed)
            except TypeError:
                return self.parameter_class(self.data_path)
        else:
            try:
                return self.parameter_class(seed)
            except TypeError:
                return self.parameter_class()

    def create_gui_manager(self, parent: GamePanel, game: Game, ac: ActionController) -> AbstractGUIManager:
        """Create a GUI manager for this game."""
        if self.gui_manager_class is None:
            raise AssertionError(f"No GUI manager class declared for the game: {self}")
        
        # Find ID of human player, if any (-1 if none)
        human = set()
        if game is not None and game.players is not None:
            for i, player in enumerate(game.players):
                if isinstance(player, HumanGUIPlayer):
                    human.add(i)
        
        return self.gui_manager_class(parent, game, ac, human)

    def create_game_instance(self, 
                           n_players: int, 
                           seed: Optional[int] = None, 
                           params: Optional[AbstractParameters] = None) -> Game:
        """Create an instance of the given game type."""
        if n_players < self.min_players or n_players > self.max_players:
            raise ValueError(f"Unsupported number of players: {n_players}. Should be in range [{self.min_players},{self.max_players}].")
        
        if params is None:
            seed = seed if seed is not None else int(time.time())
            params = self.create_parameters(seed)
        elif seed is not None:
            params.random_seed = seed
        
        return Game(
            self,
            self.create_forward_model(params, n_players),
            self.create_game_state(params, n_players)
        )

    def __str__(self) -> str:
        """String representation of the game type."""
        gui = self.gui_manager_class is not None
        fm = self.forward_model_class is not None
        gs = self.game_state_class is not None
        params = self.parameter_class is not None
        
        return (f"{self.__class__.__name__} {{\n"
                f"\tmin_players = {self.min_players}\n"
                f"\tmax_players = {self.max_players}\n"
                f"\tcategories = {self.categories}\n"
                f"\tmechanics = {self.mechanics}\n"
                f"\tGS = {gs}\n"
                f"\tFM = {fm}\n"
                f"\tParams = {params}\n"
                f"\tGUI = {gui}\n"
                "}\n")

    @classmethod
    def print_available_games(cls):
        """Print all available games in the framework."""
        print("Games available in the framework: \n")
        for subclass in cls.__subclasses__():
            print(subclass)


class GameTemplate(GameType):
    """Game template example."""
    def __init__(self):
        super().__init__(
            min_players=1,
            max_players=8,
            categories=None,
            mechanics=None,
            game_state_class=None,
            forward_model_class=None,
            parameter_class=None,
            gui_manager_class=None
        )


class TerraformingMars(GameType):
    """Terraforming Mars game implementation."""
    def __init__(self):
        super().__init__(
            min_players=1,
            max_players=5,
            categories=[
                GameType.Category.Economic,
                GameType.Category.Environmental,
                GameType.Category.Manufacturing,
                GameType.Category.TerritoryBuilding,
                GameType.Category.Cards,
                GameType.Category.Strategy,
                GameType.Category.Exploration
            ],
            mechanics=[
                GameType.Mechanic.Drafting,
                GameType.Mechanic.EndGameBonus,
                GameType.Mechanic.HandManagement,
                GameType.Mechanic.HexagonGrid,
                GameType.Mechanic.Income,
                GameType.Mechanic.SetCollection,
                GameType.Mechanic.TakeThat,
                GameType.Mechanic.TilePlacement,
                GameType.Mechanic.ProgressiveTurnOrder,
                GameType.Mechanic.VariablePlayerPowers,
                GameType.Mechanic.EngineBuilding,
                GameType.Mechanic.TableauBuilding
            ],
            game_state_class=TMGameState,
            forward_model_class=TMForwardModel,
            parameter_class=TMGameParameters,
            gui_manager_class=TMGUI
        )


if __name__ == "__main__":
    GameType.print_available_games()