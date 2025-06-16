from core.AbstractParameters import AbstractParameters
from games.terraformingmars.TMTypes import Expansion, Resource
from typing import Dict, List, Set

class TMGameParameters(AbstractParameters):
    def __init__(self):
        super().__init__()
        self.board_size: int = 9
        self.expansions: Set[Expansion] = {Expansion.CorporateEra}  # Elysium, Hellas and Venus compiling, but not fully parsed yet
        self.solo_tr: int = 14
        self.solo_max_gen: int = 14
        self.solo_cities: int = 2

        self.minimum_production: Dict[Resource, int] = {
            res: -5 if res == Resource.MegaCredit else 0 
            for res in Resource
        }
        
        self.starting_resources: Dict[Resource, int] = {
            res: 0 for res in Resource
        }
        self.starting_resources[Resource.TR] = 20
        # self.starting_resources[Resource.MegaCredit] = 500  # TODO Test
        
        self.starting_production: Dict[Resource, int] = {
            res: 1 if res.is_player_board_res() else 0
            for res in Resource
        }
        
        self.max_points: int = 500
        self.max_cards: int = 250  # TODO based on expansions

        self.project_purchase_cost: int = 3
        self.n_corp_choice_start: int = 2
        self.n_projects_start: int = 10
        self.n_projects_research: int = 4
        self.n_actions_per_player: int = 2
        self.n_mc_gained_ocean: int = 2

        # steel and titanium to MC rate
        self.n_steel_mc: float = 2.0
        self.n_titanium_mc: float = 3.0

        # standard projects
        self.n_gain_card_discard: int = 1
        self.n_cost_sp_energy: int = 11
        self.n_cost_sp_temp: int = 14
        self.n_cost_sp_ocean: int = 18
        self.n_cost_sp_greenery: int = 23
        self.n_cost_sp_city: int = 25
        self.n_sp_city_mc_gain: int = 1
        self.n_cost_venus: int = 15

        # Resource actions
        self.n_cost_greenery_plant: int = 8
        self.n_cost_temp_heat: int = 8

        # Milestones, awards
        self.n_cost_milestone: List[int] = [8, 8, 8]
        self.n_cost_awards: List[int] = [8, 14, 20]
        self.n_points_milestone: int = 5
        self.n_points_award_first: int = 5
        self.n_points_award_second: int = 2

    def _copy(self) -> 'AbstractParameters':
        return TMGameParameters()

    def _equals(self, o: object) -> bool:
        return False

    # Property getters
    @property
    def minimum_production(self) -> Dict[Resource, int]:
        return self._minimum_production

    @property
    def starting_production(self) -> Dict[Resource, int]:
        return self._starting_production

    @property
    def starting_resources(self) -> Dict[Resource, int]:
        return self._starting_resources

    @property
    def board_size(self) -> int:
        return self._board_size

    @property
    def max_points(self) -> int:
        return self._max_points

    @property
    def n_corp_choice_start(self) -> int:
        return self._n_corp_choice_start

    @property
    def project_purchase_cost(self) -> int:
        return self._project_purchase_cost

    @property
    def n_projects_research(self) -> int:
        return self._n_projects_research

    @property
    def n_projects_start(self) -> int:
        return self._n_projects_start

    @property
    def expansions(self) -> Set[Expansion]:
        return self._expansions

    @property
    def n_mc_gained_ocean(self) -> int:
        return self._n_mc_gained_ocean

    @property
    def n_cost_milestone(self) -> List[int]:
        return self._n_cost_milestone

    @property
    def n_cost_awards(self) -> List[int]:
        return self._n_cost_awards

    @property
    def n_steel_mc(self) -> float:
        return self._n_steel_mc

    @property
    def n_titanium_mc(self) -> float:
        return self._n_titanium_mc

    @property
    def n_actions_per_player(self) -> int:
        return self._n_actions_per_player

    @property
    def n_cost_greenery_plant(self) -> int:
        return self._n_cost_greenery_plant

    @property
    def n_cost_sp_city(self) -> int:
        return self._n_cost_sp_city

    @property
    def n_cost_sp_energy(self) -> int:
        return self._n_cost_sp_energy

    @property
    def n_cost_sp_greenery(self) -> int:
        return self._n_cost_sp_greenery

    @property
    def n_cost_sp_ocean(self) -> int:
        return self._n_cost_sp_ocean

    @property
    def n_cost_sp_temp(self) -> int:
        return self._n_cost_sp_temp

    @property
    def n_cost_temp_heat(self) -> int:
        return self._n_cost_temp_heat

    @property
    def n_gain_card_discard(self) -> int:
        return self._n_gain_card_discard

    @property
    def n_points_award_first(self) -> int:
        return self._n_points_award_first

    @property
    def n_points_award_second(self) -> int:
        return self._n_points_award_second

    @property
    def n_points_milestone(self) -> int:
        return self._n_points_milestone

    @property
    def n_sp_city_mc_gain(self) -> int:
        return self._n_sp_city_mc_gain

    @property
    def solo_cities(self) -> int:
        return self._solo_cities

    @property
    def solo_max_gen(self) -> int:
        return self._solo_max_gen

    @property
    def solo_tr(self) -> int:
        return self._solo_tr