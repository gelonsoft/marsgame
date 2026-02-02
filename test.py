import numpy as np
from env_all_actions import SERVER_BASE_URL, TerraformingMarsEnv
import json 
import random
import pyarrow as pa
from decode_observation import decode_observation
from myconfig import MAX_ACTIONS

player_id=None
player_state=None
waiting_for=None
player_state="""
{
    "cardsInHand": [],
    "ceoCardsInHand": [],
    "dealtCorporationCards": [
        {
            "name": "Spire",
            "calculatedCost": 0
        },
        {
            "name": "Robinson Industries",
            "calculatedCost": 0
        }
    ],
    "dealtPreludeCards": [
        {
            "name": "Colony Trade Hub",
            "calculatedCost": 0
        },
        {
            "name": "Venus Contract",
            "calculatedCost": 0
        },
        {
            "name": "Smelting Plant",
            "calculatedCost": 0
        },
        {
            "name": "Giant Solar Collector",
            "calculatedCost": 0
        }
    ],
    "dealtCeoCards": [],
    "dealtProjectCards": [
        {
            "name": "Teslaract",
            "calculatedCost": 14
        },
        {
            "name": "Neutralizer Factory",
            "calculatedCost": 7
        },
        {
            "name": "Fusion Power",
            "calculatedCost": 14
        },
        {
            "name": "Lichen",
            "calculatedCost": 7
        },
        {
            "name": "Vesta Shipyard",
            "calculatedCost": 15
        },
        {
            "name": "Community Services",
            "calculatedCost": 13
        },
        {
            "name": "Homeostasis Bureau",
            "calculatedCost": 16
        },
        {
            "name": "Cryo-Sleep",
            "calculatedCost": 10
        },
        {
            "name": "Regolith Eaters",
            "calculatedCost": 13
        },
        {
            "name": "Meat Industry",
            "calculatedCost": 5
        }
    ],
    "draftedCards": [],
    "game": {
        "awards": [
            {
                "name": "Landlord",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Scientist",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Banker",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Thermalist",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Miner",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Venuphile",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            }
        ],
        "colonies": [
            {
                "colonies": [],
                "isActive": true,
                "name": "Callisto",
                "trackPosition": 1
            },
            {
                "colonies": [],
                "isActive": true,
                "name": "Europa",
                "trackPosition": 1
            },
            {
                "colonies": [],
                "isActive": true,
                "name": "Ganymede",
                "trackPosition": 1
            },
            {
                "colonies": [],
                "isActive": true,
                "name": "Io",
                "trackPosition": 1
            },
            {
                "colonies": [],
                "isActive": false,
                "name": "Titan",
                "trackPosition": 1
            }
        ],
        "deckSize": 401,
        "discardedColonies": [
            "Ceres",
            "Enceladus",
            "Luna",
            "Miranda",
            "Pluto",
            "Triton"
        ],
        "expectedPurgeTimeMs": 1769179614612,
        "gameAge": 9,
        "gameOptions": {
            "altVenusBoard": false,
            "aresExtremeVariant": false,
            "boardName": "tharsis",
            "bannedCards": [],
            "draftVariant": true,
            "escapeVelocityMode": false,
            "escapeVelocityBonusSeconds": 2,
            "expansions": {
                "corpera": true,
                "promo": true,
                "venus": true,
                "colonies": true,
                "prelude": true,
                "prelude2": true,
                "turmoil": true,
                "community": false,
                "ares": false,
                "moon": false,
                "pathfinders": false,
                "ceo": false,
                "starwars": false,
                "underworld": false
            },
            "fastModeOption": false,
            "includedCards": [],
            "includeFanMA": false,
            "initialDraftVariant": false,
            "preludeDraftVariant": true,
            "politicalAgendasExtension": "Standard",
            "removeNegativeGlobalEvents": false,
            "showOtherPlayersVP": true,
            "showTimers": false,
            "shuffleMapOption": false,
            "solarPhaseOption": true,
            "soloTR": false,
            "randomMA": "No randomization",
            "requiresMoonTrackCompletion": false,
            "requiresVenusTrackCompletion": false,
            "twoCorpsVariant": false,
            "undoOption": false
        },
        "generation": 1,
        "globalsPerGeneration": [],
        "isSoloModeWin": false,
        "isTerraformed": false,
        "lastSoloGeneration": 12,
        "milestones": [
            {
                "name": "Terraformer",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 20
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 20
                    }
                ]
            },
            {
                "name": "Mayor",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Gardener",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Builder",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Planner",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            },
            {
                "name": "Hoverlord",
                "scores": [
                    {
                        "playerColor": "red",
                        "playerScore": 0
                    },
                    {
                        "playerColor": "green",
                        "playerScore": 0
                    }
                ]
            }
        ],
        "oceans": 0,
        "oxygenLevel": 0,
        "passedPlayers": [],
        "phase": "research",
        "spaces": [
            {
                "x": -1,
                "y": -1,
                "id": "01",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": -1,
                "y": -1,
                "id": "02",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": 4,
                "y": 0,
                "id": "03",
                "spaceType": "land",
                "bonus": [
                    1,
                    1
                ]
            },
            {
                "x": 5,
                "y": 0,
                "id": "04",
                "spaceType": "ocean",
                "bonus": [
                    1,
                    1
                ]
            },
            {
                "x": 6,
                "y": 0,
                "id": "05",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 7,
                "y": 0,
                "id": "06",
                "spaceType": "ocean",
                "bonus": [
                    3
                ]
            },
            {
                "x": 8,
                "y": 0,
                "id": "07",
                "spaceType": "ocean",
                "bonus": []
            },
            {
                "x": 3,
                "y": 1,
                "id": "08",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 4,
                "y": 1,
                "id": "09",
                "spaceType": "land",
                "bonus": [
                    1
                ]
            },
            {
                "x": 5,
                "y": 1,
                "id": "10",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 6,
                "y": 1,
                "id": "11",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 7,
                "y": 1,
                "id": "12",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 8,
                "y": 1,
                "id": "13",
                "spaceType": "ocean",
                "bonus": [
                    3,
                    3
                ]
            },
            {
                "x": 2,
                "y": 2,
                "id": "14",
                "spaceType": "land",
                "bonus": [
                    3
                ]
            },
            {
                "x": 3,
                "y": 2,
                "id": "15",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 4,
                "y": 2,
                "id": "16",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 5,
                "y": 2,
                "id": "17",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 6,
                "y": 2,
                "id": "18",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 7,
                "y": 2,
                "id": "19",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 8,
                "y": 2,
                "id": "20",
                "spaceType": "land",
                "bonus": [
                    1
                ]
            },
            {
                "x": 1,
                "y": 3,
                "id": "21",
                "spaceType": "land",
                "bonus": [
                    2,
                    0
                ]
            },
            {
                "x": 2,
                "y": 3,
                "id": "22",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 3,
                "y": 3,
                "id": "23",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 4,
                "y": 3,
                "id": "24",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 5,
                "y": 3,
                "id": "25",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 6,
                "y": 3,
                "id": "26",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 7,
                "y": 3,
                "id": "27",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 8,
                "y": 3,
                "id": "28",
                "spaceType": "ocean",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 0,
                "y": 4,
                "id": "29",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 1,
                "y": 4,
                "id": "30",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 2,
                "y": 4,
                "id": "31",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 3,
                "y": 4,
                "id": "32",
                "spaceType": "ocean",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 4,
                "y": 4,
                "id": "33",
                "spaceType": "ocean",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 5,
                "y": 4,
                "id": "34",
                "spaceType": "ocean",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 6,
                "y": 4,
                "id": "35",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 7,
                "y": 4,
                "id": "36",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 8,
                "y": 4,
                "id": "37",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 1,
                "y": 5,
                "id": "38",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 2,
                "y": 5,
                "id": "39",
                "spaceType": "land",
                "bonus": [
                    2,
                    2
                ]
            },
            {
                "x": 3,
                "y": 5,
                "id": "40",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 4,
                "y": 5,
                "id": "41",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 5,
                "y": 5,
                "id": "42",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 6,
                "y": 5,
                "id": "43",
                "spaceType": "ocean",
                "bonus": [
                    2
                ]
            },
            {
                "x": 7,
                "y": 5,
                "id": "44",
                "spaceType": "ocean",
                "bonus": [
                    2
                ]
            },
            {
                "x": 8,
                "y": 5,
                "id": "45",
                "spaceType": "ocean",
                "bonus": [
                    2
                ]
            },
            {
                "x": 2,
                "y": 6,
                "id": "46",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 3,
                "y": 6,
                "id": "47",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 4,
                "y": 6,
                "id": "48",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 5,
                "y": 6,
                "id": "49",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 6,
                "y": 6,
                "id": "50",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 7,
                "y": 6,
                "id": "51",
                "spaceType": "land",
                "bonus": [
                    2
                ]
            },
            {
                "x": 8,
                "y": 6,
                "id": "52",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 3,
                "y": 7,
                "id": "53",
                "spaceType": "land",
                "bonus": [
                    1,
                    1
                ]
            },
            {
                "x": 4,
                "y": 7,
                "id": "54",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 5,
                "y": 7,
                "id": "55",
                "spaceType": "land",
                "bonus": [
                    3
                ]
            },
            {
                "x": 6,
                "y": 7,
                "id": "56",
                "spaceType": "land",
                "bonus": [
                    3
                ]
            },
            {
                "x": 7,
                "y": 7,
                "id": "57",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 8,
                "y": 7,
                "id": "58",
                "spaceType": "land",
                "bonus": [
                    0
                ]
            },
            {
                "x": 4,
                "y": 8,
                "id": "59",
                "spaceType": "land",
                "bonus": [
                    1
                ]
            },
            {
                "x": 5,
                "y": 8,
                "id": "60",
                "spaceType": "land",
                "bonus": [
                    1,
                    1
                ]
            },
            {
                "x": 6,
                "y": 8,
                "id": "61",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 7,
                "y": 8,
                "id": "62",
                "spaceType": "land",
                "bonus": []
            },
            {
                "x": 8,
                "y": 8,
                "id": "63",
                "spaceType": "ocean",
                "bonus": [
                    0,
                    0
                ]
            },
            {
                "x": -1,
                "y": -1,
                "id": "69",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": -1,
                "y": -1,
                "id": "71",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": -1,
                "y": -1,
                "id": "70",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": -1,
                "y": -1,
                "id": "73",
                "spaceType": "colony",
                "bonus": []
            },
            {
                "x": -1,
                "y": -1,
                "id": "72",
                "spaceType": "colony",
                "bonus": []
            }
        ],
        "spectatorId": "s4f6820e166ca",
        "step": 1,
        "temperature": -30,
        "tags": [
            "science",
            "building",
            "power",
            "earth",
            "plant",
            "venus",
            "space",
            "microbe",
            "city",
            "animal",
            "jovian",
            "wild"
        ],
        "turmoil": {
            "chairman": "neutral",
            "ruling": "Greens",
            "dominant": "Greens",
            "parties": [
                {
                    "name": "Mars First",
                    "delegates": []
                },
                {
                    "name": "Scientists",
                    "partyLeader": "neutral",
                    "delegates": [
                        {
                            "color": "neutral",
                            "number": 1
                        }
                    ]
                },
                {
                    "name": "Unity",
                    "delegates": []
                },
                {
                    "name": "Greens",
                    "partyLeader": "neutral",
                    "delegates": [
                        {
                            "color": "neutral",
                            "number": 1
                        }
                    ]
                },
                {
                    "name": "Reds",
                    "delegates": []
                },
                {
                    "name": "Kelvinists",
                    "delegates": []
                }
            ],
            "lobby": [
                "red",
                "green"
            ],
            "reserve": [
                {
                    "color": "red",
                    "number": 6
                },
                {
                    "color": "green",
                    "number": 6
                },
                {
                    "color": "neutral",
                    "number": 11
                }
            ],
            "distant": "Improved Energy Templates",
            "coming": "Pandemic",
            "politicalAgendas": {
                "marsFirst": {
                    "bonusId": "mb01",
                    "policyId": "mp01"
                },
                "scientists": {
                    "bonusId": "sb01",
                    "policyId": "sp01"
                },
                "unity": {
                    "bonusId": "ub01",
                    "policyId": "up01"
                },
                "greens": {
                    "bonusId": "gb01",
                    "policyId": "gp01"
                },
                "reds": {
                    "bonusId": "rb01",
                    "policyId": "rp01"
                },
                "kelvinists": {
                    "bonusId": "kb01",
                    "policyId": "kp01"
                }
            },
            "policyActionUsers": [
                {
                    "color": "red",
                    "turmoilPolicyActionUsed": false,
                    "politicalAgendasActionUsedCount": 0
                },
                {
                    "color": "green",
                    "turmoilPolicyActionUsed": false,
                    "politicalAgendasActionUsedCount": 0
                }
            ]
        },
        "undoCount": 0,
        "venusScaleLevel": 0
    },
    "id": "p1b88a95d14a5",
    "runId": "r7ea3f4e0ec10",
    "pickedCorporationCard": [],
    "preludeCardsInHand": [],
    "thisPlayer": {
        "actionsTakenThisRound": 0,
        "actionsTakenThisGame": 0,
        "actionsThisGeneration": [],
        "availableBlueCardActionCount": 0,
        "cardCost": 3,
        "cardDiscount": 0,
        "cardsInHandNbr": 0,
        "citiesCount": 0,
        "coloniesCount": 0,
        "color": "green",
        "energy": 0,
        "energyProduction": 0,
        "fleetSize": 1,
        "heat": 0,
        "heatProduction": 0,
        "influence": 0,
        "isActive": false,
        "megaCredits": 0,
        "megaCreditProduction": 0,
        "name": "Green",
        "needsToResearch": true,
        "noTagsCount": 0,
        "plants": 0,
        "plantProduction": 0,
        "protectedResources": {
            "megacredits": "off",
            "steel": "off",
            "titanium": "off",
            "plants": "off",
            "energy": "off",
            "heat": "off"
        },
        "protectedProduction": {
            "megacredits": "off",
            "steel": "off",
            "titanium": "off",
            "plants": "off",
            "energy": "off",
            "heat": "off"
        },
        "tableau": [],
        "selfReplicatingRobotsCards": [],
        "steel": 0,
        "steelProduction": 0,
        "steelValue": 2,
        "tags": {
            "building": 0,
            "space": 0,
            "science": 0,
            "power": 0,
            "earth": 0,
            "jovian": 0,
            "venus": 0,
            "moon": 0,
            "mars": 0,
            "plant": 0,
            "microbe": 0,
            "animal": 0,
            "crime": 0,
            "city": 0,
            "wild": 0,
            "clone": 0,
            "event": 0
        },
        "terraformRating": 20,
        "timer": {
            "sumElapsed": 0,
            "startedAt": 1768315276646,
            "running": true,
            "afterFirstAction": false,
            "lastStoppedAt": 1768315276646
        },
        "titanium": 0,
        "titaniumProduction": 0,
        "titaniumValue": 3,
        "tradesThisGeneration": 0,
        "corruption": 0,
        "victoryPointsBreakdown": {
            "terraformRating": 20,
            "milestones": 0,
            "awards": 0,
            "greenery": 0,
            "city": 0,
            "escapeVelocity": 0,
            "moonHabitats": 0,
            "moonMines": 0,
            "moonRoads": 0,
            "planetaryTracks": 0,
            "victoryPoints": 0,
            "total": 20,
            "detailsCards": [],
            "detailsMilestones": [],
            "detailsAwards": [],
            "detailsPlanetaryTracks": [],
            "negativeVP": 0
        },
        "victoryPointsByGeneration": [],
        "excavations": 0
    },
    "waitingFor": {
        "title": " ",
        "buttonLabel": "Start",
        "type": "initialCards",
        "options": [
            {
                "title": "Select corporation",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {
                        "resources": 0,
                        "name": "Spire",
                        "calculatedCost": 0
                    },
                    {
                        "resources": 0,
                        "name": "Robinson Industries",
                        "calculatedCost": 0
                    }
                ],
                "max": 1,
                "min": 1,
                "showOnlyInLearnerMode": false,
                "selectBlueCardAction": false,
                "showOwner": false
            },
            {
                "title": "Select 2 Prelude cards",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {
                        "resources": 0,
                        "name": "Colony Trade Hub",
                        "calculatedCost": 0
                    },
                    {
                        "resources": 0,
                        "name": "Venus Contract",
                        "calculatedCost": 0
                    },
                    {
                        "resources": 0,
                        "name": "Smelting Plant",
                        "calculatedCost": 0
                    },
                    {
                        "resources": 0,
                        "name": "Giant Solar Collector",
                        "calculatedCost": 0
                    }
                ],
                "max": 2,
                "min": 2,
                "showOnlyInLearnerMode": false,
                "selectBlueCardAction": false,
                "showOwner": false
            },
            {
                "title": "Select initial cards to buy",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {
                        "resources": 0,
                        "name": "Teslaract",
                        "calculatedCost": 14
                    },
                    {
                        "resources": 0,
                        "name": "Neutralizer Factory",
                        "calculatedCost": 7
                    },
                    {
                        "resources": 0,
                        "name": "Fusion Power",
                        "calculatedCost": 14
                    },
                    {
                        "resources": 0,
                        "name": "Lichen",
                        "calculatedCost": 7
                    },
                    {
                        "resources": 0,
                        "name": "Vesta Shipyard",
                        "calculatedCost": 15
                    },
                    {
                        "resources": 0,
                        "name": "Community Services",
                        "calculatedCost": 13
                    },
                    {
                        "resources": 0,
                        "name": "Homeostasis Bureau",
                        "calculatedCost": 16
                    },
                    {
                        "resources": 0,
                        "name": "Cryo-Sleep",
                        "calculatedCost": 10
                    },
                    {
                        "resources": 0,
                        "name": "Regolith Eaters",
                        "calculatedCost": 13
                    },
                    {
                        "resources": 0,
                        "name": "Meat Industry",
                        "calculatedCost": 5
                    }
                ],
                "max": 10,
                "min": 0,
                "showOnlyInLearnerMode": false,
                "selectBlueCardAction": false,
                "showOwner": false
            }
        ]
    },
    "players": [
        {
            "actionsTakenThisRound": 0,
            "actionsTakenThisGame": 0,
            "actionsThisGeneration": [],
            "availableBlueCardActionCount": 0,
            "cardCost": 3,
            "cardDiscount": 0,
            "cardsInHandNbr": 0,
            "citiesCount": 0,
            "coloniesCount": 0,
            "color": "red",
            "energy": 0,
            "energyProduction": 0,
            "fleetSize": 1,
            "heat": 0,
            "heatProduction": 0,
            "influence": 0,
            "isActive": true,
            "megaCredits": 0,
            "megaCreditProduction": 0,
            "name": "Red",
            "needsToResearch": true,
            "noTagsCount": 0,
            "plants": 0,
            "plantProduction": 0,
            "protectedResources": {
                "megacredits": "off",
                "steel": "off",
                "titanium": "off",
                "plants": "off",
                "energy": "off",
                "heat": "off"
            },
            "protectedProduction": {
                "megacredits": "off",
                "steel": "off",
                "titanium": "off",
                "plants": "off",
                "energy": "off",
                "heat": "off"
            },
            "tableau": [],
            "selfReplicatingRobotsCards": [],
            "steel": 0,
            "steelProduction": 0,
            "steelValue": 2,
            "tags": {
                "building": 0,
                "space": 0,
                "science": 0,
                "power": 0,
                "earth": 0,
                "jovian": 0,
                "venus": 0,
                "moon": 0,
                "mars": 0,
                "plant": 0,
                "microbe": 0,
                "animal": 0,
                "crime": 0,
                "city": 0,
                "wild": 0,
                "clone": 0,
                "event": 0
            },
            "terraformRating": 20,
            "timer": {
                "sumElapsed": 0,
                "startedAt": 1768315276646,
                "running": true,
                "afterFirstAction": false,
                "lastStoppedAt": 1768315276646
            },
            "titanium": 0,
            "titaniumProduction": 0,
            "titaniumValue": 3,
            "tradesThisGeneration": 0,
            "corruption": 0,
            "victoryPointsBreakdown": {
                "terraformRating": 20,
                "milestones": 0,
                "awards": 0,
                "greenery": 0,
                "city": 0,
                "escapeVelocity": 0,
                "moonHabitats": 0,
                "moonMines": 0,
                "moonRoads": 0,
                "planetaryTracks": 0,
                "victoryPoints": 0,
                "total": 20,
                "detailsCards": [],
                "detailsMilestones": [],
                "detailsAwards": [],
                "detailsPlanetaryTracks": [],
                "negativeVP": 0
            },
            "victoryPointsByGeneration": [],
            "excavations": 0
        },
        {
            "actionsTakenThisRound": 0,
            "actionsTakenThisGame": 0,
            "actionsThisGeneration": [],
            "availableBlueCardActionCount": 0,
            "cardCost": 3,
            "cardDiscount": 0,
            "cardsInHandNbr": 0,
            "citiesCount": 0,
            "coloniesCount": 0,
            "color": "green",
            "energy": 0,
            "energyProduction": 0,
            "fleetSize": 1,
            "heat": 0,
            "heatProduction": 0,
            "influence": 0,
            "isActive": false,
            "megaCredits": 0,
            "megaCreditProduction": 0,
            "name": "Green",
            "needsToResearch": true,
            "noTagsCount": 0,
            "plants": 0,
            "plantProduction": 0,
            "protectedResources": {
                "megacredits": "off",
                "steel": "off",
                "titanium": "off",
                "plants": "off",
                "energy": "off",
                "heat": "off"
            },
            "protectedProduction": {
                "megacredits": "off",
                "steel": "off",
                "titanium": "off",
                "plants": "off",
                "energy": "off",
                "heat": "off"
            },
            "tableau": [],
            "selfReplicatingRobotsCards": [],
            "steel": 0,
            "steelProduction": 0,
            "steelValue": 2,
            "tags": {
                "building": 0,
                "space": 0,
                "science": 0,
                "power": 0,
                "earth": 0,
                "jovian": 0,
                "venus": 0,
                "moon": 0,
                "mars": 0,
                "plant": 0,
                "microbe": 0,
                "animal": 0,
                "crime": 0,
                "city": 0,
                "wild": 0,
                "clone": 0,
                "event": 0
            },
            "terraformRating": 20,
            "timer": {
                "sumElapsed": 0,
                "startedAt": 1768315276646,
                "running": true,
                "afterFirstAction": false,
                "lastStoppedAt": 1768315276646
            },
            "titanium": 0,
            "titaniumProduction": 0,
            "titaniumValue": 3,
            "tradesThisGeneration": 0,
            "corruption": 0,
            "victoryPointsBreakdown": {
                "terraformRating": 20,
                "milestones": 0,
                "awards": 0,
                "greenery": 0,
                "city": 0,
                "escapeVelocity": 0,
                "moonHabitats": 0,
                "moonMines": 0,
                "moonRoads": 0,
                "planetaryTracks": 0,
                "victoryPoints": 0,
                "total": 20,
                "detailsCards": [],
                "detailsMilestones": [],
                "detailsAwards": [],
                "detailsPlanetaryTracks": [],
                "negativeVP": 0
            },
            "victoryPointsByGeneration": [],
            "excavations": 0
        }
    ],
    "autopass": false
}
"""
#waiting_for={}
#player_id="p2238ead19a6b"

# player_state=json.loads(player_state)
# env=TerraformingMarsEnv(["1","2"],init_from_player_state=True,player_state=player_state if not player_id else None,player_id=player_id,waiting_for=waiting_for)
# print(env.player_states['1'].get('waitingFor'))
# i=0
# for id,action in env.action_lookup['1'].items():
#     if i>30:
#         break
#     i=i+1
#     print(f"Action id: {id}, action: {json.dumps(action)}")
# exit(0)

#env.step({'1':0,'2':0})
#if NON_STOP:




env=TerraformingMarsEnv(False,player_state,None,safe_mode=False)
MAX_ROWS=10000
num_actions=MAX_ACTIONS

#with open("")
rand=random.Random()
i=0
next_obs, rewards, terms, truncs, infos=(None,None,None,None,None)
while True:
    is_no_actions=True
    agent="1"
    max_actions=len(env.action_lookup.keys())
    action=0
    if max_actions>1:
        action=rand.choice(list(env.action_lookup.keys()))
        is_no_actions=False
    elif max_actions==1:
        action=rand.choice(list(env.action_lookup.keys()))
    else:
        is_no_actions=False
        action=0
    #print(f"F {agent}:{actions[agent]}/{max_actions}")
    if terms:
        terms=None
        env=TerraformingMarsEnv(False,None,None,None,safe_mode=False)
    else:
        next_obs, rewards, terms, truncs, infos=env.step(action)
        #result[i]=next_obs['1']
        #obs_new=decode_observation(next_obs)
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break
        #result[i]=next_obs['2']
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break   
        #print(f"Encoder data step done {i}")

for agent in env:
    print(f"player_link={SERVER_BASE_URL[env.server_id]}/player?id={env.agent_id_to_player_id[agent]}")
    
#np.save("test.npy",result)