import json
import os
import random
MAX_ACTIONS=int(os.getenv('MAX_ACTIONS', "64"))
ONE_ACTION_ARRAY_SIZE=int(os.getenv('ONE_ACTION_ARRAY_SIZE', "64"))
MAX_GAME_FEATURES_SIZE=int(os.getenv('MAX_GAME_FEATURES_SIZE', "1024000"))
TOTAL_ACTIONS=MAX_ACTIONS+1
SERVER_BASE_URL= [
    'http://localhost:8081',
    'http://localhost:8082',
    'http://localhost:8083',
    'http://localhost:8084',
 ] #os.environ.get('SERVER_BASE_URL','http://localhost:9976') #,"http://lev-rworker-3:9976")
PLAYER_COLORS=['red','green','blue','orange','yellow','black']

ALL_CARDS={}
with open("cards.json",'r',encoding='utf-8') as f:
    cards=json.loads(f.read())
    for card in cards:
        ALL_CARDS[card['name']]=card
GAME_START_JSON={
    "players": [
        {
            "name": "Red",
            "color": "red",
            "beginner": False,
            "handicap": 0,
            "first": False
        },
        {
            "name": "Green",
            "color": "green",
            "beginner": False,
            "handicap": 0,
            "first": False
        }
    ],
    "expansions": {
        "corpera": True,
        "promo": True,
        "venus": True,
        "colonies": True,
        "prelude": True,
        "prelude2": True,
        "turmoil": True,
        "community": False,
        "ares": False,
        "moon": False,
        "pathfinders": False,
        "ceo": False,
        "starwars": False,
        "underworld": False
    },
    "draftVariant": True,
    "showOtherPlayersVP": True,
    "customCorporationsList": [],
    "customColoniesList": [],
    "customPreludes": [],
    "bannedCards": [],
    "includedCards": [],
    "board": "tharsis",
    "seed": random.random(),
    "solarPhaseOption": True,
    "aresExtremeVariant": False,
    "politicalAgendasExtension": "Standard",
    "undoOption": False,
    "showTimers": False,
    "fastModeOption": False,
    "removeNegativeGlobalEventsOption": False,
    "includeFanMA": False,
    "modularMA": False,
    "startingCorporations": 2,
    "soloTR": False,
    "initialDraft": False,
    "preludeDraftVariant": True,
    "randomMA": "No randomization",
    "shuffleMapOption": False,
    "randomFirstPlayer": True,
    "requiresVenusTrackCompletion": False,
    "requiresMoonTrackCompletion": False,
    "moonStandardProjectVariant": False,
    "moonStandardProjectVariant1": False,
    "altVenusBoard": False,
    "escapeVelocityMode": False,
    "escapeVelocityBonusSeconds": 2,
    "twoCorpsVariant": False,
    "customCeos": [],
    "startingCeos": 3,
    "startingPreludes": 4
}