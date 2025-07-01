import json
import os
MAX_ACTIONS=int(os.getenv('MAX_ACTIONS', "102400"))
ONE_ACTION_ARRAY_SIZE=int(os.getenv('ONE_ACTION_ARRAY_SIZE', "1024"))
MAX_GAME_FEATURES_SIZE=int(os.getenv('MAX_GAME_FEATURES_SIZE', "1024000"))
TOTAL_ACTIONS=MAX_ACTIONS+1

ALL_CARDS={}
with open("cards.json",'r',encoding='utf-8') as f:
    cards=json.loads(f.read())
    for card in cards:
        ALL_CARDS[card['name']]=card