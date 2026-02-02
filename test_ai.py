from decision_mapper import TerraformingMarsDecisionMapper
ds=TerraformingMarsDecisionMapper()
original_player_input={
    "type": "or",
    "index": 0,
    "response": {
        "type": "projectCard",
        "card": "Orbital Cleanup",
        "payment": {
            "__deferred_action": {
                "xid": "b539b217-e75f-45cd-a791-0d06e4288ff7",
                "xtype": "xpayment",
                "xoptions": [
                    {
                        "megaCredits": 14,
                        "heat": 0,
                        "steel": 0,
                        "titanium": 0,
                        "plants": 0,
                        "microbes": 0,
                        "floaters": 0,
                        "lunaArchivesScience": 0,
                        "spireScience": 0,
                        "seeds": 0,
                        "auroraiData": 0,
                        "graphene": 0,
                        "kuiperAsteroids": 0,
                        "corruption": 0
                    }
                ]
            }
        }
    }
}
res=ds.generate_action_space({},{},True,original_player_input=original_player_input)
for r in res.values():
    print(r)
    
print(original_player_input)