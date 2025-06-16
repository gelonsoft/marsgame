#Web UI to 
import streamlit as st
import numpy as np
from old.env import TerraformingMarsEnv  # replace with your actual module
from streamlit_plotly_events2 import plotly_events
import plotly.graph_objects as go
import numpy as np
import random


print("Started")
def rerun_callback():
    print("Rerun callback triggered")
    st.rerun()
    
# Initialize session state
if 'env' not in st.session_state:
    print("init")
    st.session_state.env = TerraformingMarsEnv(num_players=2,render_callback=rerun_callback )
    st.session_state.place_tile=None
    st.session_state.env.reset()

env = st.session_state.env
current_player = env.players[env.current_player]
player = env.players[env.current_player]

# === Header ===
#🌍 Terraforming Mars Web UI
st.markdown(f"### Generation: {env.generation} | Current Player: Player {env.current_player + 1} | Actions: {env.current_player_actions_left}")
st.markdown(f"Phase: A={env.phase}")
# === Global Parameters ===
st.subheader("🌐 Global Parameters")
col1,col2,col3 = st.columns(3)
with col1:
    st.text(f"Temperature: {env.global_parameters['temperature']}°C")
with col2:
    st.text(f"Oxygen: {env.global_parameters['oxygen']}%")
with col3:
    st.text(f"Oceans: {env.global_parameters['oceans']}/9")
card_icons = {
    'Comet': '☄️',
    'Lichen': '🌿',
    'Nuclear Zone': '💥'
}

# === Player Dashboard ===
def draw_player(player,player_id,is_current_player):
    if is_current_player:
        st.subheader(f"Current player")
    else:
        st.subheader(f"Player #{player_id+1}")
    if player.get('corporation'):
        st.markdown(f"**Corporation:** 🏢 {player['corporation']}")
    st.text(f"TR: {player['terraform_rating']} | MC: {player['mc']} | Heat: {player['heat']} | Plants: {player['plants']}")
    st.text(f"Steel: {player['steel']} | Titanium: {player['titanium']} | Energy: {player['energy']}")
    st.markdown(f"**Production:** {player['production']}")

pcols = st.columns(env.num_players)
with pcols[0]:
    someplayer=player
    draw_player(player=player,player_id=env.current_player,is_current_player=True)

for i in range(env.num_players-1):
    j=0
    if j==env.current_player:
        j+=1
    with pcols[i+1]:
        draw_player(player=env.players[j],player_id=j,is_current_player=False)
        




def render_card(card):
    effect_text=f"\n- Effect: `{card.get('effects'),[]}`" if len(card.get('effects',[]))>0 else ""
    tags_text=f"\n- Tags: {', '.join(card.get('tags', []))}" if len(card.get('tags',[]))>0 else ""
    active_effect_text=f"\n- Active effects: `{card.get('active_effects',[])}`" if len(card.get('active_effects',[]))>0 else ""
    card_info = f"""**{card['name']}**:  
- Cost: {card['cost']} MC {effect_text}{active_effect_text}{tags_text}"""
    st.markdown(card_info)
# === Corporation choice ====
if env.choose_corporation_phase:
    st.subheader("🏢 Choose Corporation")
    for corp in player['corporation_choices']:
        with st.expander(corp['name']):
            st.markdown(f"**Starting MC:** {corp['mc']}")
            for effect in corp['effect']:
                st.markdown(f"- `{effect}`")
            if st.button(f"Choose {corp['name']}"):
                env.step({'type': 'choose_corporation', 'name': corp['name']})
                st.rerun()
                    
# === Draft Phase UI ===
if env.draft_phase:  # Only show for human player
    st.subheader("🃏 Draft Phase - Select 1 Card to Keep")

    for card in player['draft_hand']:
            render_card(card)
            if st.button(f"Buy {card['name']}"):
                env.step({'type': 'buy_card', 'card_name': card['name']})
                st.rerun()
    
    if st.button(f"End draft"):
        print("End draft")
        env.step({"type":"end_turn"})

# === Milestones ====
if not st.session_state.place_tile:
    st.subheader("🏁 Milestones")
    claimed = env.claimed_milestones
    cols=st.columns(len(env.milestones))
    i=0
    for name, condition in env.milestones.items():
        with cols[i]:
            i+=1
            claimed_by_any = name in claimed
            eligible = condition(player)
            if claimed_by_any:
                st.markdown(f"- ✅ {name} (claimed)")
            elif eligible and player['mc'] >= 8 and env.action_phase:
                if st.button(f"Claim Milestone: {name}"):
                    env.step({"type": "claim_milestone", "name": name})
                    st.rerun()
            elif env.action_phase:
                st.markdown(f"- ❌ {name} (not eligible or already claimed)")
            else:
                st.markdown(f"- 💰 {name}")

# === Awards ====
if not st.session_state.place_tile:
    st.subheader("🎯 Awards")
    funded = env.funded_awards
    cols=st.columns(len(env.awards))
    i=0
    for name in env.awards:
        with cols[i]:
            i=i+1
            funded_by_any = name in funded
            if funded_by_any:
                st.markdown(f"- 💰 {name} (funded)")
            elif player['mc'] >= 8 and env.action_phase:
                if st.button(f"Fund Award: {name}"):
                    env.step({"type": "fund_award", "name": name})
                    st.rerun()
            elif env.action_phase:
                st.markdown(f"- ❌ {name} (not enough MC or already funded)")
            else:
                st.markdown(f"- 💰 {name}")

# === Deffered actions ===
if env.deffered_actions_phase and not st.session_state.place_tile:
    st.markdown("#### Active Actions")
    for a in env.deffered_player_actions:
        if a['type']=="place_tile":
            if st.button(f"Place {a['tile']} tile"):
                st.session_state.place_tile=a
                st.rerun()
                

# === Standard Projects ===
if env.action_phase and not st.session_state.place_tile:
    st.markdown("#### Standard Projects")
    playable_standard_projects=[p for p in env.standard_projects if env.can_play_card(player, p)]
    cols=st.columns(len(playable_standard_projects))
    for i in range(len(playable_standard_projects)):
        with cols[i]:
            card=playable_standard_projects[i]
            render_card(card)
            if st.button(card['name']):
                action = {'type': 'play_card', 'card_name': card['name']}
                env.step(action)      

# === Cards in Hand ===
if not env.choose_corporation_phase:
    st.subheader("🃏 Cards in Hand")
    if not player['hand']:
        st.info("You have no cards in hand.")
    else:
        for card in player['hand']:
            playable = env.can_play_card(player, card)
            render_card(card)
            if env.action_phase:
                if playable:
                    if st.button(f"Play {card['name']}"):
                        env.step({'type': 'play_card', 'card_name': card['name']})
                        st.rerun()
                else:
                    st.button(f"Cannot Play ({card['name']})", disabled=True, help="Check cost or requirements")

# === Active cards  ===
if not env.choose_corporation_phase:
    st.subheader("⚙️ Active Cards")
    for card in player['played_cards']:
        if card.get('type') != 'active':
            continue

        render_card(card)

        # Show resource counters if present
        resources = card.get('resources', {})
        if resources:
            for rname, rvalue in resources.items():
                st.markdown(f"- Resource `{rname}`: **{rvalue}**")

        # Show action button if card has action trigger
        for ae in card.get('active_effects', []):
            if ae.get('trigger') == 'action':
                label = ae.get('description', 'Use Action')
                if env.action_phase:
                    if st.button(f"🛠 {label} ({card['name']})"):
                        env.step({'type': 'active_card_action', 'card_name': card['name']})

# === Played Cards ===
if not env.choose_corporation_phase:
    st.subheader("✅ Played Cards")
    if not player['played_cards']:
        st.text("None yet.")
    else:
        for card in player['played_cards']:
            if card.get('type') == 'active':
                continue
            render_card(card)

# === Tile Type Selector ===
#tile_type = st.radio("Select Tile Type to Place:", ["greenery", "city", "ocean"], horizontal=True)
# === Interactive Hex Grid ===

#@st.fragment

if not env.choose_corporation_phase:
    print("Render TM")
    st.markdown("### 🗺️ Terraforming Mars")
    if st.session_state.place_tile:
        st.markdown(f"## Place tile {st.session_state.place_tile['tile']}")

    HEX_RADIUS = 2
    HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS

    hex_centers = []
    for tile in env.map:
            y = tile['x'] * 1.5 * HEX_RADIUS
            x = tile['y'] * HEX_HEIGHT + (HEX_HEIGHT / 2 if tile['x'] % 2 else 0)
            hex_centers.append((x, y, tile['y'], tile['x']))

    def hexagon(x_center, y_center, radius):
        angle_offset = np.pi / 6
        return [
            (
                x_center + radius * np.cos(angle_offset + i * np.pi / 3),
                y_center + radius * np.sin(angle_offset + i * np.pi / 3)
            )
            for i in range(6)
        ]

    fig = go.Figure(layout=dict(   title="Terraforming Mars - Hex Grid",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        #height=700,
        width=800,
        clickmode='event+select'))
    

    for x, y, row, col in hex_centers:
        
        verts = hexagon(x, y, HEX_RADIUS)
        xs, ys = zip(*verts)
        xs += (xs[0],)
        ys += (ys[0],)

        tile = env.get_map_tile_by_coord(col,row)
        if row==0 and col==0:
            print(f"Render tile={tile}")
        color = {
            'empty': 'lightgray',
            'greenery': 'green',
            'city': 'blue',
            'ocean': 'aqua',
            'ocean_area': 'teal'
        }.get(tile['type'], 'gray')

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color='black'),
            hoverinfo='text',
            text=f"({row},{col})\n{tile['type']}",
            customdata=[(row, col)] * len(xs),
            showlegend=False
        ))
        fig_text=f"{row},{col}|"
        for r in tile.get('resources',[]):
            fig_text+=f"{r['resource'][0]}{r['amount']}"
        if tile.get('special'):
            fig_text+=tile.get('special')[0]
        #fig_text=fig_text[:-1]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='text',
            text=[fig_text],
            customdata=[(row, col)],
            showlegend=False
        ))

    fig.layout.template = None # to slim down the output
    
    #fig.update_layout(
    #    title="Terraforming Mars - Hex Grid",
    #    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    #    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    #    margin=dict(l=10, r=10, t=40, b=10),
    #    #height=700,
    #    width=800,
    #    clickmode='event+select'
    #)

    #Clear figure 
    if not st.session_state.place_tile:
        st.plotly_chart(fig)
    
    #st.plotly_chart(fig, use_container_width=True)
    if st.session_state.place_tile:
        print("Place tile active, activating events")
        action=st.session_state.place_tile
        selected_points = plotly_events(fig,select_event=False,click_event=True,hover_event=False,key=action['id'])
        fig.update_layout(
            height=700
        )
        tile_type=action['tile']
        # === Placement Logic ===
        if selected_points:
            print(f"Selected points: {selected_points}")
            #print(fig.data[selected_points[0]["curveNumber"]])
            row, col = fig.data[selected_points[0]["curveNumber"]]['customdata'][0] #selected_points[0]['customdata']
            
            observe,reward,done,info = env.step({'type': 'place_tile', 'tile_type': tile_type, 'position': (col, row),'id':action['id']})
            print(f"observe={observe} reward={reward} done={done} info={info}")

            if reward == -1:
                st.error(f"Invalid placement for {tile_type} at ({row}, {col})")
            else:
                st.success(f"Placed {tile_type} at ({row}, {col})! +1 TR")
                selected_points=None
                print(f"Set place_tile to None")
                st.session_state.place_tile=None
                print(f"Set place_tile to None done")
                st.rerun()
                
#render_tm()

#st.plotly_chart(fig, use_container_width=True)

# === End Turn / Reset ===
col1, col2 = st.columns(2)

if env.action_phase:
    if env.current_player_actions_left<2:
        if col1.button("➡️ End Turn"):
            env.step({"type":"end_turn"})
            #env.current_player = (env.current_player + 1) % env.num_players
    if st.button("🚫 Pass Turn"):
        env.step("pass")

if col2.button("🔄 Reset Game"):
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()
    st.rerun()