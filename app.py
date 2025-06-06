#Web UI to 
import streamlit as st
import numpy as np
from env import TerraformingMarsEnv  # replace with your actual module
from streamlit_plotly_events2 import plotly_events
import plotly.graph_objects as go
import numpy as np
import random


print("Started")
def set_z():
    print("Set z")
    
# Initialize session state
if 'env' not in st.session_state:
    print("init")
    st.session_state.env = TerraformingMarsEnv(num_players=2,render_callback=set_z )
    st.session_state.z=1
    st.session_state.env.reset()

env = st.session_state.env
current_player = env.players[env.current_player]
player = env.players[env.current_player]

# === Header ===
st.title("🌍 Terraforming Mars Web UI")
st.markdown(f"### Generation: {env.generation} | Current Player: Player {env.current_player + 1}")

card_icons = {
    'Comet': '☄️',
    'Lichen': '🌿',
    'Nuclear Zone': '💥'
}



# === Player Dashboard ===
st.subheader("📊 Player Resources")
st.text(f"TR: {player['terraform_rating']} | MC: {player['mc']} | Heat: {player['heat']} | Plants: {player['plants']}")
st.text(f"Steel: {player['steel']} | Titanium: {player['titanium']} | Energy: {player['energy']}")
st.markdown(f"**Production:** {player['production']}")

# === Global Parameters ===
st.subheader("🌐 Global Parameters")
st.text(f"Temperature: {env.global_parameters['temperature']}°C")
st.text(f"Oxygen: {env.global_parameters['oxygen']}%")
st.text(f"Oceans: {env.global_parameters['oceans']}/9")

# === Cards in Hand ===
st.subheader("🃏 Cards in Hand")
if not player['hand']:
    st.info("You have no cards in hand.")
else:
    for card in player['hand']:
        playable = env.can_play_card(player, card)
        card_info = f"""**{card['name']}**  
Cost: {card['cost']} MC  
Effect: `{card['effects']}`  

Tags: {', '.join(card.get('tags', []))}"""
        st.markdown(card_info)
        if playable:
            if st.button(f"Play {card['name']}"):
                env.step({'type': 'draft_card', 'card': card})
                st.rerun()
        else:
            st.button(f"Cannot Play ({card['name']})", disabled=True, help="Check cost or requirements")

# === Played Cards ===
st.subheader("✅ Played Cards")
if not player['played_cards']:
    st.text("None yet.")
else:
    for card in player['played_cards']:
        effects_preview = ", ".join(
            f"{e['type']}({e.get('target', e.get('resource', e.get('tile', e.get('scope', ''))))}: {e.get('amount', '')})"
            for e in card.get("effects", [])
        )
        st.markdown(f"- **{card['name']}** ({card['type']}) [{', '.join(card.get('tags', []))}] → {effects_preview}")


# --- Standard Projects ---
st.markdown("#### Standard Projects")

if st.button("🔥 Asteroid (14 MC → Raise Temp)"):
    action = {'type': 'standard_project', 'name': 'asteroid'}
    env.step(action)

if st.button("⚡ Power Plant (11 MC → +1 Energy Prod)"):
    action = {'type': 'standard_project', 'name': 'power_plant'}
    env.step(action)

if st.button("💧 Aquifer (18 MC → Place Ocean)"):
    action = {'type': 'standard_project', 'name': 'aquifer'}
    env.step(action)


    
# === Tile Type Selector ===
tile_type = st.radio("Select Tile Type to Place:", ["greenery", "city", "ocean"], horizontal=True)
# === Interactive Hex Grid ===

@st.fragment
def render_tm():
    print("Render TM")
    st.markdown("### 🗺️ Terraforming Mars - Click to Place Tile")

    HEX_RADIUS = 2
    HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS
    NUM_ROWS = 5
    NUM_COLS = 9

    hex_centers = []
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            y = col * 1.5 * HEX_RADIUS
            x = row * HEX_HEIGHT + (HEX_HEIGHT / 2 if col % 2 else 0)
            hex_centers.append((x, y, row, col))

    def hexagon(x_center, y_center, radius):
        angle_offset = np.pi / 6
        return [
            (
                x_center + radius * np.cos(angle_offset + i * np.pi / 3),
                y_center + radius * np.sin(angle_offset + i * np.pi / 3)
            )
            for i in range(6)
        ]


    fig = go.Figure(layout=dict(    title="Terraforming Mars - Hex Grid",
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

        tile = env.tiles[row][col]
        if row==0 and col==0:
            print(f"Render tile={tile}")
        color = {
            'empty': 'lightgray',
            'greenery': 'green',
            'city': 'blue',
            'ocean': 'aqua'
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

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='text',
            text=[f"{row},{col}"],
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





    #st.plotly_chart(fig, use_container_width=True)
    selected_points = plotly_events(fig,select_event=False,click_event=True,hover_event=False)

    # === Placement Logic ===
    if selected_points:
        print(selected_points)
        #print(fig.data[selected_points[0]["curveNumber"]])
        row, col = fig.data[selected_points[0]["curveNumber"]]['customdata'][0] #selected_points[0]['customdata']
        
        observe,reward,done,info = env.step({'type': 'place_tile', 'tile_type': tile_type, 'position': (col, row)})
        print(f"observe={observe} reward={reward} done={done} info={info}")

        if reward == -1:
            st.error(f"Invalid placement for {tile_type} at ({row}, {col})")
        else:
            st.success(f"Placed {tile_type} at ({row}, {col})! +1 TR")
            selected_points=None
            st.rerun(scope="fragment")
                

                
                
render_tm()

#st.plotly_chart(fig, use_container_width=True)

# === End Turn / Reset ===
col1, col2 = st.columns(2)

if col1.button("➡️ End Turn"):
    env.current_player = (env.current_player + 1) % env.num_players

if col2.button("🔄 Reset Game"):
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()
    st.rerun()

if st.button("🚫 Pass Turn"):
    env.step("pass")
    
