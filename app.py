#Web UI to 
import streamlit as st
import numpy as np
from env import TerraformingMarsEnv  # replace with your actual module
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import numpy as np

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()

env = st.session_state.env
current_player = env.players[env.current_player]

# === Header ===
st.title("🌍 Terraforming Mars Web UI")
st.markdown(f"### Generation: {env.generation} | Current Player: Player {env.current_player + 1}")

card_icons = {
    'Comet': '☄️',
    'Lichen': '🌿',
    'Nuclear Zone': '💥'
}

# === Player Dashboards ===
cols = st.columns(env.num_players)
for i, col in enumerate(cols):
    player = env.players[i]
    with col:
        st.subheader(f"Player {i + 1}")
        st.text(f"TR: {player['terraform_rating']}")
        st.text(f"MC: {player['mc']} | Heat: {player['heat']} | Plants: {player['plants']}")
        st.text(f"Steel: {player['steel']} | Titanium: {player['titanium']}")
        st.text(f"Energy: {player['energy']}")
        st.markdown(f"Production: {player['production']}")

        # ✅ Add this section for played cards
        if player['played_cards']:
            st.markdown("**Played Cards:**")
            for card in player['played_cards']:
                icon = card_icons.get(card['name'], '📄')
                st.markdown(f"{icon} **{card['name']}**")
        else:
            st.markdown("_No cards played yet._")


# === Global State ===
st.markdown("### 🌐 Global Parameters")
st.text(f"Temperature: {env.global_parameters['temperature']}°C")
st.text(f"Oxygen: {env.global_parameters['oxygen']}%")
st.text(f"Oceans: {env.global_parameters['oceans']}/9")

# === Action Buttons ===
st.markdown("## 🛠️ Actions")

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

# --- Cards (simplified) ---
st.markdown("#### Project Cards")

if st.button("🌠 Play 'Comet' (21 MC → Raise Temp)"):
    action = {'type': 'play_card', 'card_name': 'Comet'}
    env.step(action)

if st.button("🌿 Play 'Lichen' (7 MC → +1 Plant Prod)"):
    action = {'type': 'play_card', 'card_name': 'Lichen'}
    env.step(action)

# --- Tile Placement (simplified positions) ---
st.markdown("#### Tile Placement")

if st.button("🟩 Place Greenery at (2, 2)"):
    action = {'type': 'place_tile', 'tile_type': 'greenery', 'position': (2, 2)}
    env.step(action)

if st.button("🏙️ Place City at (3, 3)"):
    action = {'type': 'place_tile', 'tile_type': 'city', 'position': (3, 3)}
    env.step(action)

# === End Turn / Reset ===
col1, col2 = st.columns(2)

if col1.button("➡️ End Turn"):
    env.current_player = (env.current_player + 1) % env.num_players

if col2.button("🔄 Reset Game"):
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()
    st.experimental_rerun()

if st.button("🚫 Pass Turn"):
    env.step("pass")
    
   
st.markdown("### 🗺️ Interactive Tile Map")

fig = go.Figure()

colors = {
    'empty': '#ffffff',
    'city': '#9999ff',
    'greenery': '#88cc88',
    'ocean': '#66ccff'
}

symbol_map = {
    None: '',
    0: "P1",
    1: "P2"
}

# Draw hexagons
for y in range(env.map_height):
    for x in range(env.map_width):
        tile = env.tiles[y][x]
        tile_type = tile['type']
        owner = tile['owner']
        color = colors.get(tile_type, '#eeeeee')

        fig.add_shape(
            type="rect",
            x0=x, y0=y, x1=x+1, y1=y+1,
            line=dict(color="gray"),
            fillcolor=color
        )
        fig.add_trace(go.Scatter(
            x=[x+0.5],
            y=[y+0.5],
            text=[symbol_map.get(owner)],
            mode="text",
            showlegend=False
        ))

fig.update_layout(
    width=600,
    height=400,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)

click_data = st.session_state.get('click_data')
clicked_tile = st.session_state.get('clicked_tile', None)

if click_data := st.session_state.get('plotly_click'):
    point = click_data['points'][0]
    x = int(point['x'])
    y = int(point['y'])
    clicked_tile = (x, y)
    st.session_state['clicked_tile'] = clicked_tile
    
click_data = plotly_events(fig, click_event=True, hover_event=False)
if click_data:
    point = click_data[0]
    x = int(point['x'])
    y = int(point['y'])
    st.session_state['clicked_tile'] = (x, y)

# Show click + action buttons
if clicked_tile:
    st.markdown(f"Clicked tile: {clicked_tile}")

    col1, col2, col3 = st.columns(3)
    if col1.button("Place City Here"):
        env.step({'type': 'place_tile', 'tile_type': 'city', 'position': clicked_tile})
    if col2.button("Place Greenery Here"):
        env.step({'type': 'place_tile', 'tile_type': 'greenery', 'position': clicked_tile})
    if col3.button("Place Ocean Here"):
        env.step({'type': 'place_tile', 'tile_type': 'ocean', 'position': clicked_tile})


# Hexagonal Map Rendering
st.markdown("### 🗺️ Terraforming Mars - Hex Map")

HEX_RADIUS = 1
HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS
NUM_ROWS = 9
NUM_COLS = 9

hex_centers = []
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        if (row + col) % 2 == 0:
            x = col * 1.5 * HEX_RADIUS
            y = row * HEX_HEIGHT + (HEX_HEIGHT / 2 if col % 2 else 0)
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

fig = go.Figure()

for x, y, row, col in hex_centers:
    verts = hexagon(x, y, HEX_RADIUS)
    xs, ys = zip(*verts)
    xs += (xs[0],)
    ys += (ys[0],)

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='lines',
        fill='toself',
        line=dict(color='gray'),
        hoverinfo='text',
        text=f"({row},{col})",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='text',
        text=[f"{row},{col}"],
        textposition="middle center",
        showlegend=False
    ))

fig.update_layout(
    title="Terraforming Mars - Hex Grid Layout",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=10, r=10, t=40, b=10),
    height=700,
    width=800
)

st.plotly_chart(fig, use_container_width=True)