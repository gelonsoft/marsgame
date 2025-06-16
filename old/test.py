import streamlit as st
import numpy as np
from old.env import TerraformingMarsEnv
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import plotly.io as pio

print("Started")
pio.templates.default = 'plotly'

# === Init ===
if 'env' not in st.session_state:
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()

env = st.session_state.env
current_player = env.players[env.current_player]

# === Header ===
st.title("🌍 Terraforming Mars Web UI")
st.markdown(f"### Generation: {env.generation} | Current Player: Player {env.current_player + 1}")

# === Tile Type Selector ===
tile_type = st.radio("Select Tile Type to Place:", ["greenery", "city", "ocean"], horizontal=True)

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

        if player['played_cards']:
            st.markdown("**Played Cards:**")
            for card in player['played_cards']:
                st.markdown(f"📄 **{card['name']}**")
        else:
            st.markdown("_No cards played yet._")

# === Global Parameters ===
st.markdown("### 🌐 Global Parameters")
st.text(f"Temperature: {env.global_parameters['temperature']}°C")
st.text(f"Oxygen: {env.global_parameters['oxygen']}%")
st.text(f"Oceans: {env.global_parameters['oceans']}/9")

# === Interactive Hex Grid ===
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

fig = go.Figure()

for x, y, row, col in hex_centers:
    verts = hexagon(x, y, HEX_RADIUS)
    xs, ys = zip(*verts)
    xs += (xs[0],)
    ys += (ys[0],)

    tile = env.tiles[row][col]
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

fig.update_layout(
    title="Terraforming Mars - Hex Grid",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=10, r=10, t=40, b=10),
    #height=700,
    width=800,
    clickmode='event+select'
)




#st.plotly_chart(fig, use_container_width=True)
selected_points = plotly_events(fig,key="sss",select_event=True,click_event=True,hover_event=False)

# === Placement Logic ===
if selected_points:
    print(selected_points)
    print(fig.data[selected_points[0]["curveNumber"]])
    row, col = fig.data[selected_points[0]["curveNumber"]]['customdata'][0] #selected_points[0]['customdata']
    
    result = env.step({'type': 'place_tile', 'tile_type': tile_type, 'position': (col, row)})
    if result == -1:
        st.error(f"Invalid placement for {tile_type} at ({row}, {col})")
    else:
        st.success(f"Placed {tile_type} at ({row}, {col})! +1 TR")

# === Turn Controls ===
col1, col2 = st.columns(2)
if col1.button("➡️ End Turn"):
    env.current_player = (env.current_player + 1) % env.num_players
if col2.button("🔄 Reset Game"):
    st.session_state.env = TerraformingMarsEnv(num_players=2)
    st.session_state.env.reset()
    st.experimental_rerun()
if st.button("🚫 Pass Turn"):
    env.step("pass")