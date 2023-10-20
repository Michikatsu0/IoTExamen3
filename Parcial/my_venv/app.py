import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# Lectura y preparación de datos iniciales
archivo = "c:/Users/Escan/Documents/Parcial/my_venv/Datos_SIATA_Aire_pm25.json"
datos = pd.read_json(archivo, convert_dates=True)

latitudes = datos.latitud.values.tolist()
longitudes = datos.longitud.values.tolist()

xlonMin = min(longitudes)
xlonMax = max(longitudes)
ylatMin = min(latitudes)
ylatMax = max(latitudes)

grid_x, grid_y = np.meshgrid(np.linspace(xlonMin, xlonMax, 101), np.linspace(ylatMin, ylatMax, 101))

def generate_figure_for_hour(k):
    local_latitudes = []
    local_longitudes = []
    medida = []
    fecha = []
    for i in range(21):
        if datos.datos[i][k].get('valor') < 600 and datos.datos[i][k].get('valor') >= 0:
            local_latitudes.append(datos.latitud[i])
            local_longitudes.append(datos.longitud[i])
            fecha.append(datos.datos[i][k].get('fecha'))
            a = datos.datos[i][k].get('valor')

            if a <= 12:
                Conl = 0
                Conh = 12
                AQIl = 0
                AQIh = 50
            elif 12.1 < a <= 35.4:
                Conl = 12.1
                Conh = 35.4
                AQIl = 51
                AQIh = 100
            elif 35.5 <= a <= 55.4:
                Conl = 35.5
                Conh = 55.4
                AQIl = 101
                AQIh = 150
            elif 55.5 < a <= 150.4:
                Conl = 55.5
                Conh = 150.4
                AQIl = 151
                AQIh = 200
            elif 150.5 < a < 250.4:
                Conl = 150.5
                Conh = 250.4
                AQIl = 201
                AQIh = 300
            elif 250.5 < a < 500.4:
                Conl = 250.5
                Conh = 500.4
                AQIl = 301
                AQIh = 500

            AQI = (((AQIh - AQIl) * (a - Conl)) / (Conh - Conl)) + AQIl
            medida.append(AQI)
            fecha.append(fecha)

    n = np.array(medida)
    grid_z0 = griddata((local_latitudes, local_longitudes), n, (grid_y, grid_x), method='nearest')
    grid_z2 = griddata((local_latitudes, local_longitudes), n, (grid_y, grid_x), method='cubic')

    rows = grid_z0.shape[0]
    cols = grid_z0.shape[1]
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            if np.isnan(grid_z2[x, y]):
                grid_z2[x, y] = grid_z0[x, y]

    # Creación de la figura
    fig = go.Figure(go.Scattermapbox(
        lon=grid_x.reshape(-1),
        lat=grid_y.reshape(-1),
        text=grid_z2.reshape(-1),
        mode='markers',
        marker=dict(
            size=8,
            color=grid_z2.reshape(-1),
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=6.2442, lon=-75.5812),  # Coordenadas de Medellín
            zoom=10
        )
    )
    
    return fig

# Creación de la aplicación Dash y su diseño
initial_fig = generate_figure_for_hour(0)

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=initial_fig, id="my_map"),
    dcc.Slider(min=1, max=569, step=1, value=1, id="my_slider", marks={i: str(i) for i in range(569+1)})
])

@app.callback(Output('my_map', 'figure'), [Input('my_slider', 'value')])
def update_map(hour):
    return generate_figure_for_hour(hour)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, host='0.0.0.0', port=80)
