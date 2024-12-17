import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
import os

# Load initial data
df = pd.read_csv('extended_test_data.csv')


def calculate_rag_details(sub_df):
    above_threshold_count = (sub_df['Risk Value'] > sub_df['Threshold']).sum()
    total = len(sub_df)
    ratio = above_threshold_count / total if total > 0 else 0
    if ratio > 2 / 3:
        rag = 'Red'
    elif ratio > 1 / 3:
        rag = 'Amber'
    else:
        rag = 'Green'
    return pd.Series({
        'RAG': rag,
        'above_threshold_count': above_threshold_count,
        'total': total,
        'ratio': ratio
    })


def get_aggregated_rag(df, group_cols):
    grouped = df.groupby(group_cols).apply(calculate_rag_details).reset_index()
    return grouped


def cell_color_classification(value, threshold):
    diff = abs(value - threshold)
    if value > threshold:
        return 'High'
    elif diff <= 5:
        return 'Medium'
    else:
        return 'Low'


def cell_color(row):
    diff = abs(row['Risk Value'] - row['Threshold'])
    if row['Risk Value'] > row['Threshold']:
        return 'red'
    elif diff <= 5:
        return 'orange'
    else:
        return 'green'


def color_to_code(color):
    if color == 'red':
        return 2
    elif color == 'orange':
        return 1
    else:
        return 0


# Precompute RAG at each level
state_rag = get_aggregated_rag(df, ['State'])
county_rag = get_aggregated_rag(df, ['State', 'County'])
muni_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality'])
district_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality', 'District'])


def get_tile_tooltip(level_col, row, df):
    if level_col == 'State':
        filtered = df[df['State'] == row['State']]
    elif level_col == 'County':
        filtered = df[(df['State'] == row['State']) & (df['County'] == row['County'])]
    elif level_col == 'Municipality':
        filtered = df[(df['State'] == row['State']) &
                      (df['County'] == row['County']) &
                      (df['Municipality'] == row['Municipality'])]
    elif level_col == 'District':
        filtered = df[(df['State'] == row['State']) &
                      (df['County'] == row['County']) &
                      (df['Municipality'] == row['Municipality']) &
                      (df['District'] == row['District'])]
    else:
        filtered = df.copy()

    agg = filtered.groupby('Key Risk Indicator').agg({'Risk Value': 'mean', 'Threshold': 'mean'}).reset_index()

    lines = []
    for _, r in agg.iterrows():
        classification = cell_color_classification(r['Risk Value'], r['Threshold'])
        lines.append(f"{r['Key Risk Indicator']}: {classification}")

    return "\n".join(lines)


def create_tiles(dataframe, level_col):
    tiles = []
    for _, row in dataframe.iterrows():
        tooltip = get_tile_tooltip(level_col, row, df)
        rag = row['RAG']
        color = 'red' if rag == 'Red' else 'orange' if rag == 'Amber' else 'green'

        tiles.append(
            html.Button(
                row[level_col],
                id={
                    'type': 'tile',
                    'level': level_col,
                    'value': row[level_col]
                },
                n_clicks=0,
                title=tooltip,
                className='tile-button',
                style={
                    'backgroundColor': color,
                    'color': 'white',
                    'border': 'none',
                    'padding': '20px',
                    'margin': '10px',
                    'fontWeight': 'bold',
                    'width': '200px',
                    'height': '100px',
                    'textAlign': 'center',
                    'cursor': 'pointer'
                }
            )
        )
    return tiles


app = dash.Dash(__name__)
app.title = "Risk Dashboard"

app.layout = html.Div([
    html.H1("Risk Dashboard", id='page-title'),
    dcc.Store(id='selected-state'),
    dcc.Store(id='selected-county'),
    dcc.Store(id='selected-municipality'),
    dcc.Store(id='selected-district'),
    # view-mode can be 'hierarchy', 'all_indicators', or 'onboard'
    dcc.Store(id='view-mode', data='hierarchy'),
    dcc.Store(id='sort-by-color-toggle', data=False),

    html.Div([
        html.Button("Back to States", id='back-to-states', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Counties", id='back-to-counties', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Municipalities", id='back-to-municipalities', n_clicks=0, style={'display': 'none'}),
        html.Button("Show All Indicators", id='show-all-indicators', n_clicks=0),
        html.Button("Back to Hierarchy", id='back-to-hierarchy', n_clicks=0,
                    style={'display': 'none', 'margin-left': '10px'}),
        html.Button("Sort by Color", id='sort-by-color', n_clicks=0, style={'margin-left': '10px'}),
        html.Button("Onboard KRI", id='onboard-kri', n_clicks=0, style={'margin-left': '10px'}),
    ], style={'margin-bottom': '20px'}),

    html.Div(id='content-container', style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'start'
    }),

    html.Div(id='final-table-container'),

    # Onboarding form
    html.Div(
        id='onboard-container',
        style={'display': 'none'},
        children=[
            html.H2("Onboard New Key Risk Indicator"),
            html.Button("Back to Dashboard", id='back-to-dashboard', n_clicks=0),
            html.Br(),
            html.Label("Select State:"),
            dcc.Dropdown(
                id='onboard-state',
                options=[{'label': s, 'value': s} for s in sorted(df['State'].unique())],
                placeholder="Select a State"
            ),
            html.Br(),
            html.Label("Select County:"),
            dcc.Dropdown(id='onboard-county', placeholder="Select a County"),
            html.Br(),
            html.Label("Select Municipality:"),
            dcc.Dropdown(id='onboard-municipality', placeholder="Select a Municipality"),
            html.Br(),
            html.Label("Select District:"),
            dcc.Dropdown(id='onboard-district', placeholder="Select a District"),
            html.Br(),
            html.Label("Key Risk Indicator:"),
            dcc.Input(id='onboard-kri-name', type='text', placeholder='Enter KRI Name'),
            html.Br(), html.Br(),
            html.Label("Threshold:"),
            dcc.Input(id='onboard-threshold', type='number', placeholder='Enter Threshold'),
            html.Br(), html.Br(),
            html.Label("Risk Value:"),
            dcc.Input(id='onboard-risk-value', type='number', placeholder='Enter Risk Value'),
            html.Br(), html.Br(),
            html.Button("Submit", id='onboard-submit', n_clicks=0),
            html.Div(id='onboard-status', style={'color': 'green', 'marginTop': '10px'})
        ]
    )
])


@app.callback(
    Output('onboard-county', 'options'),
    Input('onboard-state', 'value')
)
def update_county_options(selected_state):
    if selected_state:
        filtered = df[df['State'] == selected_state]
        counties = sorted(filtered['County'].unique())
        return [{'label': c, 'value': c} for c in counties]
    return []


@app.callback(
    Output('onboard-municipality', 'options'),
    Input('onboard-state', 'value'),
    Input('onboard-county', 'value')
)
def update_municipality_options(selected_state, selected_county):
    if selected_state and selected_county:
        filtered = df[(df['State'] == selected_state) & (df['County'] == selected_county)]
        municipalities = sorted(filtered['Municipality'].unique())
        return [{'label': m, 'value': m} for m in municipalities]
    return []


@app.callback(
    Output('onboard-district', 'options'),
    Input('onboard-state', 'value'),
    Input('onboard-county', 'value'),
    Input('onboard-municipality', 'value')
)
def update_district_options(selected_state, selected_county, selected_muni):
    if selected_state and selected_county and selected_muni:
        filtered = df[
            (df['State'] == selected_state) & (df['County'] == selected_county) & (df['Municipality'] == selected_muni)]
        districts = sorted(filtered['District'].unique())
        return [{'label': d, 'value': d} for d in districts]
    return []


@app.callback(
    Output('selected-state', 'data'),
    Output('selected-county', 'data'),
    Output('selected-municipality', 'data'),
    Output('selected-district', 'data'),
    Output('back-to-states', 'style'),
    Output('back-to-counties', 'style'),
    Output('back-to-municipalities', 'style'),
    Output('view-mode', 'data'),
    Input({'type': 'tile', 'level': 'State', 'value': dash.ALL}, 'n_clicks'),
    Input({'type': 'tile', 'level': 'County', 'value': dash.ALL}, 'n_clicks'),
    Input({'type': 'tile', 'level': 'Municipality', 'value': dash.ALL}, 'n_clicks'),
    Input({'type': 'tile', 'level': 'District', 'value': dash.ALL}, 'n_clicks'),
    Input('back-to-states', 'n_clicks'),
    Input('back-to-counties', 'n_clicks'),
    Input('back-to-municipalities', 'n_clicks'),
    Input('show-all-indicators', 'n_clicks'),
    Input('back-to-hierarchy', 'n_clicks'),
    Input('onboard-kri', 'n_clicks'),
    Input('back-to-dashboard', 'n_clicks'),
    State('selected-state', 'data'),
    State('selected-county', 'data'),
    State('selected-municipality', 'data'),
    State('view-mode', 'data'),
    prevent_initial_call=True
)
def navigate(
        state_tile_clicks, county_tile_clicks, muni_tile_clicks, district_tile_clicks,
        back_states, back_counties, back_munis, show_all, back_hierarchy, onboard_kri,
        back_dashboard,
        sel_state, sel_county, sel_muni, view_mode
):
    ctx = dash.callback_context

    if ctx.triggered and 'back-to-dashboard' in ctx.triggered[0]['prop_id']:
        # Return to hierarchy view from onboarding page
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'onboard-kri' in ctx.triggered[0]['prop_id']:
        return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, 'onboard'

    if ctx.triggered and 'show-all-indicators' in ctx.triggered[0]['prop_id']:
        return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {
            'display': 'none'}, 'all_indicators'

    if ctx.triggered and 'back-to-hierarchy' in ctx.triggered[0]['prop_id']:
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-states' in ctx.triggered[0]['prop_id']:
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-counties' in ctx.triggered[0]['prop_id']:
        return sel_state, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-municipalities' in ctx.triggered[0]['prop_id']:
        return sel_state, sel_county, None, None, {'display': 'inline'}, {'display': 'inline'}, {
            'display': 'none'}, 'hierarchy'

    # Handle tile clicks
    if state_tile_clicks and any(c > 0 for c in state_tile_clicks):
        idx = [i for i, val in enumerate(state_tile_clicks) if val and val > 0][0]
        state_val = state_rag.iloc[idx]['State']
        return state_val, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if county_tile_clicks and any(c > 0 for c in county_tile_clicks):
        idx = [i for i, val in enumerate(county_tile_clicks) if val and val > 0][0]
        county_candidates = county_rag[county_rag['State'] == sel_state].reset_index(drop=True)
        county_val = county_candidates.iloc[idx]['County']
        return sel_state, county_val, None, None, {'display': 'inline'}, {'display': 'inline'}, {
            'display': 'none'}, 'hierarchy'

    if muni_tile_clicks and any(c > 0 for c in muni_tile_clicks):
        idx = [i for i, val in enumerate(muni_tile_clicks) if val and val > 0][0]
        muni_candidates = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].reset_index(
            drop=True)
        muni_val = muni_candidates.iloc[idx]['Municipality']
        return sel_state, sel_county, muni_val, None, {'display': 'inline'}, {'display': 'inline'}, {
            'display': 'inline'}, 'hierarchy'

    if district_tile_clicks and any(c > 0 for c in district_tile_clicks):
        idx = [i for i, val in enumerate(district_tile_clicks) if val and val > 0][0]
        dist_candidates = district_rag[(district_rag['State'] == sel_state) &
                                       (district_rag['County'] == sel_county) &
                                       (district_rag['Municipality'] == sel_muni)].reset_index(drop=True)
        dist_val = dist_candidates.iloc[idx]['District']
        return sel_state, sel_county, sel_muni, dist_val, {'display': 'inline'}, {'display': 'inline'}, {
            'display': 'inline'}, 'hierarchy'

    return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {
        'display': 'none'}, view_mode


@app.callback(
    Output('content-container', 'children'),
    Output('final-table-container', 'children'),
    Output('back-to-hierarchy', 'style'),
    Output('onboard-container', 'style'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    Input('selected-district', 'data'),
    Input('view-mode', 'data'),
    State('sort-by-color-toggle', 'data')
)
def update_view(sel_state, sel_county, sel_muni, sel_district, view_mode, sort_toggle):
    if view_mode == 'onboard':
        # Show onboarding form only
        return [], None, {'display': 'none'}, {'display': 'block'}

    if view_mode == 'all_indicators':
        final_df = pd.read_csv('extended_test_data.csv')  # re-read after any updates
        final_df['ColorBG'] = final_df.apply(cell_color, axis=1)
        final_df['ColorCode'] = final_df['ColorBG'].apply(color_to_code)

        style_data_conditional = []
        for i, row in final_df.iterrows():
            bg_color = row['ColorBG']
            style_data_conditional.append({
                'if': {
                    'filter_query': (
                        f'{{Key Risk Indicator}} = "{row["Key Risk Indicator"]}" && '
                        f'{{State}} = "{row["State"]}" && '
                        f'{{County}} = "{row["County"]}" && '
                        f'{{Municipality}} = "{row["Municipality"]}" && '
                        f'{{District}} = "{row["District"]}"'
                    )
                },
                'backgroundColor': bg_color,
                'color': 'white'
            })

        columns = [{'name': c, 'id': c} for c in final_df.columns if c not in ['ColorBG', 'ColorCode']]
        sort_by = []
        if sort_toggle:
            sort_by = [{'column_id': 'ColorCode', 'direction': 'desc'}]

        table = dash_table.DataTable(
            columns=columns,
            data=final_df.to_dict('records'),
            style_data_conditional=style_data_conditional,
            page_size=20,
            sort_action='native',
            sort_by=sort_by,
            hidden_columns=['ColorBG', 'ColorCode']
        )
        return [], table, {'display': 'inline'}, {'display': 'none'}

    # Hierarchy view
    final_df = pd.read_csv('extended_test_data.csv')  # ensure fresh data
    global state_rag, county_rag, muni_rag, district_rag
    state_rag = get_aggregated_rag(final_df, ['State'])
    county_rag = get_aggregated_rag(final_df, ['State', 'County'])
    muni_rag = get_aggregated_rag(final_df, ['State', 'County', 'Municipality'])
    district_rag = get_aggregated_rag(final_df, ['State', 'County', 'Municipality', 'District'])

    if sel_state is None:
        plot_df = state_rag.copy()
        tiles = create_tiles(plot_df, 'State')
        return tiles, None, {'display': 'none'}, {'display': 'none'}
    elif sel_state is not None and sel_county is None:
        plot_df = county_rag[county_rag['State'] == sel_state].copy()
        tiles = create_tiles(plot_df, 'County')
        return tiles, None, {'display': 'none'}, {'display': 'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is None:
        plot_df = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].copy()
        tiles = create_tiles(plot_df, 'Municipality')
        return tiles, None, {'display': 'none'}, {'display': 'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is not None and sel_district is None:
        plot_df = district_rag[(district_rag['State'] == sel_state) &
                               (district_rag['County'] == sel_county) &
                               (district_rag['Municipality'] == sel_muni)].copy()
        tiles = create_tiles(plot_df, 'District')
        return tiles, None, {'display': 'none'}, {'display': 'none'}
    else:
        filtered = final_df[(final_df['State'] == sel_state) &
                            (final_df['County'] == sel_county) &
                            (final_df['Municipality'] == sel_muni) &
                            (final_df['District'] == sel_district)].copy()
        filtered['ColorBG'] = filtered.apply(cell_color, axis=1)
        filtered['ColorCode'] = filtered['ColorBG'].apply(color_to_code)

        style_data_conditional = []
        for i, row in filtered.iterrows():
            bg_color = row['ColorBG']
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{Key Risk Indicator}} = "{row["Key Risk Indicator"]}"'
                },
                'backgroundColor': bg_color,
                'color': 'white'
            })

        columns = [{'name': c, 'id': c} for c in filtered.columns if c not in ['ColorBG', 'ColorCode']]
        sort_by = []
        if sort_toggle:
            sort_by = [{'column_id': 'ColorCode', 'direction': 'desc'}]

        table = dash_table.DataTable(
            columns=columns,
            data=filtered.to_dict('records'),
            style_data_conditional=style_data_conditional,
            page_size=20,
            sort_action='native',
            sort_by=sort_by,
            hidden_columns=['ColorBG', 'ColorCode']
        )
        return [], table, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('onboard-status', 'children'),
    Input('onboard-submit', 'n_clicks'),
    State('onboard-state', 'value'),
    State('onboard-county', 'value'),
    State('onboard-municipality', 'value'),
    State('onboard-district', 'value'),
    State('onboard-kri-name', 'value'),
    State('onboard-threshold', 'value'),
    State('onboard-risk-value', 'value'),
    prevent_initial_call=True
)
def add_new_kri(submit_clicks, state_val, county_val, muni_val, dist_val, kri_name, threshold, risk_value):
    if submit_clicks and state_val and county_val and muni_val and dist_val and kri_name and threshold is not None and risk_value is not None:
        new_row = {
            'State': state_val,
            'County': county_val,
            'Municipality': muni_val,
            'District': dist_val,
            'Key Risk Indicator': kri_name,
            'Threshold': threshold,
            'Risk Value': risk_value
        }
        df_to_append = pd.DataFrame([new_row])
        if os.path.exists('extended_test_data.csv'):
            df_to_append.to_csv('extended_test_data.csv', mode='a', header=False, index=False)
        else:
            df_to_append.to_csv('extended_test_data.csv', index=False)
        return "New KRI successfully added!"
    else:
        return ""


@app.callback(
    Output('page-title', 'children'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    Input('selected-district', 'data')
)
def update_title(state, county, municipality, district):
    title = "Risk Dashboard"
    if state is not None:
        title += f": {state}"
    if county is not None:
        title += f" > {county}"
    if municipality is not None:
        title += f" > {municipality}"
    if district is not None:
        title += f" > {district}"
    return title


if __name__ == '__main__':
    app.run_server(debug=True)
