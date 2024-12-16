import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table

df = pd.read_csv('extended_test_data.csv')  # or your CSV file

def calculate_rag_details(sub_df):
    # Determine how many risk indicators are above threshold
    above_threshold_count = (sub_df['Risk Value'] > sub_df['Threshold']).sum()
    total = len(sub_df)
    ratio = above_threshold_count / total if total > 0 else 0
    if ratio > 2/3:
        rag = 'Red'
    elif ratio > 1/3:
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
    # Color for the cells
    diff = abs(row['Risk Value'] - row['Threshold'])
    if row['Risk Value'] > row['Threshold']:
        return 'red'
    elif diff <= 5:
        return 'orange'
    else:
        return 'green'

def color_to_code(color):
    # For sorting by color
    if color == 'red':
        return 2
    elif color == 'orange':
        return 1
    else:
        return 0

# Precompute RAG at each level with details
state_rag = get_aggregated_rag(df, ['State'])
county_rag = get_aggregated_rag(df, ['State', 'County'])
muni_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality'])
district_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality', 'District'])

def get_tile_tooltip(level_col, row, df):
    # Determine which subset of df corresponds to this tile
    # Based on the level_col, we know which columns to filter by
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

    # Group by Key Risk Indicator and compute mean values
    agg = filtered.groupby('Key Risk Indicator').agg({'Risk Value': 'mean', 'Threshold': 'mean'}).reset_index()

    # Classify each indicator
    lines = []
    for _, r in agg.iterrows():
        classification = cell_color_classification(r['Risk Value'], r['Threshold'])
        lines.append(f"{r['Key Risk Indicator']}: {classification}")

    # Join lines into a multiline tooltip
    return "\n".join(lines)

def create_tiles(dataframe, level_col):
    tiles = []
    for _, row in dataframe.iterrows():
        tooltip = get_tile_tooltip(level_col, row, df)
        rag = row['RAG']
        # Convert rag to a color
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
                title=tooltip,  # show detailed reasons
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
    dcc.Store(id='view-mode', data='hierarchy'),
    dcc.Store(id='sort-by-color-toggle', data=False),

    html.Div([
        html.Button("Back to States", id='back-to-states', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Counties", id='back-to-counties', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Municipalities", id='back-to-municipalities', n_clicks=0, style={'display': 'none'}),
        html.Button("Show All Indicators", id='show-all-indicators', n_clicks=0),
        html.Button("Back to Hierarchy", id='back-to-hierarchy', n_clicks=0, style={'display':'none', 'margin-left':'10px'}),
        html.Button("Sort by Color", id='sort-by-color', n_clicks=0, style={'margin-left':'10px'})
    ], style={'margin-bottom': '20px'}),

    html.Div(id='content-container', style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'start'
    }),

    html.Div(id='final-table-container')
])

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
    State('selected-state', 'data'),
    State('selected-county', 'data'),
    State('selected-municipality', 'data'),
    State('view-mode', 'data'),
    prevent_initial_call=True
)
def navigate(
    state_tile_clicks, county_tile_clicks, muni_tile_clicks, district_tile_clicks,
    back_states, back_counties, back_munis, show_all, back_hierarchy,
    sel_state, sel_county, sel_muni, view_mode
):
    ctx = dash.callback_context

    if ctx.triggered and 'show-all-indicators' in ctx.triggered[0]['prop_id']:
        return sel_state, sel_county, sel_muni, None, {'display':'none'}, {'display':'none'}, {'display':'none'}, 'all_indicators'

    if ctx.triggered and 'back-to-hierarchy' in ctx.triggered[0]['prop_id']:
        return None, None, None, None, {'display':'none'}, {'display':'none'}, {'display':'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-states' in ctx.triggered[0]['prop_id']:
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-counties' in ctx.triggered[0]['prop_id']:
        return sel_state, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if ctx.triggered and 'back-to-municipalities' in ctx.triggered[0]['prop_id']:
        return sel_state, sel_county, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}, 'hierarchy'

    # Tile clicks
    # State-level tiles
    if state_tile_clicks and any(c > 0 for c in state_tile_clicks):
        idx = [i for i, val in enumerate(state_tile_clicks) if val and val > 0][0]
        state_val = state_rag.iloc[idx]['State']
        return state_val, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    # County-level tiles
    if county_tile_clicks and any(c > 0 for c in county_tile_clicks):
        idx = [i for i, val in enumerate(county_tile_clicks) if val and val > 0][0]
        county_candidates = county_rag[county_rag['State'] == sel_state].reset_index(drop=True)
        county_val = county_candidates.iloc[idx]['County']
        return sel_state, county_val, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}, 'hierarchy'

    # Municipality-level tiles
    if muni_tile_clicks and any(c > 0 for c in muni_tile_clicks):
        idx = [i for i, val in enumerate(muni_tile_clicks) if val and val > 0][0]
        muni_candidates = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].reset_index(drop=True)
        muni_val = muni_candidates.iloc[idx]['Municipality']
        return sel_state, sel_county, muni_val, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}, 'hierarchy'

    # District-level tiles
    if district_tile_clicks and any(c > 0 for c in district_tile_clicks):
        idx = [i for i, val in enumerate(district_tile_clicks) if val and val > 0][0]
        dist_candidates = district_rag[(district_rag['State'] == sel_state) &
                                       (district_rag['County'] == sel_county) &
                                       (district_rag['Municipality'] == sel_muni)].reset_index(drop=True)
        dist_val = dist_candidates.iloc[idx]['District']
        return sel_state, sel_county, sel_muni, dist_val, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}, 'hierarchy'

    return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, view_mode

@app.callback(
    Output('content-container', 'children'),
    Output('final-table-container', 'children'),
    Output('back-to-hierarchy', 'style'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    Input('selected-district', 'data'),
    Input('view-mode', 'data'),
    State('sort-by-color-toggle', 'data')
)
def update_view(sel_state, sel_county, sel_muni, sel_district, view_mode, sort_toggle):
    if view_mode == 'all_indicators':
        final_df = df.copy()
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
        return [], table, {'display': 'inline'}

    # Hierarchical view
    if sel_state is None:
        # Show states
        plot_df = state_rag.copy()
        tiles = create_tiles(plot_df, 'State')
        return tiles, None, {'display':'none'}
    elif sel_state is not None and sel_county is None:
        # Show counties
        plot_df = county_rag[county_rag['State'] == sel_state].copy()
        tiles = create_tiles(plot_df, 'County')
        return tiles, None, {'display':'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is None:
        # Show municipalities
        plot_df = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].copy()
        tiles = create_tiles(plot_df, 'Municipality')
        return tiles, None, {'display':'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is not None and sel_district is None:
        # Show districts
        plot_df = district_rag[(district_rag['State'] == sel_state) &
                               (district_rag['County'] == sel_county) &
                               (district_rag['Municipality'] == sel_muni)].copy()
        tiles = create_tiles(plot_df, 'District')
        return tiles, None, {'display':'none'}
    else:
        # Final level table
        final_df = df[(df['State'] == sel_state) &
                      (df['County'] == sel_county) &
                      (df['Municipality'] == sel_muni) &
                      (df['District'] == sel_district)].copy()
        final_df['ColorBG'] = final_df.apply(cell_color, axis=1)
        final_df['ColorCode'] = final_df['ColorBG'].apply(color_to_code)

        style_data_conditional = []
        for i, row in final_df.iterrows():
            bg_color = row['ColorBG']
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{Key Risk Indicator}} = "{row["Key Risk Indicator"]}"'
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
        return [], table, {'display':'none'}

@app.callback(
    Output('sort-by-color-toggle', 'data'),
    Input('sort-by-color', 'n_clicks'),
    State('sort-by-color-toggle', 'data')
)
def toggle_sort_by_color(n_clicks, current):
    if n_clicks:
        return not current
    return current

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
