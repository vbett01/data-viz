import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import random

# Load your data
df = pd.read_csv('/Users/vincentbett/Desktop/Viz/DataViz/extended_test_data.csv')

# Helper functions
def calculate_rag_status(sub_df):
    above_threshold_count = (sub_df['Risk Value'] > sub_df['Threshold']).sum()
    total = len(sub_df)
    ratio = above_threshold_count / total if total > 0 else 0
    if ratio > 2/3:
        return 'Red'
    elif ratio > 1/3:
        return 'Amber'
    else:
        return 'Green'

def get_aggregated_rag(df, group_cols):
    grouped = df.groupby(group_cols)
    rag_df = grouped.apply(calculate_rag_status).reset_index()
    rag_df.columns = group_cols + ['RAG']
    return rag_df

state_rag = get_aggregated_rag(df, ['State'])
county_rag = get_aggregated_rag(df, ['State', 'County'])
muni_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality'])
district_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality', 'District'])

def get_rag_color(rag):
    if rag == 'Red':
        return 'red'
    elif rag == 'Amber':
        return 'orange'
    else:
        return 'green'

def create_tiles(dataframe, level_col):
    tiles = []
    for _, row in dataframe.iterrows():
        entity = row[level_col]
        rag = row['RAG']
        color = get_rag_color(rag)
        tiles.append(
            html.Button(
                entity,
                id={
                    'type': 'tile',
                    'level': level_col,
                    'value': entity
                },
                n_clicks=0,
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

# Create global style_data_conditional for full view
# Color coding each row based on Risk Value > Threshold
def create_style_data_conditional(df):
    styles = []
    for i, row in df.iterrows():
        bg_color = 'green'
        color = 'white'
        if row['Risk Value'] > row['Threshold']:
            bg_color = 'red'
        styles.append({
            'if': {
                'filter_query': f'{{Key Risk Indicator}} = "{row["Key Risk Indicator"]}" && {{State}} = "{row["State"]}" && {{County}} = "{row["County"]}" && {{Municipality}} = "{row["Municipality"]}" && {{District}} = "{row["District"]}"'
            },
            'backgroundColor': bg_color,
            'color': color
        })
    return styles

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Risk Dashboard"),
    dcc.Store(id='selected-state'),
    dcc.Store(id='selected-county'),
    dcc.Store(id='selected-municipality'),
    dcc.Store(id='selected-district'),
    dcc.Store(id='view-mode', data='hierarchy'),  # 'hierarchy' or 'all'

    html.Div([
        html.Button("Back to States", id='back-to-states', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Counties", id='back-to-counties', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Municipalities", id='back-to-municipalities', n_clicks=0, style={'display': 'none'}),
        html.Button("Show All Indicators", id='show-all', n_clicks=0, style={'margin-left': '20px'}),
        html.Button("Back to Hierarchy", id='back-to-hierarchy', n_clicks=0, style={'display': 'none', 'margin-left': '20px'})
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
    Input({'type': 'tile', 'level': ALL, 'value': ALL}, 'n_clicks'),
    Input('back-to-states', 'n_clicks'),
    Input('back-to-counties', 'n_clicks'),
    Input('back-to-municipalities', 'n_clicks'),
    Input('show-all', 'n_clicks'),
    Input('back-to-hierarchy', 'n_clicks'),
    State('selected-state', 'data'),
    State('selected-county', 'data'),
    State('selected-municipality', 'data'),
    State('view-mode', 'data'),
    prevent_initial_call=True
)
def navigate(tile_clicks, back_states, back_counties, back_munis, show_all, back_hierarchy, sel_state, sel_county, sel_muni, view_mode):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle show all indicators
    if triggered_id == 'show-all':
        # Switch to all indicators mode
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'all'
    elif triggered_id == 'back-to-hierarchy':
        # Switch back to hierarchy mode
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'

    if triggered_id == 'back-to-states':
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'
    elif triggered_id == 'back-to-counties':
        return sel_state, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'
    elif triggered_id == 'back-to-municipalities':
        return sel_state, sel_county, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}, 'hierarchy'

    if 'tile' in triggered_id and view_mode == 'hierarchy':
        for i, v in enumerate(ctx.inputs_list[0]):
            if v['id']['type'] == 'tile' and v['value'] and tile_clicks[i] and tile_clicks[i] > 0:
                level = v['id']['level']
                value = v['id']['value']
                if sel_state is None:
                    return value, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}, 'hierarchy'
                elif sel_state and sel_county is None:
                    return sel_state, value, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}, 'hierarchy'
                elif sel_state and sel_county and sel_muni is None:
                    return sel_state, sel_county, value, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}, 'hierarchy'
                elif sel_state and sel_county and sel_muni:
                    return sel_state, sel_county, sel_muni, value, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}, 'hierarchy'

    # Default return if nothing matched
    # Keep current selection and mode
    return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, view_mode

@app.callback(
    Output('content-container', 'children'),
    Output('final-table-container', 'children'),
    Output('back-to-hierarchy', 'style'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    Input('selected-district', 'data'),
    Input('view-mode', 'data')
)
def update_view(sel_state, sel_county, sel_muni, sel_district, view_mode):
    if view_mode == 'all':
        # Show all indicators in a table
        style_data_conditional = create_style_data_conditional(df)
        columns = [{'name': c, 'id': c} for c in df.columns]
        table = dash_table.DataTable(
            columns=columns,
            data=df.to_dict('records'),
            style_data_conditional=style_data_conditional,
            page_size=20
        )
        return [], table, {'display': 'inline'}

    # Hierarchy mode
    if sel_state is None:
        # Show states
        plot_df = state_rag.copy()
        tiles = create_tiles(plot_df, 'State')
        return tiles, None, {'display': 'none'}
    elif sel_state is not None and sel_county is None:
        # Show counties in that state
        plot_df = county_rag[county_rag['State'] == sel_state].copy()
        tiles = create_tiles(plot_df, 'County')
        return tiles, None, {'display': 'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is None:
        # Show municipalities
        plot_df = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].copy()
        tiles = create_tiles(plot_df, 'Municipality')
        return tiles, None, {'display': 'none'}
    elif sel_state is not None and sel_county is not None and sel_muni is not None and sel_district is None:
        # Show districts
        plot_df = district_rag[(district_rag['State'] == sel_state) &
                               (district_rag['County'] == sel_county) &
                               (district_rag['Municipality'] == sel_muni)].copy()
        tiles = create_tiles(plot_df, 'District')
        return tiles, None, {'display': 'none'}
    else:
        # Final level: color-coded table for selected district
        final_df = df[(df['State'] == sel_state) &
                      (df['County'] == sel_county) &
                      (df['Municipality'] == sel_muni) &
                      (df['District'] == sel_district)].copy()

        style_data_conditional = []
        for i, row in final_df.iterrows():
            bg_color = 'green'
            color = 'white'
            if row['Risk Value'] > row['Threshold']:
                bg_color = 'red'
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{Key Risk Indicator}} = "{row["Key Risk Indicator"]}"'
                },
                'backgroundColor': bg_color,
                'color': color
            })

        columns = [{'name': c, 'id': c} for c in final_df.columns]
        table = dash_table.DataTable(
            columns=columns,
            data=final_df.to_dict('records'),
            style_data_conditional=style_data_conditional
        )
        return [], table, {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)
