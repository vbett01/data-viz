# data_file = "/Users/vincentbett/Desktop/Viz/DataViz/insurance_risk_data.csv"

import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv('/Users/vincentbett/Desktop/Viz/DataViz/extended_test_data.csv')

# Helper functions
def calculate_rag_status(sub_df):
    # Determine how many risk indicators are above threshold
    above_threshold_count = (sub_df['Risk Value'] > sub_df['Threshold']).sum()
    total = len(sub_df)
    ratio = above_threshold_count / total if total > 0 else 0
    if ratio > 2/3:
        return 'Red'
    elif ratio > 1/3:
        return 'Amber'
    else:
        return 'Green'

def get_rag_color(status):
    if status == 'Red':
        return 'red'
    elif status == 'Amber':
        return 'orange'
    else:
        return 'green'

def get_aggregated_rag(df, group_cols):
    # Group by the given columns and compute RAG for each group
    grouped = df.groupby(group_cols)
    rag_df = grouped.apply(calculate_rag_status).reset_index()
    rag_df.columns = group_cols + ['RAG']
    return rag_df

# Precompute all RAG statuses at each level for efficiency
state_rag = get_aggregated_rag(df, ['State'])
county_rag = get_aggregated_rag(df, ['State', 'County'])
muni_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality'])
district_rag = get_aggregated_rag(df, ['State', 'County', 'Municipality', 'District'])

# Initialize Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Risk Dashboard"),

    # Hidden divs for storing current selection
    dcc.Store(id='selected-state'),
    dcc.Store(id='selected-county'),
    dcc.Store(id='selected-municipality'),
    dcc.Store(id='selected-district'),

    html.Div(id='navigation-bar', children=[
        html.Button("Back to States", id='back-to-states', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Counties", id='back-to-counties', n_clicks=0, style={'display': 'none'}),
        html.Button("Back to Municipalities", id='back-to-municipalities', n_clicks=0, style={'display': 'none'}),
    ], style={'margin-bottom': '20px'}),

    dcc.Graph(id='hierarchy-graph'),

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
    Input('hierarchy-graph', 'clickData'),
    Input('back-to-states', 'n_clicks'),
    Input('back-to-counties', 'n_clicks'),
    Input('back-to-municipalities', 'n_clicks'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    prevent_initial_call=True
)
def navigate(clickData, back_states, back_counties, back_munis, sel_state, sel_county, sel_muni):
    ctx = dash.callback_context

    # Determine what triggered the callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'back-to-states':
        # Clear all selections
        return None, None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    elif triggered_id == 'back-to-counties':
        return sel_state, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}
    elif triggered_id == 'back-to-municipalities':
        return sel_state, sel_county, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}

    if clickData:
        # Depending on current selection level, drill down further
        if sel_state is None:
            # Currently at State level, user clicked a State
            selected_state = clickData['points'][0]['x']
            return selected_state, None, None, None, {'display': 'inline'}, {'display': 'none'}, {'display': 'none'}
        elif sel_state is not None and sel_county is None:
            # Currently at County level, user clicked a County
            selected_county = clickData['points'][0]['x']
            return sel_state, selected_county, None, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'none'}
        elif sel_state is not None and sel_county is not None and sel_muni is None:
            # Currently at Municipality level, user clicked a Municipality
            selected_muni = clickData['points'][0]['x']
            return sel_state, sel_county, selected_muni, None, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}
        elif sel_state is not None and sel_county is not None and sel_muni is not None:
            # Currently at District level, user clicked a District
            selected_district = clickData['points'][0]['x']
            return sel_state, sel_county, sel_muni, selected_district, {'display': 'inline'}, {'display': 'inline'}, {'display': 'inline'}

    # Default return if nothing triggered
    return sel_state, sel_county, sel_muni, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('hierarchy-graph', 'figure'),
    Output('final-table-container', 'children'),
    Input('selected-state', 'data'),
    Input('selected-county', 'data'),
    Input('selected-municipality', 'data'),
    Input('selected-district', 'data')
)
def update_figure(sel_state, sel_county, sel_muni, sel_district):
    if sel_state is None:
        # Show State level graph
        plot_df = state_rag.copy()
        fig = px.bar(plot_df, x='State', y=[1]*len(plot_df), color='RAG',
                     color_discrete_map={'Red':'red','Amber':'orange','Green':'green'},
                     title='States RAG Status')
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig, None

    elif sel_state is not None and sel_county is None:
        # Show County level for selected state
        plot_df = county_rag[county_rag['State'] == sel_state].copy()
        fig = px.bar(plot_df, x='County', y=[1]*len(plot_df), color='RAG',
                     color_discrete_map={'Red':'red','Amber':'orange','Green':'green'},
                     title=f'Counties in {sel_state}')
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig, None

    elif sel_state is not None and sel_county is not None and sel_muni is None:
        # Show Municipality level
        plot_df = muni_rag[(muni_rag['State'] == sel_state) & (muni_rag['County'] == sel_county)].copy()
        fig = px.bar(plot_df, x='Municipality', y=[1]*len(plot_df), color='RAG',
                     color_discrete_map={'Red':'red','Amber':'orange','Green':'green'},
                     title=f'Municipalities in {sel_county}, {sel_state}')
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig, None

    elif sel_state is not None and sel_county is not None and sel_muni is not None and sel_district is None:
        # Show District level
        plot_df = district_rag[(district_rag['State'] == sel_state) &
                               (district_rag['County'] == sel_county) &
                               (district_rag['Municipality'] == sel_muni)].copy()
        fig = px.bar(plot_df, x='District', y=[1]*len(plot_df), color='RAG',
                     color_discrete_map={'Red':'red','Amber':'orange','Green':'green'},
                     title=f'Districts in {sel_muni}, {sel_county}, {sel_state}')
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig, None

    else:
        # Final level: show table with Key Risk Indicators and color coding
        final_df = df[(df['State'] == sel_state) &
                      (df['County'] == sel_county) &
                      (df['Municipality'] == sel_muni) &
                      (df['District'] == sel_district)].copy()

        # We'll create conditional formatting for the table
        style_data_conditional = []
        for i, row in final_df.iterrows():
            color = 'white'
            bg_color = 'green'
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

        # Return an empty figure or a simple placeholder since we have a table now
        empty_fig = go.Figure()
        return empty_fig, table


if __name__ == '__main__':
    app.run_server(debug=True)
