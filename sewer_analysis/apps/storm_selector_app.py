import dash
from dash import html, dcc, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def run_storm_selector_app(df_rdii):
    SELECTED_DATES_FILE = "../data/selected_storm_dates.csv"

    app = dash.Dash(__name__)

    # Define App Layout
    app.layout = html.Div([
        html.Div([
            html.H1("Define Periods"),

            # Div to choose / display all the storm start and end times
            html.Div(id = "period-container", children = []),

            # Buttons to add / subtract number of storms
            html.Button("Add Period", id = "add-period-btn", n_clicks = 0),
            html.Button("Remove Period", id = "remove-period-btn", n_clicks = 0),
            html.Button("Save Periods", id = "save-periods-btn", n_clicks = 0, style={"marginTop": "10px"}),
            html.Div(id = "save-output", style={"marginTop": "10px"}),
            html.Div(id = "store-periods", style = {"display": "none"})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id = "rainfall-graph")
        ], style={'width': '68%', 'display': 'inline-block', 'padding': '10px'})
    ])

    # Define logic for adding and removing date start / end boxes
    @app.callback(
        Output('period-container', 'children'),
        Input('add-period-btn', 'n_clicks'),
        Input('remove-period-btn', 'n_clicks'),
        State({'type': 'period-picker', 'index': dash.ALL}, 'start_date'),
        State({'type': 'period-picker', 'index': dash.ALL}, 'end_date')
    )
    def update_period_inputs(add_clicks, remove_clicks, start_dates, end_dates):

        # Limit storms to 1 - 10
        delta = add_clicks - remove_clicks + 1
        
        if 1 <= delta <= 10:
            new_count = delta
        elif delta < 1:
            new_count = 1
        elif delta > 10:
            new_count = 10

        # Add / Remove Periods
        new_children = []
        for i in range(new_count):
            start_date = start_dates[i] if i < len(start_dates) else None
            end_date = end_dates[i] if i < len(end_dates) else None
            new_children.extend([
                html.Div([
                    dcc.DatePickerRange(
                        id={'type': 'period-picker', 'index': i},
                        min_date_allowed=df_rdii['timestamp'].min().date(),
                        max_date_allowed=df_rdii['timestamp'].max().date(),
                        display_format='YYYY-MM-DD',
                        start_date=start_date,
                        end_date=end_date
                    ),
                    html.Br()
                ])
            ])
        return new_children

    # Define logic for updating graph
    @app.callback(
        Output('rainfall-graph', 'figure'),
        Input({'type': 'period-picker', 'index': dash.ALL}, 'start_date'),
        Input({'type': 'period-picker', 'index': dash.ALL}, 'end_date')
    )
    def update_graph(start_dates, end_dates):
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)

        fig.add_trace(go.Scatter(
            x=df_rdii["timestamp"],
            y=df_rdii["rainfall_mm"],
            mode="lines",
            name="Rainfall (mm)",
            line=dict(color="darkblue", width=1),
            fill="tozeroy",
            opacity=0.2
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_rdii["timestamp"],
            y=df_rdii["RDII"],
            mode="lines",
            name="RDII (L/s)",
            line=dict(color="green", width=1)
        ), row=2, col=1)

        # Highlight periods on graph
        for i, (start, end) in enumerate(zip(start_dates, end_dates)):
            if start and end:
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    opacity=0.4,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Period {i+1}",
                    annotation_position="top left",
                )

        fig.update_layout(
            title="Storm Selector", 
            xaxis_title="Timestamp", 
            yaxis_title="Rainfall (mm)",
            yaxis2_title="RDII (L/s)",
            xaxis2_rangeslider_visible=True, 
            xaxis2_rangeslider_thickness=0.1
        )

        return fig

    # Callback to save selected periods to file
    @app.callback(
        Output("save-output", "children"),
        Input("save-periods-btn", "n_clicks"),
        State({'type': 'period-picker', 'index': dash.ALL}, 'start_date'),
        State({'type': 'period-picker', 'index': dash.ALL}, 'end_date')
    )
    def save_periods(n_clicks, start_dates, end_dates):
        if n_clicks > 0:
            periods_df = pd.DataFrame({
                'start_date': pd.to_datetime(start_dates),
                'end_date': pd.to_datetime(end_dates)
            })
            periods_df.to_csv(SELECTED_DATES_FILE, index=False)
            return f"Saved {len(periods_df)} periods. You may now close the app."
        return ""

    app.run(debug=True)