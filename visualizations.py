import plotly.express as px

def grafico_histograma(df):

    fig = px.histogram(
        df[df["DIAS"] <= 30],
        x="DIAS",
        nbins=30
    )

    fig.add_vline(x=3, line_dash="dash", line_color="green")
    fig.add_vline(x=5, line_dash="dash", line_color="orange")
    fig.add_vline(x=7, line_dash="dash", line_color="red")

    return fig


def grafico_sla(df, colors):

    fig = px.pie(
        df,
        names="ESTADO_SLA",
        color="ESTADO_SLA",
        color_discrete_map=colors
    )

    return fig