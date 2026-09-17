import matplotlib.pyplot as plt
import pandas as pd


def plot_monthly_series(df: pd.DataFrame, date_col: str, value_col: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df[date_col]), df[value_col])
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.2)
    return fig, ax
