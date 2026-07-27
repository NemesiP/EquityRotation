# Relative Rotation Dashboard

An interactive Streamlit and Plotly workspace for comparing the relative direction
and momentum of assets against a selected Yahoo Finance benchmark.

The benchmark is fixed at `(100, 100)`. Asset trails move through four regimes:
Improving, Leading, Weakening, and Lagging.

## Run locally

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch loads the 11 US sector ETFs against SPY. Choose a preset or enter
comma-separated Yahoo Finance symbols, select a benchmark and observation frequency,
then press **Update**. Matching requests are cached for 15 minutes.

## Present in Jupyter

The notebook edition uses the same data, calculations, charts, presets, and exports
as the Streamlit app, presented with native `ipywidgets` and IPython output areas.

```bash
source .venv/bin/activate
jupyter lab EquityRotation.ipynb
```

Run the notebook cells from top to bottom. The dashboard cell opens the interface and
loads the default sector universe. Playback uses a native Play widget linked to the
date slider, so the RRG and inspector advance together without rerunning a cell.

To embed the dashboard in another notebook:

```python
from rrg.notebook import launch_notebook_dashboard

dashboard = launch_notebook_dashboard(auto_load=True)
```

## Workspace

- Presets cover US sectors, US factors, global regions, and cross-asset rotation.
- Focus mode dims other trails and compares the selected asset with its benchmark.
- The playback scrubber keeps the rotation map and inspector on the same date.
- Fixed-period axes make dates comparable; Auto fit follows active trails with
  centered, label-aware padding.
- The latest summary and sortable snapshot remain pinned to the newest observation.
- Snapshot and visible-history CSV exports are available below the table.

Playback is capped at 126 daily, 104 weekly, or 60 monthly observations to keep the
dashboard responsive. It stops automatically at the latest observation.

## Calculation

The dashboard uses a transparent RRG-style proxy:

```text
relative = adjusted asset close / adjusted benchmark close
RS-Ratio = 100 × EMA(relative, 10) / EMA(relative, 30)
RS-Momentum = 100 × RS-Ratio / EMA(RS-Ratio, 10)
```

This is not the proprietary JdK Relative Rotation Graph formula. The axis labels
identify both values as proxy metrics.

## Tests

```bash
pytest -q
```

## Data notice

Market history comes from Yahoo Finance through
[yfinance](https://ranaroussi.github.io/yfinance/). The library states that Yahoo
data is intended for research and personal use. Prices may be delayed or incomplete;
this project is not investment advice.
