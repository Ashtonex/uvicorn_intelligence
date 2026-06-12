# Nirvana Analytics Sandbox

This folder is a read-only Python sidecar for Nirvana intelligence work. It is meant for experiments, diagnostics, and JSON reports that can later be consumed by the Next.js app after we trust the results.

## Installed stack

- JupyterLab and VS Code notebooks for research.
- pandas, NumPy, SciPy, statsmodels, and scikit-learn for analysis.
- matplotlib and seaborn for offline diagnostics.

## Run

From the repo root:

```powershell
$env:MPLCONFIGDIR="$PWD\.mplconfig"
.venv\Scripts\python.exe -m jupyter lab analytics\notebooks
```

Run the first useful brains:

```powershell
.venv\Scripts\python.exe -m analytics.nirvana_analytics.demand_forecast --days 90
.venv\Scripts\python.exe -m analytics.nirvana_analytics.expense_anomaly --days 60
.venv\Scripts\python.exe -m analytics.nirvana_analytics.inventory_velocity --days 60
.venv\Scripts\python.exe -m analytics.nirvana_analytics.capital_allocation --days 90
```

Each script prints JSON by default. Use `--output reports/analytics/name.json` to save an artifact for later app integration.
Use `--save-db` after applying `supabase/migrations/20260513_create_analytics_results.sql` to publish the snapshot into Nirvana.

```powershell
npm run analytics:snapshot
```

Riskfolio-Lib is optional for the capital allocation brain. If it is unavailable, the job uses SciPy optimization and reports that backend in its JSON.

## Safety rule

The loaders only use `select` calls. Keep this sidecar read-only until an analysis has proved itself and we intentionally promote it into the production app.
