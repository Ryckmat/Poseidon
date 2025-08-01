![CI](https://github.com/Ryckmat/Poseidon/actions/workflows/process-tcx.yml/badge.svg)
# Poseidon — Performance Analytics Platform for TCX / Power / Cadence Data

Poseidon is a self-hosted, extensible performance analytics pipeline and dashboard for endurance session files (TCX/GPX/FIT).  
It ingests raw activity files, cleans and filters noisy/outlier data (e.g., power > 250W), detects stable effort segments, computes derived metrics, performs correlation analysis (power vs cadence/speed), and tracks longitudinal progress over time.



## 🚀 Goals

- Ingest and version raw workout files  
- Clean/filter inconsistent data (ex: exclude power spikes above threshold)  
- Detect stable effort segments and characterize specific efforts  
- Compute performance metrics: normalized power, FTP estimation, pace, speed, TSS/IF (if applicable), etc.  
- Visualize individual sessions with deep analytics (regressions, segment inspection)  
- Track progression across sessions (trends, clustering, load)  
- Automate via GitHub Actions for hands-off processing  

## 🧩 High-level Architecture

- **Storage / Metadata**: PostgreSQL (e.g., Supabase, Neon, ElephantSQL)  
- **Raw file storage**: Supabase Storage or filesystem (`data/`)  
- **Pipeline**: Python for parsing, cleaning, filtering, segment detection, regressions, and metric computation  
- **CI / Orchestration**: GitHub Actions to trigger processing on new uploads  
- **Session Dashboard**: Streamlit (or similar) for interactive per-session visualization  
- **Longitudinal Dashboard**: Optional Grafana / Metabase for aggregation, or integrated into Streamlit  
- **Optional API**: FastAPI / minimal endpoint for uploads or programmatic access

## 📦 Repository Structure

```
Poseidon/
├── .github/
│   └── workflows/
│       └── process-tcx.yml        # CI workflow for automatic ingestion
├── data/                          # raw .tcx/.fit/.gpx files (can be ignored in git)
├── src/
│   ├── ingest/parser.py           # parsing + raw import / cleaning
│   ├── processing/analysis.py     # derived metrics, segment detection, regressions
│   ├── db/models.py              # SQL interactions / schema helpers
│   ├── api/server.py             # optional upload API (FastAPI)
│   ├── dashboard/app.py          # Streamlit dashboard prototype
│   └── utils/filters.py          # filters (power threshold, smoothing, etc.)
├── tests/                        # test suites
├── requirements.txt             # pinned Python dependencies
├── .env.example                # environment variable template
├── README.md                   # this file
└── .gitignore

````

## ⚙️ Quickstart / Setup (Local Development)

### 1. Clone the repository
```bash
# if using GitHub Desktop you already have it cloned locally; otherwise:
git clone https://github.com/Ryckmat/Poseidon.git
cd Poseidon
````

### 2. Create a Python environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate           # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Copy environment template and configure

```bash
cp .env.example .env
# Edit .env and fill in real values:
# DATABASE_URL, thresholds, Supabase credentials (if used), etc.
```

### 4. Initialize database

Connect to your PostgreSQL instance (Supabase, Neon, local, etc.) and run the schema DDL (below) to create tables.

### 5. Drop a TCX file into `data/` and run the pipeline

(This assumes you have scripts implemented in `src/ingest/parser.py` and `src/processing/analysis.py`.)

```bash
python src/ingest/parser.py --input data/your_session.tcx
python src/processing/analysis.py --session <session_id>
```

### 6. Launch visualization dashboard

```bash
streamlit run src/dashboard/app.py
```

## 🗄️ Database Schema (PostgreSQL DDL)

```sql
-- raw uploaded files
CREATE TABLE raw_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename TEXT NOT NULL,
  uploaded_at TIMESTAMPTZ DEFAULT now(),
  source_url TEXT,
  metadata JSONB
);

-- sessions
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  raw_file_id UUID REFERENCES raw_files(id),
  start_time TIMESTAMPTZ,
  end_time TIMESTAMPTZ,
  duration_s NUMERIC,
  distance_km NUMERIC,
  elevation_gain_m NUMERIC,
  avg_heart_rate NUMERIC,
  avg_speed_kmh NUMERIC,
  ftp_estimated NUMERIC,
  normalized_power NUMERIC,
  tss NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- trackpoints
CREATE TABLE trackpoints (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID REFERENCES sessions(id),
  time TIMESTAMPTZ,
  distance_m NUMERIC,
  altitude_m NUMERIC,
  heart_rate INTEGER,
  cadence INTEGER,
  power NUMERIC,
  power_filtered NUMERIC,
  speed_calc_kmh NUMERIC,
  pace_min_per_km NUMERIC,
  elevation_diff NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- stable power segments
CREATE TABLE stable_segments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id),
  start_time TIMESTAMPTZ,
  end_time TIMESTAMPTZ,
  duration_s NUMERIC,
  avg_power NUMERIC,
  std_power NUMERIC,
  avg_cadence NUMERIC,
  avg_speed_kmh NUMERIC,
  points_count INTEGER,
  label TEXT
);

-- regressions
CREATE TABLE regressions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id),
  type TEXT, -- e.g., 'power_vs_cadence', 'power_vs_speed'
  slope NUMERIC,
  intercept NUMERIC,
  r2 NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

*Add indexes where needed (e.g., on `session_id`, `time`) to speed up queries.*

## 🛠️ Core Features

### Session-level processing

* Parsing of TCX/GPX/FIT activity files
* Filtering out inconsistent values (e.g., power > threshold)
* Rolling statistics (e.g., rolling std / smooth of power)
* Detection of stable effort segments (configurable duration, std, power floor)
* Regression analyses: power vs cadence, power vs speed
* Derived metrics: pace, speed, elevation gain/loss, normalized power, FTP estimation, TSS/IF

### Visualization & Dashboard (Streamlit)

* Select session by date / metadata
* KPI summary (distance, duration, avg power, NP, cadence, speed, etc.)
* Before/after filtering comparison
* Time-series plots (power, cadence, speed, pace, elevation)
* Stable segment listing and drill-down
* Scatter + linear fit plots with R²
* Comparison overlay with reference session
* Parameter tweaking (power threshold, segment criteria)
* CSV / image export

### Longitudinal tracking (future / extension)

* FTP / normalized power trends
* Training load (TSS) over weeks / months
* Clustering of session types
* Regression of efficiency (power per speed, cadence impact)
* Alerts on regression / overtraining / new PRs

## 💡 Example Workflow Automation (GitHub Actions)

Place this in `.github/workflows/process-tcx.yml`:

```yaml
name: Process new TCX

on:
  push:
    paths:
      - 'data/**/*.tcx'
  workflow_dispatch:

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run parser + analysis
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          MAX_POWER: 250
          STABLE_WINDOW_S: 30
          STABLE_STD_THRESHOLD: 5
          MIN_STABLE_DURATION_S: 60
        run: |
          # Replace with your actual processing commands
          python src/ingest/parser.py --input data/*.tcx
          python src/processing/analysis.py
```

*Add `DATABASE_URL` and any other needed secrets under GitHub > Settings > Secrets.*

## 🧪 Testing

* Write unit tests for:

  * Parser robustness (missing fields, malformed XML)
  * Filtering logic (e.g., excluding power spikes)
  * Segment detection with synthetic data
  * Regression coefficient sanity
* Place tests under `tests/` and run with `pytest` (add to requirements if needed).

## 📈 Deployment Options (Free / Low-cost)

| Component              | Suggested Free Option                           |
| ---------------------- | ----------------------------------------------- |
| PostgreSQL             | Supabase free tier / Neon free tier             |
| File hosting           | Supabase Storage or versioned `data/` on GitHub |
| CI / triggers          | GitHub Actions (free for moderate usage)        |
| Dashboard              | Streamlit Community Cloud or HuggingFace Spaces |
| Optional API / service | Fly.io free tier, Railway free tier             |

## 🔧 Configuration (via env variables)

Typical variables in `.env`:

```env
DATABASE_URL=postgresql://user:pass@host:5432/db
MAX_POWER=250
STABLE_WINDOW_S=30
STABLE_STD_THRESHOLD=5
MIN_STABLE_DURATION_S=60
SUPABASE_URL=https://your.supabase.url
SUPABASE_SERVICE_KEY=secret
```

Load them in Python with `python-dotenv` or environment injection in CI.

## 📦 Example Commands

```bash
# Parse a TCX file and ingest
python src/ingest/parser.py --input data/session1.tcx

# Run analysis (segment detection + regressions)
python src/processing/analysis.py --session-id <uuid>

# Launch dashboard locally
streamlit run src/dashboard/app.py
```

## 🧠 Extension Ideas

* Auto-estimate FTP from sustained efforts
* Compute acute\:chronic workload ratio (ACWR)
* Smart comparison: find “most similar” past sessions
* Workout recommendation based on gaps/trends
* Multi-user support with account separation (via Supabase Auth)
* Export templated reports (PDF/HTML)

## 🧰 Tech Stack Summary

* Python: parsing, analytics, regressions
* PostgreSQL: time-series and relational storage
* Streamlit: session visualization UI
* GitHub Actions: automated file processing
* Supabase / Neon: optional managed DB + object storage
* (Optional) Grafana / Metabase: deeper longitudinal dashboards

## 🤝 Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-thing`
3. Implement & test
4. Commit with clear message (e.g., `feat(parser): add resilient TCX timestamp normalization`)
5. Open a pull request against `main`
6. Describe your change, include before/after if applicable

Follow [Conventional Commits](https://www.conventionalcommits.org/) for clarity.

## 🏷️ Badge Examples (add to top of README)

```markdown
![CI](https://github.com/Ryckmat/Poseidon/actions/workflows/process-tcx.yml/badge.svg)
```

## 📚 References

* TCX / FIT file format parsing patterns
* Power analysis methodologies (Normalized Power, TSS)
* Streamlit documentation
* PostgreSQL performance tips (indexes on time-series)

## 📜 License

Specify your license here. Example:

```
MIT License
```

Or replace with whichever fits your needs (Apache 2.0, proprietary, etc.).

## 🗂️ Useful Tips

* Store raw file hash in `raw_files.metadata` to avoid double-processing
* Record the exact filtering parameters used per session for reproducibility
* Consider a lightweight audit log of ingest/processing runs
* Back up the database schema and critical reference sessions regularly
