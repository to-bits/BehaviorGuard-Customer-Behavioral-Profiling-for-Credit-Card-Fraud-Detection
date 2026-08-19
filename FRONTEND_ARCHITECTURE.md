# BehaviorGuard Frontend Architecture

## Product Direction

A dark-first fraud-intelligence workspace centered on model evidence: live risk scoring, behavioral signals, SHAP contributors, and evaluation context. The interface uses a graphite background, blue signal accents, amber review states, and red fraud states. It is an analytics product rather than a generic CRUD dashboard.

## Page Hierarchy

- `Overview`: system health, monitored volume, flagged activity, risk trend, and recent investigations.
- `Transaction Analyzer`: structured 37-feature transaction input, model score, decision, and explanation drawer.
- `Fraud Investigation`: sortable queue of flagged transactions with evidence and review state.
- `Behavioral Intelligence`: velocity, amount-category, and behavioral-signal views.
- `Model Performance`: PR-AUC, recall, precision, threshold comparison, and confusion-matrix context from existing results.
- `Explainable AI`: global SHAP ranking and local contribution inspection.

## Component Hierarchy

- `AppShell`
  - `Sidebar`
  - `Topbar`
  - `PageTransition`
  - `ToastRegion`
  - Page content
- `PageHeader`
- `KpiCard`
- `ChartPanel`
- `RiskBadge`
- `StatusDot`
- `DataTable`
- `SkeletonBlock`
- `EmptyState`
- `ErrorState`
- `Drawer`
- `Toast`

## Data Flow

1. `src/lib/api.ts` owns the API boundary and maps to `/health`, `/model-info`, `/predict`, and `/explain`.
2. `src/data/mockData.ts` provides clearly marked fallback data while the API is unavailable.
3. `App.tsx` owns navigation, theme, toast state, selected transaction, and API health state.
4. `Transaction Analyzer` builds the frontend request from the model's 37 processed features, calls `/predict`, then `/explain` for evidence.
5. Shared response types in `src/types.ts` keep the API contract stable for the React frontend.
6. Charts consume typed view models, not raw API payloads, so replacing mock data does not change page components.

## Integration Boundary

The backend expects processed features including `Time_Period`, `Amount_Category`, `Amount_Log`, `Transactions_Last_1H`, `Transactions_Last_24H`, and `Amount_ZScore`. The frontend marks its demo transaction as mock data and keeps the adapter isolated so a real feature-builder can replace it later.
