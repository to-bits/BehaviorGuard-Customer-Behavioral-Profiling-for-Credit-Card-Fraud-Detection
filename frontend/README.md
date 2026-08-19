# BehaviorGuard Frontend

React + TypeScript + Vite + Tailwind frontend for the BehaviorGuard fraud-intelligence API.

## Run locally

Install Node.js 18+ first, then from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The API is expected at `http://127.0.0.1:8000`. Copy `.env.example` to `.env` to override it with `VITE_API_BASE_URL`.

## Build

```bash
npm run build
```

The UI intentionally keeps fallback data in `src/data/mockData.ts`. It is labeled in the interface and can be replaced by the adapters in `src/lib/api.ts` when the frontend request flow is connected to real transaction input.
