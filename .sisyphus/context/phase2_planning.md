# Ultra Planner Loop 2: Data Expansion & UI Modernization

## 1. Data Expansion (The "More Parameters" Goal)
To improve the `NRLOmniModel`, we need to expand beyond just team/venue embeddings. We must ingest:
- **Global Continuous Context:** Weather conditions (temp, humidity), team Elo/form, rest days, ladder positions.
- **Roster Continuous Context:** Player fatigue, historical form rating, positional experience.
- **Sequence Context Expansion:** Momentum metric, weather impact on play-by-play.

## 2. API & Distillation Integration (The "One Model" Goal)
The backend `api.py` must be completely stripped of the old models (`OutcomeModel`, `ContextualStackedLSTM`, `SGMTransformer`) and replaced with a single `torch.jit.load("dist/NRL_OmniModel_SOTA.pt")` call.

## 3. UI Modernization (The "shadcn" Goal)
The current `static/index.html` is a standard Tailwind layout. We will modernize it into a **shadcn-style** interface:
- Replace raw blue/green Tailwind with custom CSS variables (Radix UI color palette).
- Add `lucide` icons.
- Use rigid bordered cards, subtle shadows (`shadow-sm`), rounded-xl corners, and typography that mimics `@shadcn/ui`.
- Clean up the SGM betslip generator to look like a premium betting app (Ladbrokes style but minimalist).
