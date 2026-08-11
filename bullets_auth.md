# CV Bullets — MealWise authentication, users, and database persistence

- Built JWT authentication with user registration and login endpoints [1][2][3]
- Stored user data and meal plans in PostgreSQL database [4][5][6]
- Implemented database table creation on startup and data persistence [5][6][7]

### Sources

1. `mealwise@8afed88` — “feat : auth . register endpoint  Changed: alembic/versions/db0f61ee00ea_make_username_nullable.py, backend/app/main.py, backend/app/models/db_models.p…”
2. `mealwise@aa2b173` — “feat: add auth with JWT  Changed: backend/app/routers/auth.py, backend/app/routers/meals.py, backend/app/services/parser.py, backend/app/services/plan…”
3. `mealwise-README-13` — **mealwise README § API reference** — “## API reference  | Method | Endpoint | Auth | Description | |--------|----------|------|-------------| | POST | `/auth/register` | — | Create account…”
4. `mealwise-README-4` — **mealwise README § 1. Clone and configure** — “### 1. Clone and configure  ```bash git clone https://github.com/mohammadmassaf/mealwise.git cd mealwise ```  Create `.env` in the project root:  ```…”
5. `mealwise@9818592` — “Initial commit: mealwise backend  Changed: .gitignore, README.md, backend/__init__.py, backend/ai/__init__.py, backend/ai/parser.py, backend/ai/planne…”
6. `mealwise@8912ac4` — “fix: create database tables automatically on startup  Changed: backend/app/main.py (+4/-0)…”
7. `mealwise@3644aab` — “feat : added save meal plan func  Changed: backend/app/models/db_models.py, backend/app/services/repository.py (+30/-4)…”

---

## ⚠️ Struck by the Critic (not supported by evidence)

- ~~Implemented user management with database persistence for meal plans~~
  - *Struck because:* The evidence does not mention user management, it only discusses meal plan implementation and database interactions.
- ~~Designed database schema with SQLAlchemy for storing user data and meal plans~~
  - *Struck because:* The evidence mentions using SQLAlchemy ORM models, but it does not explicitly state that the database schema was designed with SQLAlchemy for storing user data and meal plans.
