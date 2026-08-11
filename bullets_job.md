# CV Bullets — MealWise

- Implemented POST /meal-plan/{id}/regenerate-day endpoint `[mealwise@d1a9001]`
- Refactored models to nested Meal → DayPlan → MealPlan structure `[mealwise@e17b34c]`
- Made generate_meal_plan async using asyncio.to_thread `[mealwise@af43c30]`
- Added recipe field to Meal model and prompt structure `[mealwise@e17b34c]`
- Created database tables automatically on startup `[mealwise@8912ac4]`
- Added get meal plan from DB endpoint `[mealwise@bb521ea]`

### Sources

- `[mealwise@d1a9001]` **mealwise@d1a9001** — “feat: add POST /meal-plan/{id}/regenerate-day endpoint  Changed: backend/app/models/meal.py, backend/app/routers/meals.py, backend/app/services/planne…”
- `[mealwise@e17b34c]` **mealwise@e17b34c** — “refactor: redesign models and update meal plan generation pipeline - Replace flat models with nested Meal → DayPlan → MealPlan structure - Add Prefere…”
- `[mealwise@af43c30]` **mealwise@af43c30** — “feat: make generate_meal_plan async using asyncio.to_thread  Changed: backend/app/services/planner.py (+4/-2)…”
- `[mealwise@8912ac4]` **mealwise@8912ac4** — “fix: create database tables automatically on startup  Changed: backend/app/main.py (+4/-0)…”
- `[mealwise@bb521ea]` **mealwise@bb521ea** — “feat: implement get meal plan from DB  Changed: backend/app/models/meal.py, backend/app/routers/meals.py, backend/app/services/parser.py, backend/app/…”

---

## ⚠️ Struck by the Critic (not supported by evidence)

- ~~Added meal-plan retrieval endpoint backed by SQLAlchemy models~~
  - *Struck because:* The evidence mentions changes to meal.py and repository.py, but does not explicitly state that a meal-plan retrieval endpoint was added, only that the backend calls the database, implying the use of SQLAlchemy models.
