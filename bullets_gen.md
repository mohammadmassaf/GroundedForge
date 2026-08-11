# CV Bullets — MealWise meal plan generation with Gemini and prompt parsing

- Built meal-plan generation pipeline with nested Meal/DayPlan/MealPlan models and async generation path [1][2][3]
- Integrated Gemini API for meal plan generation and USDA API for nutritional analysis with prompt parsing and validation [1][4][5]
- Implemented POST /meals/plan endpoint with retry logic and JSON parsing [1][6][3]

### Sources

1. `mealwise@0cf0a59` — “feat: scaffold FastAPI backend with Gemini meal planning endpoint - Set up FastAPI project structure (routers, services, models, core) - Add GET /heal…”
2. `mealwise@e17b34c` — “refactor: redesign models and update meal plan generation pipeline - Replace flat models with nested Meal → DayPlan → MealPlan structure - Add Prefere…”
3. `mealwise@af43c30` — “feat: make generate_meal_plan async using asyncio.to_thread  Changed: backend/app/services/planner.py (+4/-2)…”
4. `mealwise@ae2cf81` — “feat: add nutritional analysis via USDA API and Gemini function calling  Changed: backend/app/services/prompts.py (+2/-0)…”
5. `mealwise-README-12` — **mealwise README § Generate a meal plan** — “# Generate a meal plan curl -s -X POST http://localhost:8000/meals/preferences \   -H "Authorization: Bearer $TOKEN" \   -H "Content-Type: application…”
6. `mealwise@9ad5a78` — “removed validate_meal_plan  Changed: backend/app/services/parser.py (+3/-10)…”

---

## ⚠️ Struck by the Critic (not supported by evidence)

- ~~Implemented prompt builder with past meals consideration, JSON parser, and retry logic in services layer~~
  - *Struck because:* The evidence does not directly mention the implementation of retry logic and JSON parser together with the prompt builder in the services layer, although it mentions these components separately in different commits.
