# CV Bullets — MealWise USDA nutrition data verification and API integration

- Integrated USDA FDC API for nutrition data verification `[mealwise-README-0]` `[mealwise-README-4]` `[mealwise@ae2cf81]` `[mealwise@e641661]`
- Implemented synchronous nutrition data retrieval `[mealwise@6e97f01]` `[mealwise@e641661]`
- Verified calorie counts against USDA FDC food database `[mealwise-README-0]` `[mealwise@ae2cf81]` `[mealwise@e641661]`

### Sources

- `[mealwise-README-0]` **mealwise README § MealWise** — “# MealWise  AI-powered meal planner that generates personalized multi-day meal plans with recipes, ingredient lists, and USDA-verified nutrition data.…”
- `[mealwise-README-4]` **mealwise README § 1. Clone and configure** — “### 1. Clone and configure  ```bash git clone https://github.com/mohammadmassaf/mealwise.git cd mealwise ```  Create `.env` in the project root:  ```…”
- `[mealwise@ae2cf81]` **mealwise@ae2cf81** — “feat: add nutritional analysis via USDA API and Gemini function calling  Changed: backend/app/services/prompts.py (+2/-0)…”
- `[mealwise@e641661]` **mealwise@e641661** — “feat: get calories for each  ingredient  Changed: backend/app/core/config.py, backend/app/services/nutrition_service.py (+26/-0)…”
- `[mealwise@6e97f01]` **mealwise@6e97f01** — “feat : made get_nutrtion synchronous  Changed: backend/app/services/nutrition_service.py, backend/app/services/parser.py, backend/app/services/planner…”
