# llm_test v1 - замороженная выборка, главная метрика

Запросы внешней языковой модели (DeepSeek), отобранные автором; не запросы людей. Заморожена 2026-11-01, до обучения моделей.
Условия сбора и правило отбора - docs/llm_test_prompt.txt; разметка - docs/human_test.md, эталоны - data/human_test/labels.md (автор D01).

- `llm_test.jsonl` - 100 записей: 82 со схемой (очищенная форма запроса), 18 с эталоном-отказом (поле `refuse`).
- `llm_test_chatty.jsonl` - 35 записей: те же схемы с исходной разговорной формой запроса (id с суффиксом c).

Файлы не менять и по ним ничего не дорабатывать. SHA-256 - в manifest.json, проверка - tests/test_human_test.py.
Схемы этой выборки (`circuit_id`) исключаются из train v2.

Запуск:  python src/mcnl/evaluate.py --model <модель> --split llm_test --data data/dataset/llm_test_v1
