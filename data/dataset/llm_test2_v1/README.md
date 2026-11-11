# llm_test2 v1 - замороженная выборка запросов Kimi

Запросы внешней языковой модели (Kimi, Moonshot; автор D02), взяты все 100 без отбора; не запросы людей. Заморожена 2026-11-11,
до генерации датасета v3; разметка просмотрена автором 11.11.2026. Условия сбора - docs/llm_test_prompt.txt («Вторая выборка»),
разметка и принятые соглашения - docs/human_test.md («Выборка llm_test2»), эталоны - data/human_test/labels.md (H0151-H0250).

- `llm_test2.jsonl` - 100 записей: 90 со схемой, 10 с эталоном-отказом (поле `refuse`).
- `llm_test2_chatty.jsonl` - 2 записи: те же схемы с исходной разговорной формой запроса (id с суффиксом c).

Файлы не менять и по ним ничего не дорабатывать. SHA-256 - в manifest.json, проверка - tests/test_human_test.py.
Схемы этой выборки (`circuit_id`) исключаются из train v3 (generate_v3.excluded_circuits). Ответы моделей на ней смотреть один раз.

Запуск:  python src/mcnl/evaluate.py --model <модель> --split llm_test2 --data data/dataset/llm_test2_v1
