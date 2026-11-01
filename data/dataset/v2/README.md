# Датасет v2 (gen-2)

Заморожен 2026-11-01. Файлы не менять и по test не дорабатывать; новая версия - v3. Спецификация - docs/generator_v2_spec.md.

Команды: `python src/mcnl/generate_v2.py --out out/v2_full --seed 20261101`, затем
`python src/mcnl/paraphrase.py apply out/v2_full data/paraphrase_v2 data/dataset/v2`.

| Файл | Записей | SHA-256 |
|---|---|---|
| `train.jsonl` | 2999 | `46ff0589f278f1e4e0cf32e54f34b126dfc83a4fac77fbe2096b95b54159a0a6` |
| `dev.jsonl` | 150 | `d09a6e55e06105f53f22719bb7a74b41ee6742fae6ea4052bcb72a8ca7ba12b1` |
| `test.jsonl` | 500 | `8175d6c71d9435fdbd887563a028f9569d2cc569ddfcaadfb9f0dd543cbd79b5` |
| `gen_topology.jsonl` | 100 | `44fa77b45275948e7c0e901546bbc56069024b83efbd817af19788f328e55748` |
| `gen_family.jsonl` | 100 | `fe30542e350faea59e7c7bab60bfeeba28b77c3e6d9f16ea3936bccfd2770c81` |
| `gen_range.jsonl` | 100 | `6ce96d0f1c89daf1d3c654cd3cd3297a3a0fe2b0a0166d5094b4897d0899fd0a` |

Состав записи - как в v1, плюс `kind` (named_a1, named_a, compose, strange, refuse), `mutations`, `output` (эталонный ответ модели).
В train: `subset` (1000, 2000, 3000 - вложенные подмножества, для сравнения с v1 брать subset <= 2000), `paraphrase`, у парафраз - `prompt_template` и `family_template`.
Записи-отказы: `refuse`, `reason`, `output` = `REFUSE код: причина`, нетлиста нет.

train по семействам: {'paraphrase': 591, 'brief': 372, 'roles_brief': 338, 'freeform': 299, 'addon': 228, 'roles': 228, 'chain': 224, 'narrative': 187, 'list': 156, 'terse': 152, 'refuse': 118, 'table': 106}.
train по типу АЧХ: {'LP': 1540, 'HP': 716, 'прочие': 225, 'flat': 220, 'BP': 96, 'BS': 52}.

Парафразы написаны Claude (автор C02), 591 записей: номиналы проверены программой, соединения - выборочно и по ключевым словам (тип фильтра, наличие ОУ). Около 5% парафраз - с разговорной оболочкой. Схемы llm_test и C01 в выборки не входят.
Отложено от train: топологии khn, twin_t, mfb_hp (gen_topology), семейство path (gen_family), номиналы R выше 1 МОм (gen_range).
