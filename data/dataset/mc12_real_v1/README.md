# mc12_real v1 - реальные схемы фильтров

Заморожена 2026-11-01. 13 схем: 8 из примеров `C:\MC12\DATA\Filters`, 5 от дизайнера MC12; перенос в класс - src/mcnl/mc12_real.py.
Запросы написал DeepSeek по task.txt; все совпали со схемами, но близки к списку соединений: выборка проверяет покрытие реальных схем, не понимание языка.

Запуск:  python src/mcnl/evaluate.py --model <модель> --split mc12_real --data data/dataset/mc12_real_v1
