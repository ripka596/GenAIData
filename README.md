<div align="center">

# Pipeline генерации мультимодального датасета (Text + Image)

**Автоматический сбор данных для последующего использования**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Qwen](https://img.shields.io/badge/Qwen-%236950EF?style=flat&logo=qwen&logoColor=white)
</div>

## 📖 О проекте

Скрипт автоматизирует процесс создания синтетического мультимодального датасета. На основе заданного списка тем (робототехника, ИИ, квантовые вычисления и т.д.) генерируются текстовые описания и соответствующие им изображения.

**Результат работы:** 300 пар «текст + изображение» (5 тем × 60 подтем), упакованных в два удобных CSV-файла для дальнейшего использования в ML-пайплайнах.

## ⚙️ Как это работает (Pipeline)

1. **Генерация текста:** Языковая модель `Qwen/Qwen3.5-2B` пишет описание на 300 слов и промпт для Stable Diffusion (50–60 слов) в формате JSON.
2. **Оптимизация VRAM:** Модель Qwen выгружается из памяти (`gc.collect()` + `torch.cuda.empty_cache()`), чтобы освободить видеопамять для генератора картинок.
3. **Генерация изображений:** Модель `stabilityai/sdxl-turbo` генерирует картинки 512x512 всего за 2 шага инференса (благодаря режиму Turbo), что дает колоссальный выигрыш по времени.
4. **Постобработка:** Изображения ресайзятся до 1536x1536 (Lanczos) и сохраняются на диск.
5. **Сохранение:** Сборка финальных датасетов с метаданными (количество токенов и слов в тексте).

## 🛠️ Стек
- **Text-to-Text:** `transformers` (Qwen 2B)
- **Text-to-Image:** `diffusers` (Stable Diffusion XL Turbo)
- **Обработка данных:** `pandas`, `json`
- **Работа с изображениями:** `PIL` (Pillow)

## 📦 Установка

1. Клонируйте репозиторий и создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   # Активация на Windows: .\.venv\Scripts\activate
   # Активация на Linux/Mac: source .venv/bin/activate
2. Установите зависимости:
    ```bash
    pip install torch transformers diffusers pandas Pillow accelerate
    ```
3. Запустите генерацию
    ```
    python "Module A.py"
    ```

## 📊 Структура выходных данных
После завершения работы скрипт создаст две папки (images/, datasets/) и два CSV-файла:

`datasets/texts.csv`

* `topic_id` - уникальный ID (например, Нейронные сети_0)
* `description` - текст описания (~300 слов)
* `image_prompt` - промпт для генерации
* `tokens` - количество токенов (для контроля контекста)
* `words` - количество слов

`datasets/images.csv`

* `topic_id` - уникальный ID
* `image_path` - путь к PNG-файлу
* `image_prompt` - промпт, по которому сгенерировано изображение

## 💻 Требования к железу
GPU: Наличие дискретной видеокарты (NVIDIA) крайне желательно.

VRAM: Проект протестирован на NVIDIA RTX 4070 SUPER (12 GB VRAM). Благодаря последовательной загрузке моделей и выгрузке первой из памяти, лимита в 12 ГБ хватает с запасом.

