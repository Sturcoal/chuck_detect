# Автоматическая нарезка кадров с живыми объектами на подводных видео при помощи Detectron 2

Этот репозиторий содержит готовый скрипт на **Python 3** для извлечения кадров из подводного видео, детекции объектов моделью *Faster R‑CNN R‑50 FPN* (предобученной на COCO) в среде **Detectron 2**, и сохранения как «размеченных», так и «сырых» кадров вместе с таблицей результатов.

> *Скрипт пригоден как стартовая точка для собственных экспериментов: вы меняете только путь к видео и (при желании) интервал обработки кадров.*

---

## 1. Создание изолированного окружения

```bash
# 1) создаём и активируем окружение (название — на ваше усмотрение)
conda create -n detectron2_env python=3.10 -y
conda activate detectron2_env

# 2) устанавливаем PyTorch (подберите версию под свою CUDA)
# пример: CPU‑only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3) ставим Detectron 2
pip install "detectron2==0.6" --index-url https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.2/index.html
# или свежую версию с GitHub (если нужна сборка из исходников)
# pip install git+https://github.com/facebookresearch/detectron2.git

# 4) ставим прочие зависимости
pip install opencv-python pandas tqdm
```

> ⚠️ **Важно:** если у вас GPU с CUDA, замените строку установки PyTorch на соответствующую вашей версии драйвера.
> Если делаете через Spider, то ставьте ! перед строчкой кода

---

## 2. Загрузка конфигурации и весов модели

Detectron 2 бесплатно предоставляет готовые «зе́бры» весов и конфигов на **Model Zoo**.

В скрипте используются две функции:

```python
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
```

Они автоматически:

* загружают YAML‑конфиг модели,
* скачивают (при первом запуске) веса `.pkl` с официального хостинга.

Файлы кладутся в `~/.torch/iopath_cache/`; повторно качать не придётся.

---

## 3. Сам скрипт

Ниже приведён минимальный, но полностью рабочий пример. **Путь к файлу видео** (`VIDEO_PATH`) и **префикс интервала обработки** (`FRAME_INTERVAL_SEC`) редактируются под задачу.
Я лично запускаю в Spider. Перед запуском активирую окружение detectron2_env.

```python
"""detect_chukchi_rov.py
Скрипт детекции макроскопических организмов на ROV‑видео.
Автор: <ваше имя>  |  Журнал «Биология моря», 2025
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ----------- ПАРАМЕТРЫ ПОЛЬЗОВАТЕЛЯ -----------
VIDEO_PATH = "your_video.mp4"          # ← укажите свой файл
FRAME_INTERVAL_SEC = 3                  # шаг по времени (с)
DEVICE = "cpu"                          # или "cuda"
# ---------------------------------------------

# 1. Настройка Detectron 2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = DEVICE

predictor = DefaultPredictor(cfg)

# 2. Подготовка директорий вывода
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
raw_dir = f"{video_name}_raw_frames"
det_dir = f"{video_name}_detection_images"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(det_dir, exist_ok=True)

# 3. Чтение видео
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Не удалось открыть файл {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * FRAME_INTERVAL_SEC)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results = []  # будущий DataFrame

# 4. Проход по кадрам
for frame_id in tqdm(range(total_frames), desc="Обработка кадров"):
    ret, frame = cap.read()
    if not ret:
        break

    # обрабатываем только каждый N‑й кадр
    if frame_id % frame_interval:
        continue

    # === детекция ===
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes

    if len(boxes):
        # сохраняем «сырое» изображение
        cv2.imwrite(os.path.join(raw_dir, f"{video_name}_{frame_id}.jpg"), frame)

        # собираем данные для CSV
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for box, score, cls in zip(boxes, instances.scores, instances.pred_classes):
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[cls]
            results.append({
                "timestamp_s": timestamp,
                "frame": frame_id,
                "class": class_name,
                "score": float(score),
                "box": box.tensor.numpy().tolist()[0]
            })

        # визуализация
        vis = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        vis_img = vis.draw_instance_predictions(instances).get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(det_dir, f"{video_name}_{frame_id}_det.jpg"), vis_img)

cap.release()

# 5. Сохраняем таблицу
pd.DataFrame(results).to_csv(f"{video_name}_detections.csv", index=False)
print("\nГотово! Результаты сохранены.")
```

### Что получает пользователь

```
project/
├─ your_video.mp4
├─ detect_chukchi_rov.py
├─ your_video_raw_frames/          # кадры с объектами (без разметки)
├─ your_video_detection_images/    # те же кадры + рамки детектора
└─ your_video_detections.csv       # сводная таблица
```

*CSV‑файл* можно напрямую анализировать в Python/R или, как в статье, агрегировать по времени для расчёта **precision / recall / F₁**.

---

## 4. Настройка параметров

| Параметр                      | Назначение                                                                                               | Значение по умолчанию |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------- |
| `FRAME_INTERVAL_SEC`          | Шаг дискретизации (с). Ставьте `1`, чтобы обрабатывать каждую секунду, `0.1` — \~каждый кадр при 10 fps. | 3                     |
| `DEVICE`                      | `cpu` или `cuda`. На GPU работа заметно быстрее.                                                         | `cpu`                 |
| `cfg.MODEL.SCORE_THRESH_TEST` | Порог уверенности (по умолчанию *0.0* — все срабатывания).                                               | 0.0                   |

---

## 5. Ссылки и цитирование

При использовании данного кода в публикациях, пожалуйста, цитируйте оригинальную статью:

> Туранов С.В. Применение неспециализированной модели свёрточной нейросети для автоматической детекции макроскопических организмов из подводных видеосьёмок чукотского моря // Биология моря (или какой другой журнал, хз). 2025. В печати.


---

## 6. Лицензия

Код распространяется под лицензией **MIT**; модели Detectron 2 распространяются Facebook AI Research на условиях Apache 2.0.

---

С вопросами и предложениями открывайте *Issues* в репозитории или пишите на e‑mail автора.
