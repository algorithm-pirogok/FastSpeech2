# SS project barebones

## Установка

Для установки для начала потребуется  установить библиотеки и 
скачать модель
```shell
pip install -r ./requirements.txt
```


## Модель

Реализация основана на статье SpeX+ с использованием hydra в качестве конфига.



## Проверка на данных

Чтобы запустить тренировку модели на части LibriSpeech нужно изменить config, а именно часть, отвечающую за датасеты.
Выглядит она так
```yaml
train:
  batch_size: 4
  num_workers: 8
  datasets:
    - _target_: tts.datasets.MixedLibrispeechDataset
      part: train-clean-100
      sr: 16000
```
Все, что нужно подправить, это указать какую часть датасета мы хотим слушать и размер батча.

Также можно протестировать модель на произвольном датасете, для этого нужно поменять конфиг данных для тренировки6 устроен он следующим образом:
```yaml
test:
  batch_size: 1
  num_workers: 8
  datasets:
    - _target_: tts.datasets.MixedTestDataset
      sr: 16000
      absolute_path: True
      path_to_mix_dir: data/datasets/test_dataset/mix/mix
      path_to_ref_dir: data/datasets/test_dataset/ref/refs
      path_to_target_dir: data/datasets/test_dataset/target/targets

```
Соответсвенно поменять нужно пути, если данные лежат по абсолютному пути, то `absolute_path` нужно ставить False

Также для продолжения нужно указать checkpoint от того, что мы хотим делать

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

