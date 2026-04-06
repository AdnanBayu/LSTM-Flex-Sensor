![💻Sign_Language_Recognition Using_Flex_Sensor](https://github.com/user-attachments/assets/99d82daf-cfff-4c48-a8d2-a82d1b39ae57)
# SIBI Sign Language Recognition using Flex Sensor and LSTM Algorithm
![GitHub last commit](https://img.shields.io/github/last-commit/AdnanBayu/LSTM-Flex-Sensor) ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AdnanBayu/LSTM-Flex-Sensor)

> This repository implements a hand sign recognition system using flex sensor input, MQTT telemetry, and an LSTM-based prediction pipeline. The project was developed for the PKM-KC competition and as the author’s undergraduate final thesis project.

## Overview

- Collects gesture data from a flex sensor glove via MQTT.
- Uses an LSTM model to recognize alphabet letters from sequential sensor readings.
- Outputs recognized letters through audio playback.
- Includes data collection, configuration, prediction, and model files.

## Project structure

- `khadas/` - main Python source files for the real-time prediction program. This folder contains the application deployed on the Khadas VIM 4 single-board computer.
- `khadas/models/` - saved LSTM model definitions and weight files used by the Khadas program.
- `data/participant_*` - recorded sensor datasets for each participant.
- `config/` - calibration thresholds and configuration files used by the pipeline.
- `alphabet_audio/wav/` - audio files for alphabet pronunciation output.
- `Model Iteration Documentation/` - Jupyter notebooks, model experiments, and saved model checkpoints.

## Dependencies

Install the required Python packages before running the project.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: `torch` installation may require a specific command depending on your CUDA support. If the generic install fails, visit https://pytorch.org/ and use the platform-specific install command.

## Usage

Run the main pipeline from the `khadas/` folder.

```powershell
cd khadas
python lstm_main.py --fn <mode> [options]
```

Available modes:

- `--fn loop_default` - print raw converted sensor payloads.
- `--fn config` - collect calibration values and save them to `config/<config_path>.txt`.
- `--fn get_data` - record gesture data to `data/<letter>-<id>.txt`.
- `--fn predict` - perform live LSTM prediction and optionally save results.

Examples:

```powershell
python lstm_main.py --fn config --config_path default
python lstm_main.py --fn get_data --letter a --id 1 --name user1 --seq_len 20
python lstm_main.py --fn predict --name session1 --model lstm
```

> Note: `khadas/audio.py` contains the audio playback helper class, but the main process is launched from `khadas/lstm_main.py`.

## Data and models

- Recorded sensor dataset is stored under `data/participant_*`.
- The default LSTM weight file is loaded from `khadas/models/LSTM/lstm_flex_best_model18.pth`.
- Calibration files are stored under `config/` and read by `khadas/lstm_tools.py`.
- Prediction results are saved to `result_lstm/<name>.txt`.

## Notes

- MQTT broker settings are configured in `khadas/mqtt.py`.
- Audio files are played from `alphabet_audio/wav/`.
- The LSTM prediction pipeline is implemented in `khadas/LSTM.py`.

## Non-technical documentation

- `PPT PKM-KC GLOVITOO.pptx` - competition pitching deck (Indonesian)
- `Progress Report.docx` - competition progress report (Indonesian)`