import os
import time

import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pyworld as pw
import scipy.interpolate as interpolate

from tts.text import text_to_sequence
from tts.utils.util import ROOT_PATH


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.buffer = self._get_buffer(*args, **kwargs)
        self.length_dataset = len(self.buffer)

    def _get_buffer(self, data_path, batch_expand_size, mel_ground_truth, alignment_path, text_cleaners, size=99000):
        buffer = list()
        text = self._process_text(data_path)

        start = time.perf_counter()
        batch_expand_size = batch_expand_size
        energy_path, pitch_path = self.load_energy_pitch(size)
        for i in tqdm(range(len(text)), desc="Create buffer"):
            if i >= size:
                break
            mel_gt_name = os.path.join(
                ROOT_PATH / mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                ROOT_PATH / alignment_path, str(i) + ".npy"))
            energy = np.load(os.path.join(
                energy_path, "ljspeech-energy-%05d.npy" % (i + 1)
            ))
            pitch = np.load(os.path.join(
                pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1)
            ))
            character = text[i][0:len(text[i]) - 1]

            character = np.array(
                text_to_sequence(character, text_cleaners))

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            energy_target = torch.from_numpy(energy)
            pitch_target = torch.from_numpy(pitch)
            mel_gt_target = torch.from_numpy(mel_gt_target)

            buffer.append({"text": character,
                           "duration": duration,
                           "energy_target": energy_target,
                           "pitch_target": pitch_target,
                           "mel_target": mel_gt_target,
                           "batch_expand_size": batch_expand_size})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end - start))

        return buffer

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]

    @staticmethod
    def load_energy_pitch(max_size: int = None):
        energy_path = ROOT_PATH / "data" / "energy"
        pitch_path = ROOT_PATH / "data" / "pitch"
        if os.path.exists(energy_path) and os.path.exists(pitch_path):
            return energy_path, pitch_path
        energy_path.mkdir(exist_ok=True, parents=True)
        pitch_path.mkdir(exist_ok=True, parents=True)

        energies_bound = [None, None]
        pitches_bound = [None, None]

        iterator = sorted((ROOT_PATH / "data" / "LJSpeech-1.1" / "wavs").iterdir())

        for idx, wav_p in enumerate(tqdm(iterator, desc="Pitch and Energy")):
            if idx >= max_size:
                break
            mel = np.load(os.path.join(ROOT_PATH / "data" / "mels", "ljspeech-mel-%05d.npy" % (idx + 1)))

            energy = np.linalg.norm(mel, axis=-1)
            np.save(os.path.join(energy_path, "ljspeech-energy-%05d.npy" % (idx + 1)), energy)

            energies_bound[0] = min(energies_bound[0], energy.min()) if energies_bound[0] is not None else energy.min()
            energies_bound[1] = max(energies_bound[1], energy.max()) if energies_bound[1] is not None else energy.max()

            # pitch
            wave, sr = torchaudio.load(wav_p)
            wave = wave.to(torch.float64).numpy().sum(axis=0)

            frame_period = (wave.shape[0] / sr * 1000) / mel.shape[0]
            _f0, t = pw.dio(wave, sr, frame_period=frame_period)
            f0 = pw.stonemask(wave, _f0, t, sr)[:mel.shape[0]]

            idx_nonzero = np.nonzero(f0)
            x = np.arange(f0.shape[0])[idx_nonzero]
            values = (f0[idx_nonzero][0], f0[idx_nonzero][-1])
            f = interpolate.interp1d(x, f0[idx_nonzero], bounds_error=False, fill_value=values)
            new_f0 = f(np.arange(f0.shape[0]))

            np.save(os.path.join(pitch_path, "ljspeech-pitch-%05d.npy" % (idx + 1)), new_f0)

            pitches_bound[0] = min(pitches_bound[0], new_f0.min()) if pitches_bound[0] is not None else new_f0.min()
            pitches_bound[1] = max(pitches_bound[1], new_f0.max()) if pitches_bound[1] is not None else new_f0.max()

        print("Energy:", energies_bound)
        print("Pitch:", pitches_bound)

        return energy_path, pitch_path

    @staticmethod
    def _process_text(train_text_path):
        with open(ROOT_PATH / train_text_path, "r", encoding="utf-8") as f:
            txt = []
            for line in f.readlines():
                txt.append(line)
            return txt
