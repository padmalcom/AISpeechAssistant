from loguru import logger
import time
import multiprocessing

# Real Time Voice Cloning
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError
import sounddevice as sd
import math
		
class Voice:

	def __init__(self, fast_vocoding=True):
		self.process = None
		self.griffin_lim = fast_vocoding

		self.enc_model_fpath = Path("encoder/saved_models/my_run.pt")
		self.syn_model_fpath = Path("synthesizer/saved_models/my_run/my_run.pt")
		self.voc_model_fpath = Path("vocoder/saved_models/my_run/my_run.pt")
		
		if self.enc_model_fpath.exists():
			encoder.load_model(self.enc_model_fpath)
		else:
			logger.warning("Encoder-Model existiert nicht. Bitte runterladen von https://github.com/padmalcom/Real-Time-Voice-Cloning-German/releases.")
			
		if self.syn_model_fpath.exists():
			self.synthesizer = Synthesizer(self.syn_model_fpath)
		else:
			logger.error("Synthesizer-Model existiert nicht. Bitte runterladen von https://github.com/padmalcom/Real-Time-Voice-Cloning-German/releases.")
			
		if self.voc_model_fpath.exists():
			vocoder.load_model(self.voc_model_fpath)
		else:
			logger.warning("Vocoder-Model existiert nicht. Bitte runterladen von https://github.com/padmalcom/Real-Time-Voice-Cloning-German/releases.")
		
	def __speak__(self, text):
		texts = [text]
		embeds = [[0] * 256]
		
		if not self.syn_model_fpath.exists():
			logger.error("Synthesizer-Modell ist nicht geladen. TTS fehlgeschlagen.")
			return
			
		specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
		spec = specs[0]
	
		if not self.griffin_lim and self.voc_model_fpath.exists():
			generated_wav = vocoder.infer_waveform(spec)
			generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
			
			if self.enc_model_fpath.exists():
				generated_wav = encoder.preprocess_wav(generated_wav)
			else:
				logger.warning("Kann Ausgabe nicht bereinigen, da Encoder-Modell nicht geladen werden kann.")
		else:
			if not self.voc_model_fpath.exists():
				logger.warning("Vocoder-Model existiert nicht. Fallback zu Griffin-Lim.")
			generated_wav = Synthesizer.griffin_lim(spec)
			
		audio_length = librosa.get_duration(generated_wav, sr = self.synthesizer.sample_rate)
		sd.play(generated_wav.astype(np.float32), self.synthesizer.sample_rate)
		time.sleep(round(audio_length))
	
	def say(self, text):
		if self.process:
			self.stop()
		p = multiprocessing.Process(target=self.__speak__, args=(text,))
		p.start()
		self.process = p
		
	def set_voice(self, voiceId):
		pass
		
	def stop(self):
		if self.process:
			self.process.terminate()
		
	def get_voice_keys_by_language(self, language=''):
		return []