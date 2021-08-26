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

		enc_model_fpath = Path("encoder/saved_models/my_run.pt")
		syn_model_fpath = Path("synthesizer/saved_models/my_run/my_run.pt")
		voc_model_fpath = Path("vocoder/saved_models/my_run/my_run.pt")
		encoder.load_model(enc_model_fpath)
		self.synthesizer = Synthesizer(syn_model_fpath)
		vocoder.load_model(voc_model_fpath)
		
	def __speak__(self, text):
		texts = [text]
		embeds = [[0] * 256]
		
		specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
		spec = specs[0]
	
		if not self.griffin_lim:
			generated_wav = vocoder.infer_waveform(spec)
			generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
			generated_wav = encoder.preprocess_wav(generated_wav)
		else:
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