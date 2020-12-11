from loguru import logger
import time
import os
import ffmpeg
import sounddevice as sd
import soundfile as sf
import multiprocessing
import queue # Für die Exceptions
#from multiprocessing import Queue

class AudioPlayer:

	def __init__(self):
		self._process = None
		self._volume = 0.5
	
	def play_file(self, file):
		if self._process:
			self.stop()
		self._process = multiprocessing.Process(target=self._play_file, args=(file,))
		self._process.start()
		
	def play_stream(self, source):
		if self._process:
			self.stop()
		self._process = multiprocessing.Process(target=self._play_stream, args=(source,))
		self._process.start()
		
	def _play_file(self, file):
		sd.default.reset() # Vielleicht wurde das Ein-/Ausgabegerät geändert
		data, fs = sf.read(file, dtype='float32')
		sd.play(data * self._volume, fs, device=sd.default.device['output'])
		status = sd.wait()	
		if status:
			logger.error("Error bei der Soundwiedergabe {}.", status)
			
	def _play_stream(self, source):
		sd.default.reset() # Vielleicht wurde das Ein-/Ausgabegerät geändert
		_q = queue.Queue(maxsize=20)
		logger.info("Spiele auf Device {}.", sd.default.device['output'])
		
		def _callback_stream(outdata, frames, time, status):
			if status.output_underflow:
				raise sd.CallbackAbort
			assert not status
			try:
				data = _q.get_nowait()
			except queue.Empty as e:
				raise sd.CallbackAbort from e
			assert len(data) == len(outdata)
			outdata[:] = data
		
		try:
			info = ffmpeg.probe(source)
		except ffmpeg.Error as e:
			logger.error(e)

		streams = info.get('streams', [])
		if len(streams) != 1:
			logger.error('There must be exactly one stream available')

		stream = streams[0]

		if stream.get('codec_type') != 'audio':
			logger.error('The stream must be an audio stream')

		channels = stream['channels']
		samplerate = float(stream['sample_rate'])
		
		try:
			process = ffmpeg.input(source).filter('volume', self._volume).output(
				'pipe:',
				format='f32le',
				acodec='pcm_f32le',
				ac=channels,
				ar=samplerate,
				loglevel='quiet',
			).run_async(pipe_stdout=True)
			stream = sd.RawOutputStream(
				samplerate=samplerate, blocksize=1024,
				device=sd.default.device['output'], channels=channels, dtype='float32',
				callback=_callback_stream)
			read_size = 1024 * channels * stream.samplesize
			for _ in range(20):
				_q.put_nowait(process.stdout.read(read_size))
			logger.info("Starte Stream ...")
			with stream:
				timeout = 1024 * 20 / samplerate
				while True:
					_q.put(process.stdout.read(read_size), timeout=timeout)
		except queue.Full as e:
			logger.error("Streaming-Queue ist voll. Das sollte nicht passieren.")
		except Exception as e:
			logger.error(e)
			
	def stop(self):
		if self._process:
			self._process.terminate()
		
	def is_playing(self):
		if self._process:
			return self._process.is_alive()

	def set_volume(self, volume):
		self._volume = max(0.0, min(volume, 1.0))
		
	def get_volume(self):
		return self._volume