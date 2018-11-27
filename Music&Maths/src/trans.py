#!/usr/bin/env python3
import scipy.io.wavfile as wav
import wave
import os
from python_speech_features import mfcc
import numpy as np

def trans(s):
	return s[0] * 256 * 256 + s[1] * 256 + s[2]

readpath = r"../cut/"
writepath = r"../mfcc/"
files = os.listdir(readpath)
infiles = [readpath + f for f in files if f.endswith('.wav')]
outfiles = [writepath + f[-30:-4] + ".txt" for f in files if f.endswith('.wav')]

# print(len(infiles))
# print(len(outfiles))
for i in range(len(infiles)):

	print("Calculating " + outfiles[i] + "...")
	
	fs = wave.open(infiles[i])
	rate = fs.getframerate()
	frames = fs.getnframes()
	audio = fs.readframes(frames)

	# print(fs.getparams())
	tmp = np.fromstring(audio, dtype=np.int16)
	tmp.shape = -1, 2
	inp = tmp[:,0]
		# inp.append()
		# inp[i] = audio[i]
		# inp.append([np.uint64(audio[i * 3 + 3: i * 3 + 6])])
	# print("======================")
	# print(inp[0])
	# print(inp[1])
	# print(inp[2])
	# print(len(inp))
	# inp = np.array(inp).T
	feature_mfcc = mfcc(inp, samplerate=rate, winstep=0.1, nfft=2205)
	# feature_mfcc = mfcc(inp, nfft=8880)
	# f = open(outfiles[i], "w+")
	# f.write(str(feature_mfcc))
	np.savetxt(outfiles[i], feature_mfcc)
	# f.close()
	# print(feature_mfcc)
	# print(feature_mfcc.shape)


# from pydub import AudioSegment
# import os, re

# # 循环目录下所有文件
# for each in os.listdir('.'):
#     filename = re.findall(r"(.*?)\.mp3", each) # 取出.mp3后缀的文件名
#     if filename:
#         filename[0] += '.mp3'
#         mp3 = AudioSegment.from_mp3(filename[0]) # 打开mp3文件
#         mp3[17*1000+500:].export(filename[0], format="mp3") # 切割前17.5秒并覆盖保存
