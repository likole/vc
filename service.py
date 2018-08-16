# coding:utf-8
import json
import math
import librosa
import requests
import time
from flask import Flask, request, send_file
from tensorpack import SaverRestore, PredictConfig, ChainInit, OfflinePredictor
from werkzeug.utils import secure_filename
import os
import numpy as np
import soundfile as sf
import tensorflow as tf

from models.data_load import get_mfccs_and_spectrogram, normalize_0_1
from models.models import Net2
from hparams.hparam import hparam as hp
from utils.audio import denormalize_db, spec2wav, db2amp, inv_preemphasis, preemphasis, amp2db

app = Flask(__name__)
predictor = None


def init(logdir2):
    # 网络
    model = Net2()
    ckpt2 = tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    pred_conf = PredictConfig(
        model=model,
        input_names=['x_ppgs', 'x_mfccs', 'y_spec', 'y_mel'],
        output_names=['pred_spec', "ppgs"],
        session_init=ChainInit(session_inits))
    global predictor
    predictor = OfflinePredictor(pred_conf)


def convert(wav, ppgs):
    """
        转换
        输入:wav,ppgs(将被调整为3秒)
        输出:aduio(3秒)
    """

    # fix wav length
    wav = librosa.util.fix_length(wav, hp.default.sr * hp.default.duration)

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=hp.default.preemphasis)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=hp.default.n_fft, hop_length=hp.default.hop_length,
                     win_length=hp.default.win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    # fix ppgs length
    ppgs = librosa.util.fix_length(ppgs, ((hp.default.duration * hp.default.sr) // hp.default.hop_length + 1))

    # get_input
    x_ppgs, x_mfccs, y_spec, y_mel = (np.arange(9999) == ppgs[:, None]).astype(np.int32), \
                                     mfccs.T, mag_db.T, mel_db.T  # (t,9999)(t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
    x_ppgs, x_mfccs, y_spec, y_mel = x_ppgs[np.newaxis, :], x_mfccs[np.newaxis, :], \
                                     y_spec[np.newaxis, :], y_mel[np.newaxis, :]

    # get_output
    pred_spec, _ = predictor(x_ppgs, x_mfccs, y_spec, y_mel)

    # Denormalization
    pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)

    # db to amp
    pred_spec = db2amp(pred_spec)

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)

    # Spectrogram to waveform
    audio = np.array(
        list(map(lambda spec: spec2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length,
                                       hp.default.n_iter), pred_spec)))

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)

    return audio


def do_service(wav_file):
    # 提取路径
    basepath, filename = os.path.split(wav_file)
    filename, _ = os.path.splitext(filename)

    # 调整采样率和格式
    wav, sr = librosa.load(wav_file, mono=True, sr=None)
    wav = librosa.resample(wav, sr, 16000)
    wav_len = np.size(wav)
    sf.write(wav_file, wav, 16000, format="wav", subtype="PCM_16")

    # 获取ppgs
    multipart_form_data = {
        'wave': ('wav.wav', open(wav_file, "rb"))
    }
    try:
        response = requests.post('http://202.207.12.156:9000/asr', {'ali': 'true'}, files=multipart_form_data)
        content = json.loads(response.text)
        ppgs = np.array(json.loads(content['ali']))
        np.save(basepath + "/" + filename + ".npy", ppgs)
        # print(ppgs)
    except:
        # wav=AudioSegment.from_wav(basepath + "/" + filename)
        return "接口请求失败"

    # 拼接结果
    audio = []
    for i in range(int(math.ceil(wav_len / (hp.default.duration * hp.default.sr)))):
        print(i, i * hp.default.duration * hp.default.sr)
        _audio = convert(wav[i * hp.default.duration * hp.default.sr:],
                         ppgs[i * ((hp.default.duration * hp.default.sr) // hp.default.hop_length + 1):])
        audio = audio + _audio[0].tolist()  # _audio[0]

    # 修复长度
    audio = librosa.util.fix_length(np.array(audio), wav_len)

    # 写结果
    sf.write("uploads/" + filename + "_output.wav", audio, 16000, format="wav", subtype="PCM_16")
    return send_file("uploads/" + filename + "_output.wav", as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'uploads', str(int(time.time() * 1000)) + secure_filename(f.filename))
    f.save(upload_path)
    return do_service(upload_path)


if __name__ == '__main__':
    case2 = "20180815"
    hp.set_hparam_yaml(case2)
    logdir2 = '{}/{}/train2'.format(hp.logdir_path, case2)
    init(logdir2)
    app.run(debug=True)
