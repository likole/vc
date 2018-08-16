# coding:utf-8
import json

import librosa
import requests
import time
from flask import Flask, request
from tensorpack import SaverRestore, PredictConfig, ChainInit, OfflinePredictor
from werkzeug.utils import secure_filename
import os
import numpy as np
import soundfile as sf
import tensorflow as tf

from models.data_load import get_mfccs_and_spectrogram
from models.models import Net2
from hparams.hparam import hparam as hp
from utils.audio import denormalize_db, spec2wav, db2amp, inv_preemphasis

app = Flask(__name__)
logdir2 = None


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'uploads', str(int(time.time() * 1000)) + secure_filename(f.filename))
    f.save(upload_path)
    do_service(upload_path)
    return "Success"


def do_service(wav_file):
    # 提取路径
    basepath, filename = os.path.split(wav_file)
    filename, _ = os.path.splitext(filename)

    # 调整采样率和格式
    wav, sr = librosa.load(wav_file, mono=True,sr=None)
    wav = librosa.resample(wav, sr, 16000)
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
        print(ppgs)
    except:
        # wav=AudioSegment.from_wav(basepath + "/" + filename)
        print("失败")

    # 提取参数
    x_ppgs, x_mfccs, y_spec, y_mel = get_mfccs_and_spectrogram(basepath + "/" + filename + ".npy")
    x_ppgs = x_ppgs[np.newaxis, :]
    x_mfccs = x_mfccs[np.newaxis, :]
    y_spec = y_spec[np.newaxis, :]
    y_mel = y_mel[np.newaxis, :]

    # 网络
    model = Net2()
    ckpt2 = tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    pred_conf = PredictConfig(
        model=model,
        input_names=['x_ppgs', 'x_mfccs', 'y_spec', 'y_mel'],
        output_names=['pred_spec',"ppgs"],
        session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)

    # 转换
    pred_spec,_ = predictor(x_ppgs, x_mfccs, y_spec, y_mel)
    pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)
    pred_spec = db2amp(pred_spec)
    pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)
    audio = np.array(
        list(map(lambda spec: spec2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length,
                                       hp.default.n_iter), pred_spec)))
    audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)
    sf.write("uploads/result", audio[0], 16000, format="wav", subtype="PCM_16")


if __name__ == '__main__':
    case2 = "20180515"
    hp.set_hparam_yaml(case2)
    logdir2 = '{}/{}/train2'.format(hp.logdir_path, case2)
    app.run(debug=True)
