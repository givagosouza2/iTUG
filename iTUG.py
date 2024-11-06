import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype, analog=False)
    y = filtfilt(b, a, data)
    return y


st.set_page_config(layout="wide")

st.markdown(
    '<h1 style="text-align: center; color: blue;">Instrumented Timed Up and Go test</h1>', unsafe_allow_html=True)

valor = st.slider("Time cursor", min_value=0.0, max_value=50.0, value=2.0)
t1, t2, t3 = st.columns(3)
a = 0
b = 0
c = 0

with t1:
    uploaded_acc_iTUG = st.file_uploader(
        "Carregue o arquivo de texto do acelerômetro", type=["txt"],)
    if uploaded_acc_iTUG is not None:
        custom_separator = ';'
        df = pd.read_csv(uploaded_acc_iTUG, sep=custom_separator)
        t = df.iloc[:, 0]
        x = df.iloc[:, 1]
        y = df.iloc[:, 2]
        z = df.iloc[:, 3]
        time = t

        x = signal.detrend(x)
        y = signal.detrend(y)
        z = signal.detrend(z)

        interpf = scipy.interpolate.interp1d(time, x)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        x_ = interpf(time_)
        t, x = time_/1000, x_
        interpf = scipy.interpolate.interp1d(time, y)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        y_ = interpf(time_)
        t, y = time_/1000, y_
        interpf = scipy.interpolate.interp1d(time, z)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        z_ = interpf(time_)
        t, z = time_/1000, z_
        norm_waveform = np.sqrt(x**2+y**2+z**2)
        norm_waveform = butterworth_filter(
            norm_waveform, 4, 100, order=2, btype='low')

        for index, value in enumerate(norm_waveform):
            if value == np.max(norm_waveform):
                s = t[index]
                t = t - t[index]
                break
        a = 1

        with t2:
            uploaded_gyro_iTUG = st.file_uploader(
                "Carregue o arquivo de texto do giroscópio", type=["txt"],)
            if uploaded_gyro_iTUG is not None:
                custom_separator = ';'
                df_gyro = pd.read_csv(uploaded_gyro_iTUG, sep=custom_separator)
                t_gyro = df_gyro.iloc[:, 0]
                x_gyro = df_gyro.iloc[:, 1]
                y_gyro = df_gyro.iloc[:, 2]
                z_gyro = df_gyro.iloc[:, 3]
                time_gyro = t_gyro

                x_gyro = signal.detrend(x_gyro)
                y_gyro = signal.detrend(y_gyro)
                z_gyro = signal.detrend(z_gyro)

                interpf = scipy.interpolate.interp1d(time_gyro, x_gyro)
                time_gyro_ = np.arange(
                    start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
                x_gyro_ = interpf(time_gyro_)
                t_gyro, x_gyro = time_gyro_/1000, x_gyro_
                interpf = scipy.interpolate.interp1d(time_gyro, y_gyro)
                time_gyro_ = np.arange(
                    start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
                y_gyro_ = interpf(time_gyro_)
                t_gyro, y_gyro = time_gyro_/1000, y_gyro_
                interpf = scipy.interpolate.interp1d(time_gyro, z_gyro)
                time_gyro_ = np.arange(
                    start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
                z_gyro_ = interpf(time_gyro_)
                t_gyro, z_gyro = time_gyro_/1000, z_gyro_

                norm_waveform_gyro = np.sqrt(x_gyro**2+y_gyro**2+z_gyro**2)
                norm_waveform_gyro = butterworth_filter(
                    norm_waveform_gyro, 1.5, 100, order=2, btype='low')

                for index, value in enumerate(norm_waveform_gyro):
                    if value == np.max(norm_waveform_gyro):
                        s_gyro = t_gyro[index]
                        t_gyro = t_gyro - t_gyro[index]
                        break
                b = 1
                with t3:
                    c = c + 1
                    uploaded_file = st.file_uploader(
                        "Carregue o arquivo da cinemática", type="csv")
                    if uploaded_file is not None:
                        # Carregar o arquivo CSV sem cabeçalhos
                        df = pd.read_csv(uploaded_file, header=None)

                        # Verificar se o DataFrame tem pelo menos 12 colunas
                        if df.shape[1] >= 12:
                            # Extrair os dados das colunas 11 e 12 (indexadas como 10 e 11)
                            coluna_11 = df.iloc[:, 10].values
                            coluna_12 = df.iloc[:, 11].values

                            # Criar um vetor temporal com base no tamanho da coluna 11
                            vetor_temporal = np.arange(len(coluna_11))/100
                            for index, value in enumerate(coluna_12):
                                if value == np.max(coluna_12):
                                    s = vetor_temporal[index]
                                    vetor_temporal = vetor_temporal - \
                                        vetor_temporal[index]
                                    break
                            c = 1
t1, t2, t3 = st.columns([0.6, 1, 0.6])
with t2:
    if a == 1:
        plt.figure(figsize=(7, 2))
        plt.plot(t_gyro, norm_waveform_gyro, 'k')
        plt.plot([valor, valor], [0, 5], '--b')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        st.pyplot(plt)

    if b == 1:
        fig = plt.figure(figsize=(7, 2))
        plt.plot(t, norm_waveform, 'k')
        plt.plot([valor, valor], [0, 8], '--b')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        st.pyplot(plt)

    if c == 1:
        fig = plt.figure(figsize=(7, 2))
        plt.plot(vetor_temporal, coluna_11, 'k')
        plt.plot([valor, valor], [0, 3000], '--b')
        plt.xlabel('Time (s)')
        plt.ylabel('Anterior displacement (mm)')
        st.pyplot(plt)

        fig = plt.figure(figsize=(7, 2))
        plt.plot(vetor_temporal, coluna_12, 'k')
        plt.plot([valor, valor], [0, 1500], '--b')
        plt.xlabel('Time (s)')
        plt.ylabel('Vertical displacement (mm)')
        st.pyplot(plt)
