
using WAV
using TyPlot
using TySignalProcessing
using TyDSPSystem
using TyMath
using TySystemIdentification
using PyCall

librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")


# 1. 加载音频文件
input_neg = "D:/julia/sars/neg_wav/neg-0421-083-cough-m-53-0.wav"  
input_pos = "D:/julia/sars/pos_wav/pos-0421-084-cough-m-50-0.wav"  


# 使用 WAV 模块读取音频文件
println("加载音频文件...")
afr_neg, Fs = wavread(input_neg)
afr_pos, Fs = wavread(input_pos)
# 显示音频基本信息
println("采样率: $Fs Hz")
println("时长: $(length(afr_neg) / Fs) 秒")

figure(1)
subplot(2,2,1)
plot(afr_neg[:, 1])  # 绘制一个通道的波形
title("某个Sars-Cov-2阴性咳嗽时域波形")
grid("on")


N = length(afr_neg[:, 1])  # 获取信号长度
Y_neg = fft(afr_neg[:, 1])  # 对信号进行快速傅里叶变换
Y_neg = fftshift(Y_neg)  # 将零频分量移动到中心
freq = LinRange(-Fs/2, Fs/2, N)  # 创建频率轴

subplot(2,2,2)
plot(freq, abs.(Y_neg))  # 绘制频谱的幅度
title("某个Sars-Cov-2阴性咳嗽FFT频谱")
xlabel("频率 (Hz)")
ylabel("幅度")
grid("on")

subplot(2,2,3)
plot(afr_pos[:, 1])  # 绘制一个通道的波形
title("某个Sars-Cov-2阳性咳嗽时域波形")
grid("on")


N = length(afr_pos[:, 1])  # 获取信号长度
Y_pos = fft(afr_pos[:, 1])  # 对信号进行快速傅里叶变换
Y_pos = fftshift(Y_pos)  # 将零频分量移动到中心
freq = LinRange(-Fs/2, Fs/2, N)  # 创建频率轴

subplot(2,2,4)
plot(freq, abs.(Y_pos))  # 绘制频谱的幅度
title("某个Sars-Cov-2阳性咳嗽FFT频谱")
xlabel("频率 (Hz)")
ylabel("幅度")
grid("on")

# 声谱图分析
# 定义窗口函数和参数
window = hann(128)  # 汉明窗，窗长度128
overlap = 96  # 窗口重叠长度

# 对阴性咳嗽信号进行STFT并绘制声谱图
figure(2)
s_neg, f_neg, t_neg = stft(afr_neg, Fs; Window=window,OverlapLength=overlap,plotfig=true)


# 对阳性咳嗽信号进行STFT并绘制声谱图
figure(3)
s_pos, f_pos, t_pos = stft(afr_pos, Fs; Window=window,OverlapLength=overlap,plotfig=true)

# 使用librosa计算梅尔频谱
function compute_mel_spectrogram(file_path)
    # 加载音频文件
    y, sr = librosa.load(file_path, sr=16000)  # 采样率设为16kHz

    # 计算梅尔频谱
    n_fft = 2048  # FFT窗口大小
    hop_length = 512  # 帧移
    n_mels = 40  # 梅尔滤波器数量

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # 转换为dB刻度
    S_dB = librosa.power_to_db(S, ref=np.max)

    return S_dB, sr, hop_length
end

function plot_mel_spectrogram(S_dB, sr, hop_length, title)
    # 可视化梅尔频谱
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()
end

# 计算并绘制阴性咳嗽的梅尔频谱
S_dB_neg, sr_neg, hop_length_neg = compute_mel_spectrogram(input_neg)
plot_mel_spectrogram(S_dB_neg, sr_neg, hop_length_neg, "Mel-frequency spectrogram (Negative)")

# 计算并绘制阳性咳嗽的梅尔频谱
S_dB_pos, sr_pos, hop_length_pos = compute_mel_spectrogram(input_pos)
plot_mel_spectrogram(S_dB_pos, sr_pos, hop_length_pos, "Mel-frequency spectrogram (Positive)")