using WAV
using TyPlot
using TySignalProcessing
using TyDSPSystem
using TyMath
using TySystemIdentification
using PyCall

# 设置文件夹路径
pos_wav_folder = "D:/julia/sars/pos_wav"
neg_wav_folder = "D:/julia/sars/neg_wav"

# 获取所有音频文件路径
pos_files = readdir(pos_wav_folder)
neg_files = readdir(neg_wav_folder)

# 初始化存储特征和标签的列表
features = []
labels = []

# 处理阳性咳嗽声
println("处理阳性咳嗽声...")
for file in pos_files
    file_path = joinpath(pos_wav_folder, file)
    println("加载音频文件 $file...")
    audio, Fs = wavread(file_path)
    window = hann(128)  # 汉明窗，窗长度128
    overlap = 96  # 窗口重叠长度
    s, f, t = stft(audio, Fs; Window=window, OverlapLength=overlap, plotfig=false)
    push!(features, abs.(s))  # 将复数转换为实数（取模）
    push!(labels, 1)
end

# 处理阴性咳嗽声
println("处理阴性咳嗽声...")
for file in neg_files
    file_path = joinpath(neg_wav_folder, file)
    println("加载音频文件 $file...")
    audio, Fs = wavread(file_path)
    window = hann(128)  # 汉明窗，窗长度128
    overlap = 96  # 窗口重叠长度
    s, f, t = stft(audio, Fs; Window=window, OverlapLength=overlap, plotfig=false)
    push!(features, abs.(s))  # 将复数转换为实数（取模）
    push!(labels, 0)
end

using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle

# 将特征和标签转换为矩阵形式
X = hcat([features[i][:] for i in 1:length(features)]...)
Y = Flux.onehotbatch(labels, 0:1)

# 打印特征和标签的形状和数据类型
println("特征矩阵 X 的形状: $(size(X))")
println("特征矩阵 X 的数据类型: $(eltype(X))")
println("标签 Y 的形状: $(size(Y))")
println("标签 Y 的数据类型: $(eltype(Y))")

# 检查是否有 NaN 或 Inf 值
if any(isnan.(X)) || any(isinf.(X))
    println("警告：特征矩阵 X 包含 NaN 或 Inf 值")
end
if any(isnan.(Y)) || any(isinf.(Y))
    println("警告：标签 Y 包含 NaN 或 Inf 值")
end

# 数据归一化
X_mean = mean(X, dims=2)
X_std = std(X, dims=2)

# 增加对零标准差的处理
X_std = [std_val > 0 ? std_val : 1.0 for std_val in X_std]

X = (X .- X_mean) ./ X_std

# 再次检查是否有 NaN 或 Inf 值
if any(isnan.(X)) || any(isinf.(X))
    println("警告：归一化后特征矩阵 X 仍然包含 NaN 或 Inf 值")
end

# 划分训练集和测试集
train_ratio = 0.8  # 80%的数据用于训练
n_samples = size(X, 2)
train_size = Int(floor(train_ratio * n_samples))

# 随机打乱数据
using Random
Random.seed!(42)
perm = randperm(n_samples)
X = X[:, perm]
Y = Y[:, perm]

# 划分训练集和测试集
X_train = X[:, 1:train_size]
Y_train = Y[:, 1:train_size]
X_test = X[:, train_size+1:end]
Y_test = Y[:, train_size+1:end]

# 构建模型
input_size = size(X, 1)
model = Chain(
    Dense(input_size => 128),
    BatchNorm(128, relu),
    Dropout(0.5),
    Dense(128 => 64),
    BatchNorm(64, relu),
    Dropout(0.5),
    Dense(64 => 2),
    softmax
)

# 选择优化器和损失函数
learn_rate = 0.001
opt = ADAM(learn_rate)
loss(x, y) = crossentropy(model(x), y)


using BSON: @save
# 训练模型
println("开始训练模型...")
epochs = 100
loss_history = Float64[]
for epoch in 1:epochs
    # 训练
    Flux.train!(loss, Flux.params(model), [(X_train, Y_train)], opt)
    
    # 计算训练损失
    train_loss = loss(X_train, Y_train)
    push!(loss_history, train_loss)
    
    # 打印日志
    println("Epoch: $epoch, Train Loss: $(train_loss)")
    # if epoch % 10 == 0
    #     @save "model_epoch_$(epoch).bson" model
    # end
end

# 绘制损失曲线
println("绘制损失曲线...")
using TyPlot
figure(1)
plot(1:epochs, loss_history)
xlabel("Epoch")
ylabel("Loss")
title("Training Loss")
grid("on")
# 打印训练完成信息
println("模型训练完成")

# 评估模型
println("评估模型...")
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
train_acc = accuracy(X_train, Y_train)
test_acc = accuracy(X_test, Y_test)
println("训练集准确率: $(train_acc)")
println("测试集准确率: $(test_acc)")


# 获取模型预测结果
y_hat = onecold(model(X_test), 0:1)
y_test = onecold(Y_test, 0:1)

# 计算二分类指标
acc2 = (count(y_hat[y_test .== 0] .== 0) + count(y_hat[y_test .== 1] .== 1)) / length(y_test)
recall2 = count(y_hat[y_test .== 1] .== 1) / count(y_test .== 1)

println("二分类准确率: $acc2")
println("二分类召回率: $recall2")

# 绘制预测标签与真实标签对比图
println("绘制预测标签与真实标签对比图...")
figure(2)
subplot(2,2,1)
plot(1:length(y_test), y_test)
xlabel("样本索引")
ylabel("标签")
title("预测标签与真实标签对比")
subplot(2,2,2)
plot(1:length(y_test), y_hat)
xlabel("样本索引")
ylabel("预测标签")
title("预测标签与真实标签对比")
subplot(2,2,3)
scatter(1:length(y_test), y_test, label="")
subplot(2,2,4)
scatter(1:length(y_test), y_hat, label="")
grid("on")



# 保存最终模型
@save "final_model.bson" model

# 打印训练完成信息
println("模型训练完成，最终模型已保存到 final_model.bson")