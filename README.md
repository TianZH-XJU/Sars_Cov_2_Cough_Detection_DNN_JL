# 基于Julia的SARS-CoV-2咳嗽声分类模型

## 项目简介

本项目致力于开发一款基于Julia语言的先进咳嗽声分类模型，旨在通过分析咳嗽声信号辅助识别SARS-CoV-2感染，为公共卫生监测与快速筛查提供创新工具。

## 数据来源

我们特别感谢 [Virufy COVID-19 Open Cough Dataset](https://github.com/virufy/virufy-data) 提供的开源数据集。Virufy 是一个志愿者组织，致力于构建一个全球性的AI数据库，收集众包的咳嗽声音，以识别代表呼吸道疾病的模式，例如COVID-19。

## 背景与目标

SARS-CoV-2（新型冠状病毒）感染在全球引发了严重的公共卫生危机。及时准确地检测病毒感染对于控制疫情传播至关重要。本项目基于Virufy提供的数据集，利用Julia语言强大的数值计算能力和Flux框架高效的机器学习功能，构建了一个智能咳嗽声分类模型。该模型能够对咳嗽声进行自动分析和分类，辅助医疗专业人员快速识别潜在的SARS-CoV-2感染者，提高筛查效率，减轻传统检测方法的负担。

## 主要功能

### 数据预处理

  * **批量处理音频文件** ：支持对存储在指定文件夹中的阳性（SARS-CoV-2感染）和阴性（非感染）咳嗽声音频文件进行批量读取和预处理。
  * **短时傅里叶变换（STFT）特征提取** ：采用STFT技术对音频信号进行时频域转换，提取咳嗽声的关键特征，将时域信号转换为频域特征矩阵，为后续模型训练提供高质量的特征输入。
  * **归一化处理** ：对提取的特征数据进行归一化，消除特征间的量纲差异，提高模型训练的稳定性和收敛速度。

### 模型训练

  * **人工神经网络构建** ：基于Flux框架构建多层人工神经网络，包括输入层、多个隐藏层（如全连接层）和输出层。网络结构设计充分考虑咳嗽声特征的复杂性和分类任务的需求。
  * **正则化与 dropout 技术** ：为防止模型过拟合，提高泛化能力，训练过程中采用正则化（如 L2 正则化）和 dropout 技术。这些技术通过限制模型复杂度和随机丢弃部分神经元输出，有效减少模型对训练数据的过度拟合，使其在未知数据上具有更好的分类性能。
  * **模型保存与加载** ：定期保存训练过程中的模型参数，便于后续的模型评估、测试和部署。同时支持加载已保存的模型参数，继续训练或直接进行推理。

### 模型评估与可视化

  * **多指标评估** ：在测试集上全面评估模型性能，计算并输出准确率、召回率等多种分类评估指标，从不同角度衡量模型对 SARS-CoV-2 感染咳嗽声的识别能力。
  * **损失曲线绘制** ：绘制训练过程中的损失函数变化曲线，直观展示模型的学习效果和收敛趋势，帮助调整训练参数和优化模型结构。
  * **预测结果可视化** ：通过绘制预测标签与真实标签对比图、混淆矩阵等可视化图表，清晰呈现模型的分类结果和错误分布情况，便于进一步分析模型的优势与不足。

## 技术优势与创新点

### 高效的特征提取

  * **STFT 优化参数** ：经过反复实验和优化，选定合适的 STFT 窗函数（如汉明窗）、窗口长度（128）和窗口重叠长度（96），在保留咳嗽声关键特征的同时，有效降低特征维度，提高计算效率。
  * **特征增强** ：除了基本的 STFT 模值特征，后续还可拓展加入其他声学特征（如梅尔频谱、MFCC 等），进一步丰富特征表达，提升模型分类性能。

### 强大的模型架构

  * **深度神经网络设计** ：采用多层神经网络结构，通过堆叠多个全连接层和批归一化层，增强模型的非线性拟合能力和特征学习能力，能够有效捕捉咳嗽声中复杂的声学模式和细微差异。
  * **正则化与 dropout 融合** ：创新性地将正则化项直接添加到损失函数中，并结合 dropout 技术，在保证模型表达能力的同时，显著提高模型的泛化性能，使其在面对多样化的咳嗽声数据时具有更强的适应性和稳定性。

### 开放式架构与可扩展性

  * **模块化设计** ：整个项目采用模块化编程思想，将数据预处理、模型构建、训练评估等环节分离为独立的模块，便于代码维护、功能扩展和团队协作开发。
  * **易于集成新算法** ：支持方便地集成新的特征提取算法、深度学习架构和优化方法，能够紧跟学术界和工业界的最新研究成果，不断优化和升级模型性能。
  * **跨平台兼容性** ：基于纯 Julia 语言开发，充分利用 Julia 的跨平台特性，可在多种操作系统（如 Windows、Linux、macOS）上无缝运行，降低部署成本，提高模型的可用性。

## 数据安全与隐私保护

  * **数据匿名化处理** ：在处理咳嗽声音频数据时，对数据进行匿名化处理，去除任何可识别个人身份的信息，确保数据的使用符合隐私保护法规和伦理要求。
  * **数据加密存储** ：对存储的音频数据和模型参数采用加密措施，防止数据泄露和未授权访问，保障数据的安全性和完整性。
  * **合规性与伦理考量** ：严格遵守相关数据保护法规（如 GDPR 等），在数据收集、使用和共享过程中充分尊重用户隐私，确保项目的开展符合伦理道德和法律规范。

## 使用方法

  * **环境准备** ：确保已安装MWork.Syslab Julia 运行时环境，并通过 Julia 的包管理器安装所需的依赖包，包括 Flux、WAV、TyPlot、TySignalProcessing 等。
  * **数据组织** ：将阳性咳嗽声音频文件和阴性咳嗽声音频文件分别放置在指定的文件夹中，确保文件格式为 WAV 或其他常见音频格式。
  * **运行脚本** ：运行项目主脚本，依次执行数据预处理、模型训练、评估和可视化等操作。在运行过程中，可根据提示调整模型参数、训练轮数等超参数，以获得最佳的分类效果。
  * **结果分析与应用** ：训练完成后，查看输出的评估指标、损失曲线和可视化图表，分析模型性能。将训练好的模型应用于实际场景中，对新的咳嗽声数据进行快速分类和预测，辅助医疗决策。

## 开源许可

本项目遵循开源协议（具体协议类型详见 LICENSE 文件），旨在促进科学技术的交流与共享，推动咳嗽声分析技术在医疗健康领域的应用与发展。欢迎广大开发者、研究人员和医疗专业人士积极参与项目贡献，共同完善和优化该模型，为全球疫情防控和公共卫生事业贡献力量。

## 致谢

感谢 [Virufy COVID-19 Open Cough Dataset](https://github.com/virufy/virufy-data) 提供的开源数据集。Virufy 是一个志愿者组织，致力于构建一个全球性的AI数据库，收集众包的咳嗽声音，以识别代表呼吸道疾病的模式，例如COVID-19。感谢他们为全球疫情防控所做的贡献。
感谢新疆大学计算机科学与技术学院徐学斌老师和MWorks教学组的指导
感谢所有为项目提供数据、技术支持和建议的个人和机构，特别鸣谢相关研究机构或数据提供方在 SARS-CoV-2 咳嗽声数据收集与整理方面的辛勤工作，本项目的成功离不开各方的共同努力与协作。

## English Translate
Julia-based SARS-CoV-2 Cough Sound Classification Model
Project Introduction
This project aims to develop an advanced cough sound classification model based on the Julia language to assist in identifying SARS-CoV-2 infections through cough sound signal analysis, providing an innovative tool for public health monitoring and rapid screening.
Data Source
Special thanks to the Virufy COVID-19 Open Cough Dataset for providing the open-source dataset. Virufy is a volunteer-run organization dedicated to building a global AI database of crowdsourced cough sounds to identify patterns indicative of respiratory diseases such as COVID-19.
Background and Objectives
The COVID-19 pandemic has posed a severe threat to global public health. Timely and accurate virus detection is crucial for controlling the spread of the pandemic. Based on the Virufy dataset, this project utilizes Julia's powerful numerical computing capabilities and the efficient machine learning framework Flux to construct an intelligent cough sound classification model. The model can automatically analyze and classify cough sounds, assisting medical professionals in quickly identifying potential SARS-CoV-2-infected individuals, improving screening efficiency, and alleviating the burden of traditional testing methods.
Key Features
Data Preprocessing
Batch Audio File Processing : Supports batch reading and preprocessing of positive (SARS-CoV-2 infected) and negative (non-infected) cough sound audio files stored in designated folders.
Short-Time Fourier Transform (STFT) Feature Extraction : Adopts STFT technology to convert audio signals from the time domain to the frequency domain, extracting key features of cough sounds and providing high-quality feature inputs for subsequent model training.
Normalization : Normalizes the extracted features to eliminate dimensional differences between features, enhancing the stability and convergence speed of model training.
Model Training
Artificial Neural Network Construction : Based on the Flux framework, a multi-layer artificial neural network is constructed, including input, multiple hidden layers (such as fully connected layers), and output layers. The network structure is designed to accommodate the complexity of cough sound features and the requirements of the classification task.
Regularization and Dropout Techniques : To prevent model overfitting and enhance generalization ability, regularization (e.g., L2 regularization) and dropout techniques are employed during training. These techniques restrict model complexity and randomly discard partial neuron outputs, effectively reducing overfitting to the training data and improving the model's performance on unseen data.
Model Saving and Loading : The model parameters are regularly saved during training to facilitate subsequent model evaluation, testing, and deployment. Additionally, saved model parameters can be loaded to continue training or directly for inference.
Model Evaluation and Visualization
Multi-Metric Evaluation : The model's performance is comprehensively evaluated on the test set using multiple classification metrics, including accuracy and recall, to assess its ability to identify SARS-CoV-2-infected cough sounds from various perspectives.
Loss Curve Plotting : The loss function variation during training is visualized through a loss curve, intuitively displaying the model's learning progress and convergence trends. This helps in adjusting training parameters and optimizing the model structure.
Prediction Results Visualization : Prediction labels and true labels are compared and visualized using charts such as comparison plots and confusion matrices, clearly showing the model's classification results and error distributions for further analysis of its strengths and weaknesses.
Technical Advantages and Innovations
Efficient Feature Extraction
Optimized STFT Parameters : After extensive experiments and optimizations, suitable STFT window functions (e.g., Hamming window), window lengths (128), and window overlap lengths (96) have been selected. These parameters retain key cough sound features while effectively reducing feature dimensions and improving computational efficiency.
Feature Enhancement : In addition to basic STFT magnitude features, the integration of other acoustic features (such as Mel spectrograms and MFCCs) is planned to enrich feature representation and improve model classification performance.
Powerful Model Architecture
Deep Neural Network Design : A multi-layer neural network structure is adopted, with multiple fully connected layers and batch normalization layers stacked to enhance the model's non-linear fitting and feature learning capabilities. This enables the effective capture of complex acoustic patterns and subtle differences in cough sounds.
Integration of Regularization and Dropout : Regularization terms are innovatively incorporated into the loss function and combined with dropout techniques. This approach ensures expressive power while significantly enhancing the model's generalization performance, enabling it to better adapt to diverse cough sound data.
Open Architecture and Expandability
Modular Design : The project adopts a modular programming approach, separating data preprocessing, model construction, training, and evaluation into independent modules. This facilitates code maintenance, functional expansion, and team collaboration.
Easy Integration of New Algorithms : The project supports the convenient integration of new feature extraction algorithms, deep learning architectures, and optimization methods, enabling it to keep pace with the latest research findings from academia and industry to continuously optimize and upgrade model performance.
Cross-Platform Compatibility : Developed purely in Julia, the project leverages Julia's cross-platform capabilities to ensure seamless operation across multiple operating systems (such as Windows, Linux, and macOS), reducing deployment costs and enhancing model accessibility.
Data Security and Privacy Protection
Data Anonymization : During the processing of cough sound audio data, anonymization techniques are applied to remove any personally identifiable information, ensuring compliance with privacy protection regulations and ethical requirements.
Data Encryption : Audio data and model parameters are encrypted during storage to prevent data leakage and unauthorized access, ensuring data security and integrity.
Compliance and Ethical Considerations : Relevant data protection regulations (such as GDPR) are strictly adhered to, and user privacy is respected throughout the processes of data collection, usage, and sharing. The project is conducted in accordance with ethical and legal standards.
Usage Instructions
Environment Preparation : Ensure that the Julia runtime environment is installed, and use Julia's package manager to install the required dependencies, including Flux, WAV, TyPlot, TySignalProcessing, etc.
Data Organization : Place positive and negative cough sound audio files in designated folders, ensuring they are in WAV or other common audio formats.
Script Execution : Run the project's main script to sequentially perform data preprocessing, model training, evaluation, and visualization. Adjust model parameters and training epochs as prompted to achieve optimal classification results.
Result Analysis and Application : After training, review the output evaluation metrics, loss curves, and visualizations to analyze model performance. Apply the trained model to real-world scenarios to quickly classify and predict new cough sound data, aiding medical decision-making.
Open Source License
This project is released under an open-source license (the specific type is detailed in the LICENSE file). It aims to promote the exchange and sharing of scientific and technological knowledge and to advance the application and development of cough sound analysis technology in the field of healthcare. Contributors from the developer, research, and medical communities are welcome to participate in enhancing and refining the model to collectively contribute to global pandemic control and public health initiatives.
Acknowledgments
Gratitude is extended to the Virufy COVID-19 Open Cough Dataset for providing the open-source dataset. Virufy is a volunteer organization committed to building a global AI database of crowdsourced cough sounds to identify patterns indicative of respiratory diseases like COVID-19. Appreciation is also expressed to all individuals and institutions that have contributed data, technical support, and suggestions to the project. Special thanks are due to the relevant research institutions and data providers for their diligent work in collecting and preparing SARS-CoV-2 cough sound data, without which the success of this project would not have been possible.
