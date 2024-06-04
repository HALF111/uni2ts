# 1. 微调
# 1.1 将dataframe格式数据转换成uni2ts格式。需要先设置存储的目录
PATH_TO_SAVE=dataset_processed
echo "CUSTOM_DATA_PATH=dataset_processed" >> .env
# 1.2 开始转换。
# 1.2.1 可以修改dataset_type为wide，long或者wide_multivariate
# ETTh1数据集大小：487949 Bytes？
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --dataset_type wide
# 1.2.2 然而我们可能需要在微调期间进行验证集以执行超参数调整或提前停止。
# 为了另外将数据集分割为训练和验证分割，我们可以使用互斥的date_offset（日期时间字符串）或offset（整数）选项来确定训练集的最后一个时间步长。
# 验证集将保存为 DATASET_NAME_eval。
# ETTh1大小：变成322749 Bytes？
# 同时ETTh1_eval大小：变成487949 Bytes了？是因为eval事实上会包含全部的数据？
# * 之所以是2017-10-23，是因为：ETTh1包含2016-07-01到2018-06-26共两年的数据；而标准划分是6：2：2？也即1年：4个月：4个月
# 更确切的说是360天：120天：120天。
# 为了保证剩下的eval中恰为测试集的数据，可以发现2016-07-01过了480天后恰为2017-10-23。
# ? 但是2017-10-23到2018-06-26还有246天，这里如何保证只保留120天？？？？
# 事实上：322749/487949 == 480/(480+246) == 0.661，说明eval是包含全部数据，后面没有做截断的？
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --date_offset '2017-10-23 23:00:00'
# 1.3 使用适当的训练和验证数据配置开始微调
python -m cli.finetune run_name=example_run model=moirai_1.0_R_small data=etth1 val_data=etth1  


# 2. evaluation
# 可以使用MSE，MASE，CRPS等指标。参考conf/eval/default.yaml
# 2.1 仿照微调，可以对测试拆分进行评估：
# * 这里之所以etth1_test可以正常使用，是因为在/cli/conf/eval/data/中存在etth1_test.yaml文件！！
# * 其中写明了offset=14400, windows=2785, prediction_length=96，从而确定了实验的设置！！
# * 这里offser=14400，正好对应于12*30*34+4*24*30，表示test的起点。而2785==4*24*30-96+1，为样本的个数；其中96为pred_len。
# python -m cli.eval run_name=example_eval_1 model=moirai_1.0_R_small model.patch_size=32 model.context_length=1000 data=etth1_test
python -m cli.eval run_name=example_eval_1 model=moirai_1.0_R_small model.patch_size=32 model.context_length=1000 data=etth1_test data.mode=M
python -m cli.eval run_name=example_eval_1_new model=moirai_1.0_R_small model.patch_size=64 model.context_length=4000 data=etth1_test data.mode=M
# 2.2 或者可以通过数据配置进行切换。（参见/cli/conf/eval/data/中）
# 例子：假设想要再次对ETTh1进行评估。
# 2.2.1 首先需要设置TSLib存储库并下载预处理的数据集，并放入正确的目录中。然后，将数据集目录分配给LSF_PATH环境变量。
LSF_PATH=dataset  # 为当前存储数据的目录？应该是dataset？
echo "LSF_PATH=dataset" >> .env
# 2.2.2 使用预定义的Hydra配置文件运行以下脚本，Hydra文件可以参考conf/eval/data/lsf_test.yaml
# 而这些预定义的脚本是如何读取出数据的，可以参考：rc/uni2ts/eval_util/_lsf_dataset.py，以及src/val_util/data.py中的get_lsf_test_dataset类
python -m cli.eval run_name=example_eval_2 model=moirai_1.0_R_small model.patch_size=32 model.context_length=1000 data=lsf_test data.dataset_name=ETTh1 data.prediction_length=96 data.mode=M
python -m cli.eval run_name=example_eval_2 model=moirai_1.0_R_small model.patch_size=64 model.context_length=4000 data=lsf_test data.dataset_name=ETTh1 data.prediction_length=96 data.mode=M

python -m cli.eval run_name=example_eval_3 model=self_pretrain model.patch_size=64 model.context_length=4000 data=lsf_test data.dataset_name=ETTh1 data.prediction_length=96 data.mode=M


# ETTm2
python -m cli.eval run_name=example_eval_4 model=self_pretrain model.patch_size=128 model.context_length=16000 data=lsf_test data.dataset_name=ETTm2 data.prediction_length=96 data.mode=M


# 3. 预训练
# 3.1 下载数据
PATH_TO_SAVE=lotsa_data
export HF_ENDPOINT=https://hf-mirror.com  # 使用huggingface镜像，加快下载速度
huggingface-cli download Salesforce/lotsa_data --repo-type=dataset --local-dir lotsa_data
echo "LOTSA_V1_PATH=lotsa_data" >> .env
# 3.2 pre-train
# 具体设置可以参考cli/pretrain.py
python -m cli.train run_name=first_run model=moirai_small data=lotsa_v1_unweighted
python ./cli/train.py run_name=first_run model=moirai_small data=lotsa_v1_unweighted


# 新的evaluation：直接调用写好的脚本即可
. venv/bin/activate
sh project/moirai-1/eval/monash_small.sh

# 如何可视化loss？
tensorboard --logdir=outputs/train/moirai_small/lotsa_v1_unweighted/run_200_epochs