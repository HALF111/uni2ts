export CUDA_VISIBLE_DEVICES=0
# model_name=moirai_1.0_R_small
# model_name=pretrain_200_epochs
# model_name=simple_data_filter
model_name=iqn_data_filter_ratio0.05_iqn1.5

# for pl in 96 192 336 720; do
for pl in 96 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=64 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=ETTh1 \
    data.mode=M \
    data.prediction_length=$pl
done

# for pl in 96 192 336 720; do
for pl in 96; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=64 \
    model.context_length=3000 \
    data=lsf_test \
    data.dataset_name=ETTh2 \
    data.mode=M \
    data.prediction_length=$pl
done

# for pl in 96 192 336 720; do
for pl in 96; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=128 \
    model.context_length=4000 \
    data=lsf_test \
    data.dataset_name=ETTm1 \
    data.mode=S \
    data.prediction_length=$pl
done

# for pl in 96 192 336 720; do
for pl in 96; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=64 \
    model.context_length=3000 \
    data=lsf_test \
    data.dataset_name=ETTm2 \
    data.mode=S \
    data.prediction_length=$pl
done


# for pl in 96 192 336 720; do
for pl in 96; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=64 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=electricity \
    data.mode=S \
    data.prediction_length=$pl
done

# for pl in 96 192 336 720; do
for pl in 96; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=$model_name \
    model.patch_size=128 \
    model.context_length=2000 \
    data=lsf_test \
    data.dataset_name=weather \
    data.mode=M \
    data.prediction_length=$pl
done