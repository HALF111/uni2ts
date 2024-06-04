export CUDA_VISIBLE_DEVICES=0
# model_name=moirai_1.0_R_small
model_name=pretrain_200_epochs
# model_name=simple_data_filter
# model_name=iqn_data_filter

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=32 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=electricity

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=32 \
  model.context_length=2000 \
  data=gluonts_test \
  data.dataset_name=solar-energy

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=32 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=walmart

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=32 \
  model.context_length=4000 \
  data=gluonts_test \
  data.dataset_name=jena_weather

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=32 \
  model.context_length=4000 \
  data=gluonts_test \
  data.dataset_name=istanbul_traffic

python -m cli.eval \
  run_name=pf_eval \
  model=$model_name \
  model.patch_size=64 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=turkey_power