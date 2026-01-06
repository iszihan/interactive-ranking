python data/create_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_20.yml \
  --out-dir ./01_study/inputs \
  --template bo_top=./official_config/config.yml \
  --template gallery=./official_config/config_gallery.yml \
  --template slider=./official_config/config_slider.yml \
  --par 12 \
  --input-dir /scratch/ondemand29/chenxil/code/mood-board/evaluations/input30_dist/ \
  --seed 0 > ./01_study/run_sessions.sh;

python data/create_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_5.yml \
  --out-dir ./01_study/tutorial_inputs \
  --template bo_top=./official_config/tutorial/config.yml \
  --template gallery=./official_config/tutorial/config_gallery.yml \
  --template slider=./official_config/tutorial/config_slider.yml \
  --tutorial \
  --par 12 \
  --seed 0  > ./01_study/run_init.sh;
