python data/create_pilot_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_20.yml \
  --out-dir ./00_pilot/inputs \
  --template bo_top=./official_config/config.yml \
  --template gallery=./official_config/config_gallery.yml \
  --template slider=./official_config/config_slider.yml \
  --par 4 \
  --seed 0 > ./00_pilot/run_sessions.sh;

python data/create_pilot_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_5.yml \
  --out-dir ./00_pilot/tutorial_inputs \
  --template bo_top=./official_config/tutorial/config.yml \
  --template gallery=./official_config/tutorial/config_gallery.yml \
  --template slider=./official_config/tutorial/config_slider.yml \
  --tutorial \
  --par 4 \
  --seed 0  > ./00_pilot/run_init.sh;
