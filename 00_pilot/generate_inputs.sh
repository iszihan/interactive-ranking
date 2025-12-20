python data/create_pilot_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_20.yml \
  --out-dir /scratch/ondemand29/chenxil/code/interactive-ranking/00_pilot/inputs \
  --template bo=./config.yml \
  --template gallery=./config_gallery.yml \
  --template slider=./config_slider.yml \
  --par 2 \
  --seed 0;

python data/create_pilot_inputs.py \
  --classes-json /scratch/ondemand29/chenxil/code/mood-board/evaluations/classes.json \
  --models-yml /scratch/ondemand29/chenxil/code/mood-board/config/sdxl_loras_5.yml \
  --out-dir /scratch/ondemand29/chenxil/code/interactive-ranking/00_pilot/tutorial_inputs \
  --template bo=./config.yml \
  --template gallery=./config_gallery.yml \
  --template slider=./config_slider.yml \
  --tutorial \
  --par 2 \
  --seed 0;
