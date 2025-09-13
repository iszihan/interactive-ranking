This hosts the UI codebase for LoRA-Moodboard project.

Currently, to run the UI, one need to set an environment variable to the script we want to run when hitting the START button like

```export SCRIPT_CMD="python test.py & & scp ../lora-moodboard/mood-board/search_benchmark/pair_experiments/interactive_test_run/_s00/init* ./outputs/"```

and then run `python server.py` to start the webpage. 