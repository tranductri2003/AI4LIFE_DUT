#!bin/bash
docker run --rm --name all_newbie_round_1 \
	-e JUPYTER_TOKEN=1234 \
	-p 8888:8888 \
	-v ./output/:/home/jovyan/notebooks/output all_newbie_ai4life/round1
