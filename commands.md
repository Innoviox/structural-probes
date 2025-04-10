python3.12 structural-probes/run_experiment.py example/deepseek-small.yaml
 python3.12 scripts/generate_hdf5.py --model gpt2 --conllx data/dev.conllx --output embeddings/dev.hdf5e

scp -r structural-probes/ scherven@nexusclip.umiacs.umd.edu:/nfshomes/scherven/structural-probes/