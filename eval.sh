# Run decoder on dev set
python -m ai.tests.char_qalb --model_name=$1 --word_embeddings=wide --decode=ai/datasets/data/qalb/QALB.dev.mada.mle --output_path=output/$1/decoder.out

# Run m2scorer to get QALB shared task evaluations
python2 ai/tests/m2scripts/m2scorer.py -v --beta 1 output/$1/decoder.out ai/datasets/data/qalb/QALB.dev.m2 > output/$1/m2scorer.out

# Run error analysis for cleanup, insight and more evaluation metrics
python analysis.py output/$1/m2scorer.out > output/$1/analysis.out
