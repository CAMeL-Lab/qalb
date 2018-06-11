#embeddings="narrow"
dataset="mada.mle"

echo "Running seq2seq on dev..."
python -m ai.tests.char_qalb --model_name="$1"_mada --word_embeddings=$1 --decode=ai/datasets/data/qalb/QALB.dev.$dataset --extension=$dataset --beam_size=5 --output_path=output/"$1"_mada/decoder_dev.out

echo "Running m2scorer on dev..."
python2 ai/tests/m2scripts/m2scorer.py -v --beta 1 output/"$1"_mada/decoder_dev.out ai/datasets/data/qalb/QALB.dev.m2 > output/"$1"_mada/m2scorer_dev.out

python analysis.py output/"$1"_mada/m2scorer_dev.out > output/"$1"_mada/analysis_dev.out

exit

echo "Running seq2seq on test2014..."
python -m ai.tests.char_qalb --model_name="$1"_mada --word_embeddings=$1 --decode=ai/datasets/data/qalb/QALB.test2014.$dataset --extension=$dataset --output_path=output/"$1"_preproc/decoder_test2014.out

echo "Running m2scorer on test2014..."
python2 ai/tests/m2scripts/m2scorer.py -v --beta 1 output/"$1"_preproc/decoder_test2014.out ai/datasets/data/qalb/QALB.test2014.m2 > output/"$1"_preproc/m2scorer_test2014.out

python analysis.py output/"$1"_preproc/m2scorer_test2014.out > output/"$1"_preproc/analysis_test2014.out


echo "Running seq2seq on test2015..."
python -m ai.tests.char_qalb --model_name="$1"_mada --word_embeddings=$1 --decode=ai/datasets/data/qalb/QALB.test2015.$dataset --extension=$dataset --output_path=output/"$1"_preproc/decoder_test2015.out

echo "Running m2scorer on test2015..."
python2 ai/tests/m2scripts/m2scorer.py -v --beta 1 output/"$1"_preproc/decoder_test2015.out ai/datasets/data/qalb/QALB.test2015.m2 > output/"$1"_preproc/m2scorer_test2015.out

python analysis.py output/"$1"_preproc/m2scorer_test2015.out > output/"$1"_preproc/analysis_test2015.out

