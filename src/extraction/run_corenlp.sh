
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dygiepp
pip install stanfordcorenlp networkx

wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.5.1.zip
unzip stanford-corenlp-latest.zip
rm stanford-corenlp-latest.zip

cd stanford-corenlp-4.5.0
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9050 -timeout 15000 -threads 4 &
corenlp_process_pid=$!
cd ..
echo $corenlp_process_pid
python corenlp_extractor.py 4
kill -9 $corenlp_process_pid

