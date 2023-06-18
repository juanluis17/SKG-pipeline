#!/bin/bash

dataset_dump_dir=$1
dygiepp_output_dump_dir=$2
output_dir=$3

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dygiepp
pip install networkx
pip install stanfordcorenlp
pip install cso-classifier


if  [ ! -d stanford-corenlp-4.5.1 ]; then
  wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.5.1.zip
  unzip stanford-corenlp-4.5.1.zip
  rm stanford-corenlp-4.5.1.zip
fi

cd stanford-corenlp-4.5.1
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9050 -timeout 15000 -threads 4 &
corenlp_process_pid=$!
cd ..
echo $corenlp_process_pid
python corenlp_extractor_patents.py --dataset_dump_dir ${dataset_dump_dir} --dygiepp_output_dump_dir ${dygiepp_output_dump_dir} --output_dir ${output_dir}
kill -9 $corenlp_process_pid

