#!/bin/bash

data_path=$1
data_output_dir=$2
dygiepp_output=$3
GPU=$4

source ~/anaconda3/etc/profile.d/conda.sh
python data_preparation_dygiepp.py --data_path ${data_path} --data_output_dir ${data_output_dir}

git clone https://github.com/dwadden/dygiepp.git
cd dygiepp

conda create --name dygiepp python=3.7
conda activate dygiepp
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH

chmod 777 ./scripts/pretrained/get_dygiepp_pretrained.sh
./scripts/pretrained/get_dygiepp_pretrained.sh
  
input_directory=${data_output_dir}    #../../../outputs/dygiepp_input/
output_directory=${dygiepp_output}    #../../../outputs/dygiepp_output/

mkdir ${output_directory}

for file in ${input_directory}*
do

        filename=$(basename ${file})
        echo '> dygiepp processing: '$filename
        if  [ ! -e ${output_directory}/${filename} ]; then
                allennlp predict pretrained/scierc.tar.gz $input_directory$filename --predictor dygie --include-package dygie --use-dataset-reader --output-file $output_directory${filename}  --cuda-device ${GPU}
        fi
done


