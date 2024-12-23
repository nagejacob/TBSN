#!/bin/bash

experiment_home_dir="experiments"
config="tbsn_sidd.json"

experiment_name=$(python train/experiment_name.py --config option/${config})
#experiment_name='test'

experiment_dir="${experiment_home_dir}/${experiment_name}"
echo "experiment dir: ${experiment_dir}"

if [ ! -d "${experiment_home_dir}" ]
then
  mkdir "${experiment_home_dir}"
fi

if [ ! -d "${experiment_dir}" ]
then
  mkdir "${experiment_dir}"
else
  echo "experiment dir exists"
fi

cp -r "dataset" "${experiment_dir}"
cp -r "model" "${experiment_dir}"
cp -r "network" "${experiment_dir}"
cp -r "option" "${experiment_dir}"
cp -r "train" "${experiment_dir}"
cp -r "util" "${experiment_dir}"
cp -r "validate" "${experiment_dir}"

if [ ! -d "${experiment_dir}/log" ]
then
  mkdir "${experiment_dir}/log"
fi

cd ${experiment_dir}
export PYTHONPATH=$PWD:$PYTHONPATH
python train/base.py --config option/${config}