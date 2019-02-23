#!/bin/sh


LOGDIR="perturb_log_files/"
mkdir -p $LOGDIR

for DATASET in "personachat" "dailydialog"
do
    for MODEL_TYPE in "seq2seq" #"seq2seq_attn" "transfomers"  for DATATYPE in "valid" "test"
    do
        for PERTURB_TYPE in "swap" "repeat" "drop"
        do
            for PERTURB_LOC in "first" "last" "random"
            do
                for DATATYPE in "valid" "test"
                do
                    LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE$PERTURB_LOC".txt"
                    python -W ignore examples/eval_model.py -m $MODEL_TYPE -t $DATASET -mf "save_dir/model_"$DATASET"_s2s" -sft True -pb $PERTURB_TYPE_$PERTURB_LOC --datatype $DATATYPE > $LOGFILE
                    echo "---------------------"
                    echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE$PERTURB_LOC
                    grep FINAL_REPORT $LOGFILE
                    echo "---------------------"
                done
            done
        done 
    done
done
