#!/bin/sh


LOGDIR="perturb_log_files/"
mkdir -p $LOGDIR

for DATASET in "dailydialog" "personachat"
do
    for MODEL_TYPE in "seq2seq" #"seq2seq_attn" "transfomers"  for DATATYPE in "valid" "test"
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_NoPerturb"
            LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_no_perturb.txt"
            python -W ignore examples/eval_model.py -m $MODEL_TYPE -t $DATASET -mf "save_dir/model_"$DATASET"_s2s" -sft True -pb "None" --datatype $DATATYPE > $LOGFILE
            grep FINAL_REPORT $LOGFILE
            
            for PERTURB_TYPE in "swap" "repeat" "drop"
            do
                for PERTURB_LOC in "first" "last" "random"
                do
                    echo "---------------------"
                    echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC
                    LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC".txt"
                    python -W ignore examples/eval_model.py -m $MODEL_TYPE -t $DATASET -mf "save_dir/model_"$DATASET"_s2s" -sft True -pb $PERTURB_TYPE"_"$PERTURB_LOC --datatype $DATATYPE > $LOGFILE
                    grep FINAL_REPORT $LOGFILE
                done
            done
        done 
    done
done
