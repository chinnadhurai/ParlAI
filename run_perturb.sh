#!/bin/sh


LOGDIR="perturb_log_files/"
mkdir -p $LOGDIR

if [ "$1" = "s2s" ]
then
    echo "MODELTYPE: "$1
    TAG="s2s"
    MODEL_ARGS="-m seq2seq"
elif [ "$1" = "s2s_att_general" ]
then
    echo "MODELTYPE: "$1
    TAG="s2s_att_general"
    MODEL_ARGS="-m seq2seq -att general"
elif [ "$1" = "transformers" ]
then
    echo "MODELTYPE: "$1
    TAG="t"
    MODEL_ARGS="-m fairseq -bs 64 --arch transformer --share-all-embeddings"
else
    echo "INVALID modeltype : "$1" Supported : s2s, s2s_att_general, transformers"
    echo "Example command : sh run_perturb_model.sh <MODEL_TYPE>"
    exit 0
fi

for DATASET in "dailydialog" # "personachat"
do
    for MODEL_TYPE in $1
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_NoPerturb"
            LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_no_perturb.txt"
            python -W ignore examples/eval_model.py $MODEL_ARGS -t $DATASET -mf "save_dir/model_"$DATASET"_"$TAG -sft True -pb "None" --datatype $DATATYPE > $LOGFILE
            grep FINAL_REPORT $LOGFILE

            for PERTURB_TYPE in "swap" "repeat" "drop"
            do
                for PERTURB_LOC in "first" "last" "random"
                do
                    echo "---------------------"
                    echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC
                    LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC".txt"
                    python -W ignore examples/eval_model.py $MODEL_ARGS -t $DATASET -mf "save_dir/model_"$DATASET"_"$TAG -sft True -pb $PERTURB_TYPE"_"$PERTURB_LOC --datatype $DATATYPE > $LOGFILE
                    grep FINAL_REPORT $LOGFILE
                done
            done
        done 
    done
done
