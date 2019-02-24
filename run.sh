#!/bin/sh


LOGDIR="perturb_log_files/"
SAVEDIR="save_dir/"
mkdir -p $LOGDIR

if [ -z "$1" ]
then
    echo "No Run mode provided. Supported : train, perturb"
    echo "Example train command : sh run.sh train <model_type> <dataset>"
    echo "Example perturb command : sh run.sh perturb <model_type> <dataset>"
    exit 0

else
    RUN_MODE=$1
fi

if [ "$2" = "s2s" ]
then
    echo "MODELTYPE: "$2
    EVAL_MODEL_ARGS="-m seq2seq"
    TRAIN_MODEL_ARGS=$EVAL_MODEL_ARGS" -vmt loss -eps 60 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 --bidirectional true"
elif [ "$2" = "s2s_att_general" ]
then
    echo "MODELTYPE: "$2
    EVAL_MODEL_ARGS="-m seq2seq -att general"
    TRAIN_MODEL_ARGS=$EVAL_MODEL_ARGS" -vmt loss -eps 60 -veps 1 -stim 600 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 --bidirectional true"
elif [ "$2" = "transformer" ]
then
    echo "MODELTYPE: "$2
    EVAL_MODEL_ARGS="-m fairseq -bs 32 --arch transformer --share-all-embeddings"
    TRAIN_MODEL_ARGS=$EVAL_MODEL_ARGS" -vmt loss -eps 25 -veps 1 -stim 600 --optimizer adam --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0" 
else
    echo "INVALID modeltype : "$2" Supported : s2s, s2s_att_general, transformer"
    echo "Example train command : sh run.sh train <model_type> <dataset>"
    echo "Example perturb command : sh run.sh perturb <model_type> <dataset>"
    exit 0
fi

if [ -z "$3" ]
then
    echo "No Dataset type specified supplied"
    echo "Example train command : sh run.sh train <model_type> <dataset>"
    echo "Example perturb command : sh run.sh perturb <model_type> <dataset>"
    exit 0
else
    DATASET=$3
    MF=$SAVEDIR"/model_"$3"_"$2
fi

if [ $RUN_MODE = "perturb" ]
then
    echo "MODE : "$RUN_MODE
    for MODEL_TYPE in $2
    do
        for DATATYPE in "test" #valid
        do
            echo "---------------------"
            echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_NoPerturb"
            LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_no_perturb.txt"
            python -W ignore examples/eval_model.py $EVAL_MODEL_ARGS -t $DATASET -mf $MF -sft True -pb "None" --datatype $DATATYPE > $LOGFILE
            grep FINAL_REPORT $LOGFILE

            for PERTURB_TYPE in "swap" "repeat" "drop"
            do
                for PERTURB_LOC in "first" "last" "random"
                do
                    echo "---------------------"
                    echo "CONFIG : "$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC
                    LOGFILE=$LOGDIR/log_$DATASET"_"$MODEL_TYPE"_"$DATATYPE"_"$PERTURB_TYPE"_"$PERTURB_LOC".txt"
                    python -W ignore examples/eval_model.py $EVAL_MODEL_ARGS -t $DATASET -mf $MF -sft True -pb $PERTURB_TYPE"_"$PERTURB_LOC --datatype $DATATYPE > $LOGFILE
                    grep FINAL_REPORT $LOGFILE
                done
            done
        done 
    done
elif [ $RUN_MODE = "train" ]
then
    echo $TRAIN_MODEL_ARGS
    python examples/train_model.py -t $DATASET -mf $MF $TRAIN_MODEL_ARGS
fi
