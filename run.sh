while getopts f:m: flag
do
    case "${flag}" in
        f) gen_file=${OPTARG};;
        m) run_model=${OPTARG};;
    esac
done
if [ $gen_file == "T" ]; then
    echo "Generating files"
    python generate_files.py --text_mode ade --gen_files True --max_train 8 --max_dev 1;
else
    echo "Not generating anything"
fi
if [ $run_model == "T" ]; then
    echo "Training model"
    python trainer.py -d ade -c with_external_knowledge  --max_train 8 --max_dev 1
fi