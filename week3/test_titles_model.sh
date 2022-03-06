usage()
{
  echo "Usage: $0 [-m /workspace/datasets/fasttext/title_model.bin] [-t threshold] [-s]"
  exit 2
}

MODEL_FILE="/workspace/datasets/fasttext/title_model.bin"
THRESHOLD="0.90"
STEM=""

while getopts ':m:t:s' c
do
  case $c in
    m) MODEL_FILE=$OPTARG ;;
    s) STEM="--stem" ;;
    t) THRESHOLD=$OPTARG ;;
    [?]) usage ;;
  esac
done
shift $((OPTIND -1))

set -x
python ./week3/testTitleModel.py --model "$MODEL_FILE" --threshold $THRESHOLD $STEM