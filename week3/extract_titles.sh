
usage()
{
  echo "Usage: $0 [-i /workspace/search_with_machine_learning_course/data/pruned_products] [-o /workspace/datasets/fasttext/titles.txt] [-r sample_rate] [-s]"
  exit 2
}

INPUT_DIR="/workspace/search_with_machine_learning_course/data/pruned_products"
OUTPUT_TITLES_FILE="/workspace/datasets/fasttext/titles.txt"
SAMPLE_RATE="0.1"
STEM=""

while getopts ':i:o:r:s:' c
do
  case $c in
    i) INPUT_DIR=$OPTARG ;;
    o) OUTPUT_TITLES_FILE=$OPTARG ;;
    r) SAMPLE_RATE=$OPTARG ;;
    s) STEM="--stem" ;;
    [?]) usage ;;
  esac
done
shift $((OPTIND -1))

set -x
python ./week3/extractTitles.py --input "$INPUT_DIR" --output "$OUTPUT_TITLES_FILE" --sample_rate $SAMPLE_RATE $STEM