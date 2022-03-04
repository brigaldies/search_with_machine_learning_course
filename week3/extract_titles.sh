
usage()
{
  echo "Usage: $0 [-s /workspace/search_with_machine_learning_course/data/pruned_products] [-o /workspace/datasets/fasttext/titles.txt]"
  exit 2
}

SOURCE_DIR="/workspace/search_with_machine_learning_course/data/pruned_products"
OUTPUT_TITLES_FILE="/workspace/datasets/fasttext/titles.txt"

while getopts ':s:o' c
do
  case $c in
    s) SOURCE_DIR=$OPTARG ;;
    o) OUTPUT_TITLES_FILE=$OPTARG ;;
    [?]) usage ;;
  esac
done
shift $((OPTIND -1))

set -x
python ./week3/extractTitles.py --input "$SOURCE_DIR" --output "$OUTPUT_TITLES_FILE"