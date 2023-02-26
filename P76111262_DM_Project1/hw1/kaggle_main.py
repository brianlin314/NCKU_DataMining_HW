from pathlib import Path
from typing import List
import kaggle_utils
from kaggle_utils import l
import config
import kaggle_args
import kaggle

def main():
    # Parse command line arguments
    a = kaggle_args.parse_args()
    l.info(f"Arguments: {a}")

    # Load dataset, the below io handles ibm dataset
    input_data: List[List[str]] = kaggle_utils.read_file(config.IN_DIR / a.dataset)
    filename = Path(a.dataset).stem
    print(input_data)

    apriori_out=kaggle.apriori_brian(input_data, a.min_sup, a.min_conf)
    print("apriori_out",apriori_out)

    kaggle_utils.write_file(
        data=apriori_out,
        filename=config.OUT_DIR / f"{filename}-kaggle.csv"
    )

if __name__ == "__main__":
    main()