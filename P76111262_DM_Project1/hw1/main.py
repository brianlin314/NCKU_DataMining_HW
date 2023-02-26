from pathlib import Path
from typing import List
import utils
from utils import l
import config
import args
import apriori_LIN

# TODO: you have to implement this module by yourself
# import my_cool_algorithms

def main():
    # Parse command line arguments
    a = args.parse_args()
    l.info(f"Arguments: {a}")

    # Load dataset, the below io handles ibm dataset
    input_data: List[List[str]] = utils.read_file(config.IN_DIR / a.dataset)
    filename = Path(a.dataset).stem
    print(input_data)

    apriori_out=apriori_LIN.apriori_brian(input_data, a.min_sup, a.min_conf)
    print("apriori_out",apriori_out)

    utils.write_file(
        data=apriori_out,
        filename=config.OUT_DIR / f"{filename}-apriori.csv"
    )
    # # TODO: you have to implement this function by yourself
    # apriori_out = my_cool_algorithms.apriori(input_data, a)
    # # Write output to file
    # utils.write_file(
    #     data=apriori_out,
    #     filename=config.OUT_DIR / f"{filename}-apriori.csv"
    # )

    # # TODO: you have to implement this function by yourself
    # fp_growth_out = my_cool_algorithms.fp_growth(input_data, a)
    # # Write output to file
    # utils.write_file(
    #     data=fp_growth_out,
    #     filename=config.OUT_DIR / f"{filename}-fp_growth.csv"
    # )

if __name__ == "__main__":
    main()