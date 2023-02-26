import logging
import csv
import time
from pathlib import Path
from typing import Any, List, Union

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Running {func.__name__} ...", end='\r')
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} Done in {end - start:.2f} seconds")
        return result
    return wrapper

@timer
def read_file(filename: Union[str, Path]) -> List[List[any]]:
    """read_file

    Args:
        filename (Union[str, Path]): The filename to read

    Returns:
        List[List[int]]: The data in the file
    """
    bread_list=['Bread' ,'Scandinavian' ,'Hot chocolate' ,'Jam' ,'Cookies' ,'Muffin'
    ,'Coffee' ,'Pastry' ,'Medialuna' ,'Tea' ,'Tartine' ,'Basket'
    ,'Mineral water' ,'Farm House' ,'Fudge' ,'Juice' ,"Ella's Kitchen Pouches"
    ,'Victorian Sponge' ,'Frittata' ,'Hearty & Seasonal' ,'Soup'
    ,'Pick and Mix Bowls' ,'Smoothies' ,'Cake' ,'Mighty Protein' ,'Chicken sand'
    ,'Coke' ,'My-5 Fruit Shoot' ,'Focaccia' ,'Sandwich' ,'Alfajores' ,'Eggs'
    ,'Brownie' ,'Dulce de Leche' ,'Honey' ,'The BART' ,'Granola' ,'Fairy Doors'
    ,'Empanadas' ,'Keeping It Local' ,'Art Tray' ,'Bowl Nic Pitt' ,'Bread Pudding'
    ,'Adjustment' ,'Truffles' ,'Chimichurri Oil' ,'Bacon' ,'Spread' ,'Kids biscuit'
    ,'Siblings' ,'Caramel bites' ,'Jammie Dodgers' ,'Tiffin' ,'Olum & polenta'
    ,'Polenta' ,'The Nomad' ,'Hack the stack' ,'Bakewell' ,'Lemon and coconut'
    ,'Toast' ,'Scone' ,'Crepes' ,'Vegan mincepie' ,'Bare Popcorn' ,'Muesli'
    ,'Crisps' ,'Pintxos' ,'Gingerbread syrup' ,'Panatone' ,'Brioche and salami'
    ,'Afternoon with the baker' ,'Salad' ,'Chicken Stew' ,'Spanish Brunch'
    ,'Raspberry shortbread sandwich' ,'Extra Salami or Feta' ,'Duck egg'
    ,'Baguette' ,"Valentine's card" ,'Tshirt' ,'Vegan Feast' ,'Postcard'
    ,'Nomad bag' ,'Chocolates' ,'Coffee granules ' ,'Drinking chocolate spoons '
    ,'Christmas common' ,'Argentina Night' ,'Half slice Monster ' ,'Gift voucher'
    ,'Cherry me Dried fruit' ,'Mortimer' ,'Raw bars' ,'Tacos/Fajita']
    bread_dict={}
    for i in range(len(bread_list)):
        bread_dict[bread_list[i]]=str(i+1)
    print(bread_dict)
    print(bread_dict['Bread'])

    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    df=[]
    count=0
    for i in range(1,len(rows)):
        temp=[]
        item=rows[i][3]
        if item!="NONE":
            n=bread_dict[item]
            temp.append(rows[i][2])
            temp.append(rows[i][2])
            temp.append(n)
            df.append(temp)
        else:
            count+=1
    print(df)
    print("NONE 總數:",count)

    return df

@timer
def write_file(data: List[List[Any]], filename: Union[str, Path]) -> None:
    """write_file writes the data to a csv file and
    adds a header row with `relationship`, `support`, `confidence`, `lift`.

    Args:
        data (List[List[Any]]): The data to write to the file
        filename (Union[str, Path]): The filename to write to
    """
    header = ['antecedent', 'consequent', 'support', 'confidence', 'lift']
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def setup_logger():
    l = logging.getLogger('l')

    log_dir: Path = Path(__file__).parent / "logs"

    # create log directory if not exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # set log file name
    log_file_name = f"{time.strftime('%Y%m%d_%H%M%S')}.log"

    l.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(
        filename=log_dir / log_file_name,
        mode='w'
    )
    streamHandler = logging.StreamHandler()

    allFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    fileHandler.setFormatter(allFormatter)
    fileHandler.setLevel(logging.INFO)

    streamHandler.setFormatter(allFormatter)
    streamHandler.setLevel(logging.INFO)

    l.addHandler(streamHandler)
    l.addHandler(fileHandler)

    return l

l = setup_logger()