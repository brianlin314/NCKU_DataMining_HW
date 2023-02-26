import csv

def create_L1(mini_sup,feq): #丟L1到Frequency_Itemset，並且生成L1
    delete_item = []
    for i in feq:
        if feq[i] < mini_sup:
            delete_item.append(i)
    for u in range(len(delete_item)):
        del feq[delete_item[u]]
    temp_list=[]
    df_count_list=[]
    df_list=[]
    for w in feq:
        temp_list=[]
        temp_list.append(w)
        df_count=[temp_list,feq[w]]
        df_list.append(temp_list)
        df_count_list.append(df_count)
    #print("L1:",feq)
    print("L 1: ",df_list)
    return df_list,df_count_list
#a,b=create_L1(2)
#print(b)
def Union(list1,list2):
    final_list=list(set(list1) | set(list2))
    return final_list 
def Union_set(df_list,check):
    list_len=len(df_list)
    new_list=[]
    temp_list=[]
    for i in range(0,list_len):
        for j in range(i+1,list_len):
            temp_list=[]
            temp_list=Union(df_list[i],df_list[j])
            if len(temp_list) == check:
                new_list.append(temp_list)

    new = []
    for j in range(len(new_list)):
        xnew=[]
        for x in new_list[j]:
            xnew.append(int(x))
        xnew.sort()
        new.append(xnew)
    #print("new:",new)
    #print("new_list:",new_list)
    
    newer = []
    for j in range(len(new)):
        xnew=[]
        for x in new[j]:
            xnew.append(str(x))
        newer.append(xnew)
    #print("newer:",newer)   

    s=set(tuple(l) for l in newer)
    a=[list(t) for t in s]
    #print("交集完後的new_list",a)
    return a
def Comparsion_Create_Lk(compared_list,new_list,mini_sup):
    Lk=[]
    Lk_count=[]
    count_Lk_item=0
    for i in range(len(new_list)):
        #temp_Lk=[]
        new=set(new_list[i])
        count=0
        for j in range(len(compared_list)):
            compared=set(compared_list[j])
            if new.issubset(compared):
                count=count+1
        if count >= mini_sup:
            Lk.append(new_list[i])
            count_Lk_item=count_Lk_item+1
            
           #temp_Lk.append(new_list[i])
            lkk=[new_list[i],count]
            Lk_count.append(lkk)
    return Lk,Lk_count
def Add2FI(lister,Frequency_Itemset):
    for y in range(len(lister)):
        Frequency_Itemset.append(lister[y])
    print("insert done!")
    return 0
def find_TIDs(df):
    w = df[-1][0]
    return w
def delete_space(stringer):
    der=stringer[:-1]
    return der
# Frequency_Itemset=[[['9'], 2], [['5'], 2], [['6'], 3], [['7'], 5], [['4'], 2], [['0'], 4], [['8'], 5],
#                    [['0', '8'], 3], [['7', '8'], 4], [['4', '8'], 2], [['6', '7'], 2], [['8', '9'], 2], [['0', '7'], 4],
#                    [['5', '8'], 2], [['0', '9'], 2], [['7', '9'], 2], [['4', '6'], 2],[['6', '8'], 3],
#                    [['7', '8', '9'], 2], [['0', '7', '8'], 3], [['0', '8', '9'], 2], [['6', '7', '8'], 2],
#                    [['0', '7', '9'], 2], [['4', '6', '8'], 2], [['0', '7', '8', '9'], 2]]

def Rule2List(list1,mini_conf,df,bread_dict_r):
    csv_list=[] #全部要存進csv的
    total_tid=find_TIDs(df) #Tid總數
    total_tid=int(total_tid)
    for y in range(len(list1)):
        grave=[]
        base=set(list1[y][0]) # base is set
        X_appear=list1[y][1]
        X_appear=int(X_appear)
        #print("我是x:",base)
        for r in range(y+1,len(list1)):
            grave=[]
            search=set(list1[r][0]) # base的下一個，search is set
            if base.issubset(search):
                print(base," is subset of ",search)
                cache=list1[r][0].copy() #search的copy
                XY_appear=list1[r][1] #算support用，紀錄XY出現總次數
                XY_appear=int(XY_appear)
                print("cache",cache)

                ant_cache='{'
                for k in range(len(list1[y][0])): #base 在下
                    ant_cache=ant_cache+bread_dict_r[list1[y][0][k]]+' ' #加入antecedent/base 
                    print('list[y][0][k]:',bread_dict_r[list1[y][0][k]])
                    cache.remove(list1[y][0][k])
                    print("remove_cache:",cache)
                con_cache='{'
                ant_cache=delete_space(ant_cache)
                ant_cache=ant_cache+'}'
                print("ant_cache:",ant_cache)
                for jk in range(len(cache)):
                    con_cache=con_cache+bread_dict_r[cache[jk]]+' ' #加入consequent/modify_search
                con_cache=delete_space(con_cache)
                con_cache=con_cache+'}'
                print("con_cache:",con_cache)
                grave.append(str(ant_cache))
                grave.append(str(con_cache))
                
                cal_sup=XY_appear/total_tid
                print('XY_appear',XY_appear)
                cal_sup3 = round(cal_sup, 3)
                grave.append(cal_sup3)
                
                cal_conf=XY_appear/X_appear
                print('X_appear',X_appear)
                cal_conf3 = round(cal_conf, 3)
                if cal_conf < mini_conf:
                    continue
                else:
                    grave.append(cal_conf3)
                
                for b in range(len(list1)):
                    if cache == list1[b][0]:
                        Y_appear=int(list1[b][1])
                        break
                print('Y_appear:',Y_appear)
                Y_support=Y_appear/total_tid
                cal_lift=cal_conf/Y_support
                cal_lift3 = round(cal_lift, 3)
                grave.append(cal_lift3)
                
                print("grave:",grave)
                csv_list.append(grave)
                print("+++++++++++++++++++++++++++++++++++++++")
    return csv_list 
def store_csv(csv2list):
    header = ['antecedent', 'consequent', 'support', 'confidence', 'lift']

    with open('rules.csv', 'w',  newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(csv2list)):
            writer.writerow(csv2list[i])

def apriori_brian(df,minmum_support_frac,mini_conf):
    #生成原始的itemset，並用list中list表示
    bread_dict_r={'1': 'Bread', '2': 'Scandinavian', '3': 'Hot chocolate', '4': 'Jam', '5': 'Cookies', '6': 'Muffin', '7': 'Coffee', '8': 'Pastry', '9': 'Medialuna', '10': 'Tea', '11': 'Tartine', '12': 'Basket', '13': 'Mineral water', '14': 'Farm House', '15': 'Fudge', '16': 'Juice', '17': "Ella's Kitchen Pouches", '18': 'Victorian Sponge', '19': 'Frittata', '20': 'Hearty & Seasonal', '21': 'Soup', '22': 'Pick and Mix Bowls', '23': 'Smoothies', '24': 'Cake', '25': 'Mighty Protein', '26': 'Chicken sand', '27': 'Coke', '28': 'My-5 Fruit Shoot', '29': 'Focaccia', '30': 'Sandwich', '31': 'Alfajores', '32': 'Eggs', '33': 'Brownie', '34': 'Dulce de Leche', '35': 'Honey', '36': 'The BART', '37': 'Granola', '38': 'Fairy Doors', '39': 'Empanadas', '40': 'Keeping It Local', '41': 'Art Tray', '42': 'Bowl Nic Pitt', '43': 'Bread Pudding', '44': 'Adjustment', '45': 'Truffles', '46': 'Chimichurri Oil', '47': 'Bacon', '48': 'Spread', '49': 'Kids biscuit', '50': 'Siblings', '51': 'Caramel bites', '52': 'Jammie Dodgers', '53': 'Tiffin', '54': 'Olum & polenta', '55': 'Polenta', '56': 'The Nomad', '57': 'Hack the stack', '58': 'Bakewell', '59': 'Lemon and coconut', '60': 'Toast', '61': 'Scone', '62': 'Crepes', '63': 'Vegan mincepie', '64': 'Bare Popcorn', '65': 'Muesli', '66': 'Crisps', '67': 'Pintxos', '68': 'Gingerbread syrup', '69': 'Panatone', '70': 'Brioche and salami', '71': 'Afternoon with the baker', '72': 'Salad', '73': 'Chicken Stew', '74': 'Spanish Brunch', '75': 'Raspberry shortbread sandwich', '76': 'Extra Salami or Feta', '77': 'Duck egg', '78': 'Baguette', '79': "Valentine's card", '80': 'Tshirt', '81': 'Vegan Feast', '82': 'Postcard', '83': 'Nomad bag', '84': 'Chocolates', '85': 'Coffee granules ', '86': 'Drinking chocolate spoons ', '87': 'Christmas common', '88': 'Argentina Night', '89': 'Half slice Monster ', '90': 'Gift voucher', '91': 'Cherry me Dried fruit', '92': 'Mortimer', '93': 'Raw bars', '94': 'Tacos/Fajita'}
    check=df[0][0] #check是str
    check=int(check)
    temp_item=[]
    item=[]
    for i in range(len(df)):
        check1=int(df[i][0])
        if check1 != check:
            item.append(temp_item)
            temp_item=[]
            check=df[i][0]
            check=int(check)
            temp_item.append(df[i][2])
        else:
            temp_item.append(df[i][2])
    item.append(temp_item)
    print(item)
    cal_feq = []
    feq = {}
    for i in range(len(df)):
        cal_feq.append(df[i][2])
    unique_cal_feq_set = set(cal_feq)
    unique_cal_feq_set = list(unique_cal_feq_set)
    for j in range(len(unique_cal_feq_set)):
        item_k=unique_cal_feq_set[j]
        feq[item_k] = 0
    for k in range(len(cal_feq)):
        item_k=cal_feq[k]
        feq[item_k]=feq[item_k]+1
    print("Feq of C1: ",feq)

    temp_list=[]
    df_list=[]
    for w in feq:
        temp_list=[]
        temp_list.append(w)
        df_list.append(temp_list)
    print('df_list: ',df_list)
    #生成最原始的compared_list
    check=df[0][0] #check是str
    check=int(check)
    temp=[]
    compared=[]
    for i in range(len(df)):
        check1=int(df[i][0])
        if check1 != check:
            compared.append(temp)
            temp=[]
            check=df[i][0]
            check=int(check)
            temp.append(df[i][2])
        else:
            temp.append(df[i][2])
    compared.append(temp)
    #print("原始的ItemSet:",compared)

    Frequency_Itemset=[]
    rounder=2
    check_Ck=2
    #minmum_support_frac=0.4
    t_total=find_TIDs(df)
    print("t_total",t_total)
    minmum_support=minmum_support_frac*float(t_total)
    new_list,count_list=create_L1(minmum_support,feq)
    Add2FI(count_list,Frequency_Itemset)
    check_stop=len(new_list)
    #print(type(check_stop))
    while check_stop != 1 and check_stop != 0:
        new_list=Union_set(new_list,check_Ck)
        new_list,Lk_count=Comparsion_Create_Lk(compared,new_list,minmum_support)
        print("L",rounder,":",new_list)
        rounder=rounder+1
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        check_stop=len(new_list)
        Add2FI(Lk_count,Frequency_Itemset)
        check_Ck=check_Ck+1
    print("Frequency_Itemset",Frequency_Itemset)
    csv2list=Rule2List(Frequency_Itemset,mini_conf,df,bread_dict_r) 

    return csv2list


# if __name__ == '__main__':
#     csv2list=apriori_brian(df,0.5,0.7)
#     store_csv(csv2list)


