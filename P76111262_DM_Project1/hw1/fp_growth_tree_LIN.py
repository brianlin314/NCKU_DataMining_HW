from itertools import combinations
import copy
def delete_min(list,min_sup):
    hello=[]
    for ii in range(len(list)):
        temp_list=[]
        for jj in range(len(list[ii])):
            if list[ii][jj][1]>=min_sup:
                temp_list.append(int(list[ii][jj][0]))
        hello.append(temp_list)
    return hello
class Node:
    def __init__(self,data=None,next=None):
        self.data=data
        self.next=next
class LinkedList:
    def __init__(self,head=None):
        self.head=head
    def append(self, data):
        if not self.head:
            self.head = Node(data,None)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)
    def size(self):
        count = 0
        node = self.head
        while node:
            count += 1
            node = node.next
        return count
    def print(self):
        if not self.head:
            print(self.head)
        node = self.head
        while node:
            end = " -> "
            print(node.data, end=end)
            node = node.next

class treeNode:
    def __init__(self,val,parent=None):
        self.val=val
        self.child=None
        self.sibling=None
        self.parent=parent
        self.times=0                 
    def Sibling_insert(self,s_value):
        self.sibling=treeNode(s_value, self.parent)
        print(s_value,"插入sibling!")
    def child_insert(self,val):
        self.child=treeNode(val,self)
        print(val,"插入child!")
    def add_freq(self,feq=1):
        self.times+=feq
def find_tree(list,root):
    empty_list=[]
    print("root的位址",root)
    print("這是基於",list[0],"的樹:")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    for i in range(1,len(list)):
        print("位址:",list[i])
        temp_list=[]
        n=list[i]
        bp_value=list[i].val
        bp=n.times
        counter2=0
        while True:
            value_count=n.times
            values=n.val
            print("此點count值:",value_count)
            print("此點是:",values)
            if counter2!=0:
                temp_list.append(values)
            n=n.parent
            print("parent值:",n.val)
            print("parent位址:",n)
            if n==root:
                print("跳出!")
                break
            counter2+=1
        if temp_list!=[]:
            gen_g=[]
            gen_g.append(bp)
            dk=temp_list.copy()
            dk.reverse()
            temp_list=[]
            temp_list.append(gen_g)
            temp_list.append(dk)
            
            empty_list.append(temp_list)
        print("---------------------")
    return empty_list,bp_value
class sub_treeNode:
    def __init__(self,val,parent=None):
        self.val=val
        self.child=None
        self.sibling=None
        self.parent=parent
        self.times=0                 
    def Sibling_insert(self,s_value):
        self.sibling=treeNode(s_value, self.parent)
        print(s_value,"插入sibling!")
    def child_insert(self,val):
        self.child=treeNode(val,self)
        print(val,"插入child!")
    def add_freq(self,feq=1):
        self.times+=feq
        
def find_leaf(list_b):
    leaf_addr=[]
    for sm1 in range(len(list_b)):
        if len(list_b[sm1])>1:
            for sm2 in range(1,len(list_b[sm1])):
                #lkk=list_b[sm1][sm2]
                if list_b[sm1][sm2].child==None:
                    leaf_addr.append(list_b[sm1][sm2])
    return leaf_addr      

def build_subtree(list1,pointer_item2,sub_pointer):
    for L1 in range(len(list1)):
        subtree=sub_pointer
        fadder=list1[L1][0][0]
        print("fadder",fadder)
        for L2 in range(len(list1[L1][1])): 
            v=list1[L1][1][L2]
            if subtree.child==None:
                subtree.child_insert(v)
                for i in range(len(pointer_item2)):
                    if v==pointer_item2[i][0]:
                        pointer_item2[i].append(subtree.child)
                        break
                subtree.child.add_freq(fadder)
                subtree=subtree.child
            elif subtree.child.val==v:
                subtree.child.add_freq(fadder)
                subtree=subtree.child
            elif subtree.child.val!=v:
                checking=0
                subtree=subtree.child
                while subtree.sibling!=None:
                    if subtree.sibling.val==v:
                        print(v,"work!")
                        subtree.sibling.add_freq(fadder)
                        checking=1
                        subtree=subtree.sibling
                        break
                    subtree=subtree.sibling
                if checking==0:
                    subtree.Sibling_insert(v)
                    for i in range(len(pointer_item2)):
                        if v==pointer_item2[i][0]:
                            pointer_item2[i].append(subtree.sibling)
                            break
                    subtree.sibling.add_freq(fadder)
                    subtree=subtree.sibling
    return sub_pointer,pointer_item2
def find_frequent_itemsets(llist,min_sup,bp_valuer):
    collect_fi=[]
    collect_fi2=[]
    collect_fi3=[]
    total_l=[]
    total_l_dict=[]
    for av in range(len(llist)):
        l_addr=llist[av]
        temp_l=[]
        temp_d={}
        while l_addr.val!=None:
            temp_l.append(l_addr.val)
            temp_d[l_addr.val]=l_addr.times
            l_addr=l_addr.parent
        total_l.append(temp_l)
        total_l_dict.append(temp_d)
        print("total_l:",total_l)
        print("total_l_dict",total_l_dict)
        len_total_l_dict=len(total_l_dict)
        
    for avg in range(len(total_l)):
        length_l=len(total_l[avg])
        cc_list=[]
        for avg_l in range(1,length_l+1):
            cc=list(combinations(total_l[avg],avg_l))
            for c_count in range(len(cc)):
                cc_list.append(cc[c_count])
        print("cc_list:",cc_list)
        for avg_2 in range(len(cc_list)):
            temperr=[]
            if len(cc_list[avg_2])==1:
                print("test:",cc_list[avg_2])
                temperr.append(str(cc_list[avg_2][0]))
                temperr.append(str(bp_valuer))
                h4=[]
                h4.append(temperr)
                h4.append(total_l_dict[avg][cc_list[avg_2][0]])
                collect_fi.append(h4)
            else:
                minimum=999999
                for fk in range(len(cc_list[avg_2])):
                    print("work!!")
                    temperr.append(str(cc_list[avg_2][fk]))
                    if minimum>total_l_dict[avg][cc_list[avg_2][fk]]:
                        minimum=total_l_dict[avg][cc_list[avg_2][fk]]
                print("最小值:",minimum)
                temperr.append(str(bp_valuer))
                h4=[]
                h4.append(temperr)
                h4.append(minimum)
                collect_fi.append(h4)
    collect_fi2=copy.deepcopy(collect_fi)
    collect_fi3=copy.deepcopy(collect_fi2)
    print("collect_fi",collect_fi)
    for fi in range(len(collect_fi2)):
        collect_fi2[fi].append(0)
    print("collect_fi2",collect_fi2)
    print("collect_fi3",collect_fi3)
    what=[]
    for fi in range(len(collect_fi2)):
        check=0
        what_temp=[]
        if collect_fi2[fi][2]==0:
            summer=collect_fi2[fi][1]
            collect_fi2[fi][2]=1
            for fij in range(fi+1,len(collect_fi2)):
                if collect_fi2[fi][0]==collect_fi2[fij][0]:
                    summer=summer+collect_fi2[fij][1]
                    collect_fi2[fij][2]=1
                    check=1
        if check==1:
            what_temp.append(collect_fi2[fi][0])
            what_temp.append(summer)
            what.append(what_temp)
    print("what:",what)
    if len(what)>0:
        what_temp=[]
        for lp in range(len(what)):
            for lpp in range(len(collect_fi3)):
                if what[lp][0]==collect_fi3[lpp][0]:
                    what_temp.append(lpp)
        print("what_temp",what_temp)
        jg=True
        for lp in range(len(what_temp)):
            hi=what_temp[lp]
            if jg==True:
                del collect_fi3[hi]
                jg=False
            elif jg==False:
                hi-=1
                del collect_fi3[hi]
        print(collect_fi3)
        for lp in range(len(what)):
            collect_fi3.append(what[lp])
    return collect_fi3

def find_TIDs(df):
    w = df[-1][0]
    return w
def delete_space(stringer):
    der=stringer[:-1]
    return der

def Rule2List(list1,mini_conf,df):
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
                    ant_cache=ant_cache+list1[y][0][k]+' ' #加入antecedent/base 
                    print('list[y][0][k]:',list1[y][0][k])
                    cache.remove(list1[y][0][k])
                    print("remove_cache:",cache)
                con_cache='{'
                ant_cache=delete_space(ant_cache)
                ant_cache=ant_cache+'}'
                print("ant_cache:",ant_cache)
                for jk in range(len(cache)):
                    con_cache=con_cache+cache[jk]+' ' #加入consequent/modify_search
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
def My_fp_tree(df,min_sup_f,min_conf):
    #生成原始的itemset，並用list中list表示
    collect_fi=[]#這個要當全域變數
    collect_fi4=[]#這個要當全域變數
    check=df[0][0] #check是str
    check=int(check)
    temp_item=[]
    item=[]
    t_total=find_TIDs(df)
    print("t_total",t_total)
    min_sup=min_sup_f*float(t_total)
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

    temp=[]
    temp=list(feq.keys())
    print(temp)

    for i in range(len(temp)):
        tempp=[]
        temp3=[]
        tempp.append(temp[i])
        temp3.append(tempp)
        temp3.append(feq[temp[i]])
        collect_fi4.append(temp3)
        print("collect_fi4",collect_fi4)
        alter=[]
        feqer=copy.deepcopy(feq)
        #print(id(feqer))
        #print(id(feq))
        print("1:",feqer)
        feqer=sorted(feqer.keys())
        print("2:",feqer)
        temp_dict={}
        for i in range(len(feqer)):
            print("i=",i)
            gg=feq[feqer[i]]
            if gg>=min_sup:
                temp_dict[feqer[i]]=gg
            alter.append(temp_dict)
        print(temp_dict)
        feqer=sorted(temp_dict.items(),key = lambda x:x[1],reverse = True)

        print(feqer)
        asshole=[]
        for i in range(len(feqer)):
            asshole.append(feqer[i][0])
        print("asshole",asshole)
        all_list=[]
    for i in range(len(item)):
        temp_dict={}
        for j in range(len(item[i])):
            gg=feq[item[i][j]]
            temp_dict[item[i][j]]=gg
        all_list.append(temp_dict)
    print("Frequency:",all_list)

    for i in range(len(all_list)):
        all_list[i]=sorted(all_list[i].keys())
    print("all_list:",all_list)

    item=all_list.copy()
    print("Copy_item:",item)

    all_list=[]
    for i in range(len(item)):
        temp_dict={}
        for j in range(len(item[i])):
            gg=feq[item[i][j]]
            temp_dict[item[i][j]]=gg
        all_list.append(temp_dict)
    print("123:",all_list)
    for i in range(len(all_list)):
        all_list[i]=sorted(all_list[i].items(),key = lambda x:x[1],reverse = True)
    print("all_list:",all_list)
    ass=delete_min(all_list,2)
    print("ass",ass)
    data=asshole.copy()
    # unique=[]
    # for y in range(len(data)):
    #     for g in range(len(data[y])):
    #         unique.append(data[y][g])
    # unique=set(unique)
    # unique=list(unique)
    # print(unique)
    pointer_item=[]
    for y in range(len(data)):
        temp=[]
        temp.append(int(data[y]))
        pointer_item.append(temp)
    print("pointer_item:",pointer_item)

    data=ass.copy()
    fp=treeNode(None)
    pointer=fp
    #data=[[7, 85], [7, 38, 85]]
    for L1 in range(len(data)):
        fp=pointer
        for L2 in range(len(data[L1])): 
            v=data[L1][L2]
            if fp.child==None:
                fp.child_insert(v)
                for i in range(len(pointer_item)):
                    if v==pointer_item[i][0]:
                        pointer_item[i].append(fp.child)
                        break
                fp.child.add_freq()
                fp=fp.child
            elif fp.child.val==v:
                fp.child.add_freq()
                fp=fp.child
            elif fp.child.val!=v:
                checking=0
                fp=fp.child
                while fp.sibling!=None:
                    if fp.sibling.val==v:
                        print(v,"work!")
                        fp.sibling.add_freq()
                        checking=1
                        fp=fp.sibling
                        break
                    fp=fp.sibling
                if checking==0:
                    fp.Sibling_insert(v)
                    for i in range(len(pointer_item)):
                        if v==pointer_item[i][0]:
                            pointer_item[i].append(fp.sibling)
                            break
                    fp.sibling.add_freq()
                    fp=fp.sibling

    print(pointer_item)
    for j in range(len(pointer_item)-1,-1,-1):
        sub_tree_list,bp_valuer=find_tree(pointer_item[j],pointer)
        print("sub_tree_list:",sub_tree_list)
        print("bp_valuer:",bp_valuer)

        data2=asshole.copy()
        pointer_item2=[]
        for y in range(len(data2)):
            temp=[]
            temp.append(int(data2[y]))
            pointer_item2.append(temp)
        print("pointer_item2:",pointer_item2)

        list1=sub_tree_list.copy()
        subtree=sub_treeNode(None)
        sub_pointer=subtree

        a,b=build_subtree(sub_tree_list,pointer_item2,sub_pointer) #已經產生一個基於...的mining tree
        leaves=find_leaf(b)
        print("b:",b)
        print("/////////////////////////")
        print("leaves:",leaves)
        po=find_frequent_itemsets(leaves,2,bp_valuer)
        for fuck in range(len(po)):
            collect_fi4.append(po[fuck])
    print("collect_fi4:",collect_fi4)
    collect_fi5=[]
    maxmum=0
    for j in range(len(collect_fi4)):
        if collect_fi4[j]!=[] :
            if collect_fi4[j][1] >= min_sup:  
                collect_fi5.append(collect_fi4[j])
                if len(collect_fi4[j][0]) > maxmum:
                    maxmum=len(collect_fi4[j][0])
    print(collect_fi5)
    print(maxmum)
    collect_fi6=[]
    for i in range(1,maxmum+1):
        for j in range(len(collect_fi5)):
            if len(collect_fi5[j][0]) == i:
                collect_fi6.append(collect_fi5[j])
    print("collect_fi6:",collect_fi6)
    csv2list=Rule2List(collect_fi6,min_conf,df)   
    return csv2list
