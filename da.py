#apostrophe '
#parentheses ()
#colon :
#comma ,
#dash -
#ellipsis ...
#exclamation mark !
#period .
#slash /

#matplotlib
from matplotlib import pyplot as plt
import random
#x=range(0,120)
#y=[random.randint(20,35) for i in range(120)] #随机生成120个数
#x=range(2,26,2)
#y=[15,13,14.5,17,20,25,26,26,24,22,18,15]
plt.figure(figsize=(20,8),dpi=80)
#plt.plot(x,y)
#_x=list(x)[::3] #变成list取步长
#_xtick_labels=["10:{}".format(i) for i in range(60)]
#_xtick_labels+=["11:{}".format(i) for i in range(60)]
#plt.xticks(_x,_xtick_labels[::3],rotation=45) #刻度旋转
#_xtick_labels=[i/2 for i in range(4,49)]
#plt.xticks(_xtick_labels[::3]) #调整x轴刻度
#plt.yticks(range(min(y),max(y)+1))
#plt.savefig("./t1.png")
#plt.xlabel("time")
#plt.ylabel("temperature")
#plt.title("temperature plot")
#plt.show()
#y=[1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
#y1=[1,0,3,1,2,2,3,3,2,1,2,1,1,1,1,1,1,1,1,1]
#x=range(11,31)
#plt.xticks(range(11,31))
#plt.title("dating plot")
#plt.xlabel("age")
#plt.ylabel("number")
#plt.plot(x,y,label="me",color="red",linestyle="--",linewidth=5)
#plt.plot(x,y1,label="friend",color="cyan",linestyle=":")#两个折线一张图
#plt.grid(alpha=0.4,linestyle="--")#网格,透明度
#plt.legend(loc="upper left")#加标
#plt.show()

#scatterplot
y_3=[11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22,22,22,23]
y_10=[26,26,28,19,21,17,16,19,18,20,20,19,22,23,17,20,21,20,22,15,11,15,5,13,17,10,11,13,12,13,6]
x_3=range(1,32)
x_10=range(51,82)
#plt.scatter(x_3,y_3)
#plt.scatter(x_10,y_10)
_x=list(x_3)+list(x_10)
_xtick_labels=["mar.{}".format(i) for i in x_3]
_xtick_labels+=["oct.{}".format(i-50) for i in x_10]
plt.xticks(_x[::3],_xtick_labels[::3],rotation=45)#用format的格式，多个_x
plt.xlabel("time")
plt.ylabel("temp")
plt.scatter(x_3,y_3,label="mar")
plt.scatter(x_10,y_10,label="oct")
plt.legend(loc="upper right")
plt.show()

#条形图(纵向）
from matplotlib import pyplot as plt
plt.figure(figsize=(20,8),dpi=80)
y=[56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]
x=["movie"+ str(i) for i in range(1,20)]
plt.xticks(range(len(x)),x,rotation=45)
plt.bar(range(len(x)),y,width=0.3) #range(len(x))代替x
plt.show()

#横向
from matplotlib import pyplot as plt
plt.figure(figsize=(20,8),dpi=80)
x=[56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]
y=["movie"+ str(i) for i in range(1,20)]
plt.xticks(range(len(y)),y,rotation=45)
plt.barh(range(len(x)),y,height=0.3,color="orange") #range(len(x))代替x
plt.grid(alpha=0.3)
plt.show()

#三组数据
from matplotlib import pyplot as plt
import numpy as np
a=["e","f","g","h"]
b_16=[15746,312,4497,319] #y axis
b_15=[12357,156,2045,168]
b_14=[2358,399,2358,362]
width=0.2
x=np.arange(len(a))
#plt.figure(figsize=(20,8),dpi=80)
plt.bar(x-width,b_14,width,label='14')
plt.bar(x,b_15,width,label='15')
plt.bar(x+width,b_16,width,label='16')
plt.xticks(x,labels=a)
plt.legend()
plt.show()

#直方图
from matplotlib import pyplot as plt
a=[131, 98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]
bin_width=3 #组距
num_bins=(max(a)-min(a))//bin_width

plt.figure(figsize=(20,8),dpi=80)
plt.hist(a,num_bins,density=True)#num_bins组,density:percentage
plt.xticks(range(min(a),max(a)+bin_width,bin_width))

plt.grid()
plt.show()


#numpy
import numpy as np
t1=np.array([1,2,3,])
print(t1)
print(type(t1))

t2=np.array(range(10))
print(t2)
print(type(t2))

t3=np.arange(4,10,2) #range
print(t3)
print(type(t3))
print(t3.dtype) #类型int64

t4=np.array(range(1,4),dtype="i1")#set data type
print(t4)
print(t4.dtype)

t5=np.array([1,1,0,1,0,0],dtype=bool)
print(t5)
print(t5.dtype)

t6=t5.astype("int8")#change data type
print(t6)
print(t6.dtype)

import random
t7=np.array([random.random() for i in range(10)])#10个随机数0-1
print(t7)
print(t7.dtype)

t8=np.round(t7,2) #2 decimal points
print(t8)

t9=np.array([[1,2,3],[4,5,6]])
print(t9)
print(t9.shape)
print(t9.flatten())#1-dimension:1 row
print(t9+2) #every+2

t10=np.arange(12)
print(t10)
print(t10.reshape((3,4))) #3 row 4 column

t=np.arange(12,24).reshape(3,4)

#t10.shape[0]*t10.shape[1]  #row*column
#t.transpose()=t.T=t.swapaxes(1,0)


#read data
import numpy as np
t1=np.loadtxt("US_video_data_numbers.csv",delimiter=',',dtype='int') #unpack=True(transpose),skiprows=0,usecols,delimiter分隔
print(t1)

print(t1[2])#取行
print(t1[2:]) #=t1[2:,:]对列不做操作
print(t1[[2,8,10]])#不连续多行

print(t1[:,0])#取列
print(t1[:,2:])
print(t1[:,[0,2]]) #0列和2列

print(t1[2,3])#2行3列的值
print(t1[2:5,1:4]) #2:5=2-4,0开始

print(t1[[0,2],[0,1]]) #取点（0，0）和（2，1）的值

#t1[:,2:4]=0改值
#t1<10--boolean
#t1[t1<10]=3, <10等于3
#np.where(t<10,0,10) t<10为0，其他为10
#t.clip(10,18) <10变10，>18变18，nan不变
#nan：t2=t2.astype(float)
#t2[3,3]=np.nan/np.inf

#count nan: np.count_nonzero(t!=t)
#t[np.isnan(t)]=0  把nan变成0

#np.sum(t,axis=0)  算每列的sum
#np.sum(t,axis=1)  算每行的sum
#np.sum(t)  算全部element的sum

#t.mean(axis=)
#np.median(t,axis=)
#t.max(axis)
#t.min(axis)
#range:np.ptp(t,axis)
#t.std(axis)

def fill_ndarray(t): #把nan换成每列的mean
    for i in range(t.shape[1]):
        temp_col=t[:,i] #每一列
        nan_num=np.count_nonzero(temp_col!=temp_col)#每一列nan的数量
        if nan_num!=0:#有nan
            temp_not_nan_col=temp_col[temp_col==temp_col]#取每列上非nan的值--变array
            temp_col[np.isnan(temp_col)]=temp_not_nan_col.mean()#算mean，把nan变成mean
if__name__=='__main__'：#只在当前.py里运行下面的代码（如果新的py import当前文件，不会运行下面的代码）

#plot hist
t_us=t1[:,-1] #取最后一列
t_us=t_us[t_us<=5000] #update:选<=5000的数
from matplotlib import pyplot as plt
d=250
bin_nums=(t_us.max()-t_us.min())//d
plt.figure(figsize=(20,8),dpi=80)
plt.hist(t_us,bin_nums) #t_us:x axis, 数量：y axis
plt.show()

#plot scatter:relation
import numpy as np
t2=np.loadtxt("GB_video_data_numbers.csv",delimiter=',',dtype='int') #unpack=True(transpose),skiprows=0,usecols,delimiter分隔
print(t2)
t2=t2[t2[:,1]<=500000] #在总的里选择范围，update t2
t_uk=t2[:,-1] #var1
t_uk1=t2[:,1] #var2
plt.figure(figsize=(20,8),dpi=80)
plt.scatter(t_uk1,t_uk)
plt.show()

#拼接
#np.vstack((t1,t2)) 上t1下t2
#np.hstack((t1,t2)) 左t1右t2

#行列交换
#t[[1,2],:]=t[[2,1],:]  1,2 exchange
#t[:,[0,2]]=t[:,[2,0]]

#pandas
import pandas as pd
import string
import numpy as np
t=pd.Series(np.arange(10),index=list(string.ascii_uppercase[:10]))
print(t)
t["age"]
t[2]#行
t[2:10:2]#2-9行，2，4，6，8
t[t>4]
t[[2,3,6]] #2,3,6
t.index #索引
list(t.index)[:2]
t.values #对应值
t.where(t>1,10) #<1换10，其他不变

t={"name":"hong","age":30}

#read data
df=pd.read_csv("dogNames2.csv")
print(df)

#read mongodb
from pymongo import MongoClient
import pandas as pd
client=MongoClient()
collection=client['douban']['tv1']
data=list(collection.find())
print(data)

for i in data:
    temp={}
    temp["info"]=i["info"]
    temp["rating_value"]=i['rating']['value']
    temp['rating_count']=i['rating']['count']
    temp['title']=i['title']
    temp['country']=i['tv_category']
    temp['directors']=i['directors']
    temp['actors']=i['actors']
    data_list.append(temp)


pd.DataFrame(np.arange(12).reshape(3,4),index=list("abc"),columns=list("wxyz"))#index=row
d1={"name":["xiaohong","gang"],"age":[20,32]}#name=column
d1=[{"name":"xiaohong","age":32},{"name":"gang","age":20"}]
df=pd.DataFrame(d1)

df.head(1)#SHOW FIRST row 0
df.tail(2)#last 2 rows
df.info()
df.describe()#statistic summary

df=pd.read_csv("dogNames2.csv")
#print(df)
df=df.sort_values(by="Count_AnimalName",ascending=False)#big to small
print(df[:20]["Row_Labels"])#first 20 rows, choose row_labels column

df.loc["a","z"] #get number in a row+z column
df.loc['a']#get a row
df.loc["a",:]
df.loc[:,"y"] #get y column
df.loc[['a','c']] #get a and c rows
df.loc[['a','b'],['w','z']]#get a and b rows and w and z columns

df.iloc[1]# 1st row(index start from 0)
df.iloc[:,2]#get 2nd column
df.iloc[:,[2,1]]#get 2nd and 1st column
df.iloc[[0,2],[2,1]]#0 and 2 row and 2 and 1 col
df.iloc[1:,:2]#1 to last row and first to 1 column
df.iloc[1:,:2]=30#set values

df[(df["Count_AnimalName"]>800)&(df["Count_AnimalName"]<1000)] #&,|
df["Row_Labels"].str().len()#length
df["info"].str.split("/").tolist() #split based /, df to list[[]]

#missing data
pd.isnull(df) #true:nan, false:no nan
pd.notnull(df)
df[pd.notnull(df["w"])] #choose true row in df
df.dropna(axis=0,how='all',inplace)#delete the row if all nan
df.dropna(axis=0,how="any")#delete the row if has nan
df.dropna(axis=0,how="any",inplace=True)#auto edit

df.fillna(0)
df.fillna(df.mean())#nan not in calculation
df['age'].fillna(t2['age'].mean())# fill in one column
df['age'][1]#age column's 1st row
df[df==0]=np.nan #get 0 data, change to nan

import pandas as pd
df=pd.read_csv("IMDB-Movie-Data.csv")
#get average movie score
print(df['Rating'].mean())
#get directors number
print(len(set(df["Director"].tolist()))) #method 1:set(list):去重复的element
print(len(df['Director'].unique())) #method 2:df.unique()--to list+remove redundant
#get actors number
temp_list=df['Actors'].str.split(',').tolist() #list in list
actors_list=[i for j in temp_list for i in j] #flatten to a list
actors_num=len(set(actors_list)) #same method, len(dict), set(list)--dict
print(actors_num)

from matplotlib import pyplot as plt
#统计分类：各个genre有多少部
import numpy as np
temp=df['Genre'].str.split(',').tolist() #[[],[],..]，电影0:[..]genres
genre_list=list(set([i for j in temp for i in j])) #list(dict):change to list
#create zero matrix
zeros_df=pd.DataFrame(np.zeros((df.shape[0],len(genre_list))),columns=genre_list)#content: np.zeros()--zeros matrix, #rows:df.shape[0],#columns:len(genre_list)
#if movie has that genre,to 1
for i in range(df.shape[0]):#go through each row
    zeros_df.loc[i,temp[i]]=1 #[0,['sci-fi','musical']]-- 0 row+'',''columns相交的数=1（有这两类）
print(zeros_df.head(3))
genre_count=zeros_df.sum(axis=0)#column sum--every genre #movies
genre_count=genre_count.sort_values() #df type

#plot-bar chart
plt.figure(figsize=(20,8),dpi=80)
_x=genre_count.index #df.index： row labels
_y=genre_count.values #df.values #values
plt.bar(range(len(_x)),_y)
plt.xticks(range(len(_x)),_x)
plt.show()

#rating， runtime分布情况--直方图（连续数据）
runtime_data=df['Runtime (Minutes)'].values
max_runtime=runtime_data.max()
min_runtime=runtime_data.min()
num_bin=(max_runtime-min_runtime)//5 #step size=5
plt.figure(figsize=(20,8),dpi=80) #写在plt最前
plt.hist(runtime_data,num_bin)
plt.xticks(range(min_runtime,max_runtime+5,5))
plt.show()

#join
df1.join(df2) #df1 left,df2 right, based on df1 #rows

#merge:
df1.merge(df2,on='a')#merge 两个df‘a'列相同值所在的行,交集
df1.merge(df2,on='a',how='outer') #全部合并 nan补齐，并集
df1.merge(df2,on='a',how='left') #based on df1，#rows=#df1 rows
df1.merge(df2,on='a',how='right')#based on df2

df1.merge(df2,left_on='',right_on='')#left_on:df1 column,right_on:df2 column

#starbucks数量对比，中国每个省的数量
import pandas as pd
import numpy as np
df=pd.read_csv("starbucks_store_worldwide.csv")
print(df.head())
print(df.info())
#df[df['Country']=='US']
country_count=df.groupby('Country')['Brand'].count()#1st col: country,基于它算brand(2nd col)个数
print(country_count['CN'])#choose CN row
print(country_count['US'])

china_data=df[df['Country']=='CN'] #condition:country = cn
print(china_data.groupby(['State/Province'])['Brand'].count())
#1st col:state/province; 2nd col: #brand(for each 1st col)
.sum()/.mean()/.median()/.std()/.var()
print(df.groupby(['Country','State/Province'])['Brand'].count())#based 1st:country+2nd:state,get #brands

#to dataframe
print(df.groupby(['Country','State/Province'])[['Brand']].count()) #1 more bracket

#index
df.index=['a','b'] #换索引，值不变
df.reindex(['a','f'])#换索引，值变（如果没有f，值变nan）
df.set_index('a')#把a列变成索引index，drop a列
df.set_index('a',drop=False) #保留a列
df['d'].unique()#d列的独特值/不一样的
df.index.unique()#不重复的index
len(df.index) #index长度
list(df.index)#转成list
#multiindex
df.set_index(['a','b']) #设2个index

a=pd.DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':list('hjklmno')})
print(a)
b=a.set_index(['c','d'])
print(b)
c=b['a'] #get a 列（+2个index）
print(c)
c['one']['j'] #index=one,j--value=1
c['one']#index=one--对应的剩下index+values (series:一列值)
d=a.set_index(['d','c'])['a']
print(d)
print(d.swaplevel()['one']) #swaplevel():2个index列交换
b.loc['one'].loc['h'] #index one+h 对应的值(dataframe：多列值)
b.swaplevel().loc['h'] #same

#matplotlib：店铺#前10的国家,bar charts
data1=df.groupby("Country")['Brand'].count().sort_values(ascending=False)[:10]#0-9 top10
_x=data1.index
_y=data1.values
plt.figure(figsize=(20,8),dpi=80)
plt.bar(range(len(_x)),_y)
plt.xticks(range(len(_x)),_x)
plt.show()

#中国每个城市的店铺数
data2=df[df['Country']=='CN'].groupby("City")['Brand'].count().sort_values(ascending=False)[:50]
_x=data2.index
_y=data2.values
plt.figure(figsize=(20,8),dpi=80)
plt.bar(range(len(_x)),_y,width=0.3,color='orange')
plt.xticks(range(len(_x)),_x)
plt.show()

#books:不同年份书的数量
df=pd.read_csv("books.csv")
print(df.head(1))
print(df.info())
data1=df[pd.notnull(df['original_publication_year'])]#get not nan values
data1=data1.goupby('orignal_publication_year')['title'].count()

#不同年份的平均评分情况:按每一年的，get平均分
data1=data1.groupby('original_publication_year')['average_rating'].mean()
print(data1)
_x=data1.index #np array
_y=data1.values #np array
plt.plot(range(len(_x)),_y)
plt.xticks(list(range(len(_x)))[::10],_x[::10].astype(int),rotation=45) #取步长(隔9个取）,年份变int(nparray也行）
plt.show()


#911 不同类型的紧急情况次数(类型在title里）
import numpy as np
df=pd.read_csv("911.csv")
#print(df.head(1))
#print(df.info())
temp_list=df['title'].str.split(":").tolist() #[[],[]..]
cate_list=list(set([i[0] for i in temp_list])) #取0th的（就是类型）--set(list)去重,变dict--变list
#print(cate_list)
zeros_df=pd.DataFrame(np.zeros((df.shape[0],len(cate_list))),columns=cate_list)
for cate in cate_list:#遍历列（only 3列）
    zeros_df[cate][df['title'].str.contains(cate)]=1#.contains(i)--output:有cate：true， 无：false
    #fire列，true的--变1
sum_ret=zeros_df.sum(axis=0)
3print(sum_ret) #每列的sum=#
#method2：
for i in range(df.shape[0]): #每一行遍历
    zeros_df.loc[i,temp_list[i][0]]=1 #temp_list:ith的0th个。相交（有这类）=1

#method3:
#(在df上加一列：类型）
temp_list=df['title'].str.split(":").tolist()
cate_list = [i[0] for i in temp_list] #类型list
df['cate']=pd.DataFrame(np.array(cate_list).reshape((df.shape[0],1))) #新建一列cate。变成array reshape
print(df.head(5))
print(df.groupby('cate')['title'].count()) #每个类的，数量


#时间处理
pd.date_range(start='20171230',end='20180131',freq='D') #D=day每天，10d=10days每10天,生成时间范围
pd.date_range(start='20171230',periods=10,freq='D')#10 dates
pd.date_range(start='20171230',periods=10,freq='M')#EVERY MONTH end
pd.date_range(start='20171230',end='20180131',freq='H')#every hour
pd.date_range(start='20171230',periods=10, freq='MS')#every month begin

print(df['timeStamp'])
#每个月份的次数
#resample
df['timeStamp']=pd.to_datetime(df['timeStamp']) #标准化时间
df.set_index("timeStamp",inplace=True)#timestamp变成索引列，原地修改
#print(df.head())
count_by_month=df.resample('M')['title'].count() #每月有多少个（title可用来算多少条数据）
#print(count_by_month)

#plot
_x=count_by_month.index #df.index
_y=count_by_month.values
_x=[i.strftime('%Y%m%d') for i in _x] #adjust x-axis labels
plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(len(_x)),_y)
plt.xticks(range(len(_x)),_x,rotation=45)
plt.show()

#每个月的不同类型的次数 -- 2 indexes
df['timeStamp']=pd.to_datetime(df['timeStamp']) #标准化时间
#新建了一列category
temp_list=df['title'].str.split(":").tolist()
cate_list = [i[0] for i in temp_list] #类型list
df['cate']=pd.DataFrame(np.array(cate_list).reshape((df.shape[0],1))) #新建一列cate。list变成array reshape
df.set_index('timeStamp',inplace=True)#timestamp变成索引列，原地修改
print(df.groupby('cate').resample('M')['title'].count()) #1st cate, 2nd monthly, 3rd count#

#plot
plt.figure(figsize=(20,8),dpi=80) #放在plot第一行
def plot_img(df,label): #每月次数plot
    count_by_month = df.resample('M')['title'].count()
    _x=count_by_month.index #df.index, x data
    _y=count_by_month.values
    _x=[i.strftime('%Y%m%d') for i in _x] #adjust x-axis labels
    plt.plot(range(len(_x)),_y,label=label) #add label element
for group_name,group_data in df.groupby('cate'): #plot 3 times--in groupby每个category，画每月次数图
    plot_img(group_data,group_name) #call function:group_data=df,group_name=label

plt.xticks(range(len(_x)), _x, rotation=45)
plt.legend(loc='best')
plt.show()


#5个城市，pm2.5随时间变化
df=pd.read_csv("BeijingPM20100101_20151231.csv")
print(df.head())
print(df.info())
#时间格式分开了
period=pd.PeriodIndex(year=df['year'],month=df['month'],day=df['day'],hour=df['hour'],freq='H') #every hour
df['datetime']=period #格式化时间，加一列到df
print(df.head(10))
#设置成索引--resample
df.set_index('datetime',inplace=True)
df1=df.resample('7D')['PM_US Post'].mean()
df2=df.resample('7D')['PM_Dongsihuan'].mean()  #weekly pm avg.--auto delete nan
print(df)
#有nan值 缺失数据
#1.删除
#data=df['PM_US Post'].dropna() #df format
_x=df1.index
_x=[i.strftime('%Y%m%d') for i in _x]
_x_china=df2.index
_x_china=[i.strftime('%Y%m%d') for i in _x_china]
_y=df1.values
_y_china=df2.values
plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(len(_x)),_y,label='US POST')
plt.plot(range(len(_x_china)),_y_china,label='China post')
plt.xticks(range(0,len(_x),10),list(_x)[::10],rotation=45) #x轴坐标，同一个x轴
plt.legend()
plt.show()


