## “人工智能辅助糖尿病遗传风险预测”进展报告

### 1. 数据获取及预处理

> 主要说明一下数据的来源，原始数据的基本情况，如数量，字段，含义等
> 再就是到目前为止，对数据预处理的情况，如噪声的处理，缺失值的处理等。

#### 1.1 数据来源

数据集来源于“天池精准医疗大赛——人工智能辅助糖尿病遗传风险预测”的初赛阶段。初赛数据共包含两个文件，训练文件d_train.csv和测试文件d_test.csv.


#### 1.2 数据说明

* 训练集中包含年龄、性别、各项体检指标作为基本数据特征，每个文件的第一行是字段名，之后的每一行代表的是一个个体,共有5642个个体。文件共包含42个字段，包含数值型、字符型、日期型等众多数据类型，具体字段名称如下：



```python
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from math import ceil
from time import time
import seaborn as sns
import numpy as np
%matplotlib inline

file = './d_train.csv'
data = pd.read_csv(file, encoding='gb18030')
print(data.shape)
print(data.columns)
```

    (5642, 42)
    Index(['id', '性别', '年龄', '体检日期', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶',
           '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
           '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原',
           '乙肝e抗体', '乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
           '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
           '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
           '血糖'],
          dtype='object')
    

* 部分字段内容在部分人群中有缺失，各字段的缺失情况如下：


```python
print("------------------------------------------------------------------")
print(data.isnull().sum())
```

    ------------------------------------------------------------------
    id                0
    性别                0
    年龄                0
    体检日期              0
    *天门冬氨酸氨基转换酶    1221
    *丙氨酸氨基转换酶      1221
    *碱性磷酸酶         1221
    *r-谷氨酰基转换酶     1221
    *总蛋白           1221
    白蛋白            1221
    *球蛋白           1221
    白球比例           1221
    甘油三酯           1219
    总胆固醇           1219
    高密度脂蛋白胆固醇      1219
    低密度脂蛋白胆固醇      1219
    尿素             1378
    肌酐             1378
    尿酸             1378
    乙肝表面抗原         4279
    乙肝表面抗体         4279
    乙肝e抗原          4279
    乙肝e抗体          4279
    乙肝核心抗体         4279
    白细胞计数            16
    红细胞计数            16
    血红蛋白             16
    红细胞压积            16
    红细胞平均体积          16
    红细胞平均血红蛋白量       16
    红细胞平均血红蛋白浓度      16
    红细胞体积分布宽度        16
    血小板计数            16
    血小板平均体积          23
    血小板体积分布宽度        23
    血小板比积            23
    中性粒细胞%           16
    淋巴细胞%            16
    单核细胞%            16
    嗜酸细胞%            16
    嗜碱细胞%            16
    血糖                0
    dtype: int64
    

* 第一列为个体ID号。训练文件的最后一列为标签列，即需要预测的目标血糖值。测试集相对于训练集则缺少了对应的血糖值，也就是我们所期望预测到的值。

#### 1.3 数据预处理

(1) 缺失值的处理

* 由上面给出的数据集空值情况可以看出，对于这42个字段，均存在不同程度的缺失。缺失比较严重的字段大致可以分为4组：肝功能、肾功能、乙肝、血常规（一般缺失都是整个类别缺失，由于此人未做某项检查）。其中乙肝缺失情况极为严重，在进一步探究这五个乙肝特征的影响权重后，且没有合理的办法对这些缺失的数据进行填充，我们决定把这一组特征删除而不进行填充。
* 其他缺失数据，根据数据的分布，采用不同的填充办法，如平均值、中值、众数等。

(2) 去除离群值与无意义值

去除对预测模型无影响的基本特征和离群值。删除无意义值即删除在预测模型构建过程中，影响权重为零的基本特征，例如体检日期和性别。删除离群值是指删除明显与大部分数据分布之间存在极端差距的数据，离群值的存在往往会扭曲预测结果进而影响模型的精度，对于离群值可以用Box Plot来发现。

(3) 其他处理

* 读入数据后，为了数据显示与观察的方便，将所有的columns都用英文名称代替。

* 观察到最早的体检日期为2017-09-15，将所有的日期都转化为距离该最早体检日期的天数。

* 性别特征做one-hot coding，男性为1，女性为0。

### 2. 数据分析与可视化

> 数据探索性分析的结果，可以使用统计工具，聚类分析等工具
> 使用可视化来展示分析结果

* 血糖指标的描述及分布：


```python
tr_y = data.loc[:,'血糖']
tr_y.describe()
```




    count    5642.000000
    mean        5.631925
    std         1.544882
    min         3.070000
    25%         4.920000
    50%         5.290000
    75%         5.767500
    max        38.430000
    Name: 血糖, dtype: float64




```python
plt.plot(tr_y)
```




    [<matplotlib.lines.Line2D at 0x26e5ab86668>]




    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/blob/main/Figures/output_11_1.png)
    


可以看见，血糖指标大多数分布在3-10之间，个别异常值达到了38左右，对于这些异常数据，在后期处理的时候，考虑将其删除。

* 各字段空值占比：


```python
na_data = data.isna().sum() / data.shape[0]
fig_na, ax_na = plt.subplots(figsize=(15,9))
na_data.plot.barh(ax=ax_na)
```




    <AxesSubplot:>




    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_14_1.png)
    


* 各字段的describe基本情况可视化：


```python
def bar_pic(xname, y, title):
    # plt.legend() #sample
    x = [i for i in range(1, len(y) + 1)]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=(20,5))
    plt.bar(x, y, 0.1)
    plt.xticks(x, xname,rotation=90)
    plt.title(title)
    plt.show()

# 打印describe柱型图
def all_print_bar_pic(train_file):
    analyse_data = train_file.describe()
    col = np.array(analyse_data.columns[1:])
    ind = np.array(analyse_data.index)
    num = 0
    for name in ind:
        print(name)
        line = analyse_data.values[num, 1:]
        bar_pic(col, line, name)
        num += 1
        
sns.set(style='white')
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号

all_print_bar_pic(data)


```

    count
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_1.png)
    


    mean
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_3.png)
    


    std
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_5.png)
    


    min
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_7.png)
    


    25%
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_9.png)
    


    50%
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_11.png)
    


    75%
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_13.png)
    


    max
    


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_16_15.png)
    


### 3. 模型选取

> 围绕选题要解决的问题，考虑使用哪些模型来进行挖掘
> 说明选择的理由

这是一个有监督的回归问题，预计先使用GBDT、XGBoost和随机森林，后面集成多个模型。  因为数据少，主要注意过拟合问题，可能使用先分类后回归的方式。其中回归不是按正确率计算结果的，根据题目要求，评估指标为MSE。

* Gradient Boosting Decision Tree（GBDT）在传统机器学习算法里面是对真实分布拟合的最好的几种算法之一，在前几年深度学习还没有大行其道之前，GBDT在各种竞赛是大放异彩。原因大概有几个，一是效果确实挺不错；二是即可以用于分类也可以用于回归；三是可以筛选特征。

* eXtreme Gradient Boosting(XGBoost)是Gradient Boosting Machine的另外一种实现，其最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进，提高了精度。XGBoost的主要特点包括，基于树能够自动处理稀疏数据的提升学习算法，采用加权分位数法来搜索近似最优分裂点，并行和分布式计算，基于分块技术的大量数据高效快速处理。
    XGBoost不同于传统的GBDT的方式，其只利用了一阶的导数信息并对损失函数做了二阶的泰勒展开，并在目标函数之外加入了正则项整体求最优解，用以权衡目标函数的下降和模型的复杂程度，避免过拟合。将目标函数做泰勒展开，并引入正则项。
    
* 作为新兴起的、高度灵活的一种机器学习算法，随机森林（Random Forest，简称RF）拥有广泛的应用前景，从市场营销到医疗保健保险，既可以用来做市场营销模拟的建模，统计客户来源，保留和流失，也可用来预测疾病的风险和病患者的易感。随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。随机森林的主要优点在于算法效果好，时间效率高，能有效处理高维数据和海量数据。

### 4. 挖掘实验的结果

* 各字段相关性探索：


```python
sns.set(style='white')
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
plt.subplots(figsize=(15, 12))
             
sns.heatmap(data.loc[:, '年龄':].corr(), annot=False)
plt.show()
```


    
![Image text](https://github.com/Xiemixue/Data-Mining-Group-Assignments/tree/main/Figures/output_19_0.png)
    


### 5. 存在的问题

* 选出的主特征只能从概率角度判断该特征的影响力，如果要提高判断准确度，需要了解多个因素的共同作用力。

* 对于明显高过正常值、或者明显低于正常值的数据，预测能力较差。

### 6. 下一步工作

* 进行更深入的数据分析与可视化

* 探索集成多个模型

* 分析更多挖掘结果，形成最终报告

### 7. 任务分配与完成情况

谢蜜雪：模型构建(部分完成)、算法实现(未完成)、文档编写(未完成)；

吕芳蕊：系统设计(完成)、算法实现(未完成)、文档编写(未完成)；

龚凯雄：数据分析及预处理（完成）、结果分析(未完成)、文档编写(未完成)；

陈家珂：数据分析及预处理（完成）、关联规则挖掘（完成）、算法调研（完成）；

余丹雅：背景调研（完成）、文档编写(未完成)；
