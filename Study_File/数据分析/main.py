"""
1.设计一个类，完成数据封装
2.设计一个抽象类，定义文件读取相关功能使用子类实现具体功能
3.读取文件，生产数据对象
4.数据需求逻辑运算
5.pyecharts进行图形绘制

"""
from file_define import FileReader,TextReader,JsonFileReader
from data_define import  Record
from pyecharts.charts import Bar
from pyecharts.options import *
from pyecharts.globals import ThemeType

text_file_reader=TextReader("文本文件路径")
json_file_reader=JsonFileReader('Json文件路径')
jan_data:list[Record]=text_file_reader.read_data()
feb_data:list[Record]=json_file_reader.read_data()
#将两个月份数据合并成一个list
all_data:list[Record]=jan_data+feb_data
#数据计算
data_dict={}
for record in all_data
    if record.data in data_dict.keys():
        data_dict[record.data]+=record.money
    else:
        data_dict[record.data] = record.money
#可视化图标开发
bar=Bar()
bar.add_xaxis(list(data_dict.keys()))#添加X轴数据
bar.add_yaxis("销售额",list(data_dict.values()))
bar.set_gloal_opts(
    title_opts=TitleOpts(title="每日销售额")
)
bar.render("每日销售额柱状图.html")
