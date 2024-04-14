"""
文件相关的类定义
"""
import json

from data_define import Record
class FileReader:
    def read_data(self)->list[Record]:
        """
        将读到的数据转换成record类对象
        将它们封装到list内返回
        """
        pass

#文本文件数据读取
class TextReader(FileReader):
    def __init__(self,path):
        self.path=path      #定义成员变量记录文件路径


    #复写（实现抽象方法）父类的方法
    def read_data(self) ->list[Record]:
        f=open(self.path,"r",encoding="UTF-8")
        record_list:list[Record]=[]
        for line in f.readlines():
            line=line.strip()   #消除每一行回车符
            data_list=line.split(',')#逗号分隔存入列表
            # print(line)
            record=Record(data_list[0],data_list[1],int(data_list[2]),data_list[3])
            record_list.append(record)
        f.close()
        return record_list

#json文件数据读取
class JsonFileReader(FileReader):
    def __init__(self,path):
        self.path=path      #定义成员变量记录文件路径


    def read_data(self) -> list[Record]:
        f = open(self.path, "r", encoding="UTF-8")
        record_list: list[Record] = []
        for line in f.readlines():
            data_dict=json.loads(line)
            record=Record(data_dict["date"],data_dict["order_id"],data_dict["money"],data_dict["province"])
            record_list.append(record)

        f.close()
        return record_list



if __name__=='__main__':
    text_file_reader=TextReader("文本文件路径")
    text_file_reader.read_data()
    Json_file_reader = JsonFileReader("Json文件路径")
    Json_file_reader.read_data()


