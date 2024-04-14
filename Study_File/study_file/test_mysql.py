from pymysql import Connection
conn=Connection(
    host='localhost',
    port=3306,
    user='root',
    password='wjzs128199'
)
cursor=conn.cursor()
conn.select_db("test1")
# cursor.execute("create table teacher(id int)")
cursor.execute("select * from student")
# 获取查询结果
result:tuple=cursor.fetchall()
for i in result:
    print(i)
    #运行结果
    # (1, '张三', 23, '男')
    # (2, '李四', 21, '男')
    # (3, '王五', 25, '男')
conn.close()