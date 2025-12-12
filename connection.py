import mysql.connector

conn = mysql.connector.connect(
    host='localhost', 
    user='root', 
    password='0007'
    )

if conn.is_connected(): 
    print('connnection established')
print(conn)
print(conn.is_connected())