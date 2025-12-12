import mysql.connector

conn = mysql.connector.connect(host='localhost', user='root', password='0007')

if conn.is_connected(): 
    print('connnection established')

mycursor = conn.cursor()
mycursor.execute("create database pythondb")
print(mycursor)

