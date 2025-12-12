import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0007',
    database='pythondb'
)

mycursor = conn.cursor()

# Fetch all rows from the student table
mycursor.execute("SELECT * FROM student")

for x in mycursor:
    print(x)

# Get all rows
#result = mycursor.fetchall()

# Print rows
# for row in result:
    # print(row)

# conn.close()
