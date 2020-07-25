import sqlite3 

connection = sqlite3.connect("details.db")

crsr = connection.cursor() 

sql_command = """CREATE TABLE info (  
id INTEGER PRIMARY KEY,  
name VARCHAR(20),    
num BIGINT
);"""

crsr.execute(sql_command)

connection.commit() 
connection.close() 