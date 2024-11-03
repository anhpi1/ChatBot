import pyodbc
import numpy as np
def print_Bt(bt):
    Bt=[int(i) for i in bt]
    topics = ['topology', 'structure', 'performance_metric', 'applications', 'components', 'tools']
    content = []
    server = 'DESKTOP-1MU0IU3\\SQLEXPRESS'
    database = 'comparator'
    username = ''  # Add username if required
    password = ''  # Add password if required

    # Connect to SQL Server
    conn = pyodbc.connect(
        f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )

    # Create a cursor object
    cursor = conn.cursor()

    # Loop through each table and corresponding ID in Bt
    for topic, id_value in zip(topics, Bt):
        # Convert numpy int to standard int if necessary
        if isinstance(id_value, np.int64):
            id_value = int(id_value)  # Convert to Python int

        # Execute query for each table and ID
        cursor.execute(f"SELECT content FROM dbo.{topic} WHERE id = ?;", id_value)
        
        # Fetch the results and append to content list
        content.append(cursor.fetchall())

    # Close the connection
    conn.close()
    
    # Print the fetched content
    for index, data in enumerate(content):
        print(f"{topics[index]}: {data}")

