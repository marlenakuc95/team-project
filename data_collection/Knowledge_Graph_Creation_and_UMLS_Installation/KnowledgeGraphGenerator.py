
# Install required packages:

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("mysql.connector")

# Import sql connector:
import mysql.connector
from mysql.connector import Error


#database config values. Please edit these variables before executing the annotated file
host_name = "localhost"
database_name = "umls"
user_name = "root"
password = "Endure#7"

no_of_files = 50

for i in range(1,no_of_files+1):
    
    if i<10:
        file_extension = str(i).zfill(2)
    else:
        file_extension = str(i)
    # Annotated file name. Please edit this variable before executing the annotated file 
    current_filename = f"annotations_pubmed21n00{file_extension}_parsed.csv"
    #subject predicater object fileAjay
    spo_filename = current_filename.split('.csv')[0] + '_triples.spo'

    # Get all unique CUIs:


    import pandas as pd
    annotated_data = pd.read_csv(current_filename)
    unique_cuis = annotated_data.CUI.unique()

    #Find relations and create the Subject Predicate Object tsv file and the concept dictionary

    import mysql.connector
    current_concept_name = None

    try:
        mydb = mysql.connector.connect(
        host=host_name,
        user=user_name,
        password=password,
        database=database_name
        )
        
        for cui in unique_cuis:
            query_string_query_cui = "SELECT b.str FROM mrconso b WHERE b.cui = '" + cui + "' AND b.ts = 'P' AND b.stt = 'PF' AND b.ispref = 'Y' AND b.lat = 'ENG';"
            mycursor1 = mydb.cursor()

            mycursor1.execute(query_string_query_cui)

            myresult1 = mycursor1.fetchall()
            for x in myresult1:
                current_concept_name = x[0]
            
            query_string ="SELECT DISTINCT a.rela, b.str FROM mrrel a, mrconso b WHERE a.cui2 = '" + cui + "' AND RELA IS NOT NULL AND a.cui1 = b.cui AND b.ts = 'P' AND b.stt = 'PF' AND b.ispref = 'Y' AND b.lat = 'ENG';"

            mycursor2 = mydb.cursor()

            mycursor2.execute(query_string)

            myresult2 = mycursor2.fetchall()

            
            for x in myresult2:
                if current_concept_name is not None and x[0] is not None and x[1] is not None:
                    with open(spo_filename, "a", encoding="utf-8") as f:
                        f.write(str(current_concept_name) + "\t" + str(x[0]) + "\t" + str(x[1]) + "\n")
            
    except Error as e:
        print(e)




