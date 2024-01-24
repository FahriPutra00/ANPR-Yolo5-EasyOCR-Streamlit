import mysql.connector
import streamlit as st

@st.cache_resource()
class connect_db():
    def __init__(self):
        self.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="fahri_anprdb"
        )
        self.cursor = self.connection.cursor()

    def create_record(self,path, folder, filename, class_name):
        sql = "INSERT INTO `image_dataset` (`path`, `folder`, `filename`, `class`) VALUES (%s, %s, %s, %s)"
        val = (path,folder, filename, class_name)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record inserted successfully.")
                    
    def read_record(self, filename):
        sql = "SELECT * FROM `image_dataset` WHERE `filename` = %s LIMIT 1"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record

    def update_record(self, id, new_filename, new_class_name):
        sql = "UPDATE `image_dataset` SET `folder` = %s, `filename` = %s, `class` = %s WHERE `id` = %s LIMIT 1"
        val = (new_class_name,new_filename, new_class_name, id)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record updated successfully.")

    def delete_record(self, id):
        sql = "DELETE FROM `image_dataset` WHERE `id` = %s LIMIT 1"
        self.cursor.execute(sql, (id,))
        self.connection.commit()
        st.success("Record deleted successfully.")

    def __del__(self):
        self.cursor.close()
        self.connection.close()
    
