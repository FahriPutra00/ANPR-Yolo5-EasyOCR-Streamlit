import mysql.connector
import streamlit as st
import os
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

    def create_record(self, filename, class_name):
        sql = "INSERT INTO `image_dataset` (`filename`, `class`) VALUES (%s, %s)"
        val = (filename, class_name)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record inserted successfully.")
        
    def read_all_records(self):
        sql = "SELECT * FROM `image_dataset`"
        self.cursor.execute(sql)
        records = self.cursor.fetchall()
        if records:
            st.success("Records found")
        else:
            st.warning("Records not found.")
            
    def read_record(self, id):
        sql = "SELECT * FROM `image_dataset` WHERE `id` = %s"
        self.cursor.execute(sql, (id,))
        record = self.cursor.fetchone()
        if record:
            st.success("Record found:", record)
        else:
            st.warning("Record not found.")

    def update_record(self, id, new_filename, new_class_name):
        sql = "UPDATE `image_dataset` SET `filename` = %s, `class` = %s WHERE `id` = %s"
        val = (new_filename, new_class_name, id)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record updated successfully.")

    def delete_record(self, id):
        sql = "DELETE FROM `image_dataset` WHERE `id` = %s"
        self.cursor.execute(sql, (id,))
        self.connection.commit()
        st.success("Record deleted successfully.")

    def __del__(self):
        self.cursor.close()
        self.connection.close()
    
