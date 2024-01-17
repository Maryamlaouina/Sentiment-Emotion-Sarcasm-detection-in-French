# Load Database Pkg
from datetime import datetime

import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()


# Fxn
def create_page_visited_table():
    with sqlite3.connect('data.db') as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TEXT)')
        conn.commit()
def add_page_visited_details(pagename,timeOfvisit):
	with sqlite3.connect('data.db') as conn:
     	  c = conn.cursor()
     	  c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, datetime.now()))
     	  conn.commit()

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable')
	data = c.fetchall()
	return data


# Fxn To Track Input & Prediction
def create_emotionclf_table():
    with sqlite3.connect('data.db') as conn:
      	c = conn.cursor()
      	c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT)')
      	conn.commit()

def add_prediction_details(rawtext,prediction,probability,timeOfvisit):
	with sqlite3.connect('data.db') as conn:
     	 c = conn.cursor()
     	 c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', ("Home", datetime.now()))
     	 conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM emotionclfTable')
	data = c.fetchall()
	return data



