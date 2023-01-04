#Veri İşlemleri
import pandas as pd
import numpy as np
#Veri Kazıma
from bs4 import BeautifulSoup as bs
import requests 
import datetime
from collections import defaultdict


main_dict = defaultdict(list)

for i in range(1,5):
    url = f"https://www.imdb.com/list/ls004610270/?st_dt=&mode=detail&page={i}&ref_=ttls_vm_dtl&sort=list_order,asc"
    url = f"https://www.imdb.com/list/ls004610270/?st_dt=&mode=detail&page={i}&ref_=ttls_vm_dtl&sort=list_order,asc"
    html = requests.get(url).text
    soup = bs(html, "lxml")
    for div in soup.findAll("div", attrs={"class":"lister-item-content"}):
        # print(div.find('a').contents[0])
        main_dict["movie_name"].append(div.find('a').contents[0])
        main_dict["year_of_movie"].append(div.find('span', class_="lister-item-year text-muted unbold").contents[0].replace("(","").replace(")",""))
        main_dict["genre"].append(div.find('span', class_="genre").contents[0].replace(" ","").replace("\n",""))
        main_dict["rate"].append(div.find('span', class_="ipl-rating-star__rating").contents[0])
        main_dict["summary"].append(div.find('p', class_="").contents[0].replace("\n",""))


df = pd.DataFrame(main_dict)


from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)
df.to_sql("data",con=engine)

import sqlite3
conn = sqlite3.connect('movie_datadb.db')
c = conn.cursor()

conn.commit()
df.to_sql('Movies', conn, if_exists='replace', index = False)













































































































