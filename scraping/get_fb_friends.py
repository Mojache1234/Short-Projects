from bs4 import BeautifulSoup
from urllib import request
import csv

html = request.urlopen('file:///Users/aongoco/Downloads/friends.html').read()

soup = BeautifulSoup(html, 'html.parser')

containers = soup.find_all('div', class_='uiBoxWhite')

with open('test.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(
        csvfile,
        delimiter=','
    )
    for container in containers:
        name = container.find('div', class_='_2lek').get_text()
        email = container.find('div', class_='_2let').get_text() if container.find('div', class_='_2let') is not None else 'No Email'
        writer.writerow([name, email])
