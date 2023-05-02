import pandas as pd

import requests as rq
from bs4 import BeautifulSoup as bs

import datetime

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-y', dest='year', action='store')
# python get_calendar.py -y YYYY
args = parser.parse_args()


MONTH = {'Январь': '01', 'Февраль': '02', 'Март': '03', 'Апрель': '04', 
         'Май': '05', 'Июнь': '06', 'Июль': '07', 'Август': '08', 
         'Сентябрь': '09', 'Октябрь': '10', 'Ноябрь': '11', 'Декабрь': '12'}


if args.year is not None:
    year = args.year
else:    
    now = datetime.datetime.now()    
    year = now.year


link = f'https://www.consultant.ru/law/ref/calendar/proizvodstvennye/{year}/'
req = rq.get(link, verify=False)

tree = bs(req.text, 'lxml')
tbls = tree.find_all('table')
cals = [i for i in tbls if i.get('class', None) == ['cal']]

final = []
for i in cals:
    month_name = [j for j in i.find_all('th')\
             if j.get('class', None) == ['month']][0].contents[0]
    
    month = i.find_all('td')
    workdays = [int(j.contents[0]) for j in month 
                                    if (len(j.get('class', None)) == 0)\
                                    or ('preholiday' in j.get('class', None))]
    holidays = [int(j.contents[0]) for j in month 
                                    if ('weekend' in j.get('class', None))\
                                    or ('holiday' in j.get('class', None))]
    
    workdays = [(j, 1) for j in workdays]
    holidays = [(j, 0) for j in holidays]
    res = workdays.copy()
    res.extend(holidays)
    res = pd.DataFrame(res, columns=['date', 'weekend'])
    res.iloc[:, 0] = res.iloc[:, 0].apply(lambda x: pd.to_datetime(f'{year}-{MONTH[month_name]}-{x}'))
    res = res.set_index('date').sort_index()
    final.append(res)
final = pd.concat(final)

final.to_excel(f'{year}-holidays.xlsx')

