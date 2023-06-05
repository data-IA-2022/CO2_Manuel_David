import pandas as pd
from sqlalchemy import create_engine
from yaml import safe_load

df = pd.read_csv('data/2016_Building_Energy_Benchmarking.csv')

yml_file = safe_load(open('config.yml'))

config = yml_file['co2']

host = config['host']
user = config['user']
password = config['password']
database = config['database']


url = f'postgresql+psycopg2://{user}:{password}@{host}/{database}'
print(url)

engine = create_engine(url)
print(engine)

