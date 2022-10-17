import math
from random import randrange, random, sample
from datetime import datetime, timedelta
from time import time
from time import mktime as mktime

import connection as connection
import cursor as cursor
import pandas.io.gbq
import psycopg2
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandasql as ps
import numpy as np

from web3 import Web3
#from dunebuggy import Dune
from duneanalytics import DuneAnalytics

def get_data_to_CSV(queryID, name): # getting data from the dyne analytics for free -> sorry dune analytics...
    # fetching the data from the Dune analytics query
    # initialize client
    dune = DuneAnalytics('kirill_molch', 'password')

    # try to login
    dune.login()

    # fetch token
    dune.fetch_auth_token()

    # fetch query result id using query id
    # query id for any query can be found from the url of the query:
    # for example:
    # https://dune.com/queries/4494/8769 => 4494
    # https://dune.com/queries/3705/7192 => 3705
    # https://dune.com/queries/3751/7276 => 3751

    result_id = dune.query_result_id(query_id=queryID)

    # fetch query result
    data = dune.query_result(result_id) # dictionary with a awkward structure -> to analyze, the data should be a data frame
    filtering_result = data['data']
    results = filtering_result['get_result_by_result_id']


    cleaned_data_list = []
    for row in results:
        dict1 = row
        cleaned_data_list.append(dict1['data'])

    df = pd.DataFrame(cleaned_data_list) # clean data in the df format
    df.to_csv(name + '.csv', index=False) # saving data to csv file

# get_data_to_CSV(1257409, "onefourth") # getting 1/4 of the MEV inspect from dune analytics (since the data is too large, I had to devide it into four parts)
# get_data_to_CSV(1259325, "twofourth") # getting 2/4 of the MEV inspect from dune analytics
# get_data_to_CSV(1259333, "threefourth") # getting 3/4 of the MEV inspect from dune analytics
# get_data_to_CSV(1259341, "fourfourth") # getting 4/4 of the MEV inspect from dune analytics

# get_data_to_CSV(1259461, "ethprice") # getting ethereum prices

start_block = 14715000
end_block = 15449617
num_blocks = end_block - start_block + 1
