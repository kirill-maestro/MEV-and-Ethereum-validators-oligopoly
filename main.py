import math
from random import randrange, random, sample
from datetime import datetime, timedelta
from time import time
import dataframe_image as dfi
from IPython import display
import connection as connection
import cursor as cursor
import psycopg2
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandasql as ps
import numpy as np
from time import mktime as mktime
import csv

from web3 import Web3
from duneanalytics import DuneAnalytics


class Ecdf(pd.Series):
    def __init__(self, data):
        s = pd.Series(data)
        super().__init__(s.value_counts().sort_index().cumsum() * 1. / len(s))

    def get_quantile(self, q):
        # self[self.ge(random())].index[0]
        return self.index[np.argmax(self.array >= q)]  # faster

    def get_scaled_ecdf(self, scaling_factor):
        index = [v * scaling_factor for v in self.index.values]
        scaled_ecdf = self.set_axis(index)
        scaled_ecdf.__class__ = Ecdf
        return scaled_ecdf

start_block = 13136427  # first block of September 2021
end_block = 15449617  # last block of August 2022


# onefourth = 13136427 + 578298 = 13714725
# twofourth = 13714725 + 578298 = 14293023
# threefourth = 14293023 + 578298 = 14871321
# fourfourth = 14871321 + 578296 = 15449617

SECONDS_PER_SLOT = 12

# # plot daily fees/coinbase transfers
# daily = pd.read_csv('daily_totals.csv')
# daily['net_fees'] = daily['total_fee_revenue'] - daily['total_basefee_cost'] #basefee costs will be burned and no one gets it
# daily['burned_ether'] = -1 * daily['total_basefee_cost']
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# daily.set_index('date', inplace=True)
# daily.rename(inplace=True, columns={
#     'net_fees': 'Net fees',
#     'total_transfer_revenue': 'Coinbase transfers',
#     'burned_ether': 'Burned'
# })
# daily[['Net fees', 'Coinbase transfers', 'Burned']].plot.area(
#     ax=ax, ylabel="ETH per day"
# )
# #plt.show()
#
# ax.margins(x=0)
# ax.set_title('Daily Ethereum fee/transfer revenue March 2021 — May 2022');

df = pd.read_csv('final_correct_data.csv') # aggregated data from mev and eth
median_mev = df['miner_extracted'].median()
#print("median", median_mev)

# plot block revenue histogram and empirical cumulative density function (ecdf)

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

bins = [e/100 for e in range(201)]
miner_extracted = df['miner_extracted']
miner_exctractable = df['miner_extractable']
miner_extracted.hist(ax=ax1, bins=bins, density=True, alpha=0.5, grid=False, label='Miner revenue without extractable MEV')
miner_exctractable.hist(ax=ax1, bins=bins, density=True, alpha=0.5, grid=False, label='Miner revenue with extractable MEV')
ax1.set_title('Histogram of Miner Revenue per Block (Sep 2021—Aug 2022)')
ax1.set_xlim(0, 2) # changes: was 2 -> 0.5
ax1.set_ylabel('Frequency density')
mean_without_MEV = np.mean(df['miner_extracted'])
mean_with_MEV = np.mean(df['miner_extractable'])
print("mean_without_MEV", mean_without_MEV)
print("mean_with_MEV", mean_with_MEV)
print("mean_with_MEV", mean_with_MEV)
print(df.describe())
ax1.axvline(mean_without_MEV, color='b', linestyle='dashed', label='Miner Extracted')
ax1.axvline(mean_with_MEV, color='r', linestyle='dashed', label='Miner Extractable')
plt.legend(loc='lower right')

mev_ecdf = Ecdf(df['miner_extracted'])
mev_ecdf.plot(ax=ax2, label='ECDF meiner extracted', legend=True)
mev_extractable_ecdf = Ecdf(df['miner_extractable'])
mev_extractable_ecdf.plot(ax=ax2, label='ECDF miner extractable with MEV', legend=True)
plt.legend(loc='lower right')

quantiles = [.01, .1, .25, .5, .75, .9, .99, .999]

table = pd.DataFrame({
    'quantile': quantiles,
    'centile': [100 * q for q in quantiles],
    'revenue': [mev_ecdf.get_quantile(q) for q in quantiles]
})

table_extratcable = pd.DataFrame({
    'quantile': quantiles,
    'centile': [100 * q for q in quantiles],
    'revenue': [mev_extractable_ecdf.get_quantile(q) for q in quantiles]
})
print("1", table)
print("2", table_extratcable)

table.to_csv('table_0110.csv', index=False) # saving data to csv file
table_extratcable.to_csv('table_extratcable_0110.csv', index=False) # saving data to csv file
table.set_index('quantile', inplace=True, drop=False)
#table.plot('revenue<br>(ETH per block)', 'quantile', kind='scatter', ax=ax2)
plt.scatter(table['revenue'], table['quantile'], edgecolors='blue')
plt.scatter(table_extratcable['revenue'], table_extratcable['quantile'], edgecolors='orange')

#print(table_extratcable['revenue'].loc[0])

ax2.set_title(
    'Empirical Cumulative Density Function (ECDF) \n of Miner Revenue per Block vs Miner Revenue per Block with extractable MEV'
)
ax2.set_xlim(0,2)
ax2.set_xlabel('Net miner payment per block (ETH)')
ax2.set_ylim(0,1)
ax2.set_ylabel('Cumulative frequency density')
c1 = table['revenue'].loc[0.01]
c11 = table_extratcable['revenue'].loc[0]
ax2.annotate(f'1st centile: {c1:.3f} extracted ETH vs {c11:.3f} extractable ETH', (c1 + 0.02, 0.02))
d1 = table['revenue'].loc[0.1]
d11 = table_extratcable['revenue'].loc[1]
ax2.annotate(f'10th centile: {d1:.3f} extracted ETH vs {d11:.3f} extractable ETH', (d1 + 0.02, 0.075))
lq = table['revenue'].loc[0.25]
lq1 = table_extratcable['revenue'].loc[2]
ax2.annotate(f'25th centile (lower quartile): {lq:.3f} extracted ETH vs {lq1:.3f} extractable ETH', (lq + 0.02, 0.225))
med = table['revenue'].loc[0.5]
med1 = table_extratcable['revenue'].loc[3]
ax2.annotate(f'50th centile (median): {med:.3f} extracted ETH vs {med1:.3f} extractable ETH', (med1 + 0.02, 0.475))
uq = table['revenue'].loc[0.75]
uq1 = table_extratcable['revenue'].loc[4]
ax2.annotate(f'75th centile (upper quartile): {uq:.3f} extracted ETH vs {uq1:.3f} extractable ETH', (uq + 0.02, 0.725))
d9 = table['revenue'].loc[0.9]
d91 = table_extratcable['revenue'].loc[5]
ax2.annotate(f'90th centile: {d9:.3f} extracted ETH vs {d91:.3f} extractable ETH', (d9 + 0.02, 0.875))
c99 = table['revenue'].loc[0.99]
c991 = table_extratcable['revenue'].loc[6]
ax2.annotate(f'99th centile: {c99:.3f} extracted ETH vs {c991:.3f} extractable ETH', (c99 - 0.35, 0.925))

plt.show()

table.rename(columns={
    'centile': 'Centile<br>(%)',
    'revenue': 'Revenue per block<br>(ETH)'
}, inplace=True)

dfTime = df['block_timestamp']
element = datetime.strptime(str(dfTime[0]),"%Y-%m-%d %H:%M:%S %Z")
tuple = element.timetuple()
timestampBegin = mktime(tuple)
#print(timestampBegin)

element = datetime.strptime(str(dfTime[len(dfTime)-1]),"%Y-%m-%d %H:%M:%S %Z")
tuple = element.timetuple()
timestampEnd = mktime(tuple)
#print(timestampEnd)

# model full validator returns using previously calculated MEV ECDFs

num_validators = 420000
seconds_per_year = 31556952
slots_per_year = seconds_per_year // SECONDS_PER_SLOT

HEAD_WT = 14
SOURCE_WT = 14
TARGET_WT = 26
SYNC_WT = 2
PROPOSER_WT = 8
BASE_REWARD_FACTOR = 64
WEIGHT_DENOM = 64
EPOCHS_PER_COMMITTEE = 256
COMMITTEE_SIZE = 512
SLOTS_PER_EPOCH = 32
GWEI_PER_ETH = int(1e9)
gwei_per_validator = int(32e9)
staked_gwei = gwei_per_validator * num_validators
epochs_per_year = slots_per_year // SLOTS_PER_EPOCH

mean_interval = (timestampEnd - timestampBegin) / (len(df) - 1) #mean interval of blocks with mev
scaling_factor = SECONDS_PER_SLOT / mean_interval # should be 12 / 13.5
scaled_mev_ecdf = mev_ecdf.get_scaled_ecdf(scaling_factor) # scaling by the factor
scaled_mev_extractable_ecdf = mev_extractable_ecdf.get_scaled_ecdf(scaling_factor) # scaling by the factor

base_reward = gwei_per_validator * BASE_REWARD_FACTOR // math.isqrt(staked_gwei)
total_reward = base_reward * num_validators

att_reward = base_reward * (HEAD_WT + SOURCE_WT + TARGET_WT) // WEIGHT_DENOM # attestation rewards
annual_attestation_reward_eth = att_reward * epochs_per_year / GWEI_PER_ETH

# perfect performance so all validators get full attestation reward for the year
validators = [annual_attestation_reward_eth] * num_validators # list of 420 k times annual_attestation_reward_eth, each item of the list represents a validator rewards
validators_extractable_MEV = [annual_attestation_reward_eth] * num_validators

prop_reward = total_reward * PROPOSER_WT // WEIGHT_DENOM // SLOTS_PER_EPOCH
prop_reward_eth = prop_reward / GWEI_PER_ETH
sync_reward = total_reward * SYNC_WT // WEIGHT_DENOM // SLOTS_PER_EPOCH \
              // COMMITTEE_SIZE
sync_reward_eth = sync_reward / GWEI_PER_ETH

start_time = time()
last_update = 0
for slot in range(slots_per_year):
    # process sync committee:
    if slot % (32 * EPOCHS_PER_COMMITTEE) == 0:
        # select sync committee
        committee = sample(range(num_validators), COMMITTEE_SIZE) # for each slot we simulate the chain rewards for validators
    for ind in committee:
        validators[ind] += sync_reward_eth # sync rewards for validators participating committee
        validators_extractable_MEV[ind] += sync_reward_eth

    # random selection of validator as proposer
    ind = randrange(num_validators)
    r = random()
    validators[ind] += scaled_mev_ecdf.get_quantile(r) + prop_reward_eth
    validators_extractable_MEV[ind] += scaled_mev_extractable_ecdf.get_quantile(r) + prop_reward_eth

    t = time()
    if t - last_update > 0.1:
        percentage = 100 * (slot + 1) / slots_per_year
        elapsed = timedelta(seconds=int(t - start_time))
        print(f"{percentage:.2f}% / {elapsed} elapsed", end='\r')
        last_update = t

annual_full_rtn = pd.Series([100 * v / 32 for v in validators]) # annual rate of return
annual_full_extractable_rtn = pd.Series([100 * v / 32 for v in validators_extractable_MEV]) # annual rate of return
annual_full_ecdf = Ecdf(annual_full_rtn) # Ecdf function
annual_full_extractable_ecdf = Ecdf(annual_full_extractable_rtn) # Ecdf function

val_df = pd.DataFrame(validators)
val_MEV_df = pd.DataFrame(validators_extractable_MEV)
val_df.to_csv('val_df_0110.csv') # saving data to csv file
val_MEV_df.to_csv('val_MEV_df_0110.csv') # saving data to csv file

# plot simulated full validator returns

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

bins = [e/5 for e in range(176)]

annual_full_rtn.to_csv('annual_full_rtn_0110.csv') # saving data to csv file
annual_full_rtn.hist(ax=ax1, bins=bins, density=True, alpha=0.5, grid=False, label='Simulated validator APR without extractable MEV')
annual_full_extractable_rtn.to_csv('annual_full_extractable_rtn_0110.csv') # saving data to csv file
annual_full_extractable_rtn.hist(ax=ax1, bins=bins, density=True, alpha=0.5, grid=False, label='Simulated validator APR with extractable MEV')
plt.legend(loc='lower right')
ax1.set_title(
    'Histogram of Simulated Validator Rate of Return (420k validators, 1 year)'
)
ax1.set_xlim(0, 35) # change from 35 to 20
ax1.set_ylabel('Frequency density')

quantiles = [0, .01, .1, .25, .5, .75, .9, .99, .999, 1]

table1 = pd.DataFrame({
    'quantile': quantiles,
    'centile': [100 * q for q in quantiles],
    'all': [annual_full_ecdf.get_quantile(q) for q in quantiles],
    'all_extractable': [annual_full_extractable_ecdf.get_quantile(q) for q in quantiles]
    #'h1': [annual_full_ecdf_h1.get_quantile(q) for q in quantiles],
    #'h2': [annual_full_ecdf_h2.get_quantile(q) for q in quantiles    ]
})
table1.set_index('quantile', inplace=True, drop=False)
table1.plot('all', 'quantile', kind='scatter', ax=ax2)
table1.plot('all_extractable', 'quantile', kind='scatter', ax=ax2)
table1.to_csv('table1_0110.csv') # saving data to csv file

annual_full_ecdf.to_csv('annual_full_ecdf_0110.csv') # saving data to csv file
annual_full_ecdf.plot(ax=ax2, label="Simulated ARP of validators")
annual_full_extractable_ecdf.to_csv('annual_full_extractable_ecdf_0110.csv') # saving data to csv file
annual_full_extractable_ecdf.plot(
    ax=ax2, alpha=0.5, label="Simulated ARP of validators with extracted MEV"
)


plt.scatter(table1['all_extractable'], table1['quantile'], edgecolors='orange')
c1 = table1['all'].loc[0.01]
c11 = table1['all_extractable'].loc[0.01]
ax2.annotate(f'1st centile: {c1:.2f}% APR vs {c11:.3f}% APR with extractable MEV', (c1 + 0.7, 0.02))
d1 = table1['all'].loc[0.1]
d11 = table1['all_extractable'].loc[0.1]
ax2.annotate(f'10th centile: {d1:.2f}% APR vs {d11:.3f}% APR with extractable MEV', (d11 + 0.7, 0.075))
lq = table1['all'].loc[0.25]
lq1 = table1['all_extractable'].loc[0.25]
ax2.annotate(f'25th centile (lower quartile): {lq:.2f}% APR vs {lq1:.3f}% APR with extractable MEV', (lq + 1, 0.225))
med = table1['all'].loc[0.5]
med1 = table1['all_extractable'].loc[0.5]
ax2.annotate(f'50th centile (median): {med:.2f}% APR vs {med1:.3f}% APR with extractable MEV', (med + 1, 0.475))
uq = table1['all'].loc[0.75]
uq1 = table1['all_extractable'].loc[0.75]
ax2.annotate(f'75th centile (upper quartile): {uq:.2f}% APR vs {uq1:.3f}% APR with extractable MEV', (uq + 1.3, 0.725))
d9 = table1['all'].loc[0.9]
d91 = table1['all_extractable'].loc[0.9]
ax2.annotate(f'90th centile: {d9:.2f}% APR vs {d91:.3f}% APR with extractable MEV', (d9 + 1.8, 0.875))
c99 = table1['all'].loc[0.99]
c991 = table1['all_extractable'].loc[0.99]
ax2.annotate(f'99th centile: {c99:.2f}% APR vs {c991:.3f}% APR with extractable MEV', (c99 - 6, 0.925))

ax2.set_title('Simulated Validator Rate of Return (420k validators, 1 year)')
ax2.set_xlabel('Rate of return (% APR)')
ax2.set_xlim(0, 35)
ax2.set_ylabel('Cumulative frequency density')
ax2.set_ylim(0, 1)
ax2.legend(title='APR of validators:', loc='lower right')

plt.show()

table1.drop('quantile', axis=1, inplace=True)
cols = [
    ('','Centile<br>(%)'),
    ('Rate of return (% APR)<br>based on data from:','Sep 2021<br>to Aug 2022'),
    ('Rate of return (% APR)<br>based on data from:','Sep 2021<br>to Feb 2022'),
    ('Rate of return (% APR)<br>based on data from:','Mar 2022<br>to Aug 2022'),
]

fmts = ['{:.1f}'] + ['{:.2f}'] * 3
col_formats = {c: f for c, f in zip(cols, fmts)}
dfi.export(table1, 'table3_0110.png')
print("table two", table1)

# calculate ECDFs for 1, 2, 4, 8, 16, 32 validators

rtns = []
ecdfs = []
ecdfs1 = []
for e in range(6):
    ecdfs.append(Ecdf([
        100 * sum(validators[i:i + 2 ** e]) / (32 * 2 ** e)
        for i in range(0, num_validators, 2 ** e)
    ]))
    ecdfs1.append(Ecdf([
        100 * sum(validators_extractable_MEV[i:i + 2 ** e]) / (32 * 2 ** e)
        for i in range(0, num_validators, 2 ** e)
    ]))

mean_return = 100 * sum(validators) / (32 * len(validators))
print("mean_return", mean_return)
mean_extractable_return = 100 * sum(validators_extractable_MEV) / (32 * len(validators_extractable_MEV))
print("mean_extractable_return", mean_extractable_return)

fig, ax = plt.subplots(figsize=(10, 6))
fig1, ax1 = plt.subplots(figsize=(10, 6))
for e in range(6):
    label = '1 validator' if e == 0 else f'{2 ** e} validators'
    ecdfs[e].plot(ax=ax, label=label)
    ecdfs1[e].plot(ax=ax1, label=label)

ax.set_xlim(3, 15)
ax.set_ylim(0, 1)
ax.set_title("ECDF for Validator Returns (1 year, 420k validators)")
ax.set_xlabel("Rate of return (% APR)")
ax.set_ylabel("Cumulative frequency density")
ax.axvline(mean_return, color='r', linestyle='dashed', label='Mean return')
ax.axvline(mean_extractable_return, color='b', linestyle='dashed', label='Mean return with extractable MEV')
ax.legend()

ax1.set_xlim(3, 15)
ax1.set_ylim(0, 1)
ax1.set_title("ECDF for Validator Returns (1 year, 420k validators)")
ax1.set_xlabel("Rate of return (% APR)")
ax1.set_ylabel("Cumulative frequency density")
ax1.axvline(mean_return, color='r', linestyle='dashed', label='Mean return')
ax1.axvline(mean_extractable_return, color='b', linestyle='dashed', label='Mean return with extractable MEV')
ax1.legend()

plt.show()

quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

table2 = pd.DataFrame({'quantile': [100 * q for q in quantiles]})
table3 = pd.DataFrame({'quantile': [100 * q for q in quantiles]})
for e in range(6):
    table2[2 ** e] = [ecdfs[e].get_quantile(q) for q in quantiles]
    table3[2 ** e] = [ecdfs1[e].get_quantile(q) for q in quantiles]
cols = [('', 'Centile<br>(%)')] + [
    ('Rate of return (% APR) for number of validators:', 2 ** e) for e in range(6)
]
fmts = ['{:.1f}'] + ['{:.2f}'] * 6
col_formats = {c: f for c, f in zip(cols, fmts)}

dfi.export(table2, 'table4_0110.png')
table2.to_csv('table2_0110.csv') # saving data to csv file
table3.to_csv('table3_0110.csv') # saving data to csv file
dfi.export(table3, 'table5_0110.png')
print("table three", table2)
