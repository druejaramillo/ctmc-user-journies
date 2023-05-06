import pandas as pd
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('./criteo_attribution_dataset/criteo_attribution_dataset.tsv.gz', sep='\t', compression='gzip')
data = data[['timestamp', 'uid', 'campaign', 'conversion', 'conversion_timestamp', 'conversion_id']]

# Calculate basic numbers
num_users = data['uid'].nunique()
num_conversions = data[data['conversion'] == 1]['conversion_id'].nunique()
num_converts = data[data['conversion'] == 1]['uid'].nunique()
percent_convert = num_converts / num_users
print(f'\n\nNumber of users: {num_users}')
print(f'Number of conversions: {num_conversions}')
print(f'Number of users that converted: {num_converts}')
print(f'Percent of users that converted: {round(100*percent_convert, 2)}%')

# Sort data into user journies
journies = data.sort_values(by=['uid', 'timestamp']).reset_index(drop=True)

# Extract journies for users that converted
converts = journies[journies['conversion'] == 1]['uid'].unique()
converted_journies = journies[journies['uid'].isin(converts)]

# Find when each user first converted
conversion_times = converted_journies[converted_journies['conversion_timestamp'] != -1]
conversion_times = conversion_times.groupby('uid').first()
conversion_times = conversion_times['conversion_timestamp']
conversion_times.rename('first_conversion_timestamp', inplace=True)

# Delete touchpoints after the first conversion
converted_journies = converted_journies.join(conversion_times, on='uid', how='left')
converted_journies = converted_journies[converted_journies['timestamp'] <= converted_journies['first_conversion_timestamp']].reset_index(drop=True)

# Replace the original conversion journies with the filtered ones
non_converted_journies = journies[~(journies['uid'].isin(converts))]
converted_journies.drop('first_conversion_timestamp', axis=1, inplace=True)
journies = pd.concat([non_converted_journies, converted_journies], ignore_index=True).sort_values(by=['uid', 'timestamp'])

# Find the first touchpoint's campaign for converted users
first_touches = converted_journies.groupby('uid').first()['campaign']
first_counts = first_touches.value_counts(normalize=True, sort=False)
first_counts.plot(kind='bar', xticks=range(0, len(first_counts), 50), title='First touchpoints', xlabel='Campaign ID', ylabel='Frequency')
plt.savefig('./first_counts.png')
plt.close()
plt.cla()
plt.clf()

# Find the last touchpoint's campaign for converted users
last_touches = converted_journies.groupby('uid').last()['campaign']
last_counts = last_touches.value_counts(normalize=True, sort=False)
last_counts.plot(kind='bar', xticks=range(0, len(last_counts), 50), title='Last touchpoints', xlabel='Campaign ID', ylabel='Frequency')
plt.savefig('./last_counts.png')
plt.close()
plt.cla()
plt.clf()

# Find the starting times for converted users' journies
initial_times = converted_journies.groupby('uid').first()
initial_times = initial_times['timestamp']

# Find the total time to conversion for converted users
total_conversion_times = conversion_times.subtract(initial_times)
total_conversion_times /= 86400
total_conversion_times.plot.hist(bins=30, title='Total times to conversion', xlabel='Total time of user journey', ylabel='Frequency')
plt.savefig('./total_conversion_times.png')
plt.close()
plt.cla()
plt.clf()
print('\n\nTotal times to conversion:')
print(total_conversion_times.describe())

# Find the number of touchpoints for each converted user
num_touchpoints = converted_journies.groupby('uid').count()
num_touchpoints = num_touchpoints['timestamp']
q = num_touchpoints.quantile(0.99) # remove outlier
num_touchpoints[num_touchpoints < q].plot.hist(bins=20, title='Total number of touchpoints', xlabel='# of touchpoints in user journey', ylabel='Frequency')
plt.savefig('./num_touchpoints.png')
plt.close()
plt.cla()
plt.clf()
print('\n\nNumber of touchpoints:')
print(num_touchpoints.describe())

# Create a list of states for the Markov chain
states = list(journies['campaign'].unique())
states.sort()
states.extend(['conversion', 'null'])

# Filter out journies of length 1 (only 1 touchpoint)
journies = journies[journies.groupby('uid').transform('size') > 1]

# For each touchpoint for each user, add the previous touchpoint's campaign
journies['prev_campaign'] = journies.groupby('uid')['campaign'].shift(periods=1).astype('Int64')

# For each touchpoint for each user, add the time since the last touchpoint
journies['time_since_last'] = journies.groupby('uid')['timestamp'].diff(periods=1).astype('Int64')

# Extract the transition times
transition_times = journies[['prev_campaign', 'campaign', 'time_since_last']].copy()
transition_times.sort_values(by=['prev_campaign', 'campaign', 'time_since_last'], inplace=True)

# Find the number of transitions between each pair of states (campaigns)
num_transitions = transition_times.groupby(['prev_campaign', 'campaign']).count()

# Plot transition times for pair of states with the most transitions
example = num_transitions.idxmax()
print(f'Pair of campaigns with the most transitions: {example}')
example_times = transition_times[(transition_times['prev_campaign'] == example[0][0]) & (transition_times['campaign'] == example[0][1])]
(example_times['time_since_last'] / 86400).plot.hist(bins=30, title='Example transition times', xlabel='Transition time', ylabel='Frequency')
plt.savefig('./example_transition_times.png')
plt.close()
plt.cla()
plt.clf()

# For converted users, find the transition time from their last campaign to conversion
last_touches = converted_journies.groupby('uid').last()[['campaign', 'timestamp']]
last_touches.sort_values(by=['campaign', 'timestamp'], inplace=True)
last_touches = last_touches.join(conversion_times, how='left')
conversion_times = pd.DataFrame()
conversion_times['prev_campaign'] = last_touches['campaign']
conversion_times['campaign'] = 'conversion'
conversion_times['time_since_last'] = last_touches['first_conversion_timestamp'] - last_touches['timestamp']

# For non-converted users, fill in a transition time from their last campaign to "null"
last_touches = non_converted_journies.groupby('uid').last()['campaign']
last_touches.sort_values()
null_times = pd.DataFrame()
null_times['prev_campaign'] = last_touches
null_times['campaign'] = 'null'
null_times['time_since_last'] = 3*86400

# Put all the transition times together
transition_times = pd.concat([transition_times, conversion_times, null_times], ignore_index=True)
transition_times.sort_values(by=['prev_campaign', 'campaign', 'time_since_last'])

# Calculate the number of transitions between each pair of states (campaigns)
num_transitions = transition_times.groupby(['prev_campaign', 'campaign']).count()
print('\n\nNumber of transitions between pairs of states:')
print(num_transitions.describe())

# Sum the transition times for each pair of states
transition_time_sums = transition_times.groupby(['prev_campaign', 'campaign']).sum()

# Calculate the posterior parameters for the transition rate between each pair of states
alpha = 1
beta = 3*86400
posterior_params = {state1: {state2: () for state2 in states} for state1 in states}
for state1 in states:
    for state2 in states:
        try:
            n = num_transitions.loc[(state1, state2), 'time_since_last']
            tt_sum = transition_time_sums.loc[(state1, state2), 'time_since_last']
        except:
            n = 0
            tt_sum = 5000000
        posterior_params[state1][state2] = (alpha+n, beta+tt_sum)

# Plot the common prior distribution for the transition rates
x = np.arange(1e-8, 3e-5, 1e-8)
y = gamma.pdf(x, alpha, scale=1/beta)
plt.plot(x, y, 'b-')
plt.xlabel('Transition rate q_ij')
plt.ylabel('Probability')
plt.title('Prior distribution for transition rates')
plt.savefig('./prior_transition_rates.png')
plt.close()
plt.cla()
plt.clf()

# Plot the posterior for the pair with the most transitions
a, b = posterior_params[example[0][0]][example[0][1]]
x = np.arange(1e-8, 1e-5, 1e-8)
y = gamma.pdf(x, a, scale=1/b)
plt.plot(x, y, 'b-')
plt.xlabel('Transition rate q_ij')
plt.ylabel('Probability')
plt.title('Posterior distribution for example transition rate')
plt.savefig('./posterior_example_transition_rate.png')
plt.close()
plt.cla()
plt.clf()

# Calculate posterior means for transition rates
posterior_means = {state1: {state2: 0 for state2 in states} for state1 in states}
for state1 in states:
    for state2 in states:
        a, b = posterior_params[state1][state2]
        posterior_means[state1][state2] = a / b
for state in states:
    posterior_means['conversion'][state] = 0
    posterior_means['null'][state] = 0

# Calculate holding time parameters and expected holding times
holding_time_params = {state: 0 for state in states}
for state1 in states:
    for state2 in states:
        holding_time_params[state1] += posterior_means[state1][state2]
holding_times = {state: (1/holding_time_params[state])/86400 for state in states if state not in ['conversion', 'null']}

# Plot expected holding times
plt.bar(range(len(holding_times)), list(holding_times.values()))
plt.xticks(range(0, len(holding_times), 100))
plt.xlabel('Campaign ID')
plt.ylabel('Expected holding time (days)')
plt.title('Holding times for each campaign')
plt.savefig('./holding_times.png')
plt.close()
plt.cla()
plt.clf()

# Create generator
Q = np.zeros((len(states), len(states)))
for i in range(len(states)):
    if states[i] not in ['conversion', 'null']:
        for j in range(len(states)):
            if i == j:
                Q[i][i] = -1 * holding_time_params[states[i]]
            else:
                Q[i][j] = posterior_means[states[i]][states[j]]

# Calculate fundamental matrix and expected absorption times
V = Q[:-2, :-2]
F = -1 * np.linalg.inv(V)
absorption_times = np.sum(F, axis=1) / 86400
print('\n\nExpected absorption times:')
print(pd.Series(absorption_times).describe())

# Plot expected absorption times
plt.figure(figsize=(6, 6))
plt.bar(range(len(F)), absorption_times)
plt.xticks(range(0, len(F), 100))
plt.xlabel('Campaign ID')
plt.ylabel('Expected time to absorption (days)')
plt.title('Expected absorption time by campaign')
plt.savefig('./absorption_times.png')
plt.close()
plt.cla()
plt.clf()

# Calculate arrival times for conversions
arrival_times = data[['uid', 'conversion_timestamp']].copy()
arrival_times = arrival_times[arrival_times['conversion_timestamp'] != -1]
arrival_times.drop_duplicates(inplace=True, ignore_index=True)
arrival_times = arrival_times['conversion_timestamp'].copy()
arrival_times.sort_values(inplace=True)

# Calculate and plot interarrival times
interarrival_times = arrival_times.diff(periods=1).dropna()
q = interarrival_times.quantile(0.99) # remove outlier
interarrival_times[interarrival_times < q].plot.hist(bins=30, title='Interarrival times of conversions', xlabel='Interarrival time', ylabel='Frequency')
plt.savefig('./interarrival_times.png')
plt.close()
plt.cla()
plt.clf()
print('\n\nInterarrival times:')
print(interarrival_times.describe())

# Plot prior distribution for lambda (Poisson process parameter)
alpha = 1
beta = 300
x = np.arange(1e-4, 3e-2, 1e-4)
y = gamma.pdf(x, alpha, scale=1/beta)
plt.plot(x, y, 'b-')
plt.xlabel('Poisson parameter lambda')
plt.ylabel('Probability')
plt.title('Prior distribution for Poisson parameter')
plt.savefig('./prior_lambda.png')
plt.close()
plt.cla()
plt.clf()

# Plot posterior distribution for lambda
a = alpha + len(interarrival_times)
b = beta + np.sum(interarrival_times)
x = np.arange(1e-6, 2e-1, 1e-6)
y = gamma.pdf(x, a, scale=1/b)
plt.plot(x, y, 'b-')
plt.xlabel('Poisson parameter lambda')
plt.ylabel('Probability')
plt.title('Posterior distribution for Poisson parameter')
plt.savefig('./posterior_lambda.png')
plt.close()
plt.cla()
plt.clf()

# Calculate expected interarrival time and expected wait time (Parting Paradox)
posterior_mean = a / b
print(f'\n\nExpected interarrival time: {1/posterior_mean}')
print(f'Expected waiting time: {2/posterior_mean}')