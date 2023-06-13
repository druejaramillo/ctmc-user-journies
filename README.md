# CTMC for Modeling User Journies

## Overview

The purpose of this project is to model user journies as continuous-time Markov chains (CTMCs) in order to extract insights into user behavior. In order to extract even more insights at a macro level, we model the total number of conversions as a Poisson process.

## Modeling User Journeys

### Continuous-Time Markov Chains (CTMCs)

The project used [30 days of live traffic data from Criteo](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/). Each data point corresponded to a particular touchpoint/impression and contained information on the time, marketing campaign, and user id of the touchpoint.

For each user, we can model their journey across the marketing campaigns using a CTMC. We can associate each campaign $i$ with a state $i$ of a CTMC. Then, we can use each user's sequence of touchpoints to build a CTMC for that user's journey. Furthermore, we can use the timestamps of each touchpoint to determine how long the user was in each state -- that is, how long they went before interacting with another campaign.

Once we build the CTMCs for each user, we now have data on the holding time parameters of each state and the transition rates between pairs of states. Since the holding/transition times are exponentially distributed, we can use a Bayesian approach with a conjugate gamma prior. Combined with the data, this gives us gamma posterior distributions for each parameter, from which we can derive estimates and insights.

### Poisson Processes

We can model the total number of conversions over time as a Poisson process. By taking the timestamps of each conversion, we can find their interarrival times. This gives us data on the parameter of the Poisson process. Since the interarrival times are exponentially distributed, we can use a Bayesian approach with a conjugate gamma prior. Combined with the data, this gives us gamma posterior distributions for the parameter, from which we can derive estimates and insights.

## Exploratory Data Analysis

We can use some EDA to get preliminary insights into the data, which may be useful in certain contexts. In addition, we can use EDA to check our assumptions that 1) the holding/transition times of the user journies are exponentially distributed and 2) the interarrival times of the conversions are exponentially distributed.

EDA was performed as follows:
1. Summary of the data (number of users, number of conversions, etc.)
2. Frequency plots of first and last touchpoints by campaign (i.e. how often each campaign was the first and/or last touchpoint for a user)
3. Histogram of transition times between the pair of campaigns that had the greatest number of transitions between them (to check assumption 1).
4. Histogram of interarrival times of conversions (to check assumption 2).
5. Summary statistics for the number of touchpoints for each user journey.
6. Summary statistics for the total time until conversion (in days) for each user journey.

## Insights Extraction

Once we have posterior distributions for our parameters, we can extract some insights as follows:
1. We can use the mean of the posterior distributions as estimators for our parameters.
2. Once we have estimators, we can calculate the expected holding times of each campaign (how long we expect before a user interacts with another campaign).
3. We can also calculate the expected absorption times of each campaign (how long we expect a user's journey to be given that they start with a particular campaign).
4. We can examine the parting paradox: while the expected interarrival time is $\lambda$, for a fixed time point $t$, the expected time between the previous and next conversions is $2\lambda$.

## Getting Started

This project requires the following non-standard Python libraries:
1. `numpy`
2. `pandas`
3. `scipy`
4. `matplotlib`