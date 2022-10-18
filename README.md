# MEV-and-Ethereum-validators-oligopoly

Big respect and huge thanks to pintail who created the simulation code based the Ethereum PoS setup and historical MEV data: https://github.com/pintail-xyz/post-merge-mev/blob/master/model-post-merge-returns.ipynb
In this study I used the simulation code, visualisation, and the data preparation methods from pintails´ work. 

Data used in the study is available under: https://drive.google.com/drive/folders/1YxkYNVhO8WDQXwinp8EfLp-4bJIeUELi?usp=sharing

## Abstract## 
  Smart contacts on Ethereum allowed new types of services such as DeFi (decentralized finance), thus bringing in a new wave of applications and new adopters. On the other hand, high volumes, inexperienced users, and not perfect architecture of applications created new roles (e.g., MEV searchers) and enabled new earning opportunities and incentives.
  Ethereum's miners & validators decide in which order the pending transactions will be added to a block.  At the same time, many users want their transaction to be included in a block as soon as possible to make a profit from, e.g.,  arbitrage opportunities between two liquidity pools. Therefore, such users want to incentivize miners & validators to take their transactions first by setting a higher gas price for their transactions. That is what Maximum Extractable Value (MEV) is about.
This research briefly explains the MEV phenomenon, shows the potential benefits of MEV to miners & validators, and analyzes the effect of MEV on validators' returns.
###Keywords: MEV, flashbots, Ethereum, blockchain, miners, validators.###

## Introduction ##
  Surprisingly to many inexperienced Ethereum users, Maximal Extractable Value (from now MEV) is essential even for them, besides miners/validators, application developers, blockchain architectures, and other participants. MEV is still a very new topic in blockchain (it was firstly defined by Daian et al., 2019, p. 2). Generally, there are two established groups of people who gain profit from MEV: searchers and miners & validators. $675.524.491 (Flashbots, n.d.) is the gross sum of totally extracted MEV, the searchers, however, have to pay some part (sometimes up to 99%) to the miners & validators; that makes MEV another source of revenue to miners & validators and creates new roles (MEV searchers) with potentially high-profit opportunities. After the merge, the validators will fight for market share by trying to offer higher annual yields to people who delegate ETH to them. 
  The research studies whether the validators who decided to play the role of MEV searcher and validator simultaneously can get higher returns (and thus offer higher yield).  Key findings are: on average, validators who extract MEV themselves have a higher annual percentage rate (APR) than those who do not use this opportunity (9.484%% vs. 8.048%). Although such difference has a low size effect, the reality is that validators who extract MEV themselves have at least the same returns as those who do not. Furthermore, there is almost no risk for validators to extract MEV themselves (if they extract non-negative MEV); thus, extracting MEV opportunities is always a better option. 
  The research partner, Staking Facilities GmbH, is an ambitious validator in many blockchains (Solana, Polkadot, Cosmos, and others; Staking Facilities, n.d.). At the same time, making additional revenue from MEV is potentially a competitive advantage for the validator since he could then offer higher APY for his stakers and thus attract more stakes, which leads to higher chances of being chosen to create a block. For these reasons, the company is highly interested in the research.

### MEV ###
  When a user initiates a new transaction, it is added to a list of other pending transactions (mempool). Since miners & validators act economically rational, they tend to include transactions with the highest tip they get from including transactions to the block (Zhou et al., 2020, p. 10). On the other hand, some users might want to include their transactions as fast as possible and pay a high gas price, for example for a lucrative arbitrage opportunity: such users are called searchers, they look for emerging opportunities by analyzing chain data and mempool. 
Maximal extractable value (MEV) is the process of including, excluding, and transaction scheduling in the block to get maximal value from block creation (ethereum.org, 2022). The analogy from the classical financial world is the high frequency trading: executing profitable transactions before other traders. However, MEV is more democratized since no huge initial capital investments are required.
### MEV searcher ###
MEV searcher is someone who monitors mempool and other data to find lucrative opportunities and extracts profit from those opportunities.
### Arbitrage ###
Arbitrage is one of the most popular types of MEV. When there is a price difference between the same coins on different DEXes, one can buy coins where the price is low and sell them on the DEX at a higher price, thus making profit with no risks due to atomic transactions (ethereum.org, 2022).
Arbitrage is only available on DEXes, because all pending transactions can be found in the mempool and thus be front-run. On the other hand, on CEXes there is no way for someone external to get the list of the pending transactions, so only the CEXes can front run their users. 
### Sandwiching ###
Sandwiching is another popular type of MEV: searchers are watching mempool for big transactions on the DEXes, since they will change the balanced price of the pair, so there will be profitable opportunity. The searchers may approximately calculate the effect of the large transaction on the pair and place their buy transaction right before the big transaction and sell transaction right after the large trade (ethereum.org, 2022).
  Example: trader wants to buy 10,000 UNI with DAI on a DEX. Such trade will change the balanced price of the pair UNI/DAI on the DEX. Since the UNI/DAI balance is changed, UNI will have a much higher price after the trade. A clever searcher finds this transaction in the mempool, places his UNI buy-transaction right before the trader's transaction, and places his UNI sell-transaction right after the trader's transaction. The searcher made profit, because the UNI he bought cost more DAI after the big trade, than before it. 
Sandwiching is more dangerous, since the transactions are not atomic and something might go wrong (Defi-Cartel, 2021).
### Liquidation ###
DeFi lending services create another popular type of MEV: to lend some coins, users have to provide some amount of collateral (other coins). The value of the collateral can fluctuate. When collateral depreciates over a certain threshold (when it gets risky), the lending protocol allows anyone to liquidate the collateral. In case of liquidation, the loan is repaid from collateral, and the lender has to pay liquidation fees, part of it is the reward of the liquidator.  Searchers are scanning blockchain data to find such loans that can be liquidated, post their liquidation transactions and try to get these included to a block before other liquidators (ethereum.org, 2022).
### Importance of MEV ###
$674.214.708 is the amount of total extracted MEV on Ethereum (Flashbots, n.d.) - an impressive number and an earning source for blockchain participants. 
There are different types of MEV, some of them, like liquidations or arbitrage are beneficial, but some of them, like front running, sandwich, or time bandit attacks can harm user experience and jeopardize blockchain security (Daian et al., 2019, pp. 14-16). Many MEV searchers and MEV attempts caused high competition for the block space and thus extremely high gas prices, resulting in enormously expensive usage of Ethereum for ordinary users (ethereum.org, 2022). MEV matters to normal users, miners/validators, application developers, and blockchain itself: an important topic. 
### MEV in ETH 2.0 ###
The “merge” of Ethereum was implemented in September 2022; the main goal is to move the Ethereum blockchain from Proof of Work (PoW) to more energy-efficient Proof of Stake (PoS) consensus algorithm (ethereum.org, 2022). 
After the merge, the transaction ordering process will not change massively; the main difference is that validators will be block proposers (before the merge, it was miners) (Flashbots, 2021).
### Flashbots ###
Flashbots is a research and development organization working on mitigating the negative externalities of MEV extraction techniques and avoiding the existential risks MEV could cause to blockchains like Ethereum (Flashbots, n.d.). Before the merge, Flashbots had a private mining pool with participating miners and searchers. Mainly, they offered direct communication channels between searchers and miners in the form of an auction: a mix between an English auction and an all-pay auction so that MEV rewards are distributed more efficiently (Flashbots, n.d.). 






