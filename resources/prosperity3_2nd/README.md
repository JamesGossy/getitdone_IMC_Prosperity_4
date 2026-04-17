# Frankfurt Hedgehogs 🦔

This writeup shares our algorithm and insights that brought us to 2nd place globally in IMC Prosperity 3 (2025). Outperforming (almost) all other 12,000+ teams, we achieved a final score of 1,433,876 SeaShells but unfortunately we didn't win the 10,000$ prize for it as we had already earned prize money in last year's competition :)

<table width="50%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="200px">
        <a href="https://www.linkedin.com/in/timo-diehm">
          <img src="https://github.com/user-attachments/assets/9a919806-70ff-4672-bbde-57ec67f891b6" width="150;" alt="Timo Diehm"/>
          <br />
          <p><b>Timo Diehm</b></p></a>
      </td>
      <td align="center" valign="top" width="200px">
        <a href="https://www.linkedin.com/in/arne-witt">
          <img src="https://github.com/user-attachments/assets/c57bbac1-0448-4a6b-8ad1-cbd225790b3d" width="150;" alt="Arne Witt"/>
          <br />
          <p><b>Arne Witt</b></p></a>
      </td>
      <td align="center" valign="top" width="200px">
        <a href="https://www.linkedin.com/in/marvin-schuster">
          <img src="https://github.com/user-attachments/assets/0d885b67-11b8-4794-84d5-995918abe3f0" width="150;" alt="Marvin Schuster"/>
          <br />
          <p><b>Marvin Schuster</b></br></p></a><span>*only mental support in Prosperity 3</span>
      </td>
    </tr>
  </tbody>
</table>

<br/>

After countless requests, we decided to share our final algorithm along with all of our insights to give back to the Prosperity 3 community. We believe that sharing our code advances the competition itself — ensuring that more participants start on the same page for future iterations, and encouraging innovation on IMC’s side.
While Prosperity 3 already introduced many new products and trading styles, there is still a lot of untapped potential — especially in designing bot behaviors that create deeper and more exploitable opportunities for highly advanced teams.
We realize that fellow or future participants have varying levels of experience with quant and algorithmic trading, so we tried to make this write-up as detailed and accessible as possible. Some topics are just too deep to explain in a short paragraph, so we included links to external resources that less experienced readers should carefully study.

<br/>

This report goes far beyond just presenting our final strategies.
We not only break down those strategies and insights that worked for us, but also share the thought processes and decisions behind them.
That said, this document is mainly intended for fellow or future Prosperity participants, since it focuses specifically on Prosperity 3.
If you're more interested in how we consistently stayed at the top across multiple competitions — and want general advice on how to compete against thousands of teams — check out our separate blog post:
#### [How to (Almost) Win Against Thousands of Other Teams in Competitive Environments](https://medium.com/@td.timodiehm/how-to-almost-win-against-thousands-of-other-teams-in-competitive-environments-bc31387e4b26)

<br>

## IMC Prosperity 3

IMC Prosperity 3 (2025) was a global [algorithmic trading](https://www.investopedia.com/terms/a/algorithmictrading.asp) competition that ran over five rounds and fifteen days, with 12,000+ teams participating worldwide. The challenge tasked teams with designing trading algorithms to maximize profits across a variety of simulated products — replicating real-world opportunities such as [market making](https://www.investopedia.com/terms/m/marketmaker.asp), [statistical arbitrage](https://www.investopedia.com/terms/s/statisticalarbitrage.asp), [scalping](https://www.investopedia.com/terms/s/scalping.asp), and [locational arbitrage](https://www.investopedia.com/terms/a/arbitrage.asp).

The competition was gamified: each team represented an "island" trading fictional products like Kelp, Squid Ink, Picnic Baskets (an [ETF](https://www.investopedia.com/terms/e/etf.asp) analog), and Volcanic Rock Vouchers (an [options](https://www.investopedia.com/terms/o/option.asp) analog), using SeaShells as the in-game currency. It started with just three products in Round 1 and progressively expanded to 15 products by the final round.

In each round, teams submitted an updated version of their trading algorithm, which was then independently evaluated against a marketplace of bot participants. Teams could study and optimize their algorithms by analyzing bot behaviors and interactions (e.g., predictable quoting or trading patterns) as well as statistical patterns in the price series themselves — both within a single product and across multiple related products (such as deviations between an ETF and its underlying constituents). The profit and loss (PnL) from this evaluation determined each team's standing relative to all others on the global leaderboard.

In addition to algorithmic trading, each round featured a manual trading challenge. Although these accounted for only a small fraction of total PnL, they were a fun aspect of the competition, often involving optimization under uncertainty, game-theoretic decision-making, or news-based trading tasks.

For full documentation on the algorithmic trading environment and more competition context, please refer to the [Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4).

<br>

## Structural Overview

- [Tools](#tools)
- [Algorithmic Challenge](#algorithmic-challenge)
  - [Round 1: Market Making](#round-1-market-making)
  - [Round 2: ETF Statistical Arbitrage](#round-2-etf-statistical-arbitrage)
  - [Round 3: Options Scalping](#round-3-options-scalping)
  - [Round 4: Location Arbitrage](#round-4-location-arbitrage)
  - [Round 5: Trader IDs](#round-5-trader-ids)
- [Manual Challenge](#manual-challenge)
  - [Round 1: FX Arbirage](#round-1-fx-arbitrage)
  - [Round 2: Containers](#round-2-containers)
  - [Round 3: Reserve Price](#round-3-reserve-price)
  - [Round 4: Suitcases](#round-4-suitcases)
  - [Round 5: News Trading](#round-5-news-trading)
- [FAQ](#frequently-asked-questions)
  - [What is the Wall Mid and why did we use it?](#what-is-the-wall-mid-and-why-did-we-use-it)
  - [How to properly backtest?](#how-to-properly-backtest)
  - [How to break into quant trading?](#how-to-break-into-quant-trading)
  - [Is the Discord Channel useful?](#is-the-discord-channel-useful)
  - [What was going on with all the hardcoding in the first two rounds?](#what-was-going-on-with-all-the-hardcoding-in-the-first-two-rounds)
  - [What else did we try?](#what-else-did-we-try)

<br>


## Tools

Having the right tools prepared before the competition is critical for maximum efficiency during the competition itself.
Prosperity 2’s data was publicly available, allowing teams to familiarize themselves with the data formats, set up the tutorial environment early, and test their algorithms and logging infrastructure well before the official start of Prosperity 3.

### Backtester

For backtesting, we mainly relied on our own forked version of Jmerle’s [open-source backtester](https://github.com/jmerle/imc-prosperity-3-backtester) alongside the Prosperity website’s own backtesting functionality.
Each served different, specific purposes in our workflow — for a detailed explanation of how we approached backtesting, please refer to the [Backtesting Section](#how-to-properly-backtest).

### Dashboard

We developed our own dashboard as a preparation for Prosperity 2, and further updated and improved it before Prosperity 3 — adding features that we didn’t have time to implement during the first competition.
Since this dashboard will be heavily referenced when we explain our strategies and insights across all products, we’ll first give a detailed description of it here.

Prosperity — like real-world trading — puts strong emphasis on [market microstructure](https://en.wikipedia.org/wiki/Market_microstructure). A proper, intuitive [order book](https://www.investopedia.com/terms/o/order-book.asp) visualization tool is essential for building the deep intuition necessary to recognize and exploit profitable patterns.

Unlike many standard trading dashboards, we designed ours completely from scratch, based on what was actually most useful for this particular competition.
Aesthetics were never our priority — everything was optimized purely for functionality and speed during use.
(Please keep that in mind — we know it’s ugly!)

![dashboard explanation](https://github.com/user-attachments/assets/6c283b73-07e3-4b3a-b8b5-9b38cc51b314)

In the main plot, you can see **price levels**:

- **[Ask](https://www.investopedia.com/terms/a/ask.asp) (sell) quotes** are plotted in **red**.
- **[Bid](https://www.investopedia.com/terms/b/bid.asp) (buy) quotes** are plotted in **blue**.

Markers represent **trades**:

- **Squares** = trades by [makers](https://www.cmegroup.com/education/courses/trading-and-analysis/market-makers-vs-market-takers.html#market-maker).
- **Triangles** = trades by [takers](https://www.cmegroup.com/education/courses/trading-and-analysis/market-makers-vs-market-takers.html#market-taker).
- **Crosses** = our own trades.

Each numbered section in the dashboard corresponds to a specific functionality:

1. **Hoverable Tooltip**
   Displays who traded, how much, and at what price at the hovered timestamp.

2. **PnL Panel**
   Shows the profit and loss for the currently selected product.

3. **Position Panel**
   Displays the net position for the selected product over time.

4. **Log Viewer**
   Parses our own logger outputs into a clean, timestamp-synced view.  
   Always matches the time currently hovered over in the main plot.

5. **Selection Controls**
   Allows selecting:
   - The log file.
   - The product (e.g. Kelp).
   - Specific **logged indicators** to overlay onto prices.
   
   A powerful feature here is the **normalization dropdown**:  
   By selecting an indicator (e.g., [WallMid](#what-is-wall-mid-and-why-did-we-use-it) — our proxy for the "true price"), all prices can be normalized relative to it.  
   This is extremely useful for visualizing strategies like mean reversion (When having **PICNIC_BASKET1** selected normalizing by the sum of its constituents perfectly demonstrates the mean reversion of the [basket's premium](https://www.fidelity.com/learning-center/investment-products/etf/premiums-discounts-etfs) still maintaining the orderbook plotting style).

6. **Trade Filtering and Visualization**
   Controls what types of trades and order book elements to display:
   - Toggle order book levels.
   - Toggle all trades, specific trader groups or specific traders:
     - **M** (maker)
     - **S** (small taker)
     - **B** (big taker)
     - **I** (informed trader)
     - **F** (our own trades)
   - Set quantity filters to only show trades within a specified size range, especially helpful when trader IDs still unknown.

7. **Performance and Downsampling Controls**
   Adjusts dynamic downsampling and visibility thresholds to prevent lag when visualizing large datasets.

Notes:
- We intentionally avoided any existing known dashboard styles; instead, we focused purely on designing what helped us most during analysis or checking of our algorithm during intense rounds.
- The visualization choice (scatter plot as order book depth representation) was made based on the specific structure of Prosperity markets — where products typically have only 1–4 meaningful price levels.

<br>

# Algorithmic Challenge

## Round 1: Market Making

### Rainforest Resin

Rainforest Resin was the simplest and most beginner-friendly product in Prosperity 3, perfectly suited to teach the fundamentals of [market making](https://www.investopedia.com/terms/m/marketmaker.asp). The product’s true price was permanently fixed at 10,000, meaning there were no intrinsic price movements to worry about. This setup clearly demonstrated the roles of [makers](https://www.cmegroup.com/education/courses/trading-and-analysis/market-makers-vs-market-takers.html#market-maker) and [takers](https://www.cmegroup.com/education/courses/trading-and-analysis/market-makers-vs-market-takers.html#market-taker): takers would cross the true price by either buying above 10,000 or selling below it, while makers posted passive orders hoping to be [matched](https://www.investopedia.com/terms/m/matchingorders.asp). The only thing that mattered for profitability here was the distance between the trade price and the true price — commonly referred to as the "edge." In short, the further you could buy below 10,000 or sell above 10,000, the better.

A key insight not just for Rainforest Resin but for all Prosperity products was understanding how the simulation handled order flow. At the start of every new timestep, the simulation first cleared all previous orders. Then, it sequentially processed new submissions: first some deep-liquidity makers, then occationally some takers, then our own bot’s actions (take or make), followed by other bots — usually more takers. This structure meant that speed and order cancellation were irrelevant: you had a full snapshot of the book and could submit any combination of passive or aggressive orders without racing against anyone. For Rainforest Resin, this confirmed that all focus should be on carefully optimizing the edge versus fill probability trade-off.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 1: Rainforest Resin Orderbook over Time</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/54363d35-63ac-406f-b2de-ad6a06e7433d"
       alt="Dynamic dashboard"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>Snippet of orderbook over time for Rainforest Resin.  
  Black stars are our quotes. Orange crosses are fills we got, profitable opportunities we immediately took, or trades at 10,000 we used to unwind inventory.</em>
</td>
</tr>
</table>

#### Final Strategy

Our final strategy for Rainforest Resin was straightforward. Each timestep, we first immediately took any favorable trades available — buying below 10,000 or selling above it. Afterward, we placed passive quotes slightly better than any existing liquidity (existing orders in orderbook): overbidding on [bids](https://www.investopedia.com/terms/a/bid.asp) and undercutting on [asks](https://www.investopedia.com/terms/a/ask.asp) while maintaining positive edge. If inventory became too skewed, we flattened it at exactly 10,000 to free up risk capacity for the next opportunities. No sophisticated logic or aggressiveness was needed due to the stable true price and the clean snapshot-based trading model.

Anyone could have come up with this approach by carefully studying the competition's matching rules and observing the environment during the tutorial round. Realizing that the true price was constant, fills were processed sequentially, and that orders only lived for one timestep simplified the problem dramatically. Having a basic visualization of price levels and logging fill quality would have made it even more obvious. Rainforest Resin alone consistently contributed around 39,000 SeaShells per round to our total PnL.

<br>

### Kelp

Kelp was very similar in nature to Rainforest Resin, with the only major difference being that its price could move slightly from one timestep to the next. Instead of a fixed true price like Rainforest Resin, Kelp's true price followed a slow [random walk](https://www.investopedia.com/terms/r/randomwalktheory.asp). However, this movement was minor enough that the basic structure of the problem remained unchanged. Buyers and sellers still interacted as takers when crossing the fair price, and makers earned profits based on how far their trades deviated from the true price at the moment of execution.

The critical insight for Kelp was recognizing that, despite small movements, the future price was essentially unpredictable. Once teams realized that takers lacked predictive power and that the next true price could not be systematically forecasted, it became clear that the best available estimate for the true price was simply the current one. In fact, while there was a minor technical edge — stemming from the fact that the true price was internally a floating-point value and orders could only be posted at integer levels (creating slight mean-reversion tendencies after ticks) — this effect was too small to materially alter strategy. Just like with Rainforest Resin, the optimal approach was to treat the [WallMid](#what-is-wall-mid-and-why-did-we-use-it) as the fair price and quote around it.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 2a: Kelp Orderbook over Time (Raw)</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/2a7c36dc-76b8-482d-934b-c9ee7ff527f6"
       alt="Dynamic dashboard"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>Same as Figure 1, but showing Kelp's price movement over time.</em>
</td>
</tr>
</table>


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 2b: Kelp Orderbook over Time (Normalized)</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/5b7828a5-df9b-44ae-ab11-6461ee026a51"
       alt="Static, normalized dashboard"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>Same as Figure 2a, but with prices normalized by the Wall Mid indicator to make the series stationary.  
  Notice how it resembles Rainforest Resin, but with a tighter bid-ask spread.</em>
</td>
</tr>
</table>

#### Final Strategy

Our final strategy for Kelp was nearly identical to that for Rainforest Resin. At each timestep, we first immediately took any favorable trades available relative to the current wall mid, then placed slightly improved passive orders (overbidding and undercutting) around the fair price. If inventory became too large, we neutralized it by trading at zero edge relative to the current price estimate. No major changes were needed compared to the first product.

Teams that approached Kelp correctly would have first verified whether takers or the market exhibited any predictability, either through simple empirical analysis or by observing that naive strategies (like quoting around the current price) worked well. Realizing that there was no meaningful adverse selection risk meant that treating Kelp identically to Rainforest Resin was the optimal path. On average, Kelp generated around 5,000 SeaShells per round, primarily limited by the tighter spreads compared to the first product.

<br>

### Squid Ink

Squid Ink differed from the previous two products mainly in that it had a tighter bid-ask spread relative to its average movement, combined with occasional sharp price jumps. This made pure market-making less attractive, not because of systematic losses, but because it introduced higher variance in realized PnL. In other words, fills could swing more widely in value depending on unpredictable price jumps, even if there was no predictable [adverse selection](https://www.investopedia.com/terms/a/adverseselection.asp) in the classic sense. Officially, the product was described as mean-reverting in the short term, suggesting that mean-reversion strategies might work. However, after investigating the market dynamics more carefully, we discovered an entirely different and more reliable opportunity.

Our main insight was that one of the anonymous bot traders consistently exhibited a strikingly predictable pattern: buying 15 lots at the daily low and selling 15 lots at the daily high. We observed this behavior early on, without initially knowing who the trader was. It was only in the final round — when trader IDs were temporarily visible — that we learned this trader was named Olivia. Anticipating this kind of behavior and designing logic to detect it gave us a clear edge. Without revealing our exact identification method (to avoid encouraging blind copying), the general approach involved tracking the daily running minimum and maximum. When a trade occurred at a daily extreme — and in the expected direction relative to the mid price — we flagged it as a signal and positioned accordingly. False positives were managed by monitoring for corresponding new extrema that contradicted earlier signals.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 3a: Squid Ink Prices with Informed Trader</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/9f552b07-98e9-4488-b4b9-95b2e1435747"
       alt="Dynamic dashboard"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows that Olivia bought exactly at the daily minimum and sold exactly at the daily maximum.</em>
</td>
</tr>
</table>

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 3b: Squid Ink Prices with Anonymous Trades</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/b6e23225-fd1f-4971-ad00-729ec2bdef8f"
       alt="Static, normalized dashboard"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot filters all anonymous trades to only show trades with quantity = 15, as it appeared during early rounds.  
  Careful teams could have spotted this pattern and identified Olivia's behavior during Rounds 1–4.</em>
</td>
</tr>
</table>

#### Final Strategy

Our final strategy for Squid Ink focused purely on following this daily-extrema trading behavior, dynamically updating our positions based on detected trades and resetting when invalidations occurred. No active market making or mean reversion trading was used for this product. The result was a low-risk, high-reliability PnL contributor that did not rely on predicting price moves directly.

Anyone who carefully analyzed historical Prosperity 2 data or public write-ups — such as [Stanford Cardinal’s](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal) or [Jasper's](https://github.com/jmerle/imc-prosperity-2) — could have anticipated similar behaviors and prepared detection logic in advance. We also discovered and executed this strategy on another product in Prosperity 2 without having participated in Prosperity 1. Early identification of this behavior consistently netted us on average 8,000 SeaShells per round, providing a stable and important edge in Round 1.

<br>

## Round 2: ETF Statistical Arbitrage

### Picnic Baskets

In Round 2, three new individual products — Croissants, Jams, and Djembes — were introduced alongside two new baskets: PICNIC_BASKET1 (6x Croissants, 3x Jams, 1x Djembes) and PICNIC_BASKET2 (4x Croissants, 2x Jams).
Each basket represented a combination of different quantities of the three products, but crucially, it was not possible to directly convert baskets into their underlying constituents.
This setup clearly simulated a basic ETF (Exchange-Traded Fund) structure: linked assets that normally move together, but which might temporarily deviate, creating statistical arbitrage opportunities.
In quantitative trading, finding and exploiting such linkages — when the synthetic price of a basket diverges from the sum of its parts — is a classic technique.

A deeper look revealed two main spread opportunities: first, trading the spread between the two baskets adjusted by Djembes (ETF1 - 1.5*ETF2 - Djembes), and second, trading each basket relative to its synthetic value based on the underlying products (ETF - Constituents).
While both avenues were possible, we quickly identified that comparing baskets directly to their constituent sums was the stronger and more reliable path.


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 4: Basket Spreads over Time</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/9446a89f-fca0-4673-aec4-d65e09921129"
       alt="Basket spread plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows the spreads (basket price minus synthetic value) for both Picnic Baskets over time, revealing short-term mean reversion patterns.</em>
</td>
</tr>
</table>

When approaching this kind of structure, it's crucial not to blindly apply textbook strategies but to first ask a fundamental question:
How could the market data have been generated?
The most natural generation process seemed to be that the three constituents' prices were independently randomized, and a mean-reverting noise sequence was then added on top to produce the basket price.
If that's true — and our early testing supported it — then the baskets were mean-reverting relative to their synthetic value, while the constituents themselves were not responding to the baskets.
Thus, it made sense to treat baskets as drifting toward their synthetic value, not the other way around.
Furthermore, while "hedging" by taking opposite positions in constituents could reduce variance, it would actually lower expected value slightly, especially when accounting for spread costs.

This understanding had important implications for strategy design.
Many teams might have rushed into using moving average crossovers or z-scores for trading signals — but applying such methods without a clear theoretical justification is dangerous.
For instance, a moving average crossover only makes sense if you believe there is a short-term trend overlaying a longer-term mean, which was not suggested by the structure here.
Similarly, using a z-score normalizes the spread by recent volatility, but unless volatility is known to vary meaningfully over time (which we did not observe here), this introduces unnecessary complexity and risk of overfitting.
It's easy to fall into the trap of throwing fancy techniques at the problem after a few hours of backtesting — but if you can't explain why a strategy should work from first principles, then any "outperformance" in historical data is probably noise.
From the beginning, we placed the highest value on building a deep structural understanding and keeping strategies simple, minimizing parameters whenever possible to maximize robustness.

#### Final Strategy

Based on that philosophy, our final strategy was built around a fixed threshold model.
We entered long positions on the basket when the spread fell below a certain negative threshold, and short positions when it rose above a positive threshold.
Instead of dynamically scaling signals or chasing moving averages, we relied on fixed levels tuned through light grid search, focusing on robustness rather than maximizing historical PnL.
We further enhanced this base strategy by integrating an informed signal:
having already detected Olivia's trading behavior on Croissants (similar to Ink Squid), we used her inferred long/short position to bias our basket spread thresholds dynamically.
For example, if our base threshold was ±50, detecting Olivia as short would shift the long entry to -80 and the short entry to +20, dynamically tilting our bias in the favorable direction.
This cross-product adjustment allowed us to intelligently exploit correlations between Croissants and the baskets without overcomplicating the system.


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 5: Optimal Parameter Search Grid</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/3b0f9a5d-e21e-41e3-82df-d96789ace379"
       alt="Optimal parameter grid search"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>These plots show the backtested basket PnLs across different parameter combinations.  
  The left plot corresponds to Basket 1, and the right plot to Basket 2.  
  <code>num_param_1</code> represents the base threshold values, while <code>num_param_2</code> controls the informed threshold adjustments based on Olivia's position.</em>
</td>
</tr>
</table>


During parameter selection, we always prioritized landscape stability over pure performance peaks.
Rather than picking the best parameter set based on maximum backtested profit, we chose combinations that showed consistent, flat regions of good performance, reducing sensitivity to slight shifts and avoiding overfit disasters.
Additionally, because we noticed that the basket prices carried a slight persistent premium (the mean of the spread was not zero), we subtracted an estimated running premium from the spread during live trading, continuously updating it to prevent bias.

Also, for the final round, we were uncertain whether or not to fully hedge our basket exposure with the constituents.
Recognizing that any trading strategy can be viewed as a linear combination of two other strategies — in this case, fully hedged and fully unhedged — we decided to hedge 50% of our exposure as a balanced compromise.
Additionally, we adjusted our execution logic: instead of waiting for spreads to fully revert and cross opposite thresholds, we neutralized positions immediately upon spreads crossing zero (adjusted for the informed signal).
This change aimed to reduce variance and lock in profits more consistently, while maintaining approximately zero expected value under the assumption that spreads did not exhibit momentum when crossing zero.
It is important to note that here, "zero" still referred to the base threshold after incorporating informed adjustments.

Anyone thinking carefully about the problem — starting from generation assumptions, doing proper exploratory data analysis, and resisting the temptation to blindly overfit — could have arrived at a similar approach.
Concepts like synthetic replication, mean-reversion modeling (e.g., Ornstein-Uhlenbeck processes), and cross-product signal integration are core ideas in quantitative finance.
With the dynamic informed adjustment based on Croissants, our strategy made about 40,000–60,000 SeaShells per round on baskets, plus another 20,000 SeaShells per round directly from trading Croissants individually.

<br>

## Round 3: Options Scalping

### Options

In Round 3, the competition introduced a new class of assets: Volcanic Rock Vouchers — effectively call options on a new underlying product, Volcanic Rock (VR).
There were five vouchers available, each with a distinct strike price — 9500, 9750, 10000, 10250, and 10500 — while the underlying Volcanic Rock itself traded around 10,000.
Each voucher granted the right (but not the obligation) to buy Volcanic Rock at the specified strike at expiry.
Importantly, options had limited time to live: starting with seven days until expiry in the first round, decreasing to just two days by the final round.
Without basic familiarity with options theory, particularly concepts like implied volatility and option pricing models, it would have been difficult to design strong strategies for this product.

#### IV Scalping

Our first major insight came from following hints dropped in the competition wiki, suggesting the construction of a volatility smile: plotting implied volatility (IV) against moneyness.
By fitting a parabola to the observed IVs across strikes and then detrending (subtracting the fitted curve from observed values), we could isolate IV deviations that were no longer dependent on moneyness.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 6a: Volatility Smile</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/49be51d8-4335-4831-adb0-e811e50ce450"
       alt="Volatility smile scatter plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This scatter plot visualizes implied volatility (<em>v<sub>t</sub></em>) versus moneyness (<em>m<sub>t</sub></em>) across different strikes.  
  A fitted parabola is shown to filter out noise, producing <em>v̂<sub>t</sub></em> — the "fair" implied volatility given <em>m<sub>t</sub></em>.  
  Outliers at the bottom left were disregarded, as they corresponded to historical points where extrinsic value was too low.</em>
</td>
</tr>
</table>

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 6b: IV Deviations over Time</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/6aa60cbe-029d-49ed-b883-95c9b7e177df"
       alt="IV deviations plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows the time series of implied volatility deviations (<em>v<sub>t</sub> - v̂<sub>t</sub></em>) derived from Figure 6a, highlighting short-term patterns in relative option mispricing.</em>
</td>
</tr>
</table>


To convert these into actionable trading signals, we input the volatility-smile-implied IV into a Black-Scholes model to calculate a theoretical fair price, then compared it to the actual market price to find price deviations.
Plots of these price deviations — especially for the 10,000 strike call early on — revealed sharp short-term fluctuations, indicating scalping opportunities.



<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 6c: Price Deviations over Time</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/ca6b1614-c6b2-4026-b41e-5af408fae69c"
       alt="Price deviations plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows the same implied volatility deviations from Figure 6b, but transformed into price space using the Black-Scholes model, providing a more intuitive view of relative mispricings over time.</em>
</td>
</tr>
</table>


We initially focused on the 10,000 strike, but dynamically expanded to include other strikes as the underlying shifted and expiry approached, tracking profitability thresholds in real time to decide when to activate scalping on new options.
Statistical analysis, specifically testing for 1-lag negative autocorrelation in returns, strongly supported the existence of exploitable short-term inefficiencies across several strikes, further validating this approach.



<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 7a: 10k Call Price Fluctuations</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/756d8dab-e76a-4ea6-a986-03d15d5f3bc3"
       alt="10k call price fluctuations"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows short-term price fluctuations of the 10,000 strike call option.  
  The orange indicator represents the theoretical call price, calculated using the implied volatility from the fitted parabola (<em>v̂<sub>t</sub></em>) at the option’s current moneyness.</em>
</td>
</tr>
</table>


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 7b: 10k Call Price Fluctuations (Normalized)</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/da9ae65a-b0a4-49e0-b072-b9abdbffad68"
       alt="Normalized 10k call price fluctuations"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows the same 10,000 strike call fluctuations as in Figure 7a,  
  but with prices normalized by the theoretical value (orange indicator) to make deviations more stationary and visually clear.</em>
</td>
</tr>
</table>

#### Gamma Scalping
The expected value from gamma scalping was consistently positive, as the gains from underlying price movements outweighed the losses from time decay. This made buying options and rehedging the resulting deltas from gamma exposure a relatively low-risk way to generate profit. However, while the approach was stable and mostly safe, the absolute returns were limited. It was a reliable source of small gains, but ultimately, we had a higher risk appetite and wanted better returns.

#### Mean Reversion Trading

Simultaneously, analysis of the underlying Volcanic Rock asset suggested potential mean reversion behavior.
Return distributions and price dynamics resembled Squid Ink, which was explicitly designed to mean revert in Round 1.
Autocorrelation analysis of Volcanic Rock returns, compared against randomized normal samples, confirmed significant short-term negative autocorrelation at various horizons, although caution was needed given the presence of large jumps and non-normal return distributions.
Given the limited historical data available (only three days), and uncertainty about future dynamics, fully committing to mean reversion was considered too risky.
Instead, we implemented a lightweight mean reversion model: tracking a fast rolling Exponential Moving Average (EMA) and trading deviations from this EMA using fixed thresholds — without scaling by rolling volatility — to keep the model simple and robust.


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 8: Autocorrelation Plot for Volcanic Rock</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/ae8f01cf-9cd1-4867-ba26-dfcae781ccff"
       alt="Autocorrelation plot for Volcanic Rock"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>Rolling autocorrelation of Volcanic Rock returns compared to autocorrelations from purely random sequences,  
  suggesting statistically significant mean reversion behavior in the underlying.</em>
</td>
</tr>
</table>

#### Final Strategy

In the end, we deployed a hybrid strategy combining both alpha sources.
Our core focus remained on IV scalping, dynamically expanding across strikes and adjusting thresholds based on evolving conditions, while simultaneously maintaining a moderate mean reversion position — both in the underlying Volcanic Rock and in the deepest in-the-money call (the highest delta option available).
Importantly, this was not a delta hedge in the traditional sense: the delta exposure from scalping was relatively small, and explicit delta hedging would have been prohibitively expensive bid-ask spreads. It was rather a hedge against bad luck. Because this hybrid model was designed to minimize maximum regret across different possible market outcomes: it protected us if strong mean reversion materialized (even if other teams aggressively leveraged mean reversion delta exposure across multiple options and therefore outperforming us in a relative sense), while keeping our core reliance on the more stable, theory-supported scalping opportunities.

Someone could have arrived at a similar strategy without deep prior options expertise by carefully observing the market dynamics.
Even without constructing a full volatility smile, simply watching option prices — particularly the 10,000 strike — would reveal clear short-term mean-reversion patterns and negative autocorrelation in returns.
On the underlying asset side, basic return autocorrelation analysis and exploratory plotting would hint at mean reversion tendencies.
Thus, while a strong theoretical background was helpful, a combination of attentive observation, critical data analysis, and statistical common sense would have led to very similar conclusions.

In terms of results, IV scalping contributed approximately 100,000 - 150,000 SeaShells per round, providing strong and stable profits across all rounds. Mean reversion trading was much more volatile, delivering around 100,000, -50,000, and -10,000 SeaShells across the rounds respectively. Despite the swings, our hybrid approach allowed us to achieve consistently positive net results while keeping downside risks manageable.

Note: After the fourth round, where the mean reversion strategy resulted in a loss of approximately 50,000 SeaShells, we reassessed its validity. Although we no longer found strong empirical evidence to justify continuing with mean reversion purely on standalone expected value grounds, we knew that several top teams were actively only trading mean reversion strategies. So we figured, if they wouldn't find the IV scalping strategy, they might just accept the coinflip and go all in mean reversion because otherwise they would surely get overtaken by everyone. Facing a 200,000 SeaShell lead at that point, we made a calculated decision to maintain some mean reversion exposure — not because we believed it was necessarily positive EV anymore, but to hedge relatively against the teams still pursuing that angle. We estimated the 95% Value at Risk (VaR) of the mean reversion component to be around 50,000 SeaShells — only about 25% of our lead — leaving us with sufficient margin even if the strategy failed again. Under our assumptions, keeping this balanced exposure maximized our likelihood of securing first place by minimizing relative downside risk while preserving our core scalping profits. This turned out to be the right decision. Although, in the last round some random team very unnaturally jumped from 100+ rank to 1st place, we could keep a healthy distance to all teams that were previously close behind us. 

<br>

## Round 4: Location Arbitrage
  
### Macarons

In Round 4, Magnificent Macarons was introduced.
Their fictional value was described as depending on external factors like hours of sunlight, sugar prices, shipping costs, tariffs, and storage capacity.
Macarons could be traded on the local island exchange via a standard order book, or externally at fixed bid and ask prices, adjusted for im-/export and transportation fees.
The position limit for Macarons was 75 units, with a conversion limit of 10 units per timestep.
This setup opened up both straightforward arbitrage opportunities and, for those who studied the environment carefully, access to a much deeper hidden edge.

At first glance, the standard arbitrage logic applied: whenever the local bid exceeded the external ask (after fees), or the local ask was lower than the external bid, profitable conversions were possible.
However, there was a critical hidden detail: a taker bot existed that aggressively filled orders offered at attractive prices relative to a hidden "fair value."
Through experimentation, we discovered that offers priced at about int(externalBid + 0.5) would often get filled, even when no visible orderbook participants were present.
This taker bot executed approximately 60% of eligible trades, meaning that — in expectation — you could sell locally for a price about 3 SeaShells higher than the naive local best bid.
Over the course of 10,000 timesteps with a 10-unit conversion limit, this small price improvement could theoretically yield up to 300,000 SeaShells.
Of course, those conditions were not always present and realistic optimal profits were around 160,000 and 130,000 SeaShells across the two rounds. Still, the magnitude of this hidden edge made Macarons a very lucrative product of the competition.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 9a: Macarons Microstructure</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/9985cdce-a23c-4f89-b288-7709160c1548"
       alt="Macarons microstructure plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows approximately 60% of fills occurring at prices better than the local best bid.  
  The orange indicator represents the external ask after costs (i.e., the conversion price for negative inventory).  
  It also illustrates that straightforward local best bid to external ask arbitrage was not profitable during this period.</em>
</td>
</tr>
</table>


<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 9b: Macarons Microstructure (Normalized)</strong>
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/6822bdc7-1f44-4d43-9df3-289c6e7900a9"
       alt="Normalized Macarons microstructure plot"
       width="100%" />
</td>
</tr>

<tr valign="top">
<td width="100%" align="center">
  <em>This plot shows the same data as Figure 9a, but normalized by the external ask after costs (orange indicator).  
  It clearly demonstrates the achievable price improvement versus the local best bid:  
  while the local best bid was about -1 SeaShell unprofitable in this snippet, fills often occurred at +2 SeaShell profitable levels.</em>
</td>
</tr>
</table>

#### Machine Learning Approach
Last year, there was a similar round involving a sunlight and humidity index. As far as we know, nobody was able to extract any useful information from these indices, and they were largely considered a false lead.
This year, we expected the same outcome, but we still felt it was worth checking, just in case there was something hidden there (especially since an admin in the Discord channel had hinted at it).
Our model was a logistic regression, with a target of a trade being profitable in x timestamps.
##### Features:

| Feature                  | Coefficient | P-value   | Explanation                                                       |
|---------------------------|------------|-----------|-------------------------------------------------------------------|
| sunlight_diff              | -2.0517   | 0.0000    | Change in sunlight over the last 5 timestamps                    |
| sunlight_critical          | 0.4737    | 0.0000    | Binary, if sunlight is below a threshold, set to 45              |
| sunlight_critical_time     | -0.0014   | 0.0096    | Binary, if sunlight has been critical for more than 30 timestamps |
| sunlight_diff_critical     | -0.0020   | 0.0000    | Change in sunlight if sunlight has been critical as defined in sunlight_critical_time |
| sunlight_critical_time_2   | 0.0001    | 0.0020    | sunlight_critical_time^1.3                                        |


All variables produced highly significant p-values and plausible coefficients. These are only a subset of the tested features, and had we employed this strategy, the features would have faced greater scrutiny.
We also tested having a lagged price as a feature, which introduced trading small spikes and mean reversion to our model, but decided against it as this was highly volatile.


We then do the following based on the ouput of the logisitc regession y:
- **If \( y = 0 \)**: **Sell**
- **If \( y = 1 \)**: **Buy**
- **If \( y \) is between 0.49 and 0.51**: **Hold**

The thresholds of 0.49 and 0.51 were set through testing.

<table>
<tr valign="top">
<td width="100%" align="center">
  <strong>Figure 10: Logistic Regression Trades</strong>
</td>
</tr>
<tr valign="top">
<td width="100%" align="center">
  <img src="https://github.com/user-attachments/assets/fa0def84-a066-46c2-95d9-f28b7e56e9dc"
       alt="Logistic Regression Trades"
       width="100%" />
</td>
</tr>
</table>

This approach generated solid historic returns of ~25k per day. However, this approach had multiple issues:
- Although we employed train-test splits, we lacked confidence in the generalization of the model.
- Implementation challenges arose, particularly as longer lags required storing more past data, which significantly slowed down the trader. The serialization of trading data took considerable time, and correctly implementing the logistic regression model introduced numerous potential sources of error.
- Compatibility with export/import arbitrage posed a problem. Since positions can only be converted by reducing them, when the model indicates a long position and we wish to import, we first need to sell our entire inventory plus 10 units.



#### Final Strategy

Our final strategy focused on reliably exploiting this hidden arbitrage.
Each timestep, we placed limit sell orders for Macarons at precisely int(externalBid + 0.5), the maximum price that could still trigger fills from the taker bot.
We quoted only 10 units per timestep (the conversion limit), which meant we captured approximately 60% of the theoretical maximum profits, in line with the taker's acceptance probability.
In hindsight, quoting larger sizes (e.g., 20–30 units) would have allowed us to profitably convert surplus inventory even on non-fills, squeezing out closer to full optimal performance.
Nevertheless, even with conservative sizing, this strategy provided consistent, high-value returns with minimal risk.

Teams who prepared carefully had a clear advantage this round.
Similar hidden taker behavior had already appeared in Orchids during Prosperity 2, and public write-ups from top teams like Jasper and Linear Utility had discussed included it already.
Additionally, even without past experience, attentive teams could have detected the pattern by analyzing historical data: best asks occasionally priced close to best bid consistently getting filled was a clear signal.
Moreover, similar smart-taker behavior had appeared in assets like Rainforest Resin, providing further hints.
Thus, strong preparation, deep intuition about the Prosperity simulation, and diligent empirical observation were all key factors in unlocking the full potential of Macarons.
Although, we only made about 80,000 - 100,000 instead of theoretical optimum of 130,000 and 160,000 those who recognized and optimally exploited the hidden taker bot captured some of the highest single-product profits available in the entire competition. 

<br>

## Round 5: Trader IDs
  
In Round 5, no new products were introduced.
The main change was that historical trader IDs were made public, allowing teams to directly identify which trades were executed by specific bots.
For us, this did not fundamentally alter our strategies, as we had already identified Olivia’s behavior early in the competition.
However, we took this opportunity to update our detection logic: instead of inferring Olivia’s trades indirectly by tracking running minimums and maximums, we now simply checked the trader ID directly.
This adjustment helped eliminate false positives, reduced the risk of missing genuine Olivia trades, and saved a few hundred SeaShells over the course of the round.
As with every previous round, we also re-optimized all relevant parameters based on the latest available data to ensure robustness going into the final evaluation.
This was the last round, and we had a sizeable lead to place 2 (~190k), so we decided to play it save, incase ETF spreads don't converge, by half-hedging the baskets. We also limited our mean reversion strategy to minimze risk.

<br>

## Conclusion

Throughout the entire competition, we managed to consistently stay at the very top of the leaderboard, finishing first place in every round except the final one.
In the last round, the team that ultimately secured first place posted a particularly extraordinary result: a PnL of approximately 850,000 SeaShells, compared to the 200,000–400,000 range that was typical among the other top competitors.
Meanwhile, they had achieved essentially zero PnL in Round 4, suggesting a highly volatile performance profile.
While we cannot fully explain the divergence without access to their strategies, such a sharp swing indicates that some amount of luck could have been involved.
Although it was somewhat unfortunate to lose first place at the very end after leading consistently, we fully accept that in any probabilistic environment, especially competitions with partially stochastic elements, some degree of randomness is inevitable.

Reflecting on our own journey, we attribute our success to a few core principles that were consistently applied across all rounds.
Most importantly, we placed enormous emphasis on deep structural understanding of each product and market environment.
Weeks of careful preparation before the competition allowed us to build strong intuition for market microstructure, option pricing behaviors, and statistical pitfalls that many teams might have overlooked.
Rather than relying on blind optimization or overly complex machine learning models, we focused on simple, robust strategies, critically questioning the validity of every signal and making sure we understood why an edge should exist before committing to it.
Throughout the competition, we maintained a disciplined skepticism toward strategies that appeared to work "in backtest" but lacked theoretical or empirical justification.

Ultimately, we are proud of how we approached the challenge: balancing rigor, adaptability, and humility throughout. We believe that the methodology and principles we applied offer serve as a strong foundation for eventual future competitions and real-world trading alike. 

We genuinely hope that teams of all experience levels could learn something valuable from this algorithmic write-up — whether you're preparing for a future Prosperity competition or simply looking to deepen your understanding of algorithmic trading. Our goal was not just to present strategies, but to share the full depth of our thinking: how we approached each problem, how anyone could have discovered the same insights, and the critical reasoning behind every decision. We tried to explain everything we knew as transparently as possible, in the hope that it can serve both as a strong starting point for new participants and as a thoughtful reference for anyone aiming to push their skills further next year.

PS: How does it feel to get smoked by some business undergrads? 😜

<br>


# Manual Challenge

Our approach to manual trading was more about playing it safe, rather than taking risks. We knew from Prosperity2 that the manual challenges would involve optimization problems, like Currency Arbitrage, but also questions involving game theory, where it would be almost impossible to pick the best options without being very lucky.

## Round 1: FX Arbitrage

Round 1 involved simple Currency Arbitrage. For a given conversion matrix, find 5 trades to make the most money.

|              | Snowballs | Pizza | Silicon Nuggets | SeaShells |
|--------------|-----------|-------|-----------------|-----------|
| Snowballs    | 1.00      | 1.45  | 0.52            | 0.72      |
| Pizza        | 0.70      | 1.00  | 0.31            | 0.48      |
| Silicon Nuggets | 1.95   | 3.10  | 1.00            | 1.49      |
| SeaShells    | 1.34      | 1.98  | 0.64            | 1.00      |

We solved this by iterating over all possible trades, and picking the best option, which ended up being:<br><br>
`SeaShells -> Snowballs`<br>
`Snowballs -> Silicon Nuggets`<br>
`Silicon Nuggets -> Pizza`<br>
`Pizza -> Snowballs`<br>
`Snowballs -> SeaShells`<br><br>
This led to a profit of about 8.9%

## Round 2: Containers

Round 2 was the first round that involved game theory, where you had to pick one or two out of ten squares, and your profit was inversely proportional to the percentage of other participants who picked the same square. Additionally, there was a set number of island inhabitants also choosing containers. Each container contained 10,000 SeaShells, multiplied by a known factor between 10 and 90, which was then split between the teams and the inhabitants, leading to the following payoff formula:

$$\Pi(f) = \frac{M_f \times 10000}{(p_f \times 100) + I_f}$$


With $$M_f$$ being the multiplier, $$p_f$$ the percentage of teams, and $$I_f$$ the number of inhabitants for any field $$f$$. 
We tried to model other teams’ choices by looking at a similar challenge from Prosperity 2. This involved sorting fields by profitability for $$p_f = 0$$. It was assumed that regions with comparable profitability would display similar trends in over- or under-allocation. Applying this predicted the following percentages (by multiplier):
| Multiplier | Predicted % | Actual % | Rank Predicted | Rank Actual |
|------------|-------------|----------|----------------|-------------|
| 37         | 4.95%       | 5.12%    | 1              | 3           |
| 10         | 1.31%       | 0.94%    | 2              | 2           |
| 50         | 8.11%       | 8.52%    | 3              | 5           |


We avoided 37 to steer clear of the bias people have toward the number 37. 10 seemed too risky, as even a slight overallocation would lead to a much lower profit. We therefore settled for 50, which made about ~40k. In hindsight, this lost us 4k–10k compared to 10 or 37, or about 15k compared to the best field.
There was the option to pick a second field for 50k, but that wasn’t an option for us, as our prediction only had two fields that made slightly more than 50k. Assuming everybody only picked one field, the average profit was around ~34k, so we still made a reasonable amount more than the average.


## Round 3: Reserve Price

Round 3 was a combination of optimization and game theory. The first part was pure optimization. There was a uniform group of sellers who would sell their goods for any price higher than their reserve price. The distribution of reserve prices was uniform from 160 to 200. You could set one price, at which sellers would trade if it was higher than their reserve price. The goods bought could be resold for 320 after the challenge, so the profit was as follows:<br><br>
$$\Pi\left(p\right)=N\cdot\left(\frac{p-160}{40}\right)\cdot\left(320-p\right)$$<br><br>
With $$p$$ being you chosen price, and $$N$$ being the number of sellers. The optimum of this function is $$p = 200$$. 
The game theory part, however, was more complicated. The setup was similar, but with reserve prices ranging from 250 to 320. However, if your bid was less than the average bid, your profit would be scaled by:<br><br>
$S(p, \mu) = \left( \frac{320 - \mu}{320 - p} \right)^3$<br><br>
where p is your bid, and µ is the average bid. This led to the following payoff function:<br><br>
$$\Pi(p, \mu) = N \cdot \left(\frac{p - 250}{70} \right) \cdot (320 - p) \cdot \min \left( \left( \frac{320 - \mu}{320 - p} \right)^3, 1 \right)$$<br><br>
The optimum of this function occurs at the average, but the derivative shows that being below the average results in a greater loss than being above it. We incorporated this information into our estimate of what other teams would bid, leading us to bid 303. However, it turned out that most teams did not bid more than the optimum without considering game theory (284), so the average ended up being 287 — way lower than our bid. Luckily, the second part had a lower possible profit compared to the first round, so we only lost about 5.5% compared to the optimal solution. 


## Round 4: Suitcases

Round 4 was identical to the challenge in Round 2, just with a different number of fields and a second choice available for 25k instead of 50k. We employed the same strategy, which led to the following table:
| Multiplier | Predicted % | Actual % | Rank Predicted | Rank Actual |
|------------|-------------|----------|----------------|-------------|
| 60         | 4.25%       | 6.72%    | 1              | 8           |
| 37         | 2.09%       | 4.79%    | 2              | 14          |
| 50         | 2.98%       | 3.92%    | 3              | 6           |


Round 2 had shown that the “bias” for the number 37 did not materialize, which led us to choose 37 and 50. However, in this round, our modeling was way off compared to the actual numbers. We made around ~85k, whereas the optimal two options would have made almost 130k. Assuming two picks per team, the average profit was roughly 82k, so we only had a very slight edge over the average.


## Round 5: News Trading

Round 5 was, again, very similar to Round 5 of Prosperity 1 & 2. We compiled a table of products and estimated reasonable movements, while looking at the magnitude of moves from the previous iterations. Like all manual rounds, we tried to hedge our risk and reduced position sizes compared to the optimal allocations based on our estimates of moves.
| Product          | Expected Movement | Actual Movement | Profit | Optimal Profit |
|------------------|-------------------|-----------------|--------|----------------|
| Haystacks        | 12%               | -0.48%          | -3240  | 0              |
| Ranch Sauce      | 10%               | -0.72%          | -2208  | 0              |
| Cacti Needle     | -30%              | -41.20%         | 32160  | 35360          |
| Solar Panels     | -30%              | -8.90%          | -6600  | 1640           |
| Red Flags        | 5%                | 50.90%          | 9700   | 53970          |
| VR Monocle       | 30%               | 22.40%          | 9600   | 10440          |
| Quantum Coffee   | -50%              | -66.79%         | 87339  | 92932          |
| Moonshine        | 0%                | 3.00%           | 0      | 180            |
| Striped shirts   | 0%                | 0.21%           | 0      | 0              |
| **∑**            |                   |                 | **126751** | **194522**    |


This round involved multiple blunders. First, we had a debate about Ranch Sauce and compromised on 10%; Ranch Sauce ended up remaining unchanged. Haystacks were also overestimated, as there was a similar product in Prosperity 2 where “Sleddit” found new hope in a product, which then went up roughly 12%. Same story for Solar Panels, as they were similar to fishing rods in Prosperity 1. We massively underestimated Red Flags because we wanted to hedge against the possibility that the reserved flags combined with the promise of reprinting would be enough to calm the market down.



<br>

# Frequently Asked Questions

## What is the Wall Mid and why did we use it?

The Wall Mid is our best approximation of the true underlying price of a product during trading.
During testing on the official Prosperity website, it was possible to infer the true underlying price indirectly: by buying or selling a single lot and observing the resulting PnL, which was calculated based on the true internal price rather than market quotes.
Through careful analysis, we found that the most reliable way to estimate this true price was by identifying the bid wall and ask wall — price levels in the orderbook that consistently showed deep liquidity.
These "walls" typically corresponded to quotes from designated market makers who appeared to know the true price and simply quoted rounded versions of it (e.g., ±2 ticks around the true value).
By averaging the prices of the bid wall and ask wall, we obtained a Wall Mid value that was much more stable and accurate than using the raw mid price, which could be heavily distorted by overbidding or undercutting.
Thus, the Wall Mid provided a robust and low-noise estimate of the fair underlying value, crucial for designing effective strategies.


## How to properly backtest? 

A simple rule of thumb we followed:
if a strategy mainly depended on bot interactions, we backtested using the official Prosperity website;
if a strategy mainly involved taking or simple quoting logic, we used Jasper’s open-source backtester.

For fast, early-stage prototyping, we often performed quick, vectorized backtests inside Jupyter notebooks.
This allowed us to rapidly explore ideas before investing time into full-scale simulation.

Jasper’s backtester is very accurate and highly flexible — it allowed us to test across full historical data and easily modify the code to accept additional parameters (which was excellent for systematic optimization).
However, it cannot fully replicate the subtle nuances of bot behavior, simply because that behavior was not fully observable or modeled.
For example, the backtester could not correctly simulate the taker bot behavior for Macarons or properly handle conversions; it also could not accurately model potential fill probabilities after inserting liquidity for assets like Rainforest Resin and Kelp.

Thus, our guideline was:
- For most products (especially beyond Round 1), rely primarily on the backtester.
- For products where detailed bot interaction mattered (Rainforest Resin, Kelp, Macarons), validate key behaviors using website-based tests.

Critically, we strongly advise:
never optimize purely for website score.
Doing so is extremely prone to overfitting on simulation-specific randomness rather than building strategies that generalize.

## How to build the skills and knowledge needed to compete at a high level?

Competing at the top level in algorithmic trading competitions requires building a very strong and well-rounded skill set — much broader than just knowing how to code.
Here’s what worked for us:

Structured learning: Start with a solid technical foundation. Years ago, we took structured courses like the Udemy course on [Algorithmic Trading](https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/) (very often on sale for like 10 bucks or so) to understand the basics of trading, order books, strategy design etc.

YouTube and online resources: We supplemented formal education with countless hours of watching educational YouTube channels (e.g., QuantPy, Khan Academy for Statistics, QuantConnect tutorials) to stay exposed to new ideas and real-world examples.

Learning by doing: Nothing replaces hands-on experience. We regularly participated in competitions like Prosperity, where you are forced to implement, backtest, and optimize under real constraints.

Personal projects: Independent projects helped a lot. We recommend picking a published financial paper (for example, on statistical arbitrage or market making), trying to implement the strategy yourself, and backtesting it on real historical data.
For advanced learners, attempting to implement cutting-edge machine learning models (such as time series transformers for financial prediction) is a fantastic way to bridge pure coding with applied quantitative research.

Critical thinking: Always think critically about the "why" behind strategies — don’t just blindly optimize backtests. Understand market microstructure, incentive structures, and trading behaviors at a deep level.

In short: Master the technical basics first, but real expertise is built by iterating, experimenting, and failing forward in personal projects and competitive environments.

## How to break into quant trading?

Breaking into top-tier quant trading actually requires not too much of technical strength but rather preparation and strategic positioning:

Strong academic background: Attending a prestigious university (especially in fields like mathematics, statistics, computer science, physics, or engineering) definitely helps open doors.
It’s not strictly mandatory — but for highly competitive firms, it strongly increases your chances.

Technical and probability skills:
Firms test heavily on mental math, probability theory, brain teasers, and game theory during interviews.
Sites like tradinginterview.com are excellent resources to practice typical quant interview questions.

Coding ability:
Even "trading" roles often involve coding screens now.
You don’t need to be a LeetCode grandmaster, but you should be comfortable with Python (and optionally C++/Java for dev roles).

Personal projects and competitions:
Having actual projects (like your own trading models or backtesters) or strong placements in competitions (like Prosperity) is a massive differentiator in your resume.

Communication and intuition:
Beyond pure technical skills, firms want people who can explain intuition clearly, think about risk vs. reward critically, and adapt fast under uncertainty.
Practicing mock interviews and explaining your trading strategies or competition approaches out loud is a huge help.

In short: Build strong math and coding foundations, practise probability/game theory interviews extensively, and prove your ability to think like a trader through real-world experience.

## Is the Discord Channel useful?

Yes, definitely.
Being active on the official competition Discord server can be extremely valuable:
tips, hints, and clarifications from moderators are often shared early, and it’s a great place to ask technical questions or discuss strategies with other participants.
Staying on top of Discord discussions is crucial, as important insights — or even leaks of good strategies — sometimes happen informally in chat.

However, caution is essential.
Discord is also full of noise:
there are many inexperienced participants, as well as occasional scammers, so never share sensitive details about your strategy with random people.
Additionally, a form of psychological warfare is common: participants might post seemingly unbelievable backtesting results, either because they have massively overfitted and will likely collapse in live rounds, or because they intentionally fake screenshots to create panic among competitors.
It’s important to stay calm, stick to your own validation methods, and avoid the trap of overfitting or changing your approach impulsively just because of noise on Discord.

In short: Use Discord strategically to stay informed — but maintain a cool head and focus on your own robust development process.

## What was going on with all the hardcoding in the first two rounds?

In the first two rounds, a specific issue existed related to bot behavior replication from earlier competitions. While the underlying true prices were regenerated for Prosperity 3 (unlike Prosperity 2, where some price paths were reused, see [LinearUtility's Writeup](https://github.com/ericcccsliu/imc-prosperity-2/tree/main)), the bot behavior itself was still almost identical to previous years — matching with over 95% accuracy in most cases.

This created an unintended loophole: teams could "hardcode" their trading decisions based on historical bot actions. For example, in Rainforest Resin (similar to last year's Amethysts), if a bot historically submitted a market order at timestep 600, teams could hardcode their own algorithm to preemptively take all ask liquidity just before that bot arrived — virtually guaranteeing fills at better prices.

We discovered this early and initially implemented a frontrunning strategy ourselves, given that a similar situation had gone unaddressed in Prosperity 2 (see [LinearUtility's Writeup](https://github.com/ericcccsliu/imc-prosperity-2/tree/main)).  
However, unlike some teams, we also built a fallback system: if the bots' behavior changed or was fixed mid-competition, our algorithm would automatically revert to a normal, non-hardcoded trading strategy. This safety measure came at a cost — the fallback logic introduced exploration costs and slight performance degradation compared to default algorithm.

After Round 2, we reported the hardcoding exploit to IMC directly. In response, IMC officially fixed the bot behavior to prevent hardcoding from Round 3 onward and banned such tactics for the rest of the competition. Additionally, they allowed teams to resubmit corrected algorithms and reran the first two rounds to ensure fairness. Interestingly, one of the teams that ultimately overtook us in the final standings had also been relying on hardcoding early on — and without the resubmission opportunity we advocated for, they might not have been able to catch us by the end.

While this turned out to be a funny and slightly ironic anecdote, we're proud that we pushed for a fairer competition, even at a potential cost to ourselves.

## What else did we try?

At one point, we attempted a more unconventional idea:
we extracted the first 100 or so returns from a true PnL series and tried to reverse-engineer a possible random seed that might have been used to generate it.
Assuming the underlying data could have been produced by a pseudorandom number generator (with minor transformations),
we compared the observed returns to randomly generated sequences from all 4 billion possible seeds in NumPy, running the search on a Raspberry Pi for about 24 hours.
Unfortunately, no matching seed was found — but it was a fun experiment nonetheless!






