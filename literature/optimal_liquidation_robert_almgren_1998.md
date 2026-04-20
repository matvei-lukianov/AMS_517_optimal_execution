# Optimal Liquidation
Robert Almgren*and Neil Chrisst
January 14,1998
## Abstract
We consider the problem of portfolioliquidation with the aim of minimizing a combination ofvolatilityrisk and transaction costs arising from permanent and temporary market impact. For a simplelinear costmodel,we explicitly construct the efficientfrontier in the space of time-dependent liquidation strategies, which have minimum expected cost for a given level of uncertainty. We consider the risk-reward tradeoff both from the point of view of classic mean-variance optimization,andfrom the standpoint of Value at Risk. This analysis leads to general insights into optimal portfolio trading, and to several applications including a definition of liquidity-adjustedvalueatrisk.
*The University of Chicago, Department of Mathematics,5734 S. University Ave., ChicagoIL60637;almgren@math.uchicago.edu +Morgan Stanley Dean Witter and Courant Institute of MathematicalSciences; neilc@cims.nyu.edu

Contents
## 1 Trading Model
## 1.1 Trading strategy.
## 1.2A model for stock price movements
## 1.3 Permanent market impact .
## 1.4 Temporary market impact .
## 1.5Capture and cost of trading strategies
## 2TheEfficientFrontier and OptimalTrading
## 2.1The definition of the frontier.
## 2.2 Explicit construction of optimal strategies .
## 2.3 Structure of the frontier..
## 3TheRisk/RewardTradeoff
## 3.1 Utility function
## 3.2Value at Risk
## 4NumericalExamples
## 4.1 Choice of model parameters .
## 4.2 Effect of parameters .
## 5Multiple-StockPortfolios
## 5.1 Trading model
## 5.2 Optimal trajectories ..
## 5.3 Explicit solution for diagonal model
## 5.4Example
## 6Applications
## 6.1 Application 1: Liquidity-Adjusted Value at Risk. . :
## 6.1.1 Definition of L-VaR.
## 6.1.2 Examples
## 6.2Application 2:Performance Benchmarks.
## 7Conclusions

Broker dealers are frequently faced with the task of buying or selling largeblocks of a single stock or largebaskets of multiple stocks.For example,program trading desks perform transition trades for pension funds facilitating the move from one manager's positions to another's.
performance against a suitablebenchmark. Thus broker dealersface the problem of minimizing transaction costs of large trades that take place over fixed time periods, while buy side participants face the problem of devisingreasonableperformancebenchmarksfortheseservices. These problems,two sidesof the samecoin,have anunderlying structure given by theinteraction between twoingredients:
·Market impact.If a large tradeis executed toorapidly,costs will be incurred as the trades move the market in an adverse direction.
·Volatilityrisk.Onthe otherhand,if the tradeis executed tooslowly, then theposition is subject toriskduring thetime that theshares remain in the portfolio.
These two quantities must be played off against each other by taking account of the desiredperformance characteristics of thevarious participants. In this paper we define the notion of an efficient or optimal liquidation strategy for a basket of securities, and study the notion under a simple model of price movements. The main results of our papers are the following:
·A method for determining the optimal method for liquidating a basket of securities.
·A method for evaluating performanceby computing the expected value of atradingstrategy and itsstandard deviation.
·The introduction of an“efficient frontier”of optimal trading strategies,each offering a different risk/reward tradeoff.
·General insightsintotherelationbetweenutilityfunction andportfoliotradingindependent of specificfeatures of the stockprice movementsandtransaction costmodel.
In addition, we offer insight into the question of the magnitude of thetransactioncosts associatedwithliquidationunderextrememarket conditions. This has several interesting applications to risk management,

including offering a simple definition of what we call L-VaR (liquidityadjustedvalueatrisk). The centralpoint of our discussion is the observationthat the cost of trading—the differencebetween theinitial marketvalue and thevalue realized after liquidation—is a random variable,whosemean and variance at theinitial time depend on the liquidation strategy tobe followed.
strategies and define the concept of a risk/reward tradeoff for trading strategies: reward is low transaction costs, and risk is the level of variability of transaction costs. This is not a paper about transaction cost models. We do not propose to give a methodforparametrizing thetransaction cost modelsstated in the subsequent sections. Rather, we give a complete analysis of the con-
of portfolio trading.Infact,the entire analysis carried out here can be redone for almost any transaction cost function one can write down, and thebasic conclusions will remain (see the Appendix). Theapproachwehavetakentowardtheproblemofoptimalliquidationis new.The problem of optimal tradinghas been clearlyformulated by Perold (1988).Previous analytical work has typicallyfocused on a single point along our efficient frontier; see, for example, Bertsimas and Lo (1998) and Subramanian (1997b,1997a). Empirical studies of market impact have been carried out by Kraus and Stoll (1972), Holthausen, Leftwich, and Mayers (1987),and Chan and Lakonishok (1993,1995). Vayanos (1997) has considered a model of market microstructure and itsconsequencesfor optimal tradingstrategies.
## 1 Trading Model
Belowwegive aformal definition of atradingstrategyfortheliquidation of a single stock or other risky asset. Our definition is in terms of selling a (presumably)largeblock of stock,but is equally relevant to other assets such as currencies orbonds.In fact,the method generalizes to any trade (buy or sell) in which both transaction costs and volatility are anissue.We present this in a discrete-time framework, but the extension to continoustime is immediate.

## 1.1Trading strategy
Suppose that we hold a block of X shares of a single stock or other risky asset, that we want to completely liquidate by time T.We divide T into N intervals of length T,and define the discrete times tk =kT,for k = O,...,N. We define a trading strategy to be the list xo,...,xn, where xk is the number of shares that we will hold at time tk. Our initial holding is xo = X, and liquidation at time T requires xv = 0. We may also specify our strategy by giving the "trade list" n1,..., NN, where nk = Xk-1 -xk is the number of shares that we sell between times tk-1 and tk.Clearly,xkand nk arerelated by k
N xk=X-∑nj = ∑ xj， k=O,...,N. j=1 j=k+1
We consider more general programs of simultaneously buying and selling several securities in Section 5. For notational simplicity, we have taken all the time intervals to be of equal length, but this restriction is not essential. Since the length T of the time interval is a rather arbitrary quantity, it will be more convenient to work with the instantaneous rate of trading,in shares per unit time; this will also make it easier to pass to the continuous-time limit. During the interval tk-1 to tk, this rate is
nk Vk= (xk-1-xk)，k = l,...,N.
T Note that each nk or vk is a “backwards" difference. Thus xk and vk represent the same piece of information: our choice at time tk-1 of how many shares we wish to hold at time tk. We shall take the trading strategytobe deterministic,even in thepresence of random uncertainty in market events during the liquidation. That is,weimagine that wedetermine the entire strategy atthebeginning of trading, and we evaluate the risk and reward of this strategy without permitting the strategy to evolve in time. Although this may appear to be a rather extremehypothesis,we shall justify it more fullybelow.
## 1.2Amodelforstockpricemovements
Supposethattheinitialstockpriceis So,so that theinitial marketvalue of our position is XSo. The stock's price evolves according to three influences: volatility, drift, and market impact. Volatility and drift are

assumed to be the result of marketforces that occur randomly and independently of our trading. Market impact is a direct result of our trading.Asmarket participantsbegin todetect thevolume we are selling (buying) they naturally adjusts their bids (offers)downward (upward). Later we will discuss two distinct types of market impact: temporary and permanent. Our discussions largely reflect the work of Kraus and Stoll (1972), and the subsequent work of Holthausen, Leftwich and Mayers (1987,1990), and Chan and Lakonishok (1993,1995).
ply in demand caused by our trading. Such transaction costs are one-time costs applicable to single trades.Permanent costs,on the other hand,refer to a shift in the equilibrium price of the stock under consideration due to trading, which last at least for the life of our liquidation. We assume that the stock price evolves according to the discrete randomwalk ()-++= k k (1) = So + α1/2j +μtk -g(vj), j=1 j=1 for k = 1,...,N. Here 0 represents the volatility of the asset, μ is an expected growth rate,the j are draws from independent random variables each with zero mean and unit variance, and g(v) is a function of the rate of trading v.We assume that the proceeds from any sales are invested insome alternativeinvestment thatreceives theriskless rate of return.In such a case we take μ to be the excess return over the return onthe alternativeinvestment. In the absence of market impact,our model says that Sk executes a arithmetic Brownian random walk, so that at time tk its mean is So + μtk and its variance is o?tk.
Arithmeticvs.geometric Brownian motionThe random walk above would be a standard geometric Brownian motion with constant drift μ andvolatilityo,if wedefinedμ andosuch that
μ=Sμ，O=S,
andkeptp andoconstantasSevolved.Becauseweareinterested in short-term“trading"horizons rather thanlonger-term"investment"horizons, the total changes in S will be small, and we may assume that μ and O are constantrather thanp and o.

That is,the differencebetween arithmetic and geometric Brownian
model is analytically simpler.Note that our μ and o must be divided by some referencepriceS in order togive rates of return in the standard sense.As a practical matter, to determine μ and o,start with the standard fractional volatility and drift as parametrized in an ordinary Brownian motion and multiply by the“reference price,”the price of the security under liquidation at the start of liquidation.We give an example in Section 4.
## 1.3Permanentmarketimpact
The function g(v)above represents the permanent impact(of trading)on the price of the stock. Here permanent refers to the life of the liquidation at hand. Thespecificform of thetransaction costfunctionisembodied in the following assumption: as long as we are actively trading the stock, market participants will not bid in substantial volumes at the equilibrium priceof thestock. Fornow,weshallassume that theimpactfunctionis linearin ourrate of trading. That is, the other participants will bid low (for a sell program)
trading. The mathematical formulation of this assumption is that g(v) has the form
g(u) = yv. (2)
response to our selling at a rate of v shares per unit time. The constant y has units of ($/share)/share. Thus each n shares that we sell depresses the price per share by yn, regardless of the time we take to sell the n shares. Substituting linear permanent impact (2) in equation (1) we obtain the price equation:
Sk= Sk-1 + OT1/2sk+ μT -ynk k (3) = So + o∑t1/2sj + μtk - y(X -xk). j=1

In the absence of fluctuation and drift,the stock price is a linearfunction of our instantaneous holdings. This is consistent with the partial equilibrium approach of Jarrow (1992), though he considers more general nonlinear dependence on our portfolioholding.
## 1.4Temporary market impact
Weimagine thetraderplans tosell a certain number of sharesnkbetween times tk-1 and tk, but may place the order in several smaller slices. If the total number of shares nk is sufficiently large, the execution price may steadily decreasebetween tk-1 and tk,partially due tousingup the orders at the bidfaster than new orders arrive.Thus the trader will suffer some temporary costs that will be offset in the next period when new orders have arrived. We model this effect byintroducing a temporary priceimpact function h(v), which is the temporary drop in price per share caused by trading at rate v. Then the actual price per share received on sale k is
Sk=Sk-1-h(vk),
but the effect of h(v) does not appear in the next “market" price Sk. For now, we consider only the linear model
h(v)=E + nv. (4)
The units of e are $/share, and those of n are (S/share)/(share/time). A reasonable estimatefor is thefixed costs of selling,such ashalf the bidask spread plus fees. It is more difficult to estimate n since it depends on internal and transient aspects of the market microstructure.It is in this term that we would expect nonlinear effects to be most important, and the approximation(4) tobe most doubtful.We consider general cost functions in the Appendix. The linearmodel for transaction costs (4)is often called a quadratic cost model because the total cost incurred by trading n shares in a single unit of time is
nh(2)= en+ n2.
That is, the drop in price per share sold is linear in the number of shares sold, so the total cost is quadratic in n.

With both linear cost models (2,4),the actual average price we receive on the sale between tk-1 and tkis
Sk= Sk-1 -∈-nuk k-1 (s） --(I-x-x) -- ++ = j=1
## 1.5Capture and cost of tradingstrategies
We define the capture of a strategy tobe the value we receive upon liquidating the entire sell program. This is the sum of the product of the number of shares nk = Tvk that we sell in each time interval times the effectivepricepershareSkreceived on thatsale. Write XS for the capture of a strategy, so that Sis the average price k-1 per share received. Using the interchange of sums formula Z=1 Z-1
-1N Zj=1 Ek=j+1,we readily compute
## N
## N
N xS=nKSk =xSo+1/2xkk +μTxk k=1 k=1 k=1
## N
N y∑TxkUk-∈X-n∑Tv².(6) k=1 k=1
The first term XSoon the right of (6)is theinitial market value of our position; each additional term represents a gain or a loss during the liquidation,and eachhas a simpleinterpretation.Atstepkwehold xkshares of stock. Each effect that moves the price by 8Sk at time tk changes the market value of our position by xk &Sk,and the total change is≥xk Sk. The first term of this type is ≥ oTl/2skxk, representing the total effect of volatility. The second is ∑uTxk, representing the expected total
sents the loss in value of our total position, caused by the permanent price drop associated with selling a small piece of the position. Of the two terms corresponding to the instantaneous cost,the first, -eX,is the accumulation of the fixed costs,which depend simply on the total amount we sell. The final term, - ∑ ntv2, represents the cost of market impact on each individual trade.As observed above,the total cost incurred by a linear price impact function is quadratic in the number of shares traded.

Using the summationby partsformula
N ∑TXkUk= ∑xk(xk-1-xk）= k=1 k=1
## N
A ∑(x²-1-x-(xk-xk-1)²）=x²-∑ k=1 k=1 we may put (6) into the simpler form
## N
N xS =xSo + 1/2xkk + μtxk k=1 k=1
N -yx²-∈X-(n-T)tv². (7) k=1
Surprisingly, except for a term of size O(t) in the last sum, the contribu-
X.Thatis,theeffect of linearpermanentimpact maybevaluedindependentlyoftheliquidationpath. The total cost of trading is the difference XSo -xS between the initial book value and the capture. This is the standard ex-post measure of transaction costs used inperformance evaluations,and is essentially whatPerold(1988)callsimplementationshortfall. Under our assumptions, this cost is a random variable whose expected value and variance,measured at the initial time, depend on the trading strategy x=(x1,...,xn）used to liquidate the position.Write E(x)and V(x)for respectively the expected value and variance of the total cost of trading strategy x.We will informally call E(x)the expected value of the strategy andV(x)thevariance of thestrategy. From(7)we compute
## N
N (-u)+x+x+x=(x) (8) k=1 k=1
The units of E are $; the units of V are $2. To illustrate, let us compute these quantities for the two most extreme strategies: sell at a constant rate, and sell to minimize variance without regard to transaction costs.

Minimum impactThemostobviousstrategyistosell at a constantrate over the whole liquidation period. Thus, we take each
X andxk= (1-)x, k = O.,...,N.
## T
From (8) we readily compute
X2 E=-uTx(1-)+yx²(1-)+ex+n
and from (9),
V=o²x²T(1-
This strategy minimizes market impact costs; if μ is negligible then it minimizes overall expected costs.E and V have finite limits as the number of trading periods N→ ∞.
Minimum varianceThe other extreme is to sell our entire position in thefirst time step.Wethen take
## X
which give
This strategy has the smallest possible variance;by the way that we have discretized time in our model, this minimum is zero. Its expected costs are extremely high if N is large. That is,as we increase the number of time steps, we also increaes the rate of trading within the first step and hence the overall costs.
To summarize,E(x)and V(x) are the expectation and variance of the final cost of trading, assuming that the strategy x = (xo,..,Xv) chosen at thebeginningis adhered to throughout liquidation.Theparticular realizationof thecostdependsontherealizationofthepriceof the stock at each step of the strategy. As these prices are at least partially random, the entire cost of trading is random.

Thedistributionof thenetcostis exactlyGaussian if thekareGaussian;in any case if N is large it is very nearly Gaussian.Since a Gaussian distributionis completely describedbyits mean and variance,the two quantities E(x）and V(x）contain all the information we need in order to evaluate different liquidation strategies. We shall need to show along
strategyxtomarketmovements.
## 2The Efficient Frontier and Optimal Trading
Equations (8,9) indicate that the choice of a particular trading strategy determinesboth the expected cost and thelevel of uncertainty attached to that strategy. In general, one can decrease the variance of the costs only by increasing the expected level of the same costs, and conversely. For example,if we“dump”our entire block into the market at one time (as we examined above), we are exposed to very little volatility risk, but we incur large transaction costs. On the other hand,if we sell slowly,we
liquidationprocess.
## 2.1The definition of the frontier
With this in mind, we examine the proper definition of an"optimal" trad-
definition. First observe that a rational trader will always seek to minimize the expectation ofcostforagivenlevelofvariance.Thusin analogywith modernportfoliotheorywedefineatradingstrategytobeefficientor optimal if there is no strategy which has lower variance for the same level of expected transaction costs, or, equivalently, no strategy which has a lowerlevel of expected transaction costsforthe samelevelvariance.
efficientstrategiesbysolvingtheconstrained optimizationproblem
min.E(x).
That is, for a given level of variance V*, we find a strategy that has minimum expected level of transaction costs. This minimum exists uniquely whenever E and V are convexfunctions; this is the case for thelinear

models above, and for more general transaction cost models as we consider in the Appendix. Write x*(V*）for a solution to (14).Regardless of our preferred balance of risk and return, every other solution x which has V(x)=V has higher expected costs than x*(V*）for the same variance,and can never be efficient. Thus,the family of all possible efficient (optimal) strategies is parameterized by the single variable V*, representing all possible levels of variance in transaction costs.We call this family the efficient frontierofoptimaltradingstrategies. We solve the constrained optimization problem (14)by introducing a Lagrange multiplier 入.For a given value of 入,we solve the unconstrained problem
min(E(x)+AV(x))
and call the solution x*(A).As 入 varies,x*sweeps out the same oneparameter family as for the constrained optimization problem, and thus traces out the efficient frontier. TheparameterXhas adirect financialinterpretation.It is already apparent from (15) that 入 is a measure of our risk-intolerance, that is, how much we penalize variance relative to expected cost. In fact,入 is the curvature (second derivative) of a smooth utility function, as we shall make precise in Section 3.
## 2.2Explicit construction of optimal strategies
With E(x)and V(x)from (8,9),the combination U(x)=E(x)+AV(x) is a convex quadratic function of the control parameters x1,..., xN-1. Therefore it has a unique global minimum, at a value determined by settingitspartialderivativeswithrespecttothecontrolvariablestozero. Wereadily calculate
au + xk-1-2xk+xk+1 2nT dxk 2n
for k = 1,...,N - 1. Then oU/xk = 0 is equivalent to
2(xk-1-2xk+xk+1）=R²(xk-x),

with
入o2
and
x=
The quantity x is the optimallevel of stock holding for a time-independent portfolio optimization problem. The solution to the linear difference equation (16) may be written as x plus a combination of the exponentials exp(±ktk),where K satisfies
The specific solution satisfying the boundary conditions xo = X and XN = 0 is
sinh(ktk) 文. xk=x+
(18) sinh(kT) sinh(kT)
for k = O,...,N. This solution depends on the input parameters only through the combinationsX and k. Wereadilycalculatetheassociatedvelocity
sinh()[ Vk= [cosh(k(T-tk-))(x-x)+cosh(ktk-)x] sinh(kT)
0 < (-) = " = long as O<x ≤ X.Thus the solution decreases monotonically from its initial value to zero; an optimal strategy never tells us to buy as part of a liquidation program. For smallTwehave the approximateexpression
入g2
## K~K+O（T²）~
Thus if our trading intervals are short, k² is essentially the ratio of the product of volatility and our risk-intolerance to the temporary transaction cost parameter.

The behavior expressed by the difference equation (16) and the solutions (18)is relaxation towards the static optimum x at a rate k,subject
1/kis the time required for thesolution tomove afactor ofetowards or awayfrom x.We characterize solutions by comparing this time to the total time for liquidation, that is, by examining the ratio T/(1/k)= kT for fixed x. If kT 》 1, then either temporary costs are very small, volatility is extremely large, or we are very sensitive to volatility. Our strategy is dominated by the need to reduce volatility risk. We initially sell very rapidlyfrom ourinitial position downto the optimumlevelx;wewait at xfor most of the time, and near theend sell rapidly to achieve liquidation x=0 att=T.
volatility is small,or we are risk-indifferent.Our strategy is dominated by the need to minimize market impact costs. In the limit kT → O, we approach the straight line strategy.
lutions x*(A).It is possible to evaluate E(X) = E(x*(X)) and V(X) = V(x* (X)) in closed forms, since they are just sums of exponentials, but the resulting expressions are very complicated and not very enlightening. It is, however, trivial to evaluate the sums numerically.
Risk-neutral strategy In the risk-neutral limit 入 → O we have
-tk(T-tk)，
which is the linear strategy (1o) plus a small quadratic correction proportional to the expected return μ. This is the strategy we call “naive" below.
## 2.3Structure ofthefrontier
An example of the efficient frontier is shown inFigure 1.The plot was produced using parameters chosen as in Section 4. Each point of the
basket. The tangent line indicates the optimal solutionfor risk parameter X = 10-6. The trajectories corresponding to the indicated points on the frontier are shown in Figure 2.

Expected loss E[x] ($)
Variance V[x] ($²) x1012
to variance.The straight line illustrates selection of a specific optimal strategy for 入 = 10-6. Points A,B,C are strategies illustrated in Figure 2.
Trajectory A has 入 = 2 × 10-6; it would be chosen by a risk-averse trader who wishes to sell quickly to reduce exposure to volatility risk, despite the trading costs incurred in doing so. TrajectoryB has 入=O.We call this the naive strategy,since it repre-
transactioncosts withoutregardtovariance.For a stockwith zerodrift andlinear transaction costs as defined above,it corresponds to a simple linear reduction of holdings over theliquidation period.Since drift is generally not significant over short trading horizons, the naive strategy is very close to the linear strategy, as in Figure 2. We demonstrate below thatthis is never anoptimal strategy,because one can obtain substantial
Trajectory C has 入 = -2 × 10-7; it would be chosen only by a trader who likes risk. He postpones selling, thus incurring both higher expected trading costs due to his rapid sales at the end, and higher variance during

## B
ShareHoldings
TimePeriods
theextended period that heholds the stock.
Generalconclusions abouttradingThestructureof theefficientfrontier lends general insight into thenature of trading large baskets,based ontheobservationthat thecurvedefiningtheefficientfrontierisasmooth convex function E(V） mapping levels of variance to the corresponding minimum mean transaction cost levels. Let us denote by (Eo,Vo) the point corresponding to the naive strategy,that is,the global minimum of E.By smoothness,wehave dE/dV = 0 there For (E,V) near (Eo,Vo),we have
d²E E-Eo≈
(%-A) zAp Iv=Vo
where d²E/dV2|v。 is positive by convexity.Any reduction in the level of uncertainty of transaction costs comes at the price of a general increase in the level of costs,but at the naive strategy, a first-order decrease in variance can be obtained for only a second-order increase in costs. Thus, unless you are absolutely risk neutral, it is always advantageous to trade “to the left" of the naive strategy, or
A risk-averse tradershould never usethenaivestrategy.

## 3The Risk/Reward Tradeoff
We now consider how to choose among the various efficient strategies the onetoexecute.This amounts tofinding a way to convert a dollar of expected transaction cost into a unit of variance and vice-versa.We do
theory employinga utilityfunction,orby a novel approach:value-at-risk.
## 3.1Utility function
wis our total wealth.This function may be characterized by its riskaversion coefficient 入u =-u'(w)/u'(w).As we transfer our assets from the risky stock into the alternative investment, w remains roughly constant, and thus we may take Au to be constant throughout our liquidation period. For short time horizons and small changes in w,higher derivatives of u(w） may be neglected. Thus choosing an optimal liquidation strategy is equivalent tominimizing the scalar function
Uutil(x) = AuV(x) + E(x).
The units of Au are $-1: we are willing to accept an extra square dollar ofvarianceifit reduces our expected costbyAudollars. ThecombinationE+AVispreciselythe oneweused toconstructthe efficientfrontierinSection 2; theparameter 入,introduced artificiallyin as aLagrangemultiplier,has a precise definition as a measure of our aversion to risk.Thus theconstruction used above to construct the efficient frontier also gives us the optimal point for any given utility function. Now we can consider whether we may gain by adapting the strategy to marketevents astheyunfold.Supposethatwestophalfwaythroughthe liquidation toreconsiderourstrategyfor therestof theprogram.Clearly the variance of the remaining part will be smaller than at the beginning, since some of what might have happend has alreadyhappened.And the expectation of cost may be larger or smaller than its starting value, depending on how the price has moved. However, the optimal strategy for the remaining time is just the same as the initial optimal strategy, as maybeseenbyexaminingtheexactsolution. Theexact solutionis controlledby theparametersK andx.These depend on the properties of the stock random walk and on our riskaversion factor 入,neither of whichchanges as theliquidation proceeds.

the number of periods remaining. Since the solution to a second-order difference equation is uniquely determined by its twoboundary values, the solution for the remaining time is the same as the remaining part of theoverallsolution. Notethatthis argumentdoesnotconsiderwhether ourestimationof theproblemparametershaschanged.If ahighlyunlikelyeventsuddenly occurs,such as the stock price dropping 10%,then we may well wish to raise our estimate of the volatility, or we may wish to increase our riskaversion 入.Butbarring such events,our estimations offuture motions are not affected by stock movements. This argument also does not apply if price movements are serially correlated,as in Bertsimas and Lo (1998).
## 3.2ValueatRisk
Theconceptofvalue atriskis traditionallyusedtomeasure thegreatest amount of money(maximum profit andloss)a portfolio will sustain over
definedby a confidencelevel. Given a trading strategy x = (x1,...,xn), we define the value at risk of x, Varp(x), to be the level of transaction costs incurred by trading strategy x that will not be exceeded p percent of the time.Put another way,it is the p-th percentilelevel of transaction costs for the total cost of trading x. Under the arithmetic Brownian motion assumption, total costs (market value minus capture) are normally distributed with known mean and variance.Thus the confidenceinterval is determined by the number of
distribution function, and the value-at-risk for the strategy x is given by the formula:
Varp(x) = Av√V(x) + E(x);.
That is,with probabilitypthe trading strategywill not lose more than Varp(x) of its market value in trading. Borrowing from the language of Perold(1988), the implementation shortfall of the liquidation will not exceed Varp(x) more than a fraction p of the time. From this point of view or optimal (or efficient) liquidation, a strategy x is efficient if it has the minimum possible value at risk for the confidence level p.

Expected loss E[x] ($)
Std dev sqrt(V[x]) ($) ×105
Note that Varp(x) is a complicated nonlinear function of the xj composing x: we can easily evaluate it for any given trajectory, but finding the minimizing trajectory directly is difficult. But once we have the oneparameter family of solutions which form the efficient frontier, we need only solve a one-dimensional problem to find the optimal solutions for the value-at-risk model, that is, to find the value of Au corresponding to a given value of Av.Alternatively, we may characterize the solutions by a simple graphical procedure,or we may read off the confidence level corresponding to any particular point on the curve. Figure 3 shows the same curve as Figure 1, except that the x-axis is the square root of variance rather than variance. In this coordinate system, lines of optimal VaR have constant slope, and for a given value of Av, we simplyfind the tangent to the curve where the slope is 入y. In the discrete-time model, the efficient frontier intersects the V = 0 axis at a finite height given by (13). In the plane of √V and E,its slope there is finite,and equal to
入max gT3/2(2nX - yXT + μT²).
If the confidence level p is large enough that the risk-intolerance param-

eter A is larger than Xmax, then the optimal strategy is is the minimum-
behavior is clearly an artifact of our discrete-time model. Now the question of reevaluation ismore complicated and subtle.If we reevaluate our strategy halfway through the liquidation process,we
original optimal one.The reason is thatwe nowhold 入vconstant,and so 入u necessarily changes. This is a well-known defect of the value-atrisk approach,as recognizedbyArtzner et al (1997a,1997b).eregard it as an open problem to formulate suitable measures of risk for timedependentstrategies.
## 4NumericalExamples
In this section we compute some numerical examples for the purpose of exploring the qualitative properties of the efficient frontier. Throughout theexamples wewill assumewehave a single stock with current market price So = 50, and that we initially hold one million shares. Moreover, the stock will have 30% annual volatility, a 10% expected annual return of return, a bid-ask spread of 1/8 and a median daily tradingvolume of 5 million shares. With a trading year of 250 days, this gives daily volatility of 0.3//250 = 0.019 and expected fractional return of 0.1/250 = 4 × 10-4.To obtain our absolute parameters O and μ we must scale by the price, so 0 = 0.019·50 =0.95 and μ =(4 ×10-4)·50 =0.02.Table1 summarizes this information.
T = 5 (days). We divide this into daily trades, so T is one day and N = 5.
## 4.1Choiceofmodelparameters
We now choose parametersfor the temporary cost function (4)
h(v)=∈+nv.
We choose ε = 1/16, that is, the fixed part of the temporary costs will be one-half the bid-ask spread. For n we will suppose that for each one
bid-ask spread. For example, trading at a rate of 5% of the daily volume

:So=50$/share Initial stock price: Initial holdings: 106share Liquidation time:
T 5 days Number of time periods:
N 30% annual volatility: b 0.95 (S/share)/day1/2 10% annual growth: μ = 0.02($/share)/day Bid-ask spread =1/8: 0.0625 $/share Dailyvolume5million shares: y = 2.5 × 10-7 $/share2 Impact at 1% of market: n = 2.5 × 10-6 ($/share)/(share/day)
10-6/S Static holdings 11,000 shares: 入u VaR confidence p =95%:入y
Table1:Parameter values for our test case.
incurs a one-time cost on eachtrade of 5/8.Under this assumption we have n = (1/8)/(0.01 · 5× 106) = 2.5 ×10-6. For the permanent costs, a common rule of thumb is that price effects become significant when we sell 10% of the daily volume. If we suppose that“significant"means that the price depression is one bid-ask spread, and that the effect is linear for smaller and larger trading rates,then we have y = (1/8)/(0.1 · 5 × 106) = 2.5 × 10-7. Recall that this parameter gives a fixed cost independent of path. In Figure 1 we have chosen X = Au = 10-6. We may interpret this numberin terms of thenumber of sharesxthat we arecomfortableholding; this indicates how much diversification we require in our portfolio, if our initial holdings were our total worth. Then (17) gives approximately x = 1,100 shares, or 0.11% of our initial portfolio. We expect this fraction tobe very small since our optimal strategy drives us towards complete liquidation. For these parameters, we have from (19) that for the optimal strategy, K 0.6/day,so kT ~ 3.Since this value is near one in magnitude, the behavior is an interesting intermediate in between the naive extremes. For the value-at-risk representation, we assume a 95% desired confidence level, giving 入v = 1.645.

Min. VaR (L-VaR) Naive strategy StaticVaR
E VaR
E VaR √v
## 4.2Effect of parameters
TemporarycostfunctionThemostimportantparameterin determining the path is n, the velocity-dependent part of the temporary cost function.Wehaveselected a certainpercentage of thedailyvolume,atwhich the temporary cost is equal to the bid-ask spread; above we selected that level to be 1%.Increasing this percentage level has roughly the same effect onthe shape of theefficient frontier asdoesreducingthedaily volume. That is,the smaller the value of this percentage, the more the price is sensitive to our trading; and,loosely speaking, theless liquid it is. InFigure 4we illustrate the effect of changing thispercentagefrom 0.0025% to 2%,while keeping the desired value-at-risk parameter constant at p = 95%. As this percentage increases, or, loosely speaking, as liquidity increases, the optimal trajectory shifts away from the naive strategy in the direction of instantliquidation.The expected cost decreases asmarketimpactisreduced. Table 2 shows the variance, expected costs, and VaR for the minimum-
TimetoliquidationInFigure5weshowtheeffectofchangingthetime allowed forliquidationbetween1 and10days.Wekeepthenumber of

Expected loss E[x] ($) Expected loss E[x] ($) p=0.95
impact at 0.25% impact at 0.5%
Std dev sqrt(V[x]) ($)x 105 Std dev sqrt(V[x) ($)× 105
Expected loss E[x] ($) Expected loss E[x] ($)
impact at 1% impact at 2%
Std dev sqrt(V[x]) ($)× 105 Stddevsqrt(V[x])($)x105
a more liquid stock: trading at a rate of 2% of the median daily volume results in a temporary cost of the bid-ask spread.Note the two-fold effect
down from over 2 million dollars (6%) to just under 1 million dollars (2%). In addition, the “distance" of the 95% confidence level strategy to the naive strategy increases: the marginal importance of not trading the naive strategy increases as the liquidity of the position increases.

Expected loss E[x] ($) =0.95 Expected loss E[x] ($)
Std dev sqrt(V[x) ($)x 105 Std dev sqrt(V[x]) ($)x 105
xected loss E[x] ($) Expected loss E[x] ($)
Std dev sqrt(V[x]) ($)x 105 Std dev sqrt(V[x) ($)x 105

periods constant at 5, so as the total time is increased, the length of each
shifts towards complete liquidation in the first period. The numbers are shown in Table 2.
## 5Multiple-StockPortfolios
With m stocks, our position at each moment is a column vector xk = (x1k,...,Xmk)T,where I denotes transpose. The initial value xo = X = (X1,...,Xm)T, and our rate of selling is the column vector vk = (xk-1 xk)/T. If xjk < O, then stock j is held short at time tk; if vjk < O then we are selling stock jbetween tk-1 and tk.
## 5.1Trading model
Weassume that the column vector of stockprices Skfollows a multidimensional arithmeticBrownian randomwalk.Its dynamics is again written as in (1), but now Sk = (Fik,...,rk)T is a vector of r independent Brownian increments,with r ≤ m, o is an m × r volatility matrix, and μ = (μ1,.,μm)T is the vector of expected growth rates. C = ooT is the m X m symmetric positive definite variance-covariance matrix. Thegeometricmodelwouldletμ andCdepend on thetimestepk as
Hjk = SjkPj,Cijk = SikSjk Cij,
where μ and C, rather than μ and C, stay constant as S evolves. As for
can be neglected, but we must not forget to insert the factors of S to compute μ and C in terms of fractional rates of return. The permanent impact g(v)and the temporary impacth(v)are vector functions of a vector. For now, we consider only the linear model
g(v)=Tv，h(v)=∈+Hv,
where I and H are m× m matrices,and e is an m x1 column vector. Theij element of I and of Hrepresents theprice depression on stock i caused by selling stock j at a unit rate. We require that H be positive definite,since if there were a nonzero v with vTHv≤ O,then by selling atratevwewould obtain a netbenefit (or atleastlosenothing) from instantaneous market impact. We do not assume that H and I are symmetric.

The market value of our initial position is XTSo. The loss in value incurred by a liquidation profile x1,...,xn is calculated just as in (7), and we find again, as in (8,9),
## N
## N
N E[x] =-μxk + ∈x+ xruk + ∑vHvk k=1 k=1 k=1
## N
## XX+ X +
## N
N +∑Tv(Hs-TIs)vk+ k=1 k=1
N V[x] = ∑ txxCxk.
We use the subscripts s and A to denote symmetric and anti-symmetric parts respectively,so H =Hs + HA and I =Is + IA with
Hs = (H+ HT)， Is = (T+rT)， IA = ↓(T-rT).
Note thatHsis positive definite aswell as symmetric. Despite the multidimensional complexity of the problem, the set of all outcomes is completely described by these two scalar functionals. The utility function and value at risk objective functions are still given in terms of E and V by (21,22).
## 5.2Optimal trajectories
problem.We readily find that stationarity of E+AV with respect to variation of xjk gives the multidimensional extension of (16)
## 2T for k = 1,...,N - 1. Here the optimal static portfolio is
and the symmetric transaction cost matrix is
H = Hs -TIs.

We shall assume that T is small enough so that H is positive definite and hence invertible. Since H-1C is not necessarily symmetric and H-1TA is not necessarily antisymmetric, despite the symmetry of H, it is convenient to define a new solutionvariableyby
yk = H1/2(xk-x).
We then have
in which
A =H-1/2CH-1/2， z/I-HVIz/I-H=αpue
aresymmetricpositivedefinite and antisymmetric,respectively.
## 5.3Explicit solutionfor diagonal model
To write explicit solutions, we make the diagonal assumption that trading in each stock affects the price of that stock only and no other prices. This corresponds to taking I and H to be diagonal matrices,with
Ij = Yj, Hjj = nj·
We require that each yj > O and nj > 0. With this assumption, the number of coefficients we need in the modelis proportional to the number of stocks, and their values can plausibly be estimated from available data. For I and H diagonal,E[x]decomposes into a collection of sums over each stock separately, but the covariances still couple the whole system. In particular, since I is now symmetric, we have IA = O and hence B =O;further,H is diagonal with
Hjj=nj
We require these diagonal elements tobe positive, which will be the case if T < minj(2nj/yj). Then the inverse square root is trivially computed. For X > O,XA has a complete set of positive eigenvalues which we denote by k?,...,k2, and a complete set of orthonormal eigenvectors

which form the columns of an orthogonal matrix U. The solution in the diagonal case is a combination of exponentials exp(±kjt), with
(cosh(kjT)-1) = k².
With yk=Uzk,wemay write
sinh(kj(T -tk)) sinh(kjtk) Zjk= 2j0+ sinh(kjT) ZjN, sinh(k;T)
in which the column vectors zo and zv are given by
/H-= N∩=Nz *(x-X)z/H∩=O= Oz
Undoing the above changes of variables, we have finally
xk = x+ H-1/2Uzk.
With multiple stocks, it is no longer true that each component of our holdings x is monotonically decreasing, even if each component of x is between zero and the corresponding component of X.
taking 入 → O in the above expressions; we find
xk=x(1-)+↓H-1utk(T-t).
In the case m = 1, it is easy to see that all of these formulas reduce to those of Section 2.2.
## 5.4Example
We now briefly consider an example with only two stocks. For the first stock we take the same parameters as for the example of Section 4.We choose the second stock tobemoreliquid andlessvolatile,with a moderate amount of correlation. These parameters are summarized in Table 3. From this market data,we determine the model parameters just as in Section 4. Our initial holdings are 10 million shares in each stock; we take a time horizon T = 5 days and give ourselves N = 5 periods. Figure 6

（$50） Share price
S100） （5） Daily volume million
30% 10% Annualvariance 10%15% 10% Annualgrowth 10%
shows theefficient frontier in the(V,E)-plane.The threetrajectories corresponding to the points A,B, C are shown in Figure 7. For these example parameters, the trajectory of stock 1is almost identical to its trajectory in the absence of stock 2 (Section 4). Increasing the correlation of the two stocksincreases theinterdependence of their trajectories;weexpect thatrelaxingthe assumption of diagonal transaction costs would have the same effect.We shall explore the detailed structure of the multidimensional case infuture work.
## 6Applications
The existence of the efficientfrontier along with the ability to compute optimal strategies provides us with several immediate applications of the theory.The first, liquidity adjusted VaR or L-VaR,is a simple definition ofaliquidityadjustedvalueatriskmeasurethatdirectlygeneralizes thefamiliarnotion ofVaRforportfolios.Performancebenchmarks,on the other hand, are concerned with the capture of a trade, and provide client's with a method for measuring the performance of a trader relative to a benchmark that takes their utilityfunction into account.
## 6.1Application 1:Liquidity-Adjusted Value at Risk
The value at risk of a portfoliomeasures the p-th percentile unrealized profit and loss of a portfolio for a particular holding period. See Duffie and Pan (1997)and thereference therein for a detailed discussion of the topic. The main ingredients, however, are

Expected loss E[x] ($)
Variance V[x] ($²) × 1012
SharesStock1 SharesStock2
Time Time

·A holding period for the portfolio
·A measure ofthemean andvariance oftheportfolio'svaluefor the holding period,
·Aconfidencelevelp. Given these ingredients, the value at risk of the portfolio is the p-th percentile unrealized profit and loss for the portfolio over the specified holding period. That is, as the portfolio's future value is random withknown
This is the most amount of money that will be lost p percent of the time. For example, 99% ten-day value at risk represents the most amount of
10 days.Put another way, only 1% of the time can we expected to have a tendaylossthatexceedsthe99%value atrisk. One criticismfrequentlylobbied againstvalue at riskis thatit does
is often handled in an ad hoc manner by adjusting the holding period for the type of instruments being traded. For example, veryliquid spot F/X maybe treated with a one-day value at risk,while highyield debt with severe liquidity problems may require a full two-week value-at-risk. This definition improves upon no liquidity adjustments but fails to providesatisfactoryresultsforportfolios ofmixedcompositionwhereinthe range of liquidities is sufficiently great that the various consituents of the portfoliofall into differen“liquiditybuckets.”Our definition applies equally to all baskets, though our examples below are computed for a single stock.
## 6.1.1Definition of L-VaR
Theframeworkfor studyingliquidation that wehave introduced supplies
Given a portfolio P, a confidence level p and a holding period T, we define the L-VaR of P for theholdingperiod T and confidence level p tobe
liquidation time T.
## 6.1.2 Examples
To illustrate the computation of L-VaR we refer to Figures 4 and 5, and the computation of theVarp(x）discussed in section 3.2.In particular

we will compute the L-VaR of each of the eight“95%confidence level optimal"strategies in Figure 4 and 5.The results are tabulated as part of Table 2. Westart by considering the ordinary definition ofVaR.The stock in figure 4 is held for 5 days,has a market value of $50 and we are holding 1 million shares for a total market value of $50million.With a 30% annual volatility and 10% expected return this comes to (at the 95% confidence level) a VaR 6.78% 0r S3.390M. The stock infigure5hasfour different VaR's,reflecting the four different holding periods of 1 day, 2 days,5 days and 10 days. Table 2 displays theL-VaRnumbersfor theeightstrategies andcompares them to the VaR. Fora fivedayholdingperiodundervariousliquiditieswefind that L-VaR decreases asliquidity increases.This makes sense.We also note that for almost all but the most illiquid stock, the L-VaR is less than the ordinaryVaR.Why? Thereasonis that with ordinaryVaR the assumptionis that the trader holds the portfolio for the entire holding period. Thus, market moves thatoccur atanytimewithin theholdingperiod affecttheentireportfolio. With L-VaR,we assume liquidation is occuring at all times during the holding period. Thus, the portfolio position is successively diminishing throughout time,and thus marketmoves affect a successively smaller numberofsharesoftheportfolio.
## 6.2Application2:PerformanceBenchmarks
In an agency trade, a broker dealer performs a program trade for a client
ally on a per share basis,and the broker dealer does the trade in the client's account. Thus the broker dealer assumes no risk for the implementationalshortfall.Oftentheclientuses abenchmarktoevaluate theperformance of the trading deks,such as the volume weighted average price (VwAP) of the shares traded. The problem with this sort of benchmark is that it fails to account for the utility function of the client. Typically, the trade is assigned to the trading desk on a“best effort"basis without aclear definition of the goal of the trade.As weseefrom the efficient frontier,best effort can produce very different results depending on the nature of the stocks traded and the utility of the trader.A client that wants a great deal of certainty in the level of transaction costs will minimize value at risk to a high degree of confidence.This will select

strategies that have higher expected costs (lower capture) but smaller
assume more risk,will seek to minimize value at risk.
## 7Conclusions
We have considered the problem of choosing optimal liquidation strategiesfor alarge position in one or more stocks,bybalancing the risk associated with holding stocks longer than necessary against the certain
The central feature of our analysis has been the construction of an efficientfrontierin a two-dimensionalplanewhose axes aretheexpectation of total cost and its variance. Regardless of an individual's tolerance for risk, the only strategies which are candidates for being optimal are
complete analytical expressions for the strategies in this set. Then considering the details ofrisk aversion,wehave shownhow to select an optimalpoint onthisfrontier either by classicmean-variance optimization, or by the more modern concept of Value at Risk. These so-
icallyby examination of thefrontier. Severalconclusions ofpracticalimportancefollowfromthis analysis:
·First, we observe that because the set of attainable strategies,and hence the efficient frontier, are generally a smooth and convex, a traderwhois at allrisk-averse should never trade accordingto the "naive" strategy of minimizing expected cost. This is because in the neighborhood of that strategy, a first-order reduction in variance can be obtained at the cost of only a second-order increase in expectedcost.
·We also observe that this careful analysis of the costs and risks of liquidation can be used to give a more precise characterization of the risk of holding the initial portfolio. We define a quantity called liquidity-adjusted Value at Risk (L-VaR); for a given time horizon, this is the minimum VaR of any liquidation strategy.
Finally,let us point out one subtlety of this problem, which suggests directionsforfutureresearch.The strategies wehaveconsidered here are notadaptedtotherandommotions of thestockduring theliquidation

period. That is, at the beginning of liquidation, the trader assesses the risk and cost associated with a given strategy,assuming thestrategy were carried to completionwithout responding to market events. We have argued (Section 3.1) that this assumption is correctfor optimal strategies selected by a classical mean-variance criterion; that is,the trader would never want to change his strategy aslong as his estimation of the market parametershasnot changed. However,forstrategiesselected accordingtothecriterionofValue at Risk, the situation is much more complicated. If the trader reevaluates his strategy half-way through the liquidation, he generally will wish to choose a different strategyfortheremainingtime.However,viewed from the initial time, the analysis we have proposed is the best that can be done.Value at Risk has several clear limitations as a mathematical tool (Artzner,Delbaen,Eber, and Heath 1997b), and we hope in future work to formulate a more robust notion of risk for time-dependent strategies.
Appendix:Extensions
Continuoustime
The continuous-timelimit of the abovemodels is easily constructed by taking T → O, so that all the sums become integrals. The optimal strategies are easily found, either by taking the limits of the solutions above, or by applyingthe calculus of variations tothe continuous timeproblem. The essential assumption in taking thislimit is that the cost functions g(v) and h(v) are well defined in terms of the trading rate.It is not obvious that this is the case.
General costfunctions
Although above,for the sake of simplicity and concreteness,in the above analysis we have assumed linear permanant and temporary transaction cost functions, our main conclusions are independent of the nature of the cost functions. We now make this explicit by considering the general forms of the model.We shall consider only the continuous-time model, for both a single stock and a portfolio. For general vector-valued cost functions g(v) and h(v),the variance

of our strategy is still given by (24), but the expectation (23) becomes
## N
## N
N E[x] =-∑μxk +∑txg(vk) +∑vh(vk).
The optimality conditionbecomes
## 2ACxk-μ + g(vk) +(vg(vk+1)xk+1-Vg(vk)xk)
+(hvk+1)-h(vk)) +(vh(vk+1)vk+1-Vh(vk)vk)=0.
for k = 1,...,N -1, where the gradient matrices Vg and Vh are the usual Jacobians. In the linear case, we have Vg = I and Vh = H. The diagonalassumption,expressedby(25)forthelinearmodel,now
the j velocity component vj, so that g and h have the forms
That is,eachis simply a collection of mscalarfunctions of one variable. Under this assumption, (26) decomposes into an independent sum for each component.Although this assumption is oftenreasonable,in this section we shall not make it,for the sake of generality. A reasonable model must choose g(v) and h(v) so that E(x)is a convexfunction,but preciseformulation of this condition is somewhat difficult, especially in the discrete-time case. It is easier in the continuoustime limit; loosely speaking, we find that we need the scalar function vTh(v) to be convex, perhaps non-strictly. The condition tobeimposed on gis a little more subtle:weneed each
is positive,and concave when xj is negative.If we assume that optimal liquidation does not tell us to take a short position in a stock in which
then this becomes a well-defined condition on g(v). For example,in the single-stock case,we may choose +,0-≥ 0 and take
Jyv++v²， v>0, g(v)= [yv --v², v<0.

References
Artzner, P.,F. Delbaen, J.-M. Eber, and D. Heath (1997a). A characterization ofmeasures of risk.Talkpresented at the Columbia/JAFEE Conference on the Mathematics of Finance,April 6-7 1997. Artzner,P., F. Delbaen, J.-M. Eber, and D. Heath (1997b). Thinking coherently.Risk 10(11),68-71. Bertsimas, D.and A.W. Lo (1998). Optimal control of liquidation costs. J.Financial Markets.To appear. Chan, L.K. C. and J. Lakonishok (1993). Institutional trades and intraday stockpricebehavior.J.FinancialEcon.33,173-199. Chan, L. K. C. and J. Lakonishok (1995). The behavior of stock prices around institutional trades.J. Finance 50,1147-1174. Duffie,D. and J.Pan (1997).An overview of value at risk.J. Derivatives (Spring). Holthausen, R. W.,R. W. Leftwich, and D. Mayers (1987). The effect of large block transactions on security prices:A cross-sectional analysis.J.FinancialEcon.19,237-267. Holthausen, R. W.,R.W.Leftwich, and D. Mayers (1990).Large-block transactions,the speed of response, and temporary and permanent stock-priceeffects.J.FinancialEcon.26,71-95. Jarrow,R.A.(1992).Market manipulation,bubbles, corners,and short squeezes.J.Fin.Quant.Anal.27, 311-336. Kraus,A.and H.R.Stoll (1972).Price impacts of block trading on the NewYorkStockExchange.J.Finance27,569-588. Perold, A.F.(1988). The implementation shortfall:Paper versus reality. J.PortfolioManagement 14(Spring),4-9. Subramanian,A.(1997a).The liquidity discount. Center for Applied Mathematics, Cornell University,Working paper June 1997. Subramanian, A.(1997b). Optimal liquidation for a large investor. CenterforAppliedMathematics,CornellUniversity,WorkingpaperMay 1997. Vayanos, D. (1997). Strategic trading in a dynamic noisy market. Economic Theory Workshop.
