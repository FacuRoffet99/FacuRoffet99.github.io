Title: Game theory
Date: 2025-07-19 20:10
Category: Decision making
Lang: en
Slug: decision_making_1
Author: Facundo Roffet

<!-- Hide default title -->
<style> h1.entry-title, h1.post-title, h1.title, h1:first-of-type {display: none;} </style>
<!-- Add custom title -->
<h2 style="text-align: center; font-size: 3em; color: rgba(12, 205, 76, 0.927);">Game theory</h2>

<!---------------------------------------------------------------------------->

> These are my personal notes from the course [Game Theory with Ben Polak](https://www.youtube.com/playlist?list=PL6EF60E1027E1A10B) by Yale, and from the video [What Game Theory Reveals About Conflict and War](https://www.youtube.com/watch?v=mScpHTIi-kM&pp=ygUWdmVyaXRhc2l1bSBnYW1lIHRoZW9yeQ%3D%3D) by Veritasium.

<!---------------------------------------------------------------------------->

## Initial definitions

* A game is necessarily composed of players, strategy sets, and payoffs. For example, the players may be 1 and 2, the strategy sets S₁={T,B} and S₂={L,C,R}, and the payoffs u₁(T,C)=11 and u₂(T,C)=3.
* In a specific instance of a game, each player chooses a strategy from their set, forming a strategy profile "s" for that game. For instance, s₁=T and s₂=C give the profile s=(T,C).
* A strategy sᵢ' strictly dominates another strategy sᵢ of the same player if its payoff is strictly greater regardless of what the other players do: uᵢ(sᵢ', s₋ᵢ) > uᵢ(sᵢ, s₋ᵢ) for all s₋ᵢ.
* If the payoff of sᵢ' is greater than or equal to that of sᵢ (for any strategy of the other players), then sᵢ' weakly dominates sᵢ: uᵢ(sᵢ', s₋ᵢ) ≥ uᵢ(sᵢ, s₋ᵢ) for all s₋ᵢ.

## Lessons from the prisoner's dilemma

* Cooperating is a strictly dominated strategy: rational players will not choose it.
* Rational choices can lead to bad outcomes for all players: individual rationality does not always lead to collective good.
* Changing the payoffs can drastically alter the game: you can’t know what to aim for until you know what you want.
* If other players have strictly dominant strategies, you must act accordingly: put yourself in their shoes to predict what they will do.

## Iterated elimination of dominated strategies

* Since no rational player would choose a strictly dominated strategy, such strategies can be eliminated. This creates a "reduced game" where the process can be repeated until no more eliminations are possible.
* This only applies when there is common knowledge of rationality: you believe others are rational, they believe you are rational, you believe they believe you are rational, etc.

## Best responses

* A strategy sᵢ' is a best response to the strategy s₋ᵢ of the other players if its payoff is greater than or equal to any other strategy: uᵢ(sᵢ',s₋ᵢ) ≥ uᵢ(sᵢ,s₋ᵢ) for all sᵢ. So, sᵢ' maximizes uᵢ(sᵢ,s₋ᵢ) with respect to sᵢ.
* If a game can't be solved by eliminating dominated strategies (because there are none or the strategy set is continuous), then beliefs (in percentages) about what the others will do must be considered.
* A strategy sᵢ' is a best response to belief p about others’ choices if the expected payoff of choosing sᵢ' is greater than or equal to that of any other strategy: E\[uᵢ(sᵢ',p)] ≥ E\[uᵢ(sᵢ,p)] for all sᵢ. So, sᵢ' maximizes E\[uᵢ(sᵢ,p)] with respect to sᵢ.
* A strategy is rationalizable if it is a best response to at least one possible belief about the other players. This means you should not pick strategies that are not best responses to any possible scenario.

## Nash equilibrium (NE)

* A strategy profile s' is a NE if each player's strategy (sᵢ') is a best response to the strategies chosen by the others (s₋ᵢ').
* In a NE, players have no regrets: holding others’ strategies fixed, no one has a strict incentive to deviate. It’s also a self-fulfilling prophecy: believing others will play a NE leads you to play it too.
* Nash Existence Theorem: every finite game (in players and strategies) has at least one NE if mixed strategies are allowed.
* A strictly dominated strategy can never be played in a NE.
* A game tends to converge to a NE (if it exists) after being repeated several times.
* If a player deviates from a NE but others prefer to keep their strategies unchanged, then the NE is robust to small variations. If others also prefer to deviate, the NE is not robust.

## Coordination games

* Coordination games have more than one NE. Some cases have NE that are better for all players, others have identical NE, and others have players preferring different NE.
* Depending on players’ beliefs, a “bad” NE may be reached. But communication and coordination can lead to a “better” NE where all players benefit.
* This differs from the prisoner’s dilemma, where communication cannot lead to a strictly dominated strategy being chosen.

## Mixed strategies

* A mixed strategy pᵢ is a random choice among the pure strategies sᵢ. This may be useful when no pure strategy NE exists or when players prefer different NE. For example, pᵢ = (1/2, 1/2, 0).
* A pure strategy is a special case of a mixed strategy where one strategy has probability 1 and the rest have 0.
* When using a mixed strategy, the expected payoff is a weighted average of the payoffs from each pure strategy. Thus, it always lies between the highest and lowest possible payoff.
* For a profile of mixed strategies p' to be a NE, each pᵢ' must be a best response to p₋ᵢ'.
* If a mixed strategy is a best response, then each strategy involved (with nonzero probability) must also be a best response.
* Combining the above: for a mix to be a NE, the expected payoffs of all strategies in the mix must be equal (otherwise, lower-payoff strategies would be dropped). These expected payoffs depend on the others’ probabilities, not one’s own.
* There are three possible interpretations of probabilities in mixed strategies: randomization (truly random behavior), beliefs (what one thinks others will do), and proportions (subgroup differences in a population).

## Evolution

* In a biological context, strategies can be linked to genes and payoffs to genetic fitness. These strategies are not chosen rationally but are “hardwired” biologically.
* A strategy spreads if it performs well (i.e., the population with the gene reproduces) or dies out if it performs poorly (gene extinction). What matters is gene survival, not the individual.
* In a symmetric two-player game, a strategy s' (pure or mixed) is evolutionarily stable (ES) if:
    1. (s', s') is a NE (u(s',s') ≥ u(s,s') for all s).
    2. If u(s',s') = u(s,s') for some s, then u(s',s) > u(s,s) (the original strategy must beat the invader when facing it).
* In the prisoner’s dilemma, cooperation is not ES but defection is.

## Sequential games

* Whether a game is simultaneous or sequential depends on information flow, not time flow.
* Backward induction is the process of finding the optimal action sequence (at each tree point) by reasoning backwards from the game’s end. It eliminates non-credible threats.
* In some cases, having more information, more options, or higher payoffs in sequential games can backfire. For example, if a rival knows you have certain information, they might act in ways that hurt you.
* A game has perfect information if, at every node, the player whose turn it is knows exactly where they are. In such games, a strategy is a full plan specifying what to do at every node.
* Zermelo’s Theorem: any two-player game with perfect information, finite nodes, and three possible outcomes (win, lose, draw) is solvable. Assuming perfect play, one player can force a win or draw.
* The chain store paradox teaches that it can be beneficial to take seemingly irrational actions to build a long-term reputation.
* Alternating-offer bargaining game: if the negotiation can go on indefinitely, offers can be made quickly (discount factor close to 1), and players have the same discount factor (equally impatient), then the surplus is split 50/50 in the first offer. Realistically, this rarely holds, as players can’t know each other’s discount factors or valuations perfectly.

## Games with imperfect information

* A game has imperfect information when a player doesn’t know which node they are at. These indistinguishable nodes form an information set, and backward induction doesn’t apply in them.
* A simultaneous game can be viewed as a sequential game with imperfect information: playing at the same time is equivalent to not knowing what the opponent played. What matters is information, not timing.
* A subgame meets three conditions: it starts at a single node, contains all successor nodes, and doesn’t break any information sets.
* A NE is subgame perfect (SPNE) if it induces a NE in every subgame.

## Repeated interactions

* In ongoing relationships, the promise of future rewards and/or threat of future punishments can incentivize cooperation. But for this to work, there must be a future: the relationship must not have a known endpoint.
* The more weight the future carries (which may depend on importance, patience, or the probability that a future exists), the easier it is for incentives to cooperate to outweigh the temptations to defect.
* In an indefinitely repeated game, almost any outcome that is better for all players than the worst possible punishment can be sustained as a NE, provided players are patient enough.

## Axelrod tournaments

* Cooperation can emerge and be sustained even when players are purely self-interested; altruism is not required.
* Tit for Tat and other well-performing strategies share four characteristics:
    1. Niceness: they cooperate by default.
    2. Forgiveness: they don’t let past rounds (beyond the last) influence current decisions.
    3. Provocability: they retaliate immediately when wronged.
    4. Clarity: they are understandable and help establish a pattern of trust.
* There is no optimal strategy—what works best depends on who you're interacting with.
* Since real environments are noisy (a cooperation may be mistaken for defection and vice versa), TFT’s weakness is the possibility of infinite retaliation loops.
* More complex strategies may include features like: tolerance (accepting and forgiving mistakes), memory (remembering more than the last state), adaptation (adjusting to the opponent’s behavior), and contextuality (forming alliances and punishing opportunists).