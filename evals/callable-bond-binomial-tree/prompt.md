---
tags: [cfa-analyst]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

Our desk is looking at a 3-year 3.5% annual-pay corporate bond callable at par (100) at the end of years 1 and 2. The issuer's par curve is 2.0% (1-year), 2.5% (2-year) and 3.0% (3-year), annual pay, and we assume 15% interest-rate volatility in a lognormal binomial model. Build the arbitrage-free tree, value the bond with and without the call, tell me what the call option is worth, the option-adjusted spread if the bond trades at 100.50, and its effective duration for a +/-25bp parallel shift of the par curve holding the OAS constant. Is it cheap or rich?
