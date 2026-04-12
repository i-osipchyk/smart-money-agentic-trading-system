# PoC Results

## Methodology

The PoC was designed to validate the core trading logic of the two-agent system, focusing on the interplay between HTF and LTF agents using the FVG + BOS framework. Real trade examples from the past were taken to the system to see how it asseses them.

## HTF Agent

HTF Agent provided good reasoning about states of market, seeing bullish, bearish and ranging regimes. However, when multiple FVGs are present on a chart, its reasoning become ambiguous, even though by the look of charts it showld not, which negatively influences decision of LTF Agent.

## LTF Agent

LTF Agent also did a good job in assesing the data it was fed with. But it was mostly not sure about its decision because of the uncertainty of results from HTF Agent. When the latter provided reasoning for both bullish and bearish trends, which are most of the cases, LTF Agent was hesitating to take any action.

## Strategy Following

During the PoC the prompt of LTF Agent was changed to follow the strategy better. It still does not understand it in a good way.

## Conclusion

The PoC validated that the core logic of the system is sound, but it also revealed that the HTF Agent's reasoning can be too ambiguous. This ambiguity significantly impacts the LTF Agent's ability to make confident decisions. Future iterations will focus on refining the HTF Agent's logic to provide clearer guidance to the LTF Agent, potentially by incorporating additional contextual factors or prioritizing certain FVGs over others.

Also, considering the strategy understanding of LTF Agent, it may be better to provide One Agent with a trade(Entry and Exits), and let it decide how that trade fits into current market structure.
