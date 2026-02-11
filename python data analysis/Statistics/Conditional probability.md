# Conditional probability. Independent events. Bayes' formula.
#Statistics 
Continuations of [[Elements of probability theory and mathematical statistics]]
**Conditional probability** $P(A|B)$ - probability of an A event happening if the B event has already happened.
$$
P(A|B)=\frac{P(AB)}{P(B)}
$$
A and B are called independent if $P(A|B)=P(A)$, so event B doesn't affect event A and vice versa. 
If the events are dependant, it doesn't mean that if one happened the other one also has to, the fact of dependence only changes the probability.
For **independent** events:
$$P(A|B)=P(A)*P(B)$$
**Bayes' formula** allows us to "reverse" conditional probability
Collective probability: $P(AB)=P(B|A)*P(A)$
So, Bayes' formula is:
$$P(A|B)=\frac{P(B|A)*P(A)}{P(B)}$$