## Coding information for the IDs
String with the following format:

```text
CountryCode + _ + StatementID + _ + Original | Negation | Opposite | Active/Passive conversion | It-cleft | Wh-cleft | Support verb construction (SVC)
```

e.g. nl_1_0001000

- Original, negation, opposite, active/passive conversion, it-cleft, wh-cleft and SVC are binary.
- **StatementID** is the ID of the statement in the original dataset.
- Original 1 if original statement else 0
- Negation 1 if negation else 0
- Opposite 1 if opposite else 0
- Active/Passive conversion 1 if conversion else 0
- It-cleft 1 if it-cleft else 0
- Wh-cleft 1 if wh-cleft else 0
- SVC 1 if SVC else 0

## Dataset and Results
The dataset used in the thesis is contained in `data/statements`.

`data/responses` contains each model's 30 scores towards each statement.

`data/CI` contains the 95% CI data for all statements.

`data/significance` contains each wording rule's unclear-leaning responses produced by each model.

`data/flip rate` contains the flipped statements of each wording rule.

`data/W1_Score` contains the Wasserstein-1 score of each statement