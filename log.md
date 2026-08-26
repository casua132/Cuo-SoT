# Definition of the User Implicit State

The user's implicit state is a structured representation of who the user is and how they feel at any point in a conversation. It is maintained by the `cot` and `cot_opt` solutions.

Fields (all values should be concrete and vivid rather than brief labels; fill with "unknown" when a field cannot be inferred):

Each field holds the user's **current** state — when a field changes it is rewritten to the latest value, never accumulated into a growing history. `state.MAX_FIELD_LEN` caps any single field and `state.MAX_STATE_LEN` caps the total across all fields as backstops.

**`Great_experience` is the exception**: significant past experiences accumulate. It has its own budget (`--great-exp-max`, default 1500 chars) and, when an update would grow it past that, one extra LLM call condenses the whole field (fact-preserving) instead of truncating it.

- **name**: the name of the user
- **age**: the age of the user
- **gender**: the gender of the user
- **location**: the location of the user
- **preference**: the preferences of the user (what they like and dislike)
- **occupation**: the occupation of the user
- **interest**: the interests of the user
- **emotion**: the current emotion of the user
- **objective**: the likely objective behind the user's current query
- **knowledge**: the knowledge level of the user
- **Great_experience**: significant experiences the user has had in the past
- **character**: the character traits of the user

These twelve fields are used verbatim (same names, same order) in `prompt/cot.md`, `prompt/cot_opt.md`, and `prompt/intent_induce.md`, and are mirrored by `STATE_FIELDS` in `state.py`.
