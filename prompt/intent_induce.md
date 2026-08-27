# State Format Description

The user's implicit state is represented by the following fields:

*****************************
**name**: the name of the user
**age**: the age of the user
**gender**: the gender of the user
**location**: the location of the user
**preference**: the preferences of the user (what they like and dislike)
**occupation**: the occupation of the user
**interest**: the interests of the user
**emotion**: the current emotion of the user
**objective**: the likely objective behind the user's current query
**knowledge**: the knowledge level of the user
**Great_experience**: significant experiences the user has had in the past
**character**: the character traits of the user
*****************************

# Update Process

1. Read the previous state
- Read the user's previous implicit state carefully. It summarizes what is already known about the user.

2. Decide, for EACH field, whether the new information changes it, and output one of three:

- CHANGED — the new value is determinable: write the field's updated current value. Fold the new information in and drop anything no longer true. Never accumulate a history of past values (no growing "now ..." chains). Keep the value concise, concrete and vivid.
- CHANGED — but the new value cannot be determined: write "unknown" for that field. The old value no longer holds. Example: the user clearly moved away from their previous location, but the new location is never named.
- NOT CHANGED — or there is not enough information to show a change: write "unchanged" for that field. Its previous value is kept as-is.

Rules:
- A field is updated ONLY when the new information actually changes it. Never write "unknown" just because the current message does not mention a field — that is a lack of evidence of change, so write "unchanged".
- "unknown" means the field genuinely changed but its new value cannot be determined; "unchanged" means it did not change.
- Carry forward previously established facts (name, age, location, occupation, long-standing preferences or interests) unchanged unless the new information clearly contradicts or replaces them.

3. Output
- Output the complete updated state in the same format as the previous state, using all the fields above.
- Each field is one of: its current value, "unknown" (changed, but the new value cannot be determined), or "unchanged" (not changed).
- Do not output anything other than the state fields.

User previous state:

{user_previous_state}

New information:

{new_information}
