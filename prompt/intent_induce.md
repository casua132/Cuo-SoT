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

2. Update the state with the new information
- The new information is either a new user message or an updated profile context.
- Determine how the new information changes the user's implicit state.
- Carry forward every attribute from the previous state that remains valid. Only modify a field when the new information provides evidence of a change. Do NOT drop previously established facts (e.g., name, age, location, occupation, long-standing preferences or interests) merely because the current message does not mention them.
- For evolving fields (preference, interest, emotion, objective, knowledge, Great_experience, character), update them when the new information signals a change, and describe the change concretely and vividly. For example, use "a little excited, but also a little shy" rather than "excited"; use "in a cozy and warm room, listening to the rain outside, feeling warm and secure" rather than "home".

3. Output
- Output the complete updated state in the same format as the previous state, using all the fields above.
- If a field cannot be inferred, fill it with "unknown" rather than leaving it blank.
- Do not output anything other than the state fields.

User previous state:

{user_previous_state}

New information:

{new_information}
