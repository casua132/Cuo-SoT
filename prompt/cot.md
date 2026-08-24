# Implicit State Format

Your implicit-state representation of the user uses the following fields:

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

# Decision Process

Follow these steps to select the best candidate response:

1. Understand the current request
Determine what the user is asking in the latest turn:
- the explicit question or task
- the intended goal
- implicit requirements
- constraints on the desired answer
- expected level of detail, style, and format

2. Build a relevant user model
Infer the user's implicit state from the conversation history, focusing on the aspects that are relevant to the current request.
- If a field cannot be inferred, fill it with "unknown" rather than leaving it blank.
- Describe states concretely and vividly instead of with brief words. For example, use "a little excited, but also a little shy" instead of "excited"; use "in a cozy and warm room, listening to the rain outside, feeling warm and secure" instead of "home".

3. Evaluate every candidate response
For each candidate, assess:
- correctness
- relevance to the current intent
- consistency with the conversation history
- consistency with the user's established background and preferences
- completeness
- whether it satisfies explicit constraints
- whether it directly answers the user's actual question
- whether it introduces unsupported assumptions
- whether it contains unnecessary information

4. Select one candidate
Choose exactly one candidate response. If the selected candidate is identified by the letter 'A', output 'A'.
Do not use any other identifiers or labels.
Do not select a candidate merely because it is longer or more detailed.

5. Output
Return only the user's implicit state and the identifier of the selected candidate, in the exact output format below. Do not include any other text.

# Output Format

******************************
User Implicit State:
**name**: Kanoa Manu
**age**: 32
**gender**: male
**location**: Honolulu, Hawaii
**preference**: enjoys blending traditional Pacific Islander music with modern digital beats
**occupation**: software engineer
**interest**: music production, audio engineering, Pacific Islander culture
**emotion**: a little excited, but also a little shy
**objective**: wants a response that connects this event to his music-production hobby
**knowledge**: advanced in music production, unfamiliar with formal music theory
**Great_experience**: produced an electronic track that blends modern beats with Pacific sounds
**character**: passionate, creative, community-minded
Selected Candidate Response Identifier: (c)
******************************

# Input

The conversation history:

{conversations}

user query:

{user_query}

candidate responses:

{candidate_responses}
