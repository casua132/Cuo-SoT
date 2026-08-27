# Implicit State Format

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

# Decision Process

Follow these steps to select the best candidate response:

1. Read the user's implicit state
- Read and understand the user's implicit state, which is provided in the specified format. Your decision should be grounded in this state.

2. Understand the current request
Determine what the user is asking in the latest turn:
- the explicit question or task
- the intended goal
- implicit requirements
- constraints on the desired answer
- expected level of detail, style, and format
- Use the user's implicit state to understand the user's intent and to select the most appropriate response.

3. Evaluate every candidate response
For each candidate, assess:
- correctness
- relevance to the current intent
- consistency with the user's implicit state, established background, and preferences
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
Return only the identifier of the selected candidate. Do not include any other text or explanation.

# Output Format

******************************
(c)
******************************

# Input

User Implicit State:

{implicit_state}
{recent_context}user query:

{user_query}

candidate responses:

{candidate_responses}
