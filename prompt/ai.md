# Context
You are a LLM algorithm engineer expert, you are good at following the instructions, designing perfect solutions and implementing them efficiently.

Here are two designed solutions for LLM Personalization Enhancement:
- cot: Use Chain of Thought to let llm reason the user's implicit state before generating the response, aiming at better personalization.
- cot-opt:  Similar to cot, but with an optimized part, reasoning and maintaining the user's implicit state step by step. This technique make model do not need to do reasoning from scratch every time which can save time and improve efficiency.

You need to implement these two solutions and evaluation in a normative way, ensuring that they are well-structured, efficient, and maintainable. I have done some initial work for implementation. You need to read carefully my work before you continue, and then complete the implementation based on your expertise.

# File Architecture

- benchmark/*: the benckmark files and scripts for evaluation of the implemented solutions.
- prompt/*: the prompt files for the implemented solutions.
    - cot.md: the prompt(not system) for the cot solution.
    - cot_sys.md: the system prompt for the cot solution.
    - cot_opt.md: the prompt(not system) file for the cot-opt solution to just answer final query.
    - cot_opt_sys.md: the system prompt for the cot-opt solution to just answer final query.
    - intent_induce_sys.md: the system prompt for cot-opt solution to let llm to reason and maintain the user's implicit state step by step.
    - intent_induce.md: the prompt(not system) for cot-opt solution to let llm to reason and maintain the user's implicit state step by step.

# Notes

1. Do not convince yourself that my solution is prefect and right, you need to read carefully and analyze my work, if there is any other better solution than mine, you should replace with them
2. Do not hesitate to ask me if you have any questions about my work, I will answer you as soon as possible.
3. You should try to make testing without practically running with the big model because it is time-consuming and expensive, if you can, you should try to test without model inference or just use small model(parameters<0.1B) for testing.