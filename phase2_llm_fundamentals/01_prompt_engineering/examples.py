"""
Prompt Engineering Examples

Learn to write effective prompts for LLMs.
These examples work WITHOUT an API key - they show prompt patterns
and simulate responses so you can learn the techniques.

Run with: uv run python phase2_llm_fundamentals/01_prompt_engineering/examples.py
"""

from typing import List, Dict, Any
import json


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_prompt(prompt: str, label: str = "PROMPT") -> None:
    """print a prompt in a formatted box"""
    print(f"\n📝 {label}:")
    print("-" * 50)
    print(prompt.strip())
    print("-" * 50)


def print_response(response: str, label: str = "RESPONSE") -> None:
    """print a simulated response"""
    print(f"\n🤖 {label}:")
    print(response.strip())


# =============================================================================
# PART 1: Basic Prompts - Clarity is Key
# =============================================================================

def basic_prompts() -> None:
    """the foundation: clear, specific instructions"""
    print_section("1. Basic Prompts - Clarity is Key")

    # bad prompt
    print("\n❌ BAD PROMPT (vague):")
    bad_prompt = "Summarize this"
    print_prompt(bad_prompt)
    print_response("(LLM doesn't know: how long? what format? what focus?)")

    # good prompt
    print("\n✅ GOOD PROMPT (specific):")
    good_prompt = """
Summarize the following document in exactly 3 bullet points.
Focus on: key decisions made and action items.
Format: Start each bullet with an action verb.

Document:
The team met to discuss Q4 priorities. We decided to focus on
improving API performance by 50%. Sarah will lead the optimization
effort. Budget of $10K approved for new monitoring tools.
Timeline: complete by December 15th.
"""
    print_prompt(good_prompt)
    print_response("""
• Prioritize API performance improvement targeting 50% speed increase
• Assign Sarah as lead for the optimization initiative
• Allocate $10K budget for monitoring tools with Dec 15 deadline
""")

    print("\n💡 Key Principles:")
    print("   1. Be specific about what you want")
    print("   2. Specify format (bullets, JSON, length)")
    print("   3. Give context about purpose")
    print("   4. Include constraints (word count, focus areas)")


# =============================================================================
# PART 2: System Prompts - Setting Behavior
# =============================================================================

def system_prompts() -> None:
    """system prompts define the AI's persona and rules"""
    print_section("2. System Prompts - Setting Behavior")

    print("\n📋 Message Structure:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "User's question here"},
        {"role": "assistant", "content": "AI's response here"}
    ]
    print(json.dumps(messages, indent=2))

    # example 1: coding assistant
    print("\n--- Example 1: Coding Assistant ---")
    system_prompt = """
You are an expert Python developer.
Follow these rules:
- Always include type hints
- Add brief docstrings to functions
- Prefer simple, readable code over clever tricks
- If asked to review code, point out bugs AND suggest improvements
"""
    print_prompt(system_prompt, "SYSTEM PROMPT")

    user_message = "Write a function to check if a number is prime"
    print_prompt(user_message, "USER MESSAGE")

    print_response("""
def is_prime(n: int) -> bool:
    \"\"\"check if n is a prime number\"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""")

    # example 2: RAG assistant
    print("\n--- Example 2: RAG System Assistant ---")
    rag_system_prompt = """
You are an HR assistant for Acme Corp.
Your knowledge comes from company documents provided as context.

Rules:
- Only answer based on the provided context
- If the answer isn't in the context, say "I don't have that information"
- Always cite which document the information came from
- Be concise but complete
"""
    print_prompt(rag_system_prompt, "SYSTEM PROMPT (RAG)")

    print("\n💡 System Prompt Best Practices:")
    print("   1. Define the persona clearly")
    print("   2. Set explicit rules/constraints")
    print("   3. Specify what to do when uncertain")
    print("   4. Keep it focused (not too many rules)")


# =============================================================================
# PART 3: Few-Shot Learning - Teaching by Example
# =============================================================================

def few_shot_learning() -> None:
    """teach the model a pattern with examples"""
    print_section("3. Few-Shot Learning - Teaching by Example")

    print("Few-shot = give examples, model learns the pattern\n")

    # example: entity extraction
    few_shot_prompt = """
Extract entities from text as JSON.

Examples:

Text: "John Smith works at Google in New York"
Output: {"person": "John Smith", "company": "Google", "location": "New York"}

Text: "Sarah joined Microsoft last month in Seattle"
Output: {"person": "Sarah", "company": "Microsoft", "location": "Seattle"}

Text: "The CEO of Amazon, Andy Jassy, announced the news from Austin"
Output:
"""
    print_prompt(few_shot_prompt, "FEW-SHOT PROMPT")
    print_response('{"person": "Andy Jassy", "company": "Amazon", "location": "Austin"}')

    # example: sentiment classification
    print("\n--- Few-Shot Classification ---")
    classification_prompt = """
Classify the sentiment of customer reviews.

Review: "This product is amazing! Best purchase ever!"
Sentiment: POSITIVE

Review: "Terrible quality, broke after one day."
Sentiment: NEGATIVE

Review: "It's okay, nothing special but works fine."
Sentiment: NEUTRAL

Review: "Absolutely love it, exceeded my expectations!"
Sentiment:
"""
    print_prompt(classification_prompt, "FEW-SHOT CLASSIFICATION")
    print_response("POSITIVE")

    print("\n💡 Few-Shot Tips:")
    print("   1. 2-5 examples usually enough")
    print("   2. Cover edge cases in examples")
    print("   3. Keep examples consistent in format")
    print("   4. Order can matter - put best examples first")


# =============================================================================
# PART 4: Chain-of-Thought - Step by Step Reasoning
# =============================================================================

def chain_of_thought() -> None:
    """make the model reason step by step for better accuracy"""
    print_section("4. Chain-of-Thought - Step by Step Reasoning")

    print("CoT = ask the model to think through problems step by step\n")

    # without CoT
    print("--- Without CoT (often wrong) ---")
    simple_prompt = """
Q: A store has 3 shelves. Each shelf has 4 boxes.
   Each box has 5 items. How many items total?
A:
"""
    print_prompt(simple_prompt, "SIMPLE PROMPT")
    print_response("Answer: 35 items")  # might be wrong
    print("(Model might skip steps and make errors)")

    # with CoT
    print("\n--- With Chain-of-Thought (more accurate) ---")
    cot_prompt = """
Q: A store has 3 shelves. Each shelf has 4 boxes.
   Each box has 5 items. How many items total?

Let's solve this step by step:
1. First, find boxes per shelf
2. Then, find total boxes
3. Finally, calculate total items

A:
"""
    print_prompt(cot_prompt, "COT PROMPT")
    print_response("""
Let me work through this:
1. Each shelf has 4 boxes
2. Total boxes = 3 shelves × 4 boxes = 12 boxes
3. Total items = 12 boxes × 5 items = 60 items

Answer: 60 items
""")

    # zero-shot CoT (just add "think step by step")
    print("\n--- Zero-Shot CoT (simple trick) ---")
    zero_shot_cot = """
Q: If a train travels at 60 mph for 2.5 hours, how far does it go?

Think step by step, then give your answer.
"""
    print_prompt(zero_shot_cot, "ZERO-SHOT COT")
    print_response("""
Step 1: Identify the formula - distance = speed × time
Step 2: Speed = 60 mph, Time = 2.5 hours
Step 3: Distance = 60 × 2.5 = 150 miles

Answer: 150 miles
""")

    print("\n💡 When to Use CoT:")
    print("   ✅ Math problems")
    print("   ✅ Multi-step reasoning")
    print("   ✅ Complex logic")
    print("   ✅ Debugging code")
    print("   ❌ Simple factual questions (overkill)")


# =============================================================================
# PART 5: Output Formatting - Structured Responses
# =============================================================================

def output_formatting() -> None:
    """control the format of LLM responses"""
    print_section("5. Output Formatting - Structured Responses")

    # JSON output
    print("--- JSON Output ---")
    json_prompt = """
Analyze this code and respond in JSON format:

```python
def divide(a, b):
    return a / b
```

Response format:
{
    "has_bugs": boolean,
    "bugs": ["list of bugs found"],
    "severity": "low" | "medium" | "high",
    "fix": "suggested fix"
}
"""
    print_prompt(json_prompt, "JSON FORMAT PROMPT")
    print_response("""
{
    "has_bugs": true,
    "bugs": ["No handling for division by zero"],
    "severity": "high",
    "fix": "Add check: if b == 0: raise ValueError('Cannot divide by zero')"
}
""")

    # markdown output
    print("\n--- Markdown Output ---")
    markdown_prompt = """
Explain Python decorators.

Format your response as:
## Overview
(brief explanation)

## Example
(code example)

## When to Use
(bullet points)
"""
    print_prompt(markdown_prompt, "MARKDOWN FORMAT PROMPT")
    print_response("""
## Overview
Decorators are functions that modify the behavior of other functions.

## Example
```python
def timer(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        print(f"Took {time.time() - start}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

## When to Use
- Logging function calls
- Timing performance
- Authentication/authorization
- Caching results
""")

    print("\n💡 Format Tips:")
    print("   1. Show the exact format you want")
    print("   2. Use JSON for structured data")
    print("   3. Use markdown for readable docs")
    print("   4. Include example values in format spec")


# =============================================================================
# PART 6: RAG Prompt Patterns
# =============================================================================

def rag_prompt_patterns() -> None:
    """prompt patterns specifically for RAG systems"""
    print_section("6. RAG Prompt Patterns")

    print("These patterns are what you'll use in Phase 3!\n")

    # basic RAG pattern
    print("--- Pattern 1: Basic Context + Question ---")
    basic_rag = """
Context:
{context}

Question: {question}

Answer based only on the context provided.
"""
    print_prompt(basic_rag.format(
        context="Employees receive 15 days of paid vacation per year. "
                "Unused days roll over up to 30 days maximum.",
        question="How many vacation days do employees get?"
    ))
    print_response("Employees receive 15 days of paid vacation per year.")

    # citation pattern
    print("\n--- Pattern 2: With Citations ---")
    citation_rag = """
Context:
[1] Vacation policy: 15 days PTO annually
[2] Remote work: 3 days per week allowed
[3] Benefits: Health insurance enrollment in November

Question: What are the remote work rules?

Answer with citations in [n] format:
"""
    print_prompt(citation_rag)
    print_response("Employees can work remotely 3 days per week [2].")

    # strict grounding pattern
    print("\n--- Pattern 3: Strict Grounding (Reduce Hallucinations) ---")
    strict_rag = """
You are a helpful assistant that answers questions based on provided context.

IMPORTANT RULES:
1. Only use information from the context below
2. If the answer is not in the context, say "I don't have that information"
3. Never make up information
4. Quote relevant parts when appropriate

Context:
{context}

Question: {question}

Answer:
"""
    print_prompt(strict_rag.format(
        context="The company was founded in 2020. Headquarters is in Austin, TX.",
        question="Who is the CEO?"
    ))
    print_response("I don't have that information. The context only mentions "
                  "when the company was founded (2020) and its location (Austin, TX), "
                  "but doesn't include information about the CEO.")

    print("\n💡 RAG Prompt Best Practices:")
    print("   1. Always include 'based on context' instruction")
    print("   2. Tell model what to do when answer isn't found")
    print("   3. Request citations for verifiability")
    print("   4. Use strict grounding for factual applications")


# =============================================================================
# PART 7: Error Handling and Edge Cases
# =============================================================================

def error_handling() -> None:
    """handle edge cases gracefully"""
    print_section("7. Error Handling and Edge Cases")

    # handling uncertainty
    print("--- Handling Uncertainty ---")
    uncertainty_prompt = """
Answer the question based on the context.

If you're not sure, respond with:
{
    "answer": "your best answer or null",
    "confidence": "high" | "medium" | "low",
    "reasoning": "why this confidence level"
}

Context: Python was created by Guido van Rossum.

Question: When was Python created?
"""
    print_prompt(uncertainty_prompt)
    print_response("""
{
    "answer": null,
    "confidence": "low",
    "reasoning": "The context mentions the creator but not the creation date"
}
""")

    # handling bad input
    print("\n--- Handling Bad Input ---")
    bad_input_prompt = """
Extract the email from this text.
If no valid email is found, return {"email": null, "error": "reason"}.

Text: "Contact me at john [at] gmail"

Response:
"""
    print_prompt(bad_input_prompt)
    print_response("""
{"email": null, "error": "Obfuscated email format, not a valid address"}
""")

    # multi-language handling
    print("\n--- Language Handling ---")
    language_prompt = """
Detect the language and translate to English if needed.
Always respond in English.

Response format:
{
    "detected_language": "...",
    "original": "...",
    "english": "..."
}

Input: "Bonjour, comment allez-vous?"
"""
    print_prompt(language_prompt)
    print_response("""
{
    "detected_language": "French",
    "original": "Bonjour, comment allez-vous?",
    "english": "Hello, how are you?"
}
""")

    print("\n💡 Error Handling Tips:")
    print("   1. Always specify what to do when things go wrong")
    print("   2. Use structured output for error cases")
    print("   3. Include confidence levels when appropriate")
    print("   4. Handle edge cases explicitly in prompt")


# =============================================================================
# PART 8: Putting It All Together
# =============================================================================

def complete_example() -> None:
    """a complete, production-ready prompt example"""
    print_section("8. Complete Production-Ready Example")

    complete_prompt = """
SYSTEM:
You are a customer support assistant for TechCorp.
You help users with product questions based on our documentation.

RULES:
1. Only answer based on the provided documentation
2. If unsure, ask for clarification
3. Always be polite and professional
4. For complex issues, suggest contacting human support

DOCUMENTATION:
[Product Guide]
- Model X100: $299, supports WiFi 6, battery lasts 8 hours
- Model X200: $499, supports WiFi 6E, battery lasts 12 hours
- All models include 1-year warranty
- Extended warranty available for $49/year

[Return Policy]
- 30-day return window for unopened items
- 15-day return window for opened items
- Refunds processed within 5-7 business days

---

USER QUESTION: {question}

Respond in this format:
1. Direct answer to the question
2. Relevant additional info (if any)
3. Suggested next steps (if applicable)

RESPONSE:
"""

    print_prompt(complete_prompt.format(
        question="What's the difference between X100 and X200?"
    ))

    print_response("""
1. **Direct Answer:**
   The main differences between X100 and X200 are:
   - Price: X100 is $299, X200 is $499
   - WiFi: X100 has WiFi 6, X200 has newer WiFi 6E
   - Battery: X100 lasts 8 hours, X200 lasts 12 hours

2. **Additional Info:**
   Both models include a 1-year warranty. You can add extended
   warranty coverage for $49/year if desired.

3. **Next Steps:**
   Would you like more details about either model, or information
   about current promotions?
""")

    print("\n💡 Production Prompt Checklist:")
    print("   ✅ Clear system role defined")
    print("   ✅ Explicit rules/constraints")
    print("   ✅ Context/documentation included")
    print("   ✅ Output format specified")
    print("   ✅ Edge cases handled")
    print("   ✅ Fallback behavior defined")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """run all prompt engineering examples"""
    print("\n" + "=" * 70)
    print("  PROMPT ENGINEERING")
    print("  Master the art of communicating with LLMs")
    print("=" * 70)

    basic_prompts()
    system_prompts()
    few_shot_learning()
    chain_of_thought()
    output_formatting()
    rag_prompt_patterns()
    error_handling()
    complete_example()

    print("\n" + "=" * 70)
    print("  PROMPT ENGINEERING COMPLETE!")
    print("=" * 70)

    print("\n📚 Key Takeaways:")
    print("   1. Be specific - clarity beats brevity")
    print("   2. System prompts set the AI's behavior")
    print("   3. Few-shot examples teach patterns")
    print("   4. Chain-of-thought improves reasoning")
    print("   5. Specify output format explicitly")
    print("   6. Always handle edge cases")

    print("\n🚀 Next Steps:")
    print("   1. Practice modifying these prompts")
    print("   2. Move to 02_api_integration (connect to real LLMs!)")
    print("   3. Apply these patterns in Phase 3 RAG systems")


if __name__ == "__main__":
    main()