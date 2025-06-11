from langchain.prompts import PromptTemplate
from typing import Optional

def get_prompt(
    context: str,
    question: str,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    Creates a LangChain PromptTemplate for answering a question based on a given context.

    Args:
        context: The context or background information for the question.
        question: The question to be answered.
        template: Optional custom prompt template string. If None, uses default template.

    Returns:
        A PromptTemplate instance configured with the provided context and question.

    Raises:
        ValueError: If context or question is empty or not a string.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    # Use default template if none provided
    default_template = (
        "Based on this context: {context} Answer the question: {question}. "
        "If you do not know the answer, just say that you do not know. "
        "Do not try to make up an answer."
    )
    prompt_template = template or default_template

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"], 
    )

    return prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )