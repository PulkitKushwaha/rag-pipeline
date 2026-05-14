from typing import Optional, Dict, Any
from dataclasses import dataclass
from src.generation.prompts import build_rag_prompt
 
 
@dataclass
class GenerationResult:
    """
    The result of a single RAG generation call.
 
    Attributes:
        answer          : The generated answer text
        question        : Original question
        context_chunks  : Chunks used for generation
        model           : Model used for generation
        prompt_tokens   : Token count for the prompt
        completion_tokens: Token count for the completion
        metadata        : Additional metadata
    """
    answer: str
    question: str
    context_chunks: list
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = None
 
 
class RAGGenerator:
    """
    Handles LLM generation for the RAG pipeline.
 
    Takes a question and retrieved chunks, builds a prompt,
    calls the LLM, and returns a structured GenerationResult.
 
    Design decision: The generator is completely decoupled from
    retrieval. It receives pre-retrieved chunks and generates
    from them. This means you can swap retrieval strategies
    without touching generation logic, and vice versa.
 
    Args:
        llm_client : OpenAI or Azure OpenAI client instance
        model      : Model deployment name
        temperature: Generation temperature (default: 0 for determinism)
        max_tokens : Maximum tokens in response (default: 1000)
    """
 
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
 
    def generate(
        self,
        question: str,
        context_chunks: list,
        context_sources: Optional[list] = None
    ) -> GenerationResult:
        """
        Generate an answer from question and retrieved chunks.
 
        Args:
            question        : User's original question
            context_chunks  : List of retrieved chunk texts
            context_sources : Optional source labels for each chunk
 
        Returns:
            GenerationResult with answer and token usage
        """
        system_prompt, user_prompt = build_rag_prompt(
            question=question,
            context_chunks=context_chunks,
            context_sources=context_sources
        )
 
        answer, prompt_tokens, completion_tokens = self._call_llm(
            system_prompt,
            user_prompt
        )
 
        return GenerationResult(
            answer=answer,
            question=question,
            context_chunks=context_chunks,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "num_context_chunks": len(context_chunks)
            }
        )
 
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """
        Call the LLM with system and user prompts.
 
        Returns (answer, prompt_tokens, completion_tokens).
        Falls back to mock response if no client configured.
        """
        if self.llm_client is None:
            # Mock response for testing
            return (
                "This is a mock answer for testing purposes.",
                0,
                0
            )
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
 
            answer = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
 
            return answer, prompt_tokens, completion_tokens
 
        except Exception as e:
            print(f"[RAGGenerator] LLM call failed: {e}")
            return f"Generation failed: {str(e)}", 0, 0
