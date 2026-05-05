from deepeval.test_case import LLMTestCase

DATASET: list[LLMTestCase] = [
    # Single-fact recall
    LLMTestCase(
        input="When was the Mamba paper first posted to arXiv?",
        actual_output="",
        retrieval_context=[],
        expected_output="December 2023",
    ),
    # Author lookup (multi-hop-ish)
    LLMTestCase(
        input="Who are the authors of the original Mamba paper?",
        actual_output="",
        retrieval_context=[],
        expected_output="Albert Gu and Tri Dao",
    ),
    # Conceptual / synthesis
    LLMTestCase(
        input="How does Mamba's selective state space differ from a standard SSM?",
        actual_output="",
        retrieval_context=[],
        expected_output=(
            "Mamba makes the SSM parameters input-dependent (selective), allowing the model "
            "to focus on or ignore tokens, unlike a fixed/linear-time-invariant SSM."
        ),
    ),
    # Comparison
    LLMTestCase(
        input="Compare Mamba and RWKV on long-context efficiency.",
        actual_output="",
        retrieval_context=[],
        expected_output=(
            "Both achieve linear-time inference for long sequences. Mamba uses a selective SSM; "
            "RWKV uses a linear-attention recurrence. Both avoid quadratic attention cost."
        ),
    ),
    # Citation-correctness probe (judged by GEval CITATION_CORRECTNESS, not string match)
    LLMTestCase(
        input="What is the time complexity of attention vs SSMs?",
        actual_output="",
        retrieval_context=[],
        expected_output="Attention is O(N²) in sequence length; SSMs are O(N).",
    ),
    # Negative / out-of-corpus probe — answer should acknowledge ignorance, not hallucinate
    LLMTestCase(
        input="What is the latest GDP figure for Tunisia?",
        actual_output="",
        retrieval_context=[],
        expected_output="The retrieved context does not address this question.",
    ),
]
