import anthropic
import os
import pandas as pd
from dotenv import load_dotenv


def explain_de_results(results_df: pd.DataFrame) -> str:
    """
    Summarize a differential expression results table in plain language,
    using the Anthropic Claude API.

    This is a convenience layer for reading results faster, not a
    statistical tool — it does not re-derive or double-check any numbers,
    it just narrates what's already in ``results_df``. The underlying
    analysis is still the exploratory method described in
    ``deconcord.differential_expression.methods``, so the caveats there
    apply to whatever this function says too.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_differential_expression`` — needs ``log_fold_change``,
        ``adjusted_p_value``, and ``significant`` columns.

    Returns
    -------
    str
        A short natural-language summary of the results.

    Raises
    ------
    KeyError
        If ``ANTHROPIC_API_KEY`` is not set in the environment or a
        ``.env`` file.
    """
    load_dotenv()

    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)

    n_total = len(results_df)
    n_significant = results_df["significant"].sum()

    top_genes = (
        results_df
        .sort_values("adjusted_p_value")
        .head(5)
    )

    prompt = f"""
I am summarizing the results of an RNA-seq differential expression analysis.

Total number of genes tested: {n_total}
Number of statistically significant genes: {n_significant}

Top 5 most significant genes:
{top_genes[['log_fold_change', 'adjusted_p_value']].to_string()}

Briefly and clearly explain these results to a researcher.
Describe the overall pattern of the analysis and mention any important
considerations or potential issues that should be kept in mind when
interpreting the results.
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content[0] is typed as a union of every possible content
    # block the API can return (tool use, thinking, etc.) -- only
    # TextBlock has .text. A plain text-only prompt like this one always
    # gets a TextBlock back in practice, but asserting it explicitly turns
    # a hypothetical future API response shape into a clear error naming
    # the actual block type received, instead of an opaque AttributeError.
    block = response.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise TypeError(
            f"Expected a text response from the Claude API, got a {type(block).__name__} "
            "content block instead."
        )
    return block.text