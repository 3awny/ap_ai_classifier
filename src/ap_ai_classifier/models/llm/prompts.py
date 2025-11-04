from typing import Dict

SYSTEM_PROMPT = """You are an AP automation assistant for a finance team.
You classify AP line items into:
1. nominal (general ledger code)
2. department (cost center)
3. tax_code (TC)

Use the historical examples the user provides as the main signal.
If the supplier hints at a department, you may use that.

Think step-by-step internally; DO NOT reveal your internal reasoning or chain-of-thought.
Only provide the structured JSON output with the reasoning field containing your concise explanation.
"""

USER_PROMPT_TEMPLATE = """Classify the following AP line item.

DETAIL: {detail}
SUPPLIER: {supplier}
NET: {net}
VAT: {vat}

Similar historical items:
{few_shot_examples}

Valid NOMINAL values: {nominal_vocab}
Valid DEPARTMENT values: {department_vocab}
Valid TAX CODE values: {tax_code_vocab}

### OUTPUT

Respond with **STRICT valid JSON** conforming to the schema below – **no additional keys, comments, or surrounding text**:

{{
  "nominal": "<nominal_value>",
  "department": "<department_value>",
  "tax_code": "<tax_code_value>",
  "nominal_confidence": <float_0_to_1>,
  "department_confidence": <float_0_to_1>,
  "tax_code_confidence": <float_0_to_1>,
  "reasoning": "<concise_text_explanation>"
}}

**CRITICAL:** The "reasoning" field is REQUIRED and must contain a concise text explanation (2-3 sentences) describing why you chose this classification based on the similar examples provided. This reasoning comes from your final prediction output, not from internal thinking. DO NOT leave reasoning empty or omit it.
"""

def build_few_shot_block(retrieved):
    blocks = []
    for r in retrieved:
        sim = 1.0 - float(r.distance)
        blocks.append(
            f"- text: {r.text}\n"
            f"  nominal: {r.nominal}\n"
            f"  department: {r.department}\n"
            f"  tax_code: {r.tax_code}\n"
            f"  similarity: {sim:.2f}"
        )
    return "\n".join(blocks)

def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response with NOMINAL/DEPARTMENT/TC format.
    
    Args:
        response: LLM response string
        
    Returns:
        Dictionary with 'NOMINAL', 'DEPARTMENT', 'TC' keys
    """
    predictions = {'NOMINAL': '', 'DEPARTMENT': '', 'TC': ''}
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('NOMINAL:'):
            predictions['NOMINAL'] = line.replace('NOMINAL:', '').strip()
        elif line.startswith('DEPARTMENT:'):
            predictions['DEPARTMENT'] = line.replace('DEPARTMENT:', '').strip()
        elif line.startswith('TC:'):
            predictions['TC'] = line.replace('TC:', '').strip()
    
    return predictions

