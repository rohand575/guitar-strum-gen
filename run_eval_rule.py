# run_eval_rule.py  —  Rule-based evaluation runner
import json, sys
sys.path.insert(0, '.')
from src.rules.generate_rule_based import generate_rule_based

results = []
with open('data/processed/test.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        try:
            output = generate_rule_based(sample['prompt'])
            if isinstance(output, dict):
                output['_ref'] = sample
                results.append(output)
            else:
                # if it's a GuitarSample object
                d = output.model_dump() if hasattr(output, 'model_dump') else output.dict()
                d['_ref'] = sample
                results.append(d)
        except Exception as e:
            print(f"Error on {sample['id']}: {e}")

with open('rule_based_outputs.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated {len(results)} rule-based outputs")
print("Saved to rule_based_outputs.json")