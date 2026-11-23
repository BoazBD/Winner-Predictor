"""s00: which Anthropic models on Amazon Bedrock can this AWS account actually invoke?
(The installed aws-cli 2.13 has no `bedrock` command, so boto3 is used.)
Lists foundation models + inference profiles in il-central-1, eu-central-1, us-east-1, then sends ONE tiny
Converse call (max 5 output tokens) per candidate Claude model/profile. Output: out/bedrock_check.json
"""
import json, os, boto3, botocore

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = ["il-central-1", "eu-central-1", "us-east-1"]
res = {"regions": {}}
for reg in REGIONS:
    r = res["regions"][reg] = {}
    try:
        b = boto3.client("bedrock", region_name=reg)
        fm = b.list_foundation_models(byProvider="anthropic")["modelSummaries"]
        r["foundation_models"] = [(m["modelId"], m.get("inferenceTypesSupported"), m.get("modelLifecycle", {}).get("status"))
                                  for m in fm]
        prof = []
        tok = None
        while True:
            kw = {"maxResults": 100}
            if tok:
                kw["nextToken"] = tok
            p = b.list_inference_profiles(**kw)
            prof += [x["inferenceProfileId"] for x in p["inferenceProfileSummaries"]]
            tok = p.get("nextToken")
            if not tok:
                break
        r["inference_profiles"] = [x for x in prof if "anthropic" in x]
    except Exception as e:
        r["error"] = str(e)[:300]
        continue
    # candidates: on-demand foundation models + profiles, newest families only
    cands = [m[0] for m in r["foundation_models"] if m[1] and "ON_DEMAND" in m[1]] + r["inference_profiles"]
    keep = [c for c in cands if any(s in c for s in ["opus-4", "sonnet-4", "haiku-4", "3-7-sonnet", "3-5-haiku",
                                                     "opus-5", "sonnet-5", "haiku-5", "fable"])]
    rt = boto3.client("bedrock-runtime", region_name=reg,
                      config=botocore.config.Config(read_timeout=30, retries={"max_attempts": 1}))
    r["invoke"] = {}
    for mid in keep:
        try:
            o = rt.converse(modelId=mid, messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
                            inferenceConfig={"maxTokens": 5})
            r["invoke"][mid] = {"ok": True, "text": o["output"]["message"]["content"][0]["text"],
                                "usage": o["usage"], "latency_ms": o["metrics"]["latencyMs"]}
        except Exception as e:
            r["invoke"][mid] = {"ok": False, "err": str(e)[:200]}
        print(reg, mid, r["invoke"][mid], flush=True)
json.dump(res, open(os.path.join(HERE, "out", "bedrock_check.json"), "w"), indent=1)
