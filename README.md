# keychecker
a fast, bulk key checker for various AI services

Currently supports and validates keys for the services below, and checks for the listed attributes a key might have:

- OpenAI - (Best model, key in quota, RPM (catches increase requests), tier, list of organizations if applicable, trial key status)
- Anthropic - (Live key status, model access, usable balance status, and current rate-limit headroom)
- AI21 - (Trial check)
- Google MakerSuite (List of available models + billing status and key tier)
- AWS - (Admin status, auto-fetch the region, logging status, username, bedrock status + enabled models)
- Azure - (Auto-fetch all deployments, auto-fetch best deployment/model, filter status, dall-e deployment)
- Google Cloud Vertex AI - (Requires a key file since oauth tokens expire hourly.)
- MistralAI - (Subscription status)
- OpenRouter - (Account balance, separate key spending limit, budget from these limits, and spending in USD)
- ElevenLabs - (Key tier, remaining characters in plan, detect uncapped char quota, pro voice cloning limit, invoice details on pay as you go plans)
- DeepSeek - (Account balance)
- xAI - (Live key, ACL and model access, usable billing status, test cost, and rate-limit headroom)
- Perplexity - (Sonar access, quota/rate-limit status)
- GLM - (Key validity, model access, balance/no-balance status)
- Moonshot - (Balance status, model access, kimi-k2.6 access)

# Usage:
`pip install -r requirements.txt`

`python main.py`

# Optional Arguments:

`-proxyoutput`

Outputs keys in a format that can be easily copied and pasted into khanon's proxy instead of pretty print

`-nooutput`

Stops outputting and saving keys to the snapshot file (proxyoutput will also do this)

`-file`

Reads keys from a file instead of stdin, place either the absolute or relative path to the file in quotes after the flag.

`-verbose`

Displays an output as keys are being checked real time.

`-awslegacy`

Uses the slower legacy AWS checker (thread parallelized + boto3) instead of the new asynchronous REST API one.

`-verifyorg`

Enables the OpenAI checker to check if orgs are verified (by default this is disabled since it incurs one token of o3 usage)

# OpenRouter balance output

OpenRouter checks authentication and reads billing metadata using GET requests; it does not send paid model requests.

- `Account balance` is the account's total credits purchased minus total account spending. It is shared by the account's keys.
- `Key spending limit` is the spending cap configured for this key. `none` means no per-key cap; the account still needs credits for paid usage.
- `Remaining key limit` is the unused part of that cap, not money held by the key. A daily/weekly/monthly cap uses its reset period, not all-time usage.
- `Budget from balance/key limit` is the smaller of the account balance and the remaining key cap (never below zero). With no per-key cap, it equals the positive account balance. This is an upper bound from these two restrictions, not confirmation of model access or request availability.
- `Spent with this key` is usage, not remaining funds. Day/week/month are the current UTC calendar periods; the week starts on Monday.

For example, an account with $75 left and a key with $3 left in its spending cap has a budget of $3. A positive account balance with an exhausted key cap has a budget of $0.

```text
  Budget from balance/key limit: $3.0000 (upper bound from these two limits)
  Account balance: $75.0000 (shared across the account)
  Key spending limit: $20.0000
  Remaining key limit: $3.0000 (spending allowance, not account balance)
  Key limit resets: monthly (UTC)
```

The account `/credits` endpoint requires a management key according to the API reference. If access is denied, the checker displays `Account balance: unknown` with the reason and still shows the key's own cap and usage. A remaining key cap alone does not prove available account funds. Missing/malformed amounts and failed requests stay unknown, not zero or unlimited. A known exhausted cap or nonpositive balance is sufficient to show a zero budget even if the other amount is unknown.

Temporary failures checking `/key` remain visible as `inconclusive`; failure of the optional `/credits` lookup does not invalidate authenticated keys. The deprecated `rate_limit` object is no longer displayed as current RPM. Normal snapshots use the same readable output.

API references: [key credit limits and usage](https://openrouter.ai/docs/api_reference/limits), [account credits and permissions](https://openrouter.ai/docs/api/api-reference/credits/get-credits).
