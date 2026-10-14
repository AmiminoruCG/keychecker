# keychecker
a fast, bulk key checker for various AI services

Currently supports and validates keys for the services below, and checks for the listed attributes a key might have:

- OpenAI - (Live inference, usable/exhausted/unknown quota, model-specific RPM/TPM, estimated tier, and optional organization metadata)
- Anthropic - (Live key status, model access, usable balance status, and current rate-limit headroom)
- AI21 - (Trial check)
- Google MakerSuite (List of available models + billing status and key tier)
- AWS - (Admin status, auto-fetch the region, logging status, username, bedrock status + enabled models)
- Azure - (Live inference, deployment discovery and access, quota status, and model-specific rate limits)
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

Disables writing the snapshot file; console output still includes keys (`-proxyoutput` also disables snapshot-file writing).

`-file`

Reads keys from a file instead of stdin, place either the absolute or relative path to the file in quotes after the flag.

`-verbose`

Displays an output as keys are being checked real time.

`-awslegacy`

Uses the slower legacy AWS checker (thread parallelized + boto3) instead of the new asynchronous REST API one.

`-verifyorg`

Adds an optional small `o3` access probe. It reports an explicit organization-verification requirement if returned by the API. Successful model access does not prove identity verification. This additional request is billable and disabled by default.

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

# OpenAI checks

The default check lists models, then sends a small real inference request. It prefers inexpensive text models and stops at the first successful response or billing/quota error. Chat probes allow one output token; a Responses-only model allows up to 16 output tokens. Input tokens are also billable. Up to three model candidates are tried when a model is missing or disallowed.

`credit_balance_exhausted`, spend limits and other billing errors are distinct from temporary rate limits, permission restrictions and API/network failures. Inconclusive keys remain visible for retry; they are not reported as invalid or exported by `-proxyoutput`. Scoped `sk-proj-` and `sk-svcacct-` keys are recognized without requiring an internal marker.

RPM and TPM are the limits returned for the tested model and project. `Tier N (estimate)` means both limits match that model's published defaults; it does not verify the account's usage tier. Custom limits, missing headers or models without a reference table show `Tier not exposed by API` without affecting the key's usable-quota result. No manual limit increase is inferred.

Organization metadata from the legacy `/me` endpoint is best effort. Its failure does not invalidate a key. Alternate organization checks use IDs and fresh state, and apply only to legacy user keys. Project and service-account keys are not tested against other organizations.

Transient responses have bounded retries (up to three attempts, respecting `Retry-After`). A delay above ten seconds defers the check; it is never shortened. POST connection failures are left inconclusive because the request may already have been billed. Redirects are disabled.

The normal console/snapshot output contains the keys as before. `-nooutput` disables snapshot-file writing; it does not hide console output.

Run regression tests with `python -m unittest discover -s tests -v`.

# Azure checks

Azure input accepts `resource-name:apikey` or a full HTTPS resource endpoint followed by `:apikey`, for example `https://example.openai.azure.com:apikey`. The resource name is the name of the Azure OpenAI resource, not its resource group. A deployment URL such as `https://example.openai.azure.com/openai/deployments/my-chat:apikey` can supply a deployment name when listing deployments is unavailable.

The checker uses Azure's v1 API for small real text-generation probes, allowing up to 16 output tokens per probe. Input and generated tokens are billable. It prioritizes inexpensive supported text deployments; the selected deployment must have completed a successful inference request. Image, embedding, retired, and unsupported deployments remain visible when discovered but are not probed as chat models. Failed probes remain visible with their API status.

Deployment discovery is best effort. The checker can consult the legacy listing endpoint for deployment metadata, but an unavailable listing does not by itself invalidate authentication. If discovery fails and no deployment name is available, inference access remains unconfirmed. A valid key with no usable text deployment, permission restrictions, a temporary rate limit, or a network/server error stays visible as restricted or inconclusive rather than being silently discarded as invalid.

Requests use bounded timeouts and retries, and Azure credential checks run with limited concurrency. Error messages and verbose progress use a credential fingerprint instead of the secret. Normal result/snapshot output still includes credentials; use `-nooutput` to disable snapshot-file writing.

Each credential check considers at most 20 text deployments and 10 discovery pages, with a 90-second budget for starting requests. An in-flight request is still subject to its own connect/read timeout. Untested deployments remain visible as `not_checked`. Up to three attempts respect Azure's `Retry-After` or millisecond retry headers; waits above ten seconds are deferred. POST network failures are not replayed because inference may already have been billed. The `/openai/v1/models` catalogue is used only as an authentication fallback, never as a source of deployment names.

The presence or absence of content-filter annotations does not prove that filtering is enabled or disabled. The checker makes no `UNFILTERED` claim. `-proxyoutput` exports only a confirmed usable deployment that fits the legacy proxy credential format.

API references: [Azure v1 lifecycle](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle), [chat-completion request and response fields](https://learn.microsoft.com/en-us/azure/foundry/openai/latest), and [reasoning model parameters](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning).
