from enum import Enum


class APIKey:
    def __init__(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key

        if provider == Provider.OPENAI:
            self.model = ""
            self.has_quota = False
            self.default_org = ""
            self.organizations = []
            self.rpm = 0
            self.tier = ""
            self.has_special_models = False
            self.real_32k = False
            self.the_one = False
            self.extra_models = False
            self.extra_model_list = set()
            self.id_verified = False
            self.slop = True
            self.missing_models = set()
            self.access_to_model_listing = True

        elif provider == Provider.ANTHROPIC:
            self.pozzed = False
            self.rate_limited = False
            self.has_quota = None
            self.tier = ""
            self.remaining_tokens = 0
            self.validation_status = "unknown"
            self.quota_status = "unknown"
            self.permission_restricted = False
            self.model = ""
            self.models = []
            self.rpm = 0
            self.requests_limit = 0
            self.requests_remaining = None
            self.requests_reset = ""
            self.tokens_limit = 0
            self.tokens_remaining = None
            self.tokens_reset = ""
            self.input_tokens_limit = 0
            self.input_tokens_remaining = None
            self.input_tokens_reset = ""
            self.output_tokens_limit = 0
            self.output_tokens_remaining = None
            self.output_tokens_reset = ""
            self.retry_after = ""
            self.usage_tokens = 0
            self.input_tokens_used = 0
            self.output_tokens_used = 0
            self.error_type = ""
            self.error_message = ""
            self.request_id = ""

        elif provider == Provider.AI21:
            self.trial_elapsed = False

        elif provider == Provider.AWS:
            self.username = ""
            self.useless = True
            self.admin_priv = False
            self.bedrock_enabled = False
            self.region = ""
            self.alt_regions = []
            self.useless_reasons = []
            self.logged = False
            self.models = {}

        elif provider == Provider.AZURE:
            self.endpoint = ""
            self.best_deployment = ""
            self.model = ""
            self.deployments = []
            self.unfiltered = False
            self.dalle_deployments = ""
            self.has_gpt4_turbo = []

        elif provider == Provider.VERTEXAI:
            self.project_id = ""
            self.has_opus = False

        elif provider == Provider.MISTRAL:
            self.subbed = False

        elif provider == Provider.MAKERSUITE:
            self.models = []
            self.enabled_billing = False
            self.tier = ""

        elif provider == Provider.OPENROUTER:
            self.validation_status = "unknown"
            self.validation_message = "not checked"
            self.usage = None
            self.usage_daily = None
            self.usage_weekly = None
            self.usage_monthly = None
            self.credit_limit = None
            self.key_limit_known = False
            self.limit_remaining = None
            self.limit_reset = None
            self.limit_reset_known = False
            self.include_byok_in_limit = None
            self.is_management_key = None
            self.account_balance = None
            self.total_credits = None
            self.total_usage = None
            self.rpm = 0
            self.balance = None
            self.limit_reached = False
            self.bought_credits = None
            self.credits_api_available = False
            self.credits_api_status = "not_checked"
            self.credits_api_http_status = None

        elif provider == Provider.ELEVENLABS:
            self.characters_left = 0
            self.usage = ""
            self.tier = ""
            self.unlimited = False
            self.pro_voice_limit = 0

        elif provider == Provider.DEEPSEEK:
            self.balance = "$0.0 USD"
            self.available = False
            self.rate_limited = False

        elif provider == Provider.XAI:
            self.blocked = False
            self.subbed = False
            self.validation_status = "unknown"
            self.access_status = "unknown"
            self.has_usable_billing = None
            self.rate_limited = False
            self.permission_restricted = False
            self.team_blocked = False
            self.api_key_blocked = False
            self.api_key_disabled = False
            self.acls = []
            self.full_endpoint_access = False
            self.full_model_access = False
            self.model = ""
            self.models = []
            self.text_models = []
            self.requests_limit = 0
            self.requests_remaining = None
            self.requests_reset = ""
            self.tokens_limit = 0
            self.tokens_remaining = None
            self.tokens_reset = ""
            self.retry_after = ""
            self.usage_tokens = 0
            self.input_tokens_used = 0
            self.output_tokens_used = 0
            self.cost_usd_ticks = 0
            self.test_cost_usd = None
            self.service_tier = ""
            self.error_type = ""
            self.error_message = ""
            self.request_id = ""

        elif provider == Provider.PERPLEXITY:
            self.model = ""
            self.has_quota = True
            self.rate_limited = False
            self.usage_tokens = 0
            self.cost = 0

        elif provider == Provider.GLM:
            self.model = ""
            self.models = []
            self.has_balance = True
            self.has_air_access = False
            self.air_model = ""
            self.air_usage_tokens = 0
            self.coding_model = ""
            self.has_coding_access = False
            self.coding_plan_expired = False
            self.coding_error_code = ""
            self.coding_error_message = ""
            self.rate_limited = False
            self.usage_tokens = 0
            self.error_code = ""
            self.error_message = ""

        elif provider == Provider.MOONSHOT:
            self.models = []
            self.has_balance = False
            self.available_balance = 0
            self.cash_balance = 0
            self.voucher_balance = 0
            self.has_kimi_k26 = False
            self.kimi_k26_usable = False
            self.rate_limited = False
            self.usage_tokens = 0
            self.error_type = ""
            self.error_message = ""

    def clone(self):
        cloned_key = APIKey(self.provider, self.api_key)
        cloned_key.__dict__ = self.__dict__.copy()
        return cloned_key


class Provider(Enum):
    OPENAI = 1,
    ANTHROPIC = 2
    AI21 = 3
    MAKERSUITE = 4
    AWS = 5
    AZURE = 6
    VERTEXAI = 7
    MISTRAL = 8
    OPENROUTER = 9
    ELEVENLABS = 10
    DEEPSEEK = 11
    XAI = 12
    PERPLEXITY = 13
    GLM = 14
    MOONSHOT = 15
