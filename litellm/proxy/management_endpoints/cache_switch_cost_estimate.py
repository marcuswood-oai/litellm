from dataclasses import dataclass
from datetime import datetime
from typing import Final

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _cost_map_billed_rates,  # pyright: ignore[reportPrivateUsage]  # deployment pricing needs pre-resolved info
    get_billed_token_rates,
)
from litellm.proxy._types import (
    CacheSwitchCostEstimateArmRequest,
    CacheSwitchCostEstimateArmResponse,
    CacheSwitchCostEstimateRequest,
    CacheSwitchCostEstimateResponse,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage

router: Final = APIRouter()
_USER_API_KEY_AUTH_DEPENDENCY: Final = Depends(user_api_key_auth)


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    model: str
    model_id: str | None
    provider: str | None


def _resolved_direct_model(model: str) -> _ResolvedModel:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception as error:
        raise HTTPException(
            status_code=404,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not price model '{model}'"
            },
        ) from error
    return _ResolvedModel(model=resolved_model, model_id=None, provider=provider)


def _resolved_deployment(model: str, model_id: str | None) -> _ResolvedModel:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        if model_id is not None:
            raise HTTPException(
                status_code=422,
                detail={  # mutable-ok: HTTPException detail is a public response mapping
                    "error": f"Unknown model_id '{model_id}'"
                },
            )
        return _resolved_direct_model(model)
    deployments: Final = tuple(llm_router.get_model_list(model_name=model) or ())
    if model_id is None and len(deployments) > 1:
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Model '{model}' resolves to multiple deployments; provide model_id"
            },
        )
    matches: Final = (
        deployments
        if model_id is None
        else tuple(
            deployment
            for deployment in deployments
            if (deployment.get("model_info") or {}).get("id")  # mutable-ok: router metadata mapping
            == model_id
        )
    )
    if model_id is not None and len(matches) != 1:
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"model_id '{model_id}' does not uniquely identify a deployment for model '{model}'"
            },
        )
    if not matches:
        return _resolved_direct_model(model)
    deployment: Final = matches[0]
    params: Final = (
        deployment.get("litellm_params") or {}  # mutable-ok: router params mapping
    )
    info: Final = deployment.get("model_info") or {}  # mutable-ok: deployment info default is a read-only local view
    backend: Final = info.get("base_model") or params.get("base_model") or params.get("model")
    if not isinstance(backend, str):
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not resolve model '{model}'"
            },
        )
    resolved_model, provider, _, _ = litellm.get_llm_provider(
        model=backend,
        custom_llm_provider=params.get("custom_llm_provider"),
    )
    resolved_id: Final = info.get("id")
    return _ResolvedModel(
        model=resolved_model,
        model_id=resolved_id if isinstance(resolved_id, str) else None,
        provider=provider,
    )


def _usage(request: CacheSwitchCostEstimateArmRequest) -> Usage:
    creation: Final = request.cache_creation_input_tokens_5m + request.cache_creation_input_tokens_1h
    total: Final = request.uncached_input_tokens + request.cache_read_input_tokens + creation
    return Usage(
        prompt_tokens=total,
        completion_tokens=0,
        total_tokens=total,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=request.uncached_input_tokens,
            cached_tokens=request.cache_read_input_tokens,
            cache_creation_tokens=creation,
            cache_creation_token_details=CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=request.cache_creation_input_tokens_5m,
                ephemeral_1h_input_tokens=request.cache_creation_input_tokens_1h,
            ),
        ),
    )


def _estimate_arm(
    request: CacheSwitchCostEstimateArmRequest, billing_time: datetime
) -> CacheSwitchCostEstimateArmResponse:
    from litellm.proxy.proxy_server import llm_router

    resolved: Final = _resolved_deployment(request.model, request.model_id)
    usage: Final = _usage(request)
    model_info: Final = (
        llm_router.get_deployment_model_info(resolved.model_id, resolved.model)
        if llm_router is not None and resolved.model_id is not None
        else None
    )
    rates: Final = (
        _cost_map_billed_rates(
            model_info=model_info,
            usage=usage,
            custom_llm_provider=resolved.provider,
            service_tier=None,
            data_residency=None,
            vertex_location=None,
            current_time=billing_time,
        )
        if model_info is not None
        else get_billed_token_rates(
            model=resolved.model,
            custom_llm_provider=resolved.provider,
            usage=usage,
            current_time=billing_time,
        )
    )
    if rates is None:
        raise HTTPException(
            status_code=404,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not price model '{request.model}'"
            },
        )
    uncached: Final = request.uncached_input_tokens * rates.input_cost_per_token
    cache_read: Final = request.cache_read_input_tokens * rates.cache_read_input_token_cost
    write_5m: Final = request.cache_creation_input_tokens_5m * rates.cache_creation_input_token_cost
    write_1h: Final = request.cache_creation_input_tokens_1h * rates.cache_creation_input_token_cost_above_1hr
    return CacheSwitchCostEstimateArmResponse(
        **request.model_dump(),
        resolved_model=resolved.model,
        resolved_model_id=resolved.model_id,
        provider=resolved.provider,
        input_tokens=usage.prompt_tokens,
        input_cost=uncached + cache_read + write_5m + write_1h,
        uncached_input_cost=uncached,
        cache_read_input_cost=cache_read,
        cache_creation_input_cost_5m=write_5m,
        cache_creation_input_cost_1h=write_1h,
        input_cost_per_token=rates.input_cost_per_token,
        cache_read_input_token_cost=rates.cache_read_input_token_cost,
        cache_creation_input_token_cost_5m=rates.cache_creation_input_token_cost,
        cache_creation_input_token_cost_1h=rates.cache_creation_input_token_cost_above_1hr,
    )


@router.post(
    "/cost/estimate/cache-switch",
    tags=("Cost Tracking",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=CacheSwitchCostEstimateResponse,
)
async def estimate_cache_switch_cost(
    request: CacheSwitchCostEstimateRequest,
    user_api_key_dict: UserAPIKeyAuth = _USER_API_KEY_AUTH_DEPENDENCY,
) -> CacheSwitchCostEstimateResponse:
    billing_time: Final = current_billing_time()
    stay: Final = _estimate_arm(request.stay, billing_time)
    switch: Final = _estimate_arm(request.switch, billing_time)
    return CacheSwitchCostEstimateResponse(
        stay=stay,
        switch=switch,
        switch_cost_delta=switch.input_cost - stay.input_cost,
    )
