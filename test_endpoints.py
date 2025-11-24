#!/usr/bin/env python3
"""
Quick script to test the Modal endpoints and understand their API structure.
"""
import requests
import json
import sys

LLAMA_GUARD_BASE = "https://smpnet74-1--llama-guard-3-8b-serve.modal.run"
LLAMA_INSTRUCT_BASE = "https://smpnet74-1--llama-3-8b-instruct-serve.modal.run"
PASSWORD = "h3110w0r1d"

def test_endpoint(base_url, name, payload):
    """Test an endpoint with different paths and auth methods."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}")

    # Try different URL paths - prioritize /v1
    paths = [
        "/v1",
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/generate",
        "/v1/inference",
        "/",
        "/generate",
        "/inference",
        "/predict",
    ]

    # Try different authentication methods
    auth_configs = [
        ("Bearer token", {"headers": {"Authorization": f"Bearer {PASSWORD}"}}),
        ("API Key header", {"headers": {"X-API-Key": PASSWORD}}),
        ("Password header", {"headers": {"Password": PASSWORD}}),
        ("Basic auth", {"auth": ("user", PASSWORD)}),
        ("In payload", {}),  # password will be added to payload
    ]

    for path in paths:
        url = f"{base_url}{path}"
        print(f"\n--- Trying path: {path or '(root)'} ---")

        for auth_name, auth_config in auth_configs:
            # For the last auth method, add password to payload
            test_payload = payload.copy()
            if auth_name == "In payload":
                test_payload["password"] = PASSWORD

            try:
                response = requests.post(url, json=test_payload, timeout=10, **auth_config)

                if response.status_code != 404:
                    print(f"  {auth_name}: Status {response.status_code}")

                    if response.status_code == 200:
                        print(f"  ✓ SUCCESS!")
                        print(f"  Response preview: {response.text[:300]}")
                        try:
                            return {"url": url, "auth": auth_name, "response": response.json()}
                        except:
                            return {"url": url, "auth": auth_name, "response": response.text}
                    elif response.status_code in [401, 403]:
                        print(f"  Auth failed but endpoint exists!")
                    else:
                        print(f"  Response: {response.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"  {auth_name}: Timeout")
            except Exception as e:
                print(f"  {auth_name}: Error - {type(e).__name__}")

    return None

if __name__ == "__main__":
    # Test payloads - try different formats
    payloads_to_test = [
        {
            "prompt": "Hello, how are you?",
            "max_tokens": 100
        },
        {
            "inputs": "Hello, how are you?",
            "parameters": {"max_new_tokens": 100}
        },
        {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100
        }
    ]

    print("Testing with different payload formats...")

    guard_result = None
    for i, payload in enumerate(payloads_to_test, 1):
        print(f"\n\n{'#'*60}")
        print(f"PAYLOAD FORMAT {i}: {list(payload.keys())}")
        print(f"{'#'*60}")
        guard_result = test_endpoint(LLAMA_GUARD_BASE, f"Llama Guard (Format {i})", payload)
        if guard_result:
            print(f"\n✓✓✓ Found working configuration for Guard! ✓✓✓")
            break

    instruct_result = None
    if guard_result:
        # Use the same format that worked for guard
        working_payload = payloads_to_test[i-1]
    else:
        working_payload = payloads_to_test[0]

    for i, payload in enumerate(payloads_to_test, 1):
        print(f"\n\n{'#'*60}")
        print(f"PAYLOAD FORMAT {i}: {list(payload.keys())}")
        print(f"{'#'*60}")
        instruct_result = test_endpoint(LLAMA_INSTRUCT_BASE, f"Llama Instruct (Format {i})", payload)
        if instruct_result:
            print(f"\n✓✓✓ Found working configuration for Instruct! ✓✓✓")
            break

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if guard_result:
        print(f"✓ Guard endpoint: {guard_result['url']}")
        print(f"  Auth method: {guard_result['auth']}")
    else:
        print("✗ Guard endpoint: Failed to connect")

    if instruct_result:
        print(f"✓ Instruct endpoint: {instruct_result['url']}")
        print(f"  Auth method: {instruct_result['auth']}")
    else:
        print("✗ Instruct endpoint: Failed to connect")

    # Save configuration if successful
    if guard_result or instruct_result:
        config = {
            "guard": guard_result if guard_result else None,
            "instruct": instruct_result if instruct_result else None,
        }
        with open("endpoint_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("\n✓ Saved working configuration to endpoint_config.json")
