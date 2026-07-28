"""Encrypted Axon endpoint commitments for opt-in miner IP protection."""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

from poker44.utils.runtime_info import build_signed_runtime_request


MASKED_AXON_IP = "192.0.2.1"
MASKED_AXON_PORT = 1234
MAX_COMMITMENT_BYTES = 128
COMMITMENT_MAGIC = "P44E1"
COMMITMENT_VERIFY_ATTEMPTS = 5
COMMITMENT_VERIFY_DELAY_SECONDS = 2.0

PROTECTION_ENABLED_ENV = "POKER44_ENCRYPTED_AXON_ENABLED"
PUBLIC_KEY_ENV = "POKER44_ENDPOINT_PUBLIC_KEY"
PRIVATE_KEY_ENV = "POKER44_ENDPOINT_PRIVATE_KEY"
PRIVATE_KEY_FILE_ENV = "POKER44_ENDPOINT_PRIVATE_KEY_FILE"
EXTERNAL_IP_ENV = "POKER44_AXON_EXTERNAL_IP"
EXTERNAL_PORT_ENV = "POKER44_AXON_EXTERNAL_PORT"
REFRESH_SECONDS_ENV = "POKER44_ENDPOINT_REFRESH_SECONDS"
AUTO_PROVISION_ENV = "POKER44_ENDPOINT_AUTO_PROVISION"
PROVISIONING_URL_ENV = "POKER44_ENDPOINT_PROVISIONING_URL"
PROVISIONING_CACHE_FILE_ENV = "POKER44_ENDPOINT_CACHE_FILE"
PROVISIONING_TIMEOUT_ENV = "POKER44_ENDPOINT_PROVISIONING_TIMEOUT_SECONDS"

DEFAULT_PROVISIONING_URL = (
    "https://api.poker44.net/internal/validators/runtime/endpoint-key"
)
PROVISIONING_RESPONSE_LIMIT_BYTES = 64 * 1024
PROVISIONING_ENVELOPE_VERSION = 1
PROVISIONING_ALGORITHM = "curve25519-xsalsa20-poly1305"

DEFAULT_PUBLIC_KEYS: Dict[int, str] = {
    126: "3fa48ff09c0c4c55473122600cc85c47bc09f4c6e7c2f3ed2554183e82d99344",
}
EXPECTED_PROVISIONED_KEY_FINGERPRINTS: Dict[int, str] = {
    126: "d4a77a56d68268df",
}


class EndpointProtectionError(RuntimeError):
    """Raised when endpoint protection cannot be configured safely."""


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_enabled_by_default(name: str) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decode_key(value: str, label: str) -> bytes:
    try:
        key = bytes.fromhex(value.strip())
    except ValueError as exc:
        raise EndpointProtectionError(f"{label} must be hexadecimal") from exc
    if len(key) != 32:
        raise EndpointProtectionError(f"{label} must contain exactly 32 bytes")
    return key


def _private_key_hex_from_env() -> str:
    inline_key = os.getenv(PRIVATE_KEY_ENV, "").strip()
    key_path = os.getenv(PRIVATE_KEY_FILE_ENV, "").strip()
    if inline_key and key_path:
        raise EndpointProtectionError(
            f"Configure only one of {PRIVATE_KEY_ENV} or {PRIVATE_KEY_FILE_ENV}"
        )
    if not key_path:
        return inline_key

    try:
        key_stat = os.stat(key_path)
        if not stat.S_ISREG(key_stat.st_mode):
            raise EndpointProtectionError(
                f"{PRIVATE_KEY_FILE_ENV} must reference a regular file"
            )
        if stat.S_IMODE(key_stat.st_mode) & 0o077:
            raise EndpointProtectionError(
                f"{PRIVATE_KEY_FILE_ENV} must not be accessible by group or others"
            )
        with open(key_path, encoding="ascii") as key_file:
            return key_file.read().strip()
    except EndpointProtectionError:
        raise
    except (OSError, UnicodeError) as exc:
        raise EndpointProtectionError(
            f"{PRIVATE_KEY_FILE_ENV} could not be read"
        ) from exc


def _validate_provisioning_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise EndpointProtectionError(
            f"{PROVISIONING_URL_ENV} must use an HTTPS URL without credentials"
        )
    return value.strip().rstrip("/")


def _secure_key_file_hex(path: Path, label: str) -> str:
    try:
        key_lstat = path.lstat()
        if not stat.S_ISREG(key_lstat.st_mode) or stat.S_ISLNK(key_lstat.st_mode):
            raise EndpointProtectionError(f"{label} must reference a regular file")
        if stat.S_IMODE(key_lstat.st_mode) & 0o077:
            raise EndpointProtectionError(
                f"{label} must not be accessible by group or others"
            )
        if hasattr(os, "geteuid") and key_lstat.st_uid != os.geteuid():
            raise EndpointProtectionError(f"{label} must be owned by the validator user")

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            key_fstat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(key_fstat.st_mode)
                or key_fstat.st_dev != key_lstat.st_dev
                or key_fstat.st_ino != key_lstat.st_ino
            ):
                raise EndpointProtectionError(f"{label} changed while being opened")
            with os.fdopen(descriptor, "r", encoding="ascii", closefd=False) as key_file:
                value = key_file.read(256).strip()
        finally:
            os.close(descriptor)
        _decode_key(value, label)
        return value
    except EndpointProtectionError:
        raise
    except (OSError, UnicodeError) as exc:
        raise EndpointProtectionError(f"{label} could not be read") from exc


def _write_private_key_cache(path: Path, private_key_hex: str) -> None:
    _decode_key(private_key_hex, "Provisioned endpoint private key")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temp_path, flags, 0o600)
        try:
            encoded = f"{private_key_hex.strip()}\n".encode("ascii")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise EndpointProtectionError(
                        "Provisioned endpoint key cache could not be written"
                    )
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _provisioning_cache_path(cache_path: Optional[Path]) -> Optional[Path]:
    configured = os.getenv(PROVISIONING_CACHE_FILE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return cache_path


def _provision_endpoint_private_key(
    *,
    wallet: Any,
    netuid: int,
    cache_path: Path,
) -> Tuple[str, str]:
    try:
        from nacl.public import Box, PrivateKey, PublicKey
    except ImportError as exc:
        raise EndpointProtectionError(
            "PyNaCl is required for endpoint key provisioning"
        ) from exc

    url = _validate_provisioning_url(
        os.getenv(PROVISIONING_URL_ENV, DEFAULT_PROVISIONING_URL)
    )
    timeout_text = os.getenv(PROVISIONING_TIMEOUT_ENV, "5").strip()
    try:
        timeout_seconds = min(15.0, max(1.0, float(timeout_text)))
    except ValueError:
        timeout_seconds = 5.0

    transport_private_key = PrivateKey.generate()
    payload = {
        "netuid": int(netuid),
        "transport_public_key": bytes(transport_private_key.public_key).hex(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    signed = build_signed_runtime_request(
        wallet=wallet,
        url=url,
        payload=payload,
        method="POST",
    )
    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "x-validator-hotkey": signed["hotkey_ss58"],
            "x-validator-signature": signed["signature_hex"],
            "x-validator-nonce": signed["nonce"],
            "x-validator-timestamp": str(signed["timestamp"]),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if not 200 <= int(getattr(response, "status", 200)) < 300:
                raise EndpointProtectionError("Provisioning server rejected the request")
            raw = response.read(PROVISIONING_RESPONSE_LIMIT_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise EndpointProtectionError(
            f"Provisioning server returned HTTP {exc.code}"
        ) from exc
    except EndpointProtectionError:
        raise
    except Exception as exc:
        raise EndpointProtectionError("Provisioning server could not be reached") from exc

    if len(raw) > PROVISIONING_RESPONSE_LIMIT_BYTES:
        raise EndpointProtectionError("Provisioning response is too large")
    try:
        decoded = json.loads(raw.decode("utf-8"))
        envelope = decoded["data"] if decoded.get("success") is True else decoded
        if (
            int(envelope["version"]) != PROVISIONING_ENVELOPE_VERSION
            or envelope["algorithm"] != PROVISIONING_ALGORITHM
        ):
            raise ValueError("unsupported envelope")
        server_public_key = _decode_key(
            envelope["server_public_key"], "Provisioning server public key"
        )
        nonce = bytes.fromhex(envelope["nonce"])
        ciphertext = bytes.fromhex(envelope["ciphertext"])
        if len(nonce) != Box.NONCE_SIZE:
            raise ValueError("invalid nonce")
        plaintext = Box(
            transport_private_key,
            PublicKey(server_public_key),
        ).decrypt(ciphertext, nonce)
        key_record = json.loads(plaintext.decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise EndpointProtectionError("Provisioning response is invalid") from exc
    except Exception as exc:
        raise EndpointProtectionError("Provisioning response could not be decrypted") from exc

    hotkey = str(wallet.hotkey.ss58_address)
    if (
        int(key_record.get("version", 0)) != PROVISIONING_ENVELOPE_VERSION
        or key_record.get("key_id") != envelope.get("key_id")
        or key_record.get("validator_hotkey") != hotkey
    ):
        raise EndpointProtectionError(
            "Provisioned endpoint key is not bound to this validator"
        )

    private_key_hex = str(key_record.get("private_key", ""))
    private_key = _decode_key(private_key_hex, "Provisioned endpoint private key")
    endpoint_public_key = bytes(PrivateKey(private_key).public_key)
    fingerprint = hashlib.sha256(endpoint_public_key).hexdigest()[:16]
    expected_fingerprint = EXPECTED_PROVISIONED_KEY_FINGERPRINTS.get(int(netuid), "")
    if (
        envelope.get("key_fingerprint") != fingerprint
        or (expected_fingerprint and fingerprint != expected_fingerprint)
    ):
        raise EndpointProtectionError(
            "Provisioned endpoint key fingerprint does not match the trusted release"
        )

    _write_private_key_cache(cache_path, private_key_hex)
    return private_key_hex, str(envelope.get("key_id", ""))


def _resolve_validator_private_key(
    *,
    wallet: Any,
    netuid: int,
    cache_path: Optional[Path],
) -> Tuple[str, str, str]:
    configured_key = _private_key_hex_from_env()
    if configured_key:
        return configured_key, "configured", ""

    resolved_cache_path = _provisioning_cache_path(cache_path)
    provisioning_error = ""
    if (
        wallet is not None
        and resolved_cache_path is not None
        and _env_enabled_by_default(AUTO_PROVISION_ENV)
    ):
        try:
            private_key_hex, key_id = _provision_endpoint_private_key(
                wallet=wallet,
                netuid=netuid,
                cache_path=resolved_cache_path,
            )
            return private_key_hex, f"provisioned:{key_id}", ""
        except EndpointProtectionError as exc:
            provisioning_error = str(exc)

    if resolved_cache_path is not None:
        try:
            cached_key = _secure_key_file_hex(
                resolved_cache_path,
                "Provisioned endpoint key cache",
            )
            private_key = _decode_key(cached_key, "Provisioned endpoint key cache")
            _, PrivateKey, _, _ = _load_nacl()
            fingerprint = hashlib.sha256(
                bytes(PrivateKey(private_key).public_key)
            ).hexdigest()[:16]
            expected = EXPECTED_PROVISIONED_KEY_FINGERPRINTS.get(int(netuid), "")
            if expected and fingerprint != expected:
                raise EndpointProtectionError(
                    "Provisioned endpoint key cache fingerprint is not trusted"
                )
            return cached_key, "cache", provisioning_error
        except EndpointProtectionError:
            pass

    return "", "disabled", provisioning_error


def _load_nacl():
    try:
        from nacl import exceptions as nacl_exceptions
        from nacl.public import PrivateKey, PublicKey, SealedBox
    except ImportError as exc:
        raise EndpointProtectionError(
            "PyNaCl is required when encrypted Axon endpoints are enabled"
        ) from exc
    return nacl_exceptions, PrivateKey, PublicKey, SealedBox


def _validate_endpoint(ip: str, port: int) -> Tuple[str, int]:
    try:
        parsed = ipaddress.ip_address(str(ip).strip())
    except ValueError as exc:
        raise EndpointProtectionError("Axon endpoint must use a valid IP address") from exc
    if parsed.version != 4 or not parsed.is_global:
        raise EndpointProtectionError(
            "Encrypted Axon endpoints currently require a public IPv4 address"
        )
    if not 1 <= int(port) <= 65535:
        raise EndpointProtectionError("Axon endpoint port must be between 1 and 65535")
    return str(parsed), int(port)


def encrypt_endpoint(
    ip: str,
    port: int,
    public_key: bytes,
    hotkey: str,
) -> bytes:
    """Encrypt a hotkey-bound endpoint using a NaCl sealed box."""
    ip, port = _validate_endpoint(ip, port)
    _, _, PublicKey, SealedBox = _load_nacl()
    plaintext = f"{COMMITMENT_MAGIC}|{hotkey}|{ip}:{port}".encode("utf-8")
    ciphertext = SealedBox(PublicKey(public_key)).encrypt(plaintext)
    if len(ciphertext) > MAX_COMMITMENT_BYTES:
        raise EndpointProtectionError(
            f"Encrypted endpoint exceeds the {MAX_COMMITMENT_BYTES}-byte chain limit"
        )
    return ciphertext


def decrypt_endpoint(
    ciphertext: bytes,
    private_key: bytes,
    expected_hotkey: str,
) -> Tuple[str, int]:
    """Decrypt an endpoint and reject commitments copied across hotkeys."""
    _, PrivateKey, _, SealedBox = _load_nacl()
    try:
        plaintext = SealedBox(PrivateKey(private_key)).decrypt(ciphertext).decode("utf-8")
    except Exception as exc:
        raise EndpointProtectionError("Endpoint commitment could not be decrypted") from exc

    parts = plaintext.split("|", 2)
    if len(parts) != 3 or parts[0] != COMMITMENT_MAGIC:
        raise EndpointProtectionError("Endpoint commitment has an invalid format")
    _, committed_hotkey, endpoint = parts
    if committed_hotkey != expected_hotkey:
        raise EndpointProtectionError("Endpoint commitment hotkey does not match its owner")
    try:
        ip, port_text = endpoint.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise EndpointProtectionError("Endpoint commitment has an invalid endpoint") from exc
    return _validate_endpoint(ip, port)


def publish_endpoint_commitment(
    subtensor: Any,
    wallet: Any,
    netuid: int,
    ciphertext: bytes,
) -> bool:
    """Publish encrypted endpoint metadata without masking on failure."""
    try:
        try:
            from bittensor.core.extrinsics.serving import (
                publish_metadata_extrinsic as publish_metadata,
            )
        except ImportError:
            from bittensor.core.extrinsics.serving import publish_metadata

        result = publish_metadata(
            subtensor=subtensor,
            wallet=wallet,
            netuid=int(netuid),
            data_type=f"Raw{len(ciphertext)}",
            data=ciphertext,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if isinstance(result, tuple):
            return bool(result[0])
        if hasattr(result, "success"):
            return bool(result.success)
        return result is not False
    except Exception:
        return False


def _unwrap(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _raw_field_to_bytes(value: Any) -> Optional[bytes]:
    value = _unwrap(value)
    if isinstance(value, str):
        try:
            return bytes.fromhex(value[2:] if value.startswith("0x") else value)
        except ValueError:
            return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        inner = value
        if len(value) == 1 and isinstance(
            value[0], (list, tuple, bytes, bytearray, str)
        ):
            inner = value[0]
        if isinstance(inner, str):
            try:
                return bytes.fromhex(inner[2:] if inner.startswith("0x") else inner)
            except ValueError:
                return None
        try:
            return bytes(inner)
        except (TypeError, ValueError):
            return None
    return None


def extract_commitment_ciphertext(commitment_data: Any) -> Optional[bytes]:
    """Read RawN metadata across supported substrate decoding shapes."""
    try:
        data = _unwrap(commitment_data)
        entry = data["info"]["fields"][0]
        if isinstance(entry, (list, tuple)):
            entry = entry[0]
        raw_key = next(iter(entry))
        ciphertext = _raw_field_to_bytes(entry[raw_key])
        if ciphertext is None or len(ciphertext) > MAX_COMMITMENT_BYTES:
            return None
        return ciphertext
    except (KeyError, IndexError, TypeError, StopIteration):
        return None


def _commitment_block(commitment_data: Any) -> int:
    data = _unwrap(commitment_data)
    if hasattr(data, "get"):
        try:
            return int(data.get("block", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _key_to_ss58(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        return value

    raw = value
    if (
        isinstance(value, tuple)
        and len(value) == 1
        and isinstance(value[0], (tuple, list, bytes, bytearray))
    ):
        raw = value[0]
    try:
        raw_bytes = bytes(raw)
    except (TypeError, ValueError) as exc:
        raise EndpointProtectionError("Unsupported commitment account key") from exc

    try:
        from bittensor_wallet import Keypair
    except ImportError as exc:
        raise EndpointProtectionError("Bittensor wallet SS58 encoder is unavailable") from exc
    return Keypair(public_key=raw_bytes.hex(), ss58_format=42).ss58_address


def read_endpoint_commitment(
    subtensor: Any,
    netuid: int,
    hotkey: str,
) -> Optional[bytes]:
    """Read one hotkey's endpoint commitment from finalized chain state."""
    try:
        rows = subtensor.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[int(netuid)],
        )
        for account_key, commitment_data in rows:
            try:
                account_hotkey = _key_to_ss58(account_key)
            except EndpointProtectionError:
                continue
            if account_hotkey == str(hotkey):
                return extract_commitment_ciphertext(commitment_data)
    except Exception:
        return None
    return None


def verify_endpoint_commitment(
    subtensor: Any,
    netuid: int,
    hotkey: str,
    expected_ciphertext: bytes,
    *,
    attempts: int = COMMITMENT_VERIFY_ATTEMPTS,
    delay_seconds: float = COMMITMENT_VERIFY_DELAY_SECONDS,
) -> bool:
    """Require an exact finalized read-back before a miner masks its Axon."""
    for attempt in range(max(1, int(attempts))):
        observed = read_endpoint_commitment(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
        )
        if observed == expected_ciphertext:
            return True
        if attempt + 1 < max(1, int(attempts)):
            time.sleep(max(0.0, float(delay_seconds)))
    return False


@dataclass(frozen=True)
class EndpointRecord:
    block: int
    ip: str
    port: int


class ValidatorEndpointResolver:
    """Resolve opt-in encrypted endpoints while preserving public Axons."""

    def __init__(
        self,
        subtensor: Any,
        netuid: int,
        private_key_hex: str,
        refresh_seconds: float = 300.0,
    ):
        self.subtensor = subtensor
        self.netuid = int(netuid)
        self.private_key = (
            _decode_key(private_key_hex, PRIVATE_KEY_ENV)
            if private_key_hex.strip()
            else None
        )
        self.key_fingerprint = ""
        if self.private_key is not None:
            _, PrivateKey, _, _ = _load_nacl()
            public_key = bytes(PrivateKey(self.private_key).public_key)
            self.key_fingerprint = hashlib.sha256(public_key).hexdigest()[:16]
        self.refresh_seconds = max(30.0, float(refresh_seconds))
        self._records: Dict[str, EndpointRecord] = {}
        self._last_refresh = 0.0
        self.last_refresh_succeeded: Optional[bool] = None
        self.last_refresh_error = ""
        self.key_source = "configured" if self.private_key is not None else "disabled"
        self.provisioning_error = ""

    @classmethod
    def from_env(
        cls,
        subtensor: Any,
        netuid: int,
        *,
        wallet: Any = None,
        cache_path: Optional[Path] = None,
    ) -> "ValidatorEndpointResolver":
        refresh_text = os.getenv(REFRESH_SECONDS_ENV, "300").strip()
        try:
            refresh_seconds = float(refresh_text)
        except ValueError:
            refresh_seconds = 300.0
        private_key_hex, key_source, provisioning_error = _resolve_validator_private_key(
            wallet=wallet,
            netuid=netuid,
            cache_path=cache_path,
        )
        resolver = cls(
            subtensor=subtensor,
            netuid=netuid,
            private_key_hex=private_key_hex,
            refresh_seconds=refresh_seconds,
        )
        resolver.key_source = key_source
        resolver.provisioning_error = provisioning_error
        return resolver

    @property
    def enabled(self) -> bool:
        return self.private_key is not None

    @property
    def protected_hotkeys(self) -> frozenset[str]:
        return frozenset(self._records)

    def public_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "key_fingerprint": self.key_fingerprint,
            "protected_miners": len(self._records),
            "last_refresh_succeeded": self.last_refresh_succeeded,
            "last_refresh_error": self.last_refresh_error,
            "key_source": self.key_source,
            "provisioning_error": self.provisioning_error,
        }

    def refresh(self, hotkeys: Iterable[str], force: bool = False) -> int:
        if not self.enabled:
            return 0
        now = time.monotonic()
        if not force and now - self._last_refresh < self.refresh_seconds:
            return len(self._records)
        self._last_refresh = now

        hotkey_set = {str(hotkey) for hotkey in hotkeys}
        try:
            rows = list(
                self.subtensor.query_map(
                    module="Commitments",
                    name="CommitmentOf",
                    params=[self.netuid],
                )
            )
        except Exception as exc:
            self.last_refresh_succeeded = False
            self.last_refresh_error = type(exc).__name__
            return len(self._records)

        records: Dict[str, EndpointRecord] = {}
        for account_key, commitment_data in rows:
            try:
                hotkey = _key_to_ss58(account_key)
            except EndpointProtectionError:
                continue
            if hotkey not in hotkey_set:
                continue

            block = _commitment_block(commitment_data)
            cached = self._records.get(hotkey)
            if cached is not None and cached.block == block:
                records[hotkey] = cached
                continue

            ciphertext = extract_commitment_ciphertext(commitment_data)
            if ciphertext is None:
                continue
            try:
                ip, port = decrypt_endpoint(
                    ciphertext,
                    self.private_key,
                    expected_hotkey=hotkey,
                )
            except EndpointProtectionError:
                continue
            records[hotkey] = EndpointRecord(block=block, ip=ip, port=port)

        self._records = records
        self.last_refresh_succeeded = True
        self.last_refresh_error = ""
        return len(records)

    def resolve(self, hotkey: str, axon: Any) -> Tuple[Any, bool]:
        if not is_masked_axon(axon):
            return axon, False
        record = self._records.get(str(hotkey))
        if record is None:
            return axon, False
        resolved = copy.copy(axon)
        resolved.ip = record.ip
        resolved.port = record.port
        return resolved, True


def is_masked_axon(axon: Any) -> bool:
    return (
        str(getattr(axon, "ip", "") or "") == MASKED_AXON_IP
        and int(getattr(axon, "port", 0) or 0) == MASKED_AXON_PORT
    )


def _miner_external_endpoint(axon: Any) -> Tuple[str, int]:
    ip = os.getenv(EXTERNAL_IP_ENV, "").strip()
    if not ip:
        ip = str(
            getattr(axon, "external_ip", None)
            or getattr(axon, "ip", "")
            or ""
        ).strip()

    port_text = os.getenv(EXTERNAL_PORT_ENV, "").strip()
    if port_text:
        try:
            port = int(port_text)
        except ValueError as exc:
            raise EndpointProtectionError(
                f"{EXTERNAL_PORT_ENV} must be an integer"
            ) from exc
    else:
        port = int(
            getattr(axon, "external_port", None)
            or getattr(axon, "port", 0)
            or 0
        )
    return _validate_endpoint(ip, port)


def enable_miner_endpoint_protection(miner: Any) -> bool:
    """Publish the real endpoint and mask it only after confirmed publication."""
    if not _env_enabled(PROTECTION_ENABLED_ENV):
        return False

    netuid = int(miner.config.netuid)
    public_key_hex = os.getenv(PUBLIC_KEY_ENV, "").strip()
    if not public_key_hex:
        public_key_hex = DEFAULT_PUBLIC_KEYS.get(netuid, "")
    if not public_key_hex:
        raise EndpointProtectionError(
            f"No encrypted endpoint public key is configured for netuid {netuid}"
        )

    public_key = _decode_key(public_key_hex, PUBLIC_KEY_ENV)
    ip, port = _miner_external_endpoint(miner.axon)
    hotkey = miner.wallet.hotkey.ss58_address
    ciphertext = encrypt_endpoint(ip, port, public_key, hotkey)
    published = publish_endpoint_commitment(
        subtensor=miner.subtensor,
        wallet=miner.wallet,
        netuid=netuid,
        ciphertext=ciphertext,
    )
    if not published:
        return False
    if not verify_endpoint_commitment(
        subtensor=miner.subtensor,
        netuid=netuid,
        hotkey=hotkey,
        expected_ciphertext=ciphertext,
    ):
        return False

    miner.axon.external_ip = MASKED_AXON_IP
    miner.axon.external_port = MASKED_AXON_PORT
    return True
