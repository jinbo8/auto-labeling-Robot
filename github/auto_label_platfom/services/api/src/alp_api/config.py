from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[4]  # .../auto_label_platfom
DEFAULT_DB = ROOT / "data" / "alp.db"
DEFAULT_ARTIFACTS = ROOT / "data" / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALP_", env_file=".env", extra="ignore")

    app_name: str = "Auto Label Platform"
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60 * 24
    database_url: str = f"sqlite:///{DEFAULT_DB}"
    artifacts_dir: Path = DEFAULT_ARTIFACTS
    sam2_url: str = "http://127.0.0.1:8102"
    sam3_url: str = "http://127.0.0.1:8103"
    default_dataset: Path | None = None
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"


settings = Settings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
Path(DEFAULT_DB).parent.mkdir(parents=True, exist_ok=True)
