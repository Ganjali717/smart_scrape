import requests
import json
from typing import Dict, Any, Optional
from urllib.parse import quote

from config import (
    API_BASE_URL,
    TARGET_REPOSITORY_NAME,
    SERVICE_RENDER_ID,
    SERVICE_SEGMENTATION_ID,
    SERVICE_AUTH_TOKEN,
)


class FitLayoutClient:
    """
    Клиент для FitLayout REST API:
    - находит репозиторий по description
    - ищет артефакт по b:sourceUrl
    - при необходимости создаёт артефакт
    - возвращает содержимое артефакта
    """

    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.repo_name = TARGET_REPOSITORY_NAME
        self.repo_id: Optional[str] = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "*/*",
                "Authorization": SERVICE_AUTH_TOKEN,
            }
        )

    # --------- internal helpers ---------

    def _ensure_repo(self):
        if not self.repo_id:
            self.get_repo_id()
        if not self.repo_id:
            raise RuntimeError("[FitLayout] Repository not found by description")

    # --------- repository ---------

    def get_repo_id(self):
        try:
            url = f"{self.base_url}/repository"
            response = self.session.get(url)
            response.raise_for_status()

            repos = response.json()
            for repo in repos:
                if repo.get("description") == self.repo_name:
                    self.repo_id = repo.get("id")
                    return self.repo_id

            print(
                f"[FitLayout] Repository with description '{self.repo_name}' not found"
            )

        except requests.RequestException as e:
            print(f"[FitLayout] Connection failed: {e}")

        return None

    # --------- artifacts search ---------

    def check_artifact(self, page_url: str) -> Optional[str]:
        """
        Ищет артефакт с данным b:sourceUrl.
        Возвращает полный URI артефакта (http://fitlayout.github.io/resource/artXX) или None.
        """
        self._ensure_repo()

        try:
            url = f"{self.base_url}/r/{self.repo_id}/artifact"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            graphs = data.get("@graph", [])

            latest_match_uri: Optional[str] = None

            for graph in graphs:
                inner_graphs = graph.get("@graph", [])
                for inner in inner_graphs:
                    base_url = inner.get("b:sourceUrl")
                    if base_url != page_url:
                        continue

                    inner_id = inner.get("@id")
                    if inner_id and inner_id.startswith("r:"):
                        suffix = inner_id.split(":", 1)[1]
                        latest_match_uri = (
                            f"http://fitlayout.github.io/resource/{suffix}"
                        )
                    else:
                        latest_match_uri = inner_id
            return latest_match_uri

        except requests.RequestException as e:
            print(f"[FitLayout] Connection failed: {e}")
            return None

    # --------- artifact creation ---------

    def create_artifact(self, service_id: str, page_url: str) -> str:
        """
        Создаёт артефакт через /artifact/create.
        Предполагаем, что сервис принимает параметры через строку params (как в fl:creatorParams).
        Возвращает полный URI артефакта (result).
        """
        self._ensure_repo()

        try:
            url = f"{self.base_url}/r/{self.repo_id}/artifact/create"

            payload = {
                "serviceId": service_id,
                "params": {
                    "acquireImages": False,
                    "width": 1200,
                    "persist": 1,
                    "includeScreenshot": True,
                    "height": 800,
                    "url": page_url,
                    "startPage": 0,
                    "zoom": 1,
                    "endPage": 10,
                },
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": SERVICE_AUTH_TOKEN,
            }

            print(payload)

            # если у тебя create работает через GET с query-параметрами — можно заменить на params=payload
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            if data.get("status") != "ok":
                raise RuntimeError(f"[FitLayout] Artifact creation failed: {data}")

            return data["result"]  # типа "http://fitlayout.github.io/resource/art66"

        except requests.RequestException as e:
            raise RuntimeError(f"[FitLayout] Connection failed: {e}") from e

    # --------- artifact content ---------

    def get_artifact_content(self, artifact_uri: str) -> Dict[str, Any]:
        """
        Возвращает JSON-содержимое артефакта по его полному URI.
        """
        self._ensure_repo()

        try:
            encoded = quote(artifact_uri, safe="")
            url = f"{self.base_url}/r/{self.repo_id}/artifact/item/{encoded}"
            self.session.headers.update(
                {
                    "Accept": "application/ld+json",
                    "Authorization": SERVICE_AUTH_TOKEN,
                }
            )
            response = self.session.get(url)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            raise RuntimeError(f"[FitLayout] Connection failed: {e}") from e

    # --------- high-level API ---------

    def get_page_content(
        self, page_url: str, service_id: str = SERVICE_RENDER_ID
    ) -> Dict[str, Any]:
        """
        1) Проверяет, есть ли артефакт для page_url.
        2) Если есть — берёт его содержимое.
        3) Если нет — создаёт артефакт указанным service_id и потом берёт содержимое.
        """
        artifact_uri = self.check_artifact(page_url)
        if not artifact_uri:
            artifact_uri = self.create_artifact(service_id, page_url)

        return self.get_artifact_content(artifact_uri)


if __name__ == "__main__":
    client = FitLayoutClient()
    url = "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html"
    content = client.get_page_content(url)
    print(json.dumps(content, indent=2, ensure_ascii=False))
