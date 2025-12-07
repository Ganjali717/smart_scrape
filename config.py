"""
Configuration file for SmartScrape.
Centralizes all environment variables and service identifiers.
"""

# --- FITLAYOUT CONNECTION SETTINGS ---
# Базовый URL API. Проверь порт (обычно 8080)
API_BASE_URL = "https://layout.fit.vutbr.cz/api"

# Имя твоего репозитория, которое мы видели на скриншоте
TARGET_REPOSITORY_NAME = "IE AI"

# --- SERVICE IDENTIFIERS ---
# ID сервисов могут меняться. Если код упадет, мы проверим их через getServiceList.
# Обычно для Puppeteer ID именно такой:
SERVICE_RENDER_ID = "FitLayout.Puppeteer"

# Для сегментации (разбиения на блоки). VIPS - самый надежный вариант.
SERVICE_SEGMENTATION_ID = "FitLayout.VIPS"

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2NTA5MzE0MywiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjUxMDAzNDMsImlhdCI6MTc2NTA5MzE0MywianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.vUuRazlP7m07eA7mod-XszciOW0Ij7_U0Ohv4n2jd7SjKOR2LReE3uapRHtz7nbAzFinbtJJTE-Jn6Ij9Bd-ssDtJLPWoIAIvCmF4IJGKn5sUPCR_DEjrHcUImmtHaLhqFA6rBCiU--Qhx1G5QVmAupE8ETj3E_9HW1HBs8oobINqyEtofo-cqnGMHLNJhksjNU5uwLi4H_EcqCdMCcxjG-sPpQ--BvuMv5J-AxlVa3uSkKAGynxKeH6xTBzFrutnWynX8myxnrdXbpfH9qNC-iyofEYvGNiKERQVnFB3KLXCYckVu7NRz2_9QDwTQrwGmcRlMV5suLR7T8VjxxJuw"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
