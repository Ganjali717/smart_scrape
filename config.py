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

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2Mzg1NDk3OCwiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjM4NjIxNzgsImlhdCI6MTc2Mzg1NDk3OCwianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.MSW4QxsKVmW-LLcT8ETZXWJVjDOCDBtAE_Kg4ZEWeCsrSH3nGH0J8o0vEj4zAVhtc8z6qC3KSJWjK8al1P26oToECGbkUTwApnf9I6jqlAX-BymppYHcMvbpMA-RwHORLDnhLuNr0XpU42ahyeC5dHfqcoWna5209sxwX1oBy_gtTaSI2zRUeGX0C1_QuMcIlDv4dGWNhMIuetZxPsXqtAMyB5sRj7Tc7fqOMWuYp07HG0diy7nE-TUuiXYZOSxgm1yIuyhi8aAS3kDapaAPpFIhLrXi6GzcrO3KQJk0Mxk_nEIh4pKasgA4_cCohUWNy0ek2ww3dn3qsZX-EXucKQ"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
